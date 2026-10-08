import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import torch
import cv2
import soundfile as sf
import imageio.v3 as iio

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.config import load_config, get_default_config_path
from vaultbreaker.inference.router import MediaRouter, RouterError
from vaultbreaker.features.extractor import MultiModalFeaturePipeline
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator
from vaultbreaker.explain import explain_image, explain_audio, explain_video, get_top_features

logger = get_logger("Predictor")

class VaultBreakerPredictor:
    """
    Unified Inference Engine.
    Inspects input media using magic bytes, extracts multi-modal features,
    applies format standardization, computes latent embedding and calibrated verdict,
    and produces forensic explainability diagnostics.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        models_dir: Union[str, Path] = "models_saved",
        device: Optional[torch.device] = None
    ):
        self.cfg_path = config_path or get_default_config_path()
        self.config = load_config(self.cfg_path)
        self.models_dir = Path(models_dir)
        self.device = device or torch.device("cpu")

        # 1. Router
        self.router = MediaRouter()

        # 2. Feature pipeline & Scalers
        self.feature_pipeline = MultiModalFeaturePipeline(
            cache_dir=Path(self.config.get("paths", {}).get("features_dir", "data/features")) / "cache",
            num_workers=1
        )
        self.scalers = self.feature_pipeline.load_scalers(self.models_dir)

        # 3. Model
        feat_cfg = self.config.get("features", {})
        model_cfg = self.config.get("models", {})

        self.model = VaultBreakerUnifiedNet(
            img_dim=feat_cfg.get("image_dim", 72),
            aud_dim=feat_cfg.get("audio_dim", 56),
            vid_dim=feat_cfg.get("video_dim", 64),
            latent_dim=model_cfg.get("latent_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            use_aux_head=model_cfg.get("use_aux_head", True),
            num_methods=8
        ).to(self.device)

        weights_path = self.models_dir / "unified_model.pt"
        if weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            logger.info(f"Loaded unified model from {weights_path}")
        else:
            logger.warning(f"Model weights not found at {weights_path}; running in unweighted mode.")
        self.model.eval()

        # 4. Calibrator
        cal_path = self.models_dir / "calibrator.joblib"
        if cal_path.exists():
            self.calibrator = TemperatureScalingCalibrator.load(cal_path)
        else:
            self.calibrator = TemperatureScalingCalibrator(target_fpr=model_cfg.get("target_fpr", 0.05))

    def predict_file(
        self,
        filepath: Union[str, Path],
        generate_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Scan a single file from disk.
        Returns complete forensic analysis payload.
        """
        p = Path(filepath)
        # 1. Magic byte inspection
        media_type, format_name = self.router.inspect_file(p)

        # 2. Extract features
        raw_vec = self.feature_pipeline.extract_single(p, media_type)
        norm_vec = self.feature_pipeline.transform_single(raw_vec, media_type, self.scalers)

        # 3. Model forward pass
        x_tensor = torch.from_numpy(norm_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit, aux_logits, z = self.model(x_tensor, media_type)
            raw_logit = float(logit.item())
            raw_prob = float(torch.sigmoid(logit).item())
            latent_code = z.squeeze(0).cpu().numpy().tolist()

        # 4. Calibrate probability and assign risk
        cal_prob = self.calibrator.calibrate_prob(raw_prob)
        verdict, risk, details = self.calibrator.get_verdict_and_risk(cal_prob)

        # 5. Top contributory features
        top_feats = get_top_features(norm_vec, media_type, top_k=6)

        report = {
            "file_name": p.name,
            "media_type": media_type,
            "format": format_name,
            "verdict": verdict,
            "risk_level": risk,
            "calibrated_probability": float(cal_prob),
            "raw_probability": float(raw_prob),
            "raw_logit": float(raw_logit),
            "decision_threshold": float(self.calibrator.decision_threshold),
            "temperature": float(self.calibrator.temperature),
            "top_features": top_feats,
            "latent_code_summary": {
                "norm": float(np.linalg.norm(latent_code)),
                "dim": len(latent_code)
            }
        }

        # 6. Optional media-specific explanation artifacts
        if generate_explanation:
            report["explanation"] = self._generate_visual_explanation(p, media_type)

        return report

    def _generate_visual_explanation(self, filepath: Path, media_type: str) -> Dict[str, Any]:
        """Build format-tailored visual explanation data."""
        try:
            if media_type == "image":
                img = cv2.imread(str(filepath))
                if img is not None:
                    return explain_image(img)
            elif media_type == "audio":
                samples, sr = sf.read(str(filepath), dtype="int16")
                if samples.ndim > 1:
                    samples = samples[:, 0]
                return explain_audio(samples, sr=sr)
            elif media_type == "video":
                try:
                    frames = iio.imread(str(filepath), plugin="pyav")
                except Exception:
                    frames = iio.imread(str(filepath))
                return explain_video(frames)
        except Exception as e:
            logger.warning(f"Could not generate visual explanation: {e}")
        return {}
