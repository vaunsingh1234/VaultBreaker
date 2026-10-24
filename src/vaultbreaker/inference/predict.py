import os
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import torch
import cv2
import soundfile as sf
import imageio.v3 as iio
import joblib

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.config import load_config, get_default_config_path
from vaultbreaker.inference.router import MediaRouter, RouterError
from vaultbreaker.features.extractor import MultiModalFeaturePipeline
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.models.deep_net import VaultBreakerDeepNet
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator, load_format_calibrators
from vaultbreaker.explain import explain_image, explain_audio, explain_video, get_top_features

from vaultbreaker.inference.structure_check import check_structure

logger = get_logger("Predictor")

class VaultBreakerPredictor:
    """
    Unified Inference Engine with Format-Optimal Model Selection (Item 5).
    Inspects input media using magic bytes, selects the best model per format
    according to validation set ROC-AUC (Deep Model, GBDT, or Handcrafted MLP),
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

        # 2. Feature pipeline & Scalers (used for GBDT, handcrafted MLP, and explainability)
        self.feature_pipeline = MultiModalFeaturePipeline(
            cache_dir=Path(self.config.get("paths", {}).get("features_dir", "data/features")) / "cache",
            num_workers=1
        )
        self.scalers = self.feature_pipeline.load_scalers(self.models_dir)

        # 3. Validation Model Selection Decisions (Item 5)
        self.model_selections: Dict[str, Any] = {}
        selection_file = self.models_dir / "model_selection.json"
        if selection_file.exists():
            try:
                with open(selection_file, "r") as f:
                    self.model_selections = json.load(f).get("formats", {})
                logger.info(f"Loaded validation model selection decisions from {selection_file}")
            except Exception as e:
                logger.warning(f"Could not read model_selection.json: {e}")

        # 4. Gradient Boosting Baselines
        self.gb_models: Dict[str, Any] = {}
        for fmt in ["image", "audio", "video"]:
            gb_path = self.models_dir / f"baseline_{fmt}_GradientBoosting.joblib"
            if gb_path.exists():
                try:
                    self.gb_models[fmt] = joblib.load(gb_path)
                    logger.info(f"Loaded Gradient Boosting baseline for {fmt}")
                except Exception as e:
                    logger.warning(f"Could not load GBDT for {fmt}: {e}")

        # 5. Deep Model
        deep_path = self.models_dir / "deep_model.pt"
        self.is_deep_model = False
        self.model_name = "UnifiedMLP_Fallback"
        self.checkpoint_date = None
        self.deep_model = None

        if deep_path.exists():
            try:
                ckpt = torch.load(deep_path, map_location=self.device)
                model_state = ckpt.get("model_state", ckpt)
                self.checkpoint_date = ckpt.get("timestamp", None)
                self.model_name = "VaultBreakerDeepNet"

                # Infer base_channels from state dict if available
                base_ch = 16
                if "image_deep.block1.0.weight" in model_state:
                    base_ch = model_state["image_deep.block1.0.weight"].shape[0]

                self.deep_model = VaultBreakerDeepNet(
                    base_channels=base_ch,
                    latent_dim=128,
                    use_fusion=False,
                    dropout=0.0
                ).to(self.device)
                self.deep_model.load_state_dict(model_state)
                self.deep_model.eval()
                self.is_deep_model = True
                logger.info(f"Loaded VaultBreakerDeepNet from {deep_path} (timestamp: {self.checkpoint_date})")
            except Exception as e:
                logger.warning(f"Failed to load deep model from {deep_path}: {e}. Falling back to unified MLP.")
                self.is_deep_model = False

        # 6. Handcrafted Unified Net (MLP on handcrafted features)
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
        self.model.eval()

        # 7. Format-Specific Calibrators
        deep_cal_path = self.models_dir / "deep_calibrator.pt"
        self.calibrators = load_format_calibrators(deep_cal_path)
        # Default fallback calibrator
        self.default_calibrator = self.calibrators.get("image", TemperatureScalingCalibrator(target_fpr=0.05))

    @property
    def calibrator(self):
        """Backwards compatibility accessor for default calibrator."""
        return self.default_calibrator

    def predict_file(
        self,
        filepath: Union[str, Path],
        generate_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Scan a single file from disk.
        Returns complete forensic analysis payload including rule-based structure check.
        """
        p = Path(filepath)

        # 1. Rule-based structure check (works across all formats)
        structure_res = check_structure(p)
        structure_findings = structure_res.get("finding_messages", [])
        is_suspicious_structure = structure_res.get("is_suspicious", False)

        # 2. Magic byte inspection & routing
        try:
            media_type, format_name = self.router.inspect_file(p)
        except RouterError:
            # Format cannot be inspected by statistical router -> Unsupported verdict
            return {
                "file_name": p.name,
                "media_type": "unknown",
                "format": structure_res.get("format", "unsupported"),
                "model_name": self.model_name,
                "checkpoint_date": self.checkpoint_date,
                "verdict": "Unsupported",
                "risk_level": "UNKNOWN",
                "notice": "Unsupported format for statistical detector; structure check only.",
                "structure_findings": structure_findings,
                "structure_check": structure_res,
                "calibrated_probability": None,
                "raw_probability": None,
                "raw_logit": None,
                "decision_threshold": float(self.default_calibrator.decision_threshold),
                "temperature": float(self.default_calibrator.temperature),
                "top_features": [],
                "latent_code_summary": None,
                "explanation": {}
            }

        # 3. Lossy inputs (JPEG, MP3, MP4): structure check only
        if format_name.lower() in ("jpeg", "mp3", "mp4"):
            notice = "The statistical detector is not reliable on compressed files; structure check only"
            if is_suspicious_structure:
                verdict = "SUSPECTED STEGO"
                risk = "HIGH RISK"
            else:
                verdict = "No hidden data detected"
                risk = "BENIGN / CLEAN"

            cal = self.calibrators.get(media_type, self.default_calibrator)
            report = {
                "score": 0.0,
                "verdict": verdict,
                "threshold": float(cal.decision_threshold),
                "model_used": "structure_check_only",
                "model_name": "Structure Guardrail (Format Compressed)",
                "model_selection_reason": "Compressed container; statistical models disabled to avoid false positives",
                "file_name": p.name,
                "media_type": media_type,
                "format": format_name,
                "checkpoint_date": self.checkpoint_date,
                "risk_level": risk,
                "notice": notice,
                "structure_findings": structure_findings,
                "structure_check": structure_res,
                "calibrated_probability": None,
                "raw_probability": None,
                "raw_logit": None,
                "decision_threshold": float(cal.decision_threshold),
                "temperature": float(cal.temperature),
                "top_features": [],
                "latent_code_summary": None
            }
            if generate_explanation:
                report["explanation"] = self._generate_visual_explanation(p, media_type)
            return report

        # 4. Uncompressed/lossless media: Route to best model per format (Item 5)
        sel_info = self.model_selections.get(media_type, {})
        selected_model = sel_info.get("selected_model", "deep_model" if self.is_deep_model else "handcrafted_mlp")
        model_selection_reason = sel_info.get("reason", "Unified model pipeline")

        cal = self.calibrators.get(media_type, self.default_calibrator)
        decision_threshold = float(cal.decision_threshold)
        temperature = float(cal.temperature)

        norm_vec = None

        if selected_model == "gradient_boosting" and media_type in self.gb_models and media_type in self.scalers:
            # Gradient Boosting Pipeline
            raw_vec = self.feature_pipeline.extract_single(p, media_type)
            norm_vec = self.feature_pipeline.transform_single(raw_vec, media_type, self.scalers)
            gb = self.gb_models[media_type]
            raw_prob = float(gb.predict_proba(norm_vec.reshape(1, -1))[0, 1])
            raw_logit = float(np.log(max(1e-6, raw_prob) / max(1e-6, 1.0 - raw_prob)))
            cal_prob = raw_prob
            model_used = "gradient_boosting"
            model_display_name = "Gradient Boosting (GBDT)"
            latent_code = [0.0] * 128
        elif selected_model == "handcrafted_mlp" or (not self.is_deep_model and self.model is not None):
            # Handcrafted MLP Pipeline
            raw_vec = self.feature_pipeline.extract_single(p, media_type)
            norm_vec = self.feature_pipeline.transform_single(raw_vec, media_type, self.scalers)
            x_tensor = torch.from_numpy(norm_vec).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit, aux_logits, z = self.model(x_tensor, media_type)
                raw_logit = float(logit.item())
                raw_prob = float(torch.sigmoid(logit).item())
                latent_code = z.squeeze(0).cpu().numpy().tolist()
            cal_prob = cal.calibrate_prob(raw_prob)
            model_used = "handcrafted_mlp"
            model_display_name = self.model_name
        else:
            # Deep Model Pipeline
            if media_type == "image":
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    from PIL import Image
                    with Image.open(p) as pil_img:
                        img = np.array(pil_img.convert("L"))
                if img.shape != (256, 256):
                    img = cv2.resize(img, (256, 256))
                x_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.device)
            elif media_type == "audio":
                samples, _ = sf.read(str(p), dtype="int16")
                if samples.ndim > 1:
                    samples = samples[:, 0]
                if len(samples) < 48000:
                    samples = np.pad(samples, (0, 48000 - len(samples)), mode="wrap")
                elif len(samples) > 48000:
                    samples = samples[:48000]
                x_tensor = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            else:  # video
                try:
                    frames = iio.imread(str(p), plugin="pyav")
                except Exception:
                    frames = iio.imread(str(p))
                if frames.ndim == 4:
                    frames = np.dot(frames[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                if len(frames) < 8:
                    pad = np.zeros((8 - len(frames), frames.shape[1], frames.shape[2]), dtype=np.uint8)
                    frames = np.concatenate([frames, pad], axis=0)
                if frames.shape[1:] != (128, 128):
                    frames = np.array([cv2.resize(f, (128, 128)) for f in frames[:8]])
                else:
                    frames = frames[:8]
                x_tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(2).float().to(self.device)

            with torch.no_grad():
                logit, aux_logits, z = self.deep_model(x_tensor, media_type)
                raw_logit = float(logit.item())
                raw_prob = float(torch.sigmoid(logit).item())
                latent_code = z.squeeze(0).cpu().numpy().tolist()
            cal_prob = cal.calibrate_prob(raw_prob)
            model_used = "deep_model"
            model_display_name = self.model_name

        # 5. Verdict and Risk
        ml_verdict, risk, details = cal.get_verdict_and_risk(cal_prob)

        if is_suspicious_structure:
            verdict = "SUSPECTED STEGO"
            if risk in ("BENIGN / CLEAN", "LOW RISK"):
                risk = "HIGH RISK"
        else:
            verdict = "SUSPECTED STEGO" if cal_prob >= decision_threshold else "No hidden data detected"

        # 6. Top features (if available)
        top_feats = get_top_features(norm_vec, media_type, top_k=6) if norm_vec is not None else []

        # Unified Standard Output Schema (Item 5)
        report = {
            "score": float(cal_prob),
            "verdict": verdict,
            "threshold": float(decision_threshold),
            "model_used": model_used,
            "model_name": model_display_name,
            "model_selection_reason": model_selection_reason,
            "file_name": p.name,
            "media_type": media_type,
            "format": format_name,
            "checkpoint_date": self.checkpoint_date,
            "risk_level": risk,
            "structure_findings": structure_findings,
            "structure_check": structure_res,
            "calibrated_probability": float(cal_prob),
            "raw_probability": float(raw_prob),
            "raw_logit": float(raw_logit),
            "decision_threshold": float(decision_threshold),
            "temperature": float(temperature),
            "top_features": top_feats,
            "latent_code_summary": {
                "norm": float(np.linalg.norm(latent_code)),
                "dim": len(latent_code)
            }
        }

        # 7. Media explanation
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


# Alias for concise import
Predictor = VaultBreakerPredictor
