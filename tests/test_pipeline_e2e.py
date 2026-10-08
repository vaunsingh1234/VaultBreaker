import pytest
import tempfile
from pathlib import Path
import numpy as np

from vaultbreaker.data.generator import (
    generate_image_dataset, generate_audio_dataset, generate_video_dataset
)
from vaultbreaker.data.split import split_dataset_by_source, save_splits
from vaultbreaker.features.extractor import MultiModalFeaturePipeline
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.training.dataset import MultiModalStegoDataset
from vaultbreaker.training.trainer import UnifiedTrainer
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator
from vaultbreaker.inference.predict import VaultBreakerPredictor

def test_pipeline_e2e_tiny():
    """
    End-to-end smoke test:
    Generates miniature dataset -> extracts features -> trains unified model ->
    fits calibrator -> verifies predictor inference.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        gen_dir = tmp / "generated"
        splits_dir = tmp / "splits"
        feats_dir = tmp / "features"
        models_dir = tmp / "models"

        # 1. Generate tiny data
        img_records = generate_image_dataset(gen_dir, num_samples=4, target_size=(32, 32))
        aud_records = generate_audio_dataset(gen_dir, num_samples=4, sample_rate=16000, duration=0.5)
        vid_records = generate_video_dataset(gen_dir, num_samples=4, target_size=(32, 32), fps=5, duration=0.5)

        all_records = img_records + aud_records + vid_records
        assert len(all_records) == 12

        # 2. Split
        df = split_dataset_by_source(all_records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        save_splits(df, splits_dir)

        # 3. Features
        pipeline = MultiModalFeaturePipeline(cache_dir=feats_dir / "cache", num_workers=1)
        feats = pipeline.extract_dataset(df)
        df_train = df[df["split"] == "train"]
        scalers = pipeline.fit_scalers(feats, df_train)
        pipeline.save_scalers(scalers, models_dir)

        # 4. Prepare training dataset
        train_map = {}
        for m in ["image", "audio", "video"]:
            sub = df_train[df_train["media_type"] == m]
            if not sub.empty:
                train_map[f"X_{m}"] = scalers[m].transform(np.stack([feats[p] for p in sub["filepath"]]))
                train_map[f"y_{m}"] = sub["label"].to_numpy()
                train_map[f"methods_{m}"] = sub["method"].to_numpy()
                train_map[f"rates_{m}"] = sub["payload_rate"].to_numpy()

        train_ds = MultiModalStegoDataset(train_map)
        assert len(train_ds) > 0

        # 5. Train
        net = VaultBreakerUnifiedNet(img_dim=72, aud_dim=56, vid_dim=64, latent_dim=32, hidden_dim=64, num_methods=8)
        trainer = UnifiedTrainer(net)
        loss = trainer.train_epoch(train_ds, batch_size=4)
        assert np.isfinite(loss)

        # 6. Calibrate
        cal = TemperatureScalingCalibrator(target_fpr=0.05)
        cal.fit(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
        cal.save(models_dir / "calibrator.joblib")

        # 7. Predict
        clean_img_path = gen_dir / "images" / img_records[0]["filename"]
        report = VaultBreakerPredictor(models_dir=models_dir).predict_file(clean_img_path, generate_explanation=True)
        assert report["verdict"] in ["CLEAN", "SUSPECTED STEGO"]
        assert 0.0 <= report["calibrated_probability"] <= 1.0
