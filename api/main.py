import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

from vaultbreaker.inference.predict import VaultBreakerPredictor
from vaultbreaker.inference.router import RouterError
from vaultbreaker.utils.logger import get_logger

logger = get_logger("API")

app = FastAPI(
    title="VaultBreaker API",
    description="Unified Multi-Modal Steganography Detection Service for Images, Audio, and Video files.",
    version="1.0.0"
)

# Global predictor instance
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = VaultBreakerPredictor()
        logger.info("VaultBreaker API initialized predictor successfully.")
    except Exception as e:
        logger.error(f"Failed to load predictor on startup: {e}")

@app.get("/health", summary="Service Health and Metadata")
def health() -> Dict[str, Any]:
    global predictor
    is_ready = predictor is not None
    return {
        "status": "healthy" if is_ready else "initializing",
        "service": "VaultBreaker Steganography Detection Engine",
        "version": "1.0.0",
        "supported_media": ["image (PNG, JPEG)", "audio (WAV, FLAC)", "video (MKV, AVI, MP4)"],
        "device": str(predictor.device) if predictor else "unknown",
        "calibrated_threshold": float(predictor.calibrator.decision_threshold) if predictor else 0.5
    }

@app.post("/scan", summary="Scan Media File for Hidden Payloads")
async def scan_file(
    file: UploadFile = File(...),
    explain: bool = Query(False, description="Whether to include explanation summaries")
) -> Dict[str, Any]:
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service is not initialized yet.")

    # Secure temp file storage
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        report = predictor.predict_file(tmp_path, generate_explanation=explain)
        
        # Sanitize explanation for JSON response
        if "explanation" in report:
            exp_clean = {}
            for k, v in report["explanation"].items():
                if isinstance(v, (int, float, str, list)):
                    exp_clean[k] = v
                elif hasattr(v, "tolist"):
                    exp_clean[k] = v.tolist()
            report["explanation"] = exp_clean
            
        report["file_name"] = file.filename
        return report

    except RouterError as e:
        raise HTTPException(status_code=400, detail=f"File validation rejected: {str(e)}")
    except Exception as e:
        logger.error(f"Inference error scanning {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
