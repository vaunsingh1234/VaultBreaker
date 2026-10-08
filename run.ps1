# VaultBreaker PowerShell Automation Script
param (
    [string]$Target = "help"
)

switch ($Target) {
    "setup" {
        pip install -r requirements.txt
        pip install -e . --no-deps
    }
    "data" {
        python scripts/make_dataset.py
    }
    "features" {
        python scripts/extract_features.py --num-workers 2
    }
    "train" {
        python scripts/train.py
    }
    "eval" {
        python scripts/evaluate.py
    }
    "fast" {
        python scripts/make_dataset.py --fast
        python scripts/extract_features.py --num-workers 2
        python scripts/train.py --fast
        python scripts/evaluate.py
    }
    "test" {
        pytest tests/ -v
    }
    "app" {
        streamlit run app/streamlit_app.py --server.port 8501
    }
    "api" {
        uvicorn api.main:app --host 0.0.0.0 --port 8000
    }
    Default {
        Write-Host "Usage: .\run.ps1 -Target [setup | data | features | train | eval | fast | test | app | api]"
    }
}
