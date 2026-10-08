@echo off
REM VaultBreaker Automation Script for Windows Command Prompt

IF "%1"=="setup" (
    pip install -r requirements.txt
    pip install -e . --no-deps
    GOTO end
)

IF "%1"=="data" (
    python scripts\make_dataset.py
    GOTO end
)

IF "%1"=="features" (
    python scripts\extract_features.py --num-workers 2
    GOTO end
)

IF "%1"=="train" (
    python scripts\train.py
    GOTO end
)

IF "%1"=="eval" (
    python scripts\evaluate.py
    GOTO end
)

IF "%1"=="fast" (
    python scripts\make_dataset.py --fast
    python scripts\extract_features.py --num-workers 2
    python scripts\train.py --fast
    python scripts\evaluate.py
    GOTO end
)

IF "%1"=="test" (
    pytest tests\ -v
    GOTO end
)

IF "%1"=="app" (
    streamlit run app\streamlit_app.py --server.port 8501
    GOTO end
)

IF "%1"=="api" (
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    GOTO end
)

echo Usage: run.bat [setup ^| data ^| features ^| train ^| eval ^| fast ^| test ^| app ^| api]

:end
