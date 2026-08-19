@echo off
echo ========================================
echo AgriTwin v3.0 Dependency Installer
echo ========================================
echo.

echo Step 1: Cleaning up existing installations...
pip uninstall pandas numpy -y
pip cache purge

echo.
echo Step 2: Installing numpy 1.24.3...
pip install numpy==1.24.3
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install numpy. Trying alternative...
    python -m pip install numpy==1.24.3
    if %ERRORLEVEL% NEQ 0 (
        echo ❌❌ Critical: Cannot install numpy. Please use conda instead.
        echo Run: conda create -n agritwin python=3.10
        echo Then: conda install numpy pandas scikit-learn xgboost lightgbm
        pause
        exit /b 1
    )
)

echo.
echo Step 3: Installing pandas 2.0.3...
pip install pandas==2.0.3
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Failed with pip, trying python -m pip...
    python -m pip install pandas==2.0.3
)

echo.
echo Step 4: Installing ML libraries...
pip install scikit-learn xgboost lightgbm pyarrow joblib

echo.
echo Step 5: Installing FastAPI and other dependencies...
pip install fastapi uvicorn pydantic sqlalchemy requests python-dotenv

echo.
echo Step 6: Testing pandas installation...
python test_pandas.py

echo.
if %ERRORLEVEL% EQ 0 (
    echo ✅✅✅ INSTALLATION SUCCESSFUL! ✅✅✅
    echo.
    echo Next steps:
    echo 1. Generate training data: python backend\scripts\generate_training_data_simple.py
    echo 2. Train the model: python backend\scripts\train_model_custom.py
    echo 3. Start API: uvicorn backend.app.main:app --reload
) else (
    echo ❌❌❌ INSTALLATION FAILED ❌❌❌
    echo.
    echo Please try these alternatives:
    echo Option A: Use Miniconda (recommended for Windows)
    echo   1. Download Miniconda from https://docs.conda.io/en/latest/miniconda.html
    echo   2. Run: conda create -n agritwin python=3.10
    echo   3. Run: conda activate agritwin
    echo   4. Run: conda install pandas numpy scikit-learn xgboost lightgbm
    echo.
    echo Option B: Use Python 3.9 instead of 3.14
    echo   Python 3.14 might have compatibility issues with some packages.
)

echo.
pause