@echo off
cd /d "%~dp0"
echo ========================================
echo   Churn Prediction App
echo ========================================
echo.
echo Starting Flask server...
echo Server will be available at: http://127.0.0.1:5001
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
python app.py
echo.
echo Server stopped.
pause

