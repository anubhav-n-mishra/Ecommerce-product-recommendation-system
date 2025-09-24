@echo off
REM 🚀 Advanced E-Commerce Recommendation System - Quick Start
REM Author: Anubhav Mishra | Date: September 2025
REM GitHub: https://github.com/anubhav-n-mishra/Ecommerce-product-recommendation-system

echo.
echo ===============================================
echo 🚀 Advanced E-Commerce Recommendation System
echo ===============================================
echo Author: Anubhav Mishra
echo Date: September 2025
echo Models: EPR1 (99.66%%) EPR2 EPR3 (75.16%%) EPR4 (77.35%%)
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo 📦 Installing required packages...

REM Install required packages
pip install flask pandas numpy scikit-learn scipy matplotlib seaborn jupyter

if %errorlevel% neq 0 (
    echo ⚠️ Some packages may already be installed
)

echo.
echo 🔍 Checking for trained models...

if exist "epr1.pkl" (
    echo ✅ EPR1 model found
) else (
    echo ⚠️ EPR1 model not found - run Model_based_collaborative_filtering.ipynb
)

if exist "epr2.pkl" (
    echo ✅ EPR2 model found
) else (
    echo ⚠️ EPR2 model not found - run rank_based_product_recommendation.ipynb
)

if exist "epr3.pkl" (
    echo ✅ EPR3 model found
) else (
    echo ⚠️ EPR3 model not found - run User_based_collaborative_filtering.ipynb
)

if exist "epr4.pkl" (
    echo ✅ EPR4 model found
) else (
    echo ⚠️ EPR4 model not found - run ECommerce_Product_Recommendation_System.ipynb
)

echo.
echo 🌐 Starting Flask web application...
echo 📍 URL: http://localhost:5000
echo 📚 API: http://localhost:5000/api/models/status
echo.
echo Press Ctrl+C to stop the server
echo ===============================================

python flask_app.py

pause