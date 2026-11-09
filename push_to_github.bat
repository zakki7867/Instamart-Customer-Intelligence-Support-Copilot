@echo off
cd /d "%~dp0"
set "PATH=%~dp0..\git\cmd;%PATH%"
echo ========================================================
echo Pushing Instamart Copilot project to GitHub (branch main)
echo ========================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Project pushed successfully!
    echo Submission URL: https://github.com/zakki7867/Instamart-Customer-Intelligence-Support-Copilot/tree/main
) else (
    echo [ERROR] Push did not complete. Please see message above.
)
echo.
pause
