@echo off
echo Starting Lumi App on Android Emulator...
echo.

:: Start backend server in a new window
echo [1/3] Starting backend server...
start "Lumi Backend" cmd /k "cd /d c:\Users\sergc\Desktop\lumi-demo && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

:: Wait for backend to start
timeout /t 3 /nobreak > nul

:: Launch the Android emulator if not already running
echo [2/3] Starting Android Emulator...
start "" "C:\Users\sergc\AppData\Local\Android\Sdk\emulator\emulator.exe" -avd Medium_Phone_API_36.1

:: Wait for emulator to boot
echo Waiting for emulator to boot (this may take a minute)...
timeout /t 30 /nobreak > nul

:: Run Flutter app on emulator
echo [3/3] Launching Flutter app on emulator...
cd /d c:\Users\sergc\Desktop\lumi-demo\flutter_app
flutter run -d emulator-5554

pause
