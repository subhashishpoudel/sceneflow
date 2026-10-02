@echo off
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')"`) do set GEMINI_API_KEY=%%A
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('GEMINI_IMAGE_API_KEY','User')"`) do set GEMINI_IMAGE_API_KEY=%%A
python main.py
