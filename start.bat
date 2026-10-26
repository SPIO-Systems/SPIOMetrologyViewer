@echo off
rem DirtSearcher5000: get the latest version from GitHub, install what it needs, start the GUI.
cd /d "%~dp0"

set "GIT="
where git >nul 2>nul && set "GIT=git"
if not defined GIT if exist "%ProgramFiles%\Git\cmd\git.exe" set "GIT=%ProgramFiles%\Git\cmd\git.exe"
if not exist .git (
    echo Not a git clone: can't check for updates. See README, Getting started.
) else if not defined GIT (
    echo Git is not installed: can't check for updates. Install Git for Windows from https://git-scm.com
) else (
    echo Checking for updates...
    "%GIT%" pull --ff-only
    if errorlevel 1 echo *** Could not update ^(offline, or files changed locally^). Starting the current version. & timeout /t 5
)

if not exist .venv\Scripts\python.exe (
    echo First start: setting up Python packages, this takes a few minutes...
    py -3.14 -m venv .venv
    if errorlevel 1 (
        if exist .venv rmdir /s /q .venv
        echo *** Could not set up Python. Is Python 3.14 installed? Get it from https://www.python.org/downloads/
        echo     ^(tick "py launcher"^), then run this again.
        pause
        exit /b 1
    )
)

rem Install packages only when requirements.txt changed since the last install
fc /b requirements.txt .venv\requirements.installed >nul 2>nul
if errorlevel 1 (
    .venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo *** Installing the packages failed, see above.
        pause
        exit /b 1
    )
    copy /y requirements.txt .venv\requirements.installed >nul
)

start "" .venv\Scripts\pythonw.exe -m dirtsearcher.gui
