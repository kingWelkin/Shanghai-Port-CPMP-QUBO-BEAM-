@echo off
setlocal
REM Run from the project root. Set PYTHON to override the interpreter.
set "ROOT=%~dp0"
if defined PYTHON (
  set "PY=%PYTHON%"
) else (
  set "PY=python"
)
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
cd /d "%ROOT%"
"%PY%" -m greedy %*
