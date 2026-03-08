@echo off
REM FireRed-OpenStoryline 服务启动脚本
chcp 65001 >nul
setlocal enabledelayedexpansion

set projectPath=%~dp0
set pythonPath=src

echo.
echo ======================================
echo   FireRed-OpenStoryline 服务启动
echo ======================================
echo.

echo 启动 MCP 服务器...
start "MCP Server" cmd /k "cd /d %projectPath% && conda activate storyline && set PYTHONPATH=%pythonPath% && python -m open_storyline.mcp.server"

timeout /t 2 /nobreak

echo 启动 FireRed-OpenStoryline 应用...
start "FireRed-OpenStoryline" cmd /k "cd /d %projectPath% && conda activate storyline && uvicorn agent_fastapi:app --host 0.0.0.0 --port 7860"

echo.
echo ✓ 两个服务都已启动！
echo.
echo   MCP 服务器: 在 "MCP Server" 窗口中运行
echo.
echo   OpenStoryline 应用: 在 "FireRed-OpenStoryline" 窗口中运行
echo.
echo   访问: http://127.0.0.1:7860
echo.
pause
