@echo off
REM FireRed-OpenStoryline 服务启动脚本
chcp 65001 >nul
setlocal enabledelayedexpansion

set projectPath=D:\software\FireRed-OpenStoryline
set pythonPath=src

echo.
echo ======================================
echo   FireRed-OpenStoryline 服务启动
echo ======================================
echo.

REM 启动 MCP 服务器
echo 启动 MCP 服务器...
start "MCP Server" cmd /k "cd /d %projectPath% && conda activate storyline && set PYTHONPATH=%pythonPath% && python -m open_storyline.mcp.server"

timeout /t 2 /nobreak

REM 启动 FireRed-OpenStoryline 应用
echo 启动 FireRed-OpenStoryline 应用...
start "FireRed-OpenStoryline" cmd /k "cd /d %projectPath% && conda activate storyline && uvicorn agent_fastapi:app --host 0.0.0.0 --port 7888"

echo.
echo ✓ 两个服务都已启动！
echo.
echo   MCP 服务器: 在 "MCP Server" 窗口中运行
echo   FireRed-OpenStoryline 应用: 在 "FireRed-OpenStoryline" 窗口中运行
echo   访问: http://127.0.0.1:7888
echo.
pause
