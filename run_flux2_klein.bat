@echo off
setlocal EnableExtensions

title FLUX.2 Klein 4B - Local Image Test

REM ============================================================
REM CONFIGURATION
REM ============================================================

set "VENV=D:\all project\venv_cuda"

set "ROOT=D:\all project\Projects\AI projects\ai assentent\Local-agents"

REM stable-diffusion.cpp
set "SD_DIR=%ROOT%\sd-master-3f8527a-bin-win-cuda12-x64"
set "SD_CLI=%SD_DIR%\sd-cli.exe"

REM Image models
set "IMAGE_MODEL_DIR=D:\all project\Projects\AI projects\ai assentent\Local_codex\models\image"

REM Chat models
set "CHAT_MODEL_DIR=D:\all project\Projects\AI projects\ai assentent\Local_codex\models\chat\qwen3-4b"

REM FLUX.2 Klein 4B Q4_0
set "DIFFUSION_MODEL=%IMAGE_MODEL_DIR%\flux-2-klein-4b-Q4_0.gguf"

REM Qwen3-4B Q4_K_M
set "QWEN_MODEL=%CHAT_MODEL_DIR%\Qwen3-4B-Q4_K_M.gguf"

REM FLUX.2 Klein VAE
set "VAE_MODEL=%IMAGE_MODEL_DIR%\full_encoder_small_decoder.safetensors"

REM Output
set "OUTPUT_DIR=%ROOT%\generated\assets"
set "OUTPUT=%OUTPUT_DIR%\flux2_klein_test.png"


REM ============================================================
REM HEADER
REM ============================================================

echo.
echo ============================================================
echo              FLUX.2 KLEIN 4B TEST
echo ============================================================
echo.
echo Project:
echo %ROOT%
echo.
echo Python:
echo %VENV%
echo.
echo SD CLI:
echo %SD_CLI%
echo.
echo FLUX.2 Model:
echo %DIFFUSION_MODEL%
echo.
echo Qwen3-4B:
echo %QWEN_MODEL%
echo.
echo VAE:
echo %VAE_MODEL%
echo.


REM ============================================================
REM 1. ACTIVATE PYTHON ENVIRONMENT
REM ============================================================

echo ============================================================
echo [1/7] Activating Python environment
echo ============================================================
echo.

if exist "%VENV%\Scripts\activate.bat" (
    call "%VENV%\Scripts\activate.bat"
    echo OK: venv_cuda activated.
) else (
    echo WARNING: Python environment not found:
    echo %VENV%
)

echo.


REM ============================================================
REM 2. CHECK STABLE-DIFFUSION.CPP
REM ============================================================

echo ============================================================
echo [2/7] Checking stable-diffusion.cpp
echo ============================================================
echo.

if not exist "%SD_CLI%" (
    echo ERROR: sd-cli.exe not found:
    echo %SD_CLI%
    echo.
    pause
    exit /b 1
)

echo OK: sd-cli.exe found.
echo.

"%SD_CLI%" --version

echo.


REM ============================================================
REM 3. CHECK FLUX.2 KLEIN MODEL
REM ============================================================

echo ============================================================
echo [3/7] Checking FLUX.2 Klein Q4_0
echo ============================================================
echo.

if not exist "%DIFFUSION_MODEL%" (
    echo ERROR: FLUX.2 Klein model not found:
    echo %DIFFUSION_MODEL%
    echo.
    pause
    exit /b 1
)

echo OK: FLUX.2 Klein model found.
echo.


REM ============================================================
REM 4. CHECK QWEN3-4B
REM ============================================================

echo ============================================================
echo [4/7] Checking Qwen3-4B
echo ============================================================
echo.

if not exist "%QWEN_MODEL%" (
    echo ERROR: Qwen3-4B model not found:
    echo %QWEN_MODEL%
    echo.
    pause
    exit /b 1
)

echo OK: Qwen3-4B found:
echo %QWEN_MODEL%
echo.


REM ============================================================
REM 5. CHECK FLUX.2 VAE
REM ============================================================

echo ============================================================
echo [5/7] Checking FLUX.2 VAE
echo ============================================================
echo.

if not exist "%VAE_MODEL%" (
    echo ERROR: FLUX.2 VAE not found:
    echo %VAE_MODEL%
    echo.
    pause
    exit /b 1
)

echo OK: FLUX.2 VAE found:
echo %VAE_MODEL%
echo.


REM ============================================================
REM 6. PREPARE OUTPUT DIRECTORY
REM ============================================================

echo ============================================================
echo [6/7] Preparing output directory
echo ============================================================
echo.

if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

echo Output:
echo %OUTPUT%
echo.


REM ============================================================
REM 7. GENERATE IMAGE
REM ============================================================

echo ============================================================
echo [7/7] Starting FLUX.2 Klein 4B
echo ============================================================
echo.

echo ------------------------------------------------------------
echo MODEL
echo ------------------------------------------------------------
echo FLUX.2:
echo %DIFFUSION_MODEL%
echo.
echo Qwen:
echo %QWEN_MODEL%
echo.
echo VAE:
echo %VAE_MODEL%
echo.

echo ------------------------------------------------------------
echo SETTINGS
echo ------------------------------------------------------------
echo Backend: CUDA
echo CPU offload: enabled
echo Diffusion flash attention: enabled
echo Resolution: 512x512
echo Steps: 4
echo CFG: 1.0
echo Seed: 42
echo Sampler: Euler
echo.

echo ------------------------------------------------------------
echo PROMPT
echo ------------------------------------------------------------
echo A highly detailed scientific visualization of a
echo superconducting quantum processor inside a dilution
echo refrigerator. Show a realistic quantum processor at
echo the center, surrounded by cryogenic components and
echo control wiring. Professional scientific visualization,
echo realistic engineering hardware, metallic surfaces,
echo dark laboratory environment, blue and white lighting,
echo accurate physical proportions, no text, no labels,
echo no logos, no watermark.
echo ------------------------------------------------------------
echo.

echo Starting generation...
echo.


"%SD_CLI%" ^
    --diffusion-model "%DIFFUSION_MODEL%" ^
    --vae "%VAE_MODEL%" ^
    --llm "%QWEN_MODEL%" ^
    --backend cuda ^
    --offload-to-cpu ^
    --diffusion-fa ^
    --cfg-scale 1.0 ^
    --sampling-method euler ^
    --steps 4 ^
    -W 512 ^
    -H 512 ^
    --seed 42 ^
    -p "A highly detailed scientific visualization of a superconducting quantum processor inside a dilution refrigerator. Show a realistic quantum processor at the center, surrounded by cryogenic components and control wiring. Professional scientific visualization, realistic engineering hardware, metallic surfaces, dark laboratory environment, blue and white lighting, accurate physical proportions, no text, no labels, no logos, no watermark." ^
    -o "%OUTPUT%" ^
    -v


REM ============================================================
REM CHECK GENERATION RESULT
REM ============================================================

if errorlevel 1 (
    echo.
    echo ============================================================
    echo              FLUX.2 GENERATION FAILED
    echo ============================================================
    echo.
    echo Check the error message above.
    echo.
    pause
    exit /b 1
)

if exist "%OUTPUT%" (
    echo.
    echo ============================================================
    echo              GENERATION SUCCESSFUL
    echo ============================================================
    echo.
    echo Output image:
    echo %OUTPUT%
    echo.
    start "" "%OUTPUT%"
) else (
    echo.
    echo ============================================================
    echo              OUTPUT NOT FOUND
    echo ============================================================
    echo.
    echo sd-cli finished, but the expected image was not found:
    echo %OUTPUT%
)

echo.
echo ============================================================
echo                     DONE
echo ============================================================
echo.

pause