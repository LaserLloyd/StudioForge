@echo off
REM Template for a machine-specific override. Copy this file to the REPO ROOT
REM as local-env.bat (it is gitignored there) and edit the values. Every
REM launcher in this folder calls it before starting anything.
REM
REM Keep your data -- config.yaml, the model registry, engines, logs, download
REM staging -- OUTSIDE the checkout, so a git operation can never touch it and
REM the repository never accidentally contains a credential:
set "SF_DATA_DIR=C:\Users\%USERNAME%\StudioForge-data"

REM Other knobs the launchers honour (uncomment to use):
REM set "SF_CONFIG=C:\somewhere\config.yaml"      a specific config file
REM set "SF_TEST_MODELS_DIR=E:\LLM\Models"         where the contract tests find GGUFs
