@echo off
setlocal enabledelayedexpansion
conda activate face-parsing
python -m src.pipeline.run_pipeline --config config/config.yaml
pause
