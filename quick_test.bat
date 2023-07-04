@echo off
pushd %~dp0
call conda activate fire_general
python src/main.py -i sample_inputs/quick_test.json -o src/dump
call conda deactivate
pause