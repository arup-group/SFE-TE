@echo off
pushd %~dp0
call conda activate te_awf_dev
python te_awf_start.py -i sample_inputs/quick_test.json -o src/dump
call conda deactivate
pause