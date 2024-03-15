@echo off
pushd %~dp0
echo Welcome to TE tool local starter
echo[
set /p "inputFile=Enter path to input file: "
set /p "outputDir=Enter path to output directory: "
echo[
call conda activate fire_general_forge
python src/main.py -i %inputFile% -o %outputDir%
call conda deactivate
pause