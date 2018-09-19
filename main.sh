#!/usr/bin/bash

cd /app/env/bin

source ./activate env

cd ../AlarmPrediction/src/

echo "Starting the Parent script"
python3 WS_ParentScript.py
echo "Parent script is compelted"

python3 WS_ActualData.py
