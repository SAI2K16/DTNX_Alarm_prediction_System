#!/usr/bin/bash

cd /app/env/bin

source ./activate env

cd ../AlarmPrediction/src/

echo "Starting the BulkPredict script"
python3 WS_BulkPred.py
echo "BulkPredict script is compelted"

