#!/usr/bin/bash

cd /app/env/bin

source ./activate env

cd ../AlarmPrediction/src/

echo "Anomaly model building started"
python3 WS_ModelBuilding.py
echo "Anomaly model building completed"

rm /app/env/AlarmPrediction/data/interim/h_OCHCTP.csv
