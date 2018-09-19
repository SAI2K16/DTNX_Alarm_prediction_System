#!/usr/bin/bash

cd /app/env/bin

source ./activate env

cd ../AlarmPrediction/src/

echo "ARIMA parameter computation started"
python3 WS_ArimaParameterComputation.py
echo "ARIMA parameter successfully computed"

