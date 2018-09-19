import pandas as pd

pm_data = pd.read_csv("/app/env/AlarmPrediction/output/actuals.csv", names=['performance_metrics.ts', 'performance_metrics.module', 'BerPreFecMax','PhaseCorrectionAve', 'PmdMin', 'Qmin', 'SoPmdAve'])

tmp_list = []
for device in list(pm_data["performance_metrics.module"].unique()):
    tmp = pm_data[pm_data["performance_metrics.module"]==device]
    tmp.drop_duplicates(['performance_metrics.ts'], inplace=True)
    tmp_list.append(tmp)

pm_data = pd.concat(tmp_list)
pm_data.to_csv("/app/env/AlarmPrediction/output/actuals.csv", index=False)
