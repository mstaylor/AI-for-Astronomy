import json, os
import pandas as pd
import numpy as np

# sync aws and local folders
# aws s3 sync s3://cosmicai-data/results results

def average_varying_batch_size():
    batch_sizes = [32, 64, 128, 256, 512]
    runs = 3
    all_results = []

    keys = ['total_cpu_time (seconds)', "total_cpu_memory (MB)", "throughput_bps"]
    data_size = '1GB'
    column_name = 'batch_size'

    for batch_size in batch_sizes:
        for run in range(1, runs+1):
            with open(f'./results/{data_size}/Batches/{batch_size}/run {run}/combined_data.json', 'r') as f:
                data = json.load(f)
                
                results = {
                    key: [] for key in keys
                }
                
                for d in data:
                    for key in keys:
                        results[key].append(d[key])
                
                results = pd.DataFrame(results)
                results['run'] = run
                results[column_name] = batch_size
                all_results.append(results)
                
    all_results = pd.concat(all_results)
    # all_results = all_results[['data_size', 'run'] + keys]
    all_results = all_results.groupby([column_name, 'run'])[keys].agg({
        'total_cpu_time (seconds)': 'mean', 
        "total_cpu_memory (MB)": 'sum', 
        "throughput_bps": 'sum'
    }).reset_index()
    
    mean = all_results.groupby(column_name)[keys].mean().reset_index()
    mean.insert(1, 'run', 'mean')
    all_results = pd.concat([all_results, mean], axis=0)
    all_results.sort_values([column_name, 'run'], inplace=True)
    
    all_results.round(2).to_csv('./results/batch_varying_results.csv', index=False)

def average_varying_data_size():
    data_sizes = ['0.1GB', '1GB', '2GB', '4GB', '6GB', '8GB', '10GB', 'total']
    runs = 3
    all_results = []

    keys = ['total_cpu_time (seconds)', "total_cpu_memory (MB)", "throughput_bps"]

    for data_size in data_sizes:
        for run in range(1, runs+1):
            with open(f'./results/{data_size}/run {run}/combined_data.json', 'r') as f:
                data = json.load(f)
                
                results = {
                    key: [] for key in keys
                }
                
                for d in data:
                    for key in keys:
                        results[key].append(d[key])
                
                results = pd.DataFrame(results)
                results['run'] = run
                results['data_size'] = data_size
                all_results.append(results)
                
    all_results = pd.concat(all_results)
    # all_results = all_results[['data_size', 'run'] + keys]
    all_results = all_results.groupby(['data_size', 'run'])[keys].agg({
        'total_cpu_time (seconds)': 'mean', 
        "total_cpu_memory (MB)": 'sum', 
        "throughput_bps": 'sum'
    }).reset_index()

    mean = all_results.groupby(['data_size'])[keys].mean().reset_index()
    mean.insert(1, 'run', 'mean')
    all_results = pd.concat([all_results, mean], axis=0)
    all_results.sort_values(['data_size', 'run'], inplace=True)

    all_results.round(2).to_csv('./results/result_stats.csv', index=False)

if __name__ == '__main__':
    average_varying_batch_size()
    average_varying_data_size()