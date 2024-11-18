# Data Parallel Inference

Following shows the model design on AWS State Machine.

<img src='./figures/cai workflow.jpg' width='60%'/>

## Results

## Varying data size

The total data size is 12.6GB. We run the inference for different sizes to evaluate the scaling performance with increasing data load. This experiment runs with size 1GB, 2GB, 4GB, 6GB, 8GB, 10GB and 12.6GB. Batch size 512.

Please check the [result_stats.csv](./results/result_stats.csv) for the average results.

### Fig: Dataset size vs Inference time for each partition

<img src='./figures/inference_time.jpg' width='70%'/>


### Fig: Data size vs Throughput for each partition

<img src='./figures/throughput.jpg' width='70%'/>

## Varying batch sizes

Following is with 8GB data and changing batch sizes

| Batch Size | Run | Job Duration (s) | Avg Inference Time (s) |
|:---:|:---:|:---:|:---:|
| 32 | 1 | 02:26.604 |  |
|  | 2 | 02:41.871 |  |
|  | 3 | 02:24.656 |  |
| 64 | 1 | 01:30.402 |  |
|  | 2 | 02:04.803 |  |
|  | 3 | 01:32.340 |  |
| 128 | 1 | 00:59.550 |  |
|  | 2 | 01:07.719 |  |
|  | 3 | 01:03.962 |  |
| 256 | 1 | 01:04.292 |  |
|  | 2 | 01:10.250 |  |
|  | 3 | 01:04.082 |  |
| 512 | 1 | 00:58.863 |  |
|  | 2 | 00:57.032 |  |
|  | 3 | 01:19.324 |  |

Following is with 2GB data and changing batch sizes

| Batch Size | Run | Job Duration (s) | Avg Inference Time (s) |
|:---:|:---:|:---:|:---:|
| 32 | 1 | 01:43.699 |  |
|  | 2 | 01:55.086 |  |
|  | 3 | 01:58.196 |  |
| 64 | 1 | 01:20.951 |  |
|  | 2 | 01:16.662 |  |
|  | 3 | 01:10.439 |  |
| 128 | 1 | 00:55.409 |  |
|  | 2 | 00:55.194 |  |
|  | 3 | 00:49.609 |  |
| 256 | 1 | 00:42.435 |  |
|  | 2 | 00:45.326 |  |
|  | 3 | 00:41.348 |  |
| 512 | 1 | 00:45.474 |  |
|  | 2 | 00:46.783 |  |
|  | 3 | 00:34.566 |  |

Following is with 1GB data and changing batch sizes

| Batch Size | Run | Job Duration (s) | Avg Inference Time (s) |
|:---:|:---:|:---:|:---:|
| 32 | 1 | 02:18.282 |  |
|  | 2 | 02:19.304 |  |
|  | 3 | 02:30.770 |  |
| 64 | 1 | 01:38.751 |  |
|  | 2 | 01:47.680 |  |
|  | 3 | 01:43.734 |  |
| 128 | 1 | 01:13.763 |  |
|  | 2 | 01:15.658 |  |
|  | 3 | 01:13.670 |  |
| 256 | 1 | 01:02.622 |  |
|  | 2 | 01:00.940 |  |
|  | 3 | 01:00.611 |  |
| 512 | 1 | 00:45.474 |  |
|  | 2 | 00:46.783 |  |
|  | 3 | 00:34.566 |  |

## Notes

* [Economics of 'Serverless](https://www.bbva.com/en/innovation/economics-of-serverless/)