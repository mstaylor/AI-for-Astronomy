# Data Parallel Inference

Following shows the model design on AWS State Machine.

<img src='./figures/workflow.jpg' width='50%'/>

## Results

## Varying data size

The total data size is 12.6GB. We run the inference for different sizes to evaluate the scaling performance with increasing data load. This experiment runs with size 100MB, 1GB, 2GB, 4GB, 6GB, 8GB, 10GB and 12.6GB. Batch size 512.

| Data | Run | Num worlds | Job Duration (s) | Avg Inference Time (s) |
|:---|:---|:---|:---|:---|
| 100MB | 1 | 1 | 00:30.30 | |
|  | 2 | 1 | 00:27.02 | |
|  | 3 | 1 | 00:36.00 | |
| 1GB | 1 | 10 | 00:38.314 |  |
|  | 2 | 10 | 00:28.85 | |
|  | 3 | 10 | 00:30.77 | |
| 2GB | 1 | 20 | 00:30.54 | |
|  | 2 | 20 | 00:31.73 | |
|  | 3 | 20 | 00:38.55 | |
| 4GB | 1 | 40 | 01:01.291 |  |
|  | 2 | 40 | 01:01.69 | |
|  | 3 | 40 | 01:00.97 | |
| 6GB | 1 | 60 | 00:56.67 |  |
|  | 2 | 60 | 01:02.34 | |
|  | 3 | 60 | 01:14.21 | |
| 8GB | 1 | 80 | 01:06.34 |  |
|  | 2 | 80 | 00:56.70 | |
|  | 3 | 80 | 01:06.24 | |
| 10GB | 1 | 100 | 00:59.84 | |
|  | 2 | 100 | 01:08.01 | |
|  | 3 | 100 | 01:00.01 | |
| Total | 1 | 130 | 01:07.39 | |
|  | 2 | 130 | 01:10.10 | |
|  | 3 | 130 | 01:08.19 | |

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