import math
import numpy as np
import torch, json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
import torch, GPUtil, psutil, platform



def plot_density(x, y, save_plot_path):
    x = np.array(x)
    y = np.array(y)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    
    scatter = ax.scatter(x, y, c=z, s=1, cmap='inferno')

    # Adding a dashed line
    ax.plot([min(min(x), min(y)), max(max(x), max(y))], [min(min(x), min(y)), max(max(x), max(y))], '--', color='red') # Change coordinates as needed

    plt.xlim(min(min(x), min(y)), max(max(x), max(y)))
    plt.ylim(min(min(x), min(y)), max(max(x), max(y)))
    
    plt.xlabel('spectroscopic z')
    plt.ylabel('predicted z')
    plt.grid(True)
    plt.colorbar(scatter, ax=ax)
    
    plt.savefig(save_plot_path + 'inference.png')
    
    
def err_calculate(prediction, z, execution_info, save_plot_path):
    prediction = np.array(prediction)
    z = np.array(z)
    mae = np.sum(abs(prediction - z)) / z.shape[0]
    mse = np.sum((prediction - z)**2) / z.shape[0]
    
    deltaz = (prediction - z) / (1 + z)
    bias = np.sum(deltaz) / z.shape[0]
    nmad = 1.48 * np.median(abs(deltaz - np.median(deltaz)))
    r2 = r2_score(z, prediction)
    
    
    errs = {
    'average execution time (milliseconds) per batch': execution_info['execution_time'] * 1000,
    'batch size': execution_info['batch_size'],
    'number of batches': execution_info['num_batches'],
    'device': execution_info['device'],
    'MAE': mae,
    'MSE': mse,
    'Bias': bias,
    'Precision': nmad,
    'R2': r2
    }
    
    with open(save_plot_path + 'Results.json', 'w') as file:
        json.dump(errs, file, indent=5)

    system_info(save_plot_path) 
        

        


class system_info:
    def __init__(self, path_to_save):
        self.save_info(path_to_save)

    # Function to gather GPU information
    def get_gpu_info(self):
        gpu_info = []
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "ID": gpu.id,
                    "Name": gpu.name,
                    "Load": f"{gpu.load * 100:.1f}%",
                    "Free Memory": f"{gpu.memoryFree}MB",
                    "Used Memory": f"{gpu.memoryUsed}MB",
                    "Total Memory": f"{gpu.memoryTotal}MB",
                    "Temperature": f"{gpu.temperature} °C",
                    "UUID": gpu.uuid
                })
        else:
            gpu_info = "No GPU available"

        return gpu_info

    # Function to gather CPU information
    def get_cpu_info(self):
        cpu_info = {
            "Physical Cores": psutil.cpu_count(logical=False),
            "Total Cores": psutil.cpu_count(logical=True),
            "Max Frequency": f"{psutil.cpu_freq().max:.2f}Mhz",
            "Min Frequency": f"{psutil.cpu_freq().min:.2f}Mhz",
            "Current Frequency": f"{psutil.cpu_freq().current:.2f}Mhz",
            "Total CPU Usage": f"{psutil.cpu_percent()}%"
        }

        # Per core CPU usage
        core_usages = psutil.cpu_percent(percpu=True)
        cpu_info["Per Core Usage"] = [f"{usage}%" for usage in core_usages]

        return cpu_info

    def save_info(self, path_to_save):
        # Combine GPU and CPU info into one dictionary
        system_info = {
            "System": platform.system(),
            "Node Name": platform.node(),
            "Release": platform.release(),
            "Version": platform.version(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "GPU Info": self.get_gpu_info(),
            "CPU Info": self.get_cpu_info(),
            "Memory Info": {
                "Total Memory": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
                "Available Memory": f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
                "Memory Usage": f"{psutil.virtual_memory().percent}%",
            }
        }
        
        with open(path_to_save + 'System_Info.json', 'w') as file:
            json.dump(system_info, file, indent=5)