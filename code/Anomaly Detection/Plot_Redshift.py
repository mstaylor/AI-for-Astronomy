import math, json
import numpy as np
import torch, GPUtil, psutil, platform

def err_calculate(prediction, z, execution_info, save_path):
    
    def r2_score(y_true, y_pred):
        # Calculate the residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Calculate the total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        # Calculate R^2 score
        r2 = 1 - (ss_res / ss_tot)

        return r2
 
    prediction = np.array(prediction)
    z = np.array(z)
    
    #MAE
    mae = np.sum(abs(prediction - z)) / z.shape[0]
    
    #MSE
    mse = np.sum((prediction - z)**2) / z.shape[0]    
    
    #Delta
    deltaz = (prediction - z) / (1 + z)
    
    #Bias
    bias = np.sum(deltaz) / z.shape[0]
    
    #Precision
    nmad = 1.48 * np.median(abs(deltaz - np.median(deltaz)))
    
    #R^2 score
    r2 = r2_score(z, prediction)
    
    #All errors are stored in a json and saved in  save_path directory
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
    
    with open(save_path + 'Results.json', 'w') as file:
        json.dump(errs, file, indent=5)

    #System info is invoked
    system_info(save_path) 
        

        


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