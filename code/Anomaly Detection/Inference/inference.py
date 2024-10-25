import sys, argparse

sys.path.append('/scratch/aww9gh/Cosmic_Cloud/AI-for-Astronomy/code/Anomaly Detection/') #adjust based on your system's directory
import torch, time
import numpy as np
import Plot_Redshift as plt_rdshft
from torch.utils.data import DataLoader
from blocks.model_vit_inception import ViT_Astro 


#Load Data
def load_data(data_path, device):
    return torch.load(data_path, map_location = device)

#Load Model
def load_model(model_path, device):
    model = torch.load(model_path, map_location = device)
    return model.module.eval()

#Use DataLoader for iterating over batches
def data_loader(data, batch_size):
    return DataLoader(data, batch_size = batch_size, drop_last = False)   #Drop samples out of the batch size


# Iterate over data for predicting the redshift and invoke the evaluation modules
def inference(model, dataloader, real_redshift, plot_to_save_path, device, batch_size):
    redshift_analysis = []
    total_time = 0.0   # Initialize total time for execution
    num_batches = 0    # Initialize number of batches
    total_data_bits = 0   # Initialize total data bits processed

    for i, data in enumerate(dataloader):
        image = data[0].to(device)  # Image is permuted, cropped and moved to cuda
        magnitude = data[1].to(device)  # Magnitude of channels
        redshift = data[2].to(device)  # Target, which is the redshift
        
        start_time = time.time()       # Put the start time of the execution
        predict_redshift = model([image, magnitude])  # Model predicts the redshift using two inputs (image and magnitudes)
        end_time = time.time()  # Put the end time of the execution

        redshift_analysis.append(predict_redshift.view(-1, 1))
        num_batches += 1
        
        # Calculate data size for this batch
        current_batch_size = image.shape[0]  # Get the actual batch size in case it's different from the expected one
        image_bits = current_batch_size * 5 * 32 * 32 * 32  # Batch * channels * height * width * float bits (32 bits)
        magnitude_bits = current_batch_size * 5 * 64  # Batch * number of magnitudes * float bits (64 bits)
        total_data_bits += (image_bits + magnitude_bits)  # Accumulate data bits for this batch
        
        total_time += (end_time - start_time)
        

    avg_time_batch = total_time / num_batches  # Average execution time per batch
    
    num_samples = len(real_redshift)    
    redshift_analysis = torch.cat(redshift_analysis, dim=0)
    redshift_analysis = redshift_analysis.cpu().detach().numpy().reshape(num_samples,) 
    
    # Execution Info and Throughput Calculation
    execution_info = {
        'total_time': total_time,
        'execution_time': avg_time_batch,   # Average execution time per batch
        'num_batches': num_batches,   # Number of batches
        'batch_size': batch_size,   # Batch size
        'device': device,   # Selected device
        'throughput_bps': total_data_bits / total_time,   # Throughput in bits per second (using total_time for all batches)
        'sample_persec': num_samples / total_time  # Number of samples processed per second
    }

    # Invoke the evaluation metrics
    plt_rdshft.err_calculate(redshift_analysis, real_redshift, execution_info, plot_to_save_path)  

#This is the engine module for invoking and calling various modules
def engine(args):
    data = load_data(args.data_path, args.device)
    dataloader = data_loader(data, args.batch_size)
    model = load_model(args.model_path, args.device)
    inference(model, dataloader, data[:][2].to('cpu'), args.plot_path, device = args.device, batch_size = args.batch_size)

    
# Pathes and other inference hyperparameters can be adjusted below
if __name__ == '__main__':
    prj_dir = '/scratch/aww9gh/Cosmic_Cloud/AI-for-Astronomy/code/Anomaly Detection/' #adjust based on your system's directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type = str, default = '/scratch/aww9gh/Cosmic_Cloud/AI-for-Astronomy/data/fraction_data/0.5.pt')
    parser.add_argument('--model_path', type = str, default  = prj_dir + 'Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt')
    parser.add_argument('--device', type = str, default = 'cpu')    # To run on GPU, put cuda, and on CPU put cpu

    parser.add_argument('--plot_path', type = str, default = prj_dir + 'Plots/')
    args = parser.parse_args()
    
    engine(args)