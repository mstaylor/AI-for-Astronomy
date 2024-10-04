import sys
import argparse
sys.path.append('/scratch/aww9gh/Cosmic_Cloud/Big_Data_Conference/') #adjust based on your system's directory
import torch
import numpy as np
import Plot_Redshift as plt_rdshft
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from blocks.model_vit_inception import ViT_Astro 



#Load Data
def load_data(data_path):
    return torch.load(data_path)

#Load Model
def load_model(model_path):
    model = torch.load(model_path)
    return model.module.eval()

#Use DataLoader for iterating over batches
def data_loader(data, batch_size):
    return DataLoader(data, batch_size = batch_size, drop_last = False)

#Crop images to the suitable input size of the model(32 * 32 * 5)
def image_resize(image_size):
    return Compose([
            CenterCrop((image_size, image_size)),
    ])


#Iterate over data for predicting the redshift and invoke the evaluation modules
def inference(model, dataloader, real_redshift, resizing, plot_to_save_path, device = 'cuda'):
    
    redshift_analysis = []
    for i, data in enumerate(dataloader):
        image = resizing(data[0].permute(0, 3, 1, 2)).to(device) #Image is permuted, cropped and moved to cuda
        magnitude = data[1].to(device) #magnitude of of channels
        redshift = data[2].to(device) #target, which is the redshift
        
        predict_redshift = model([image, magnitude]) #model predicts the redshft using two inputs (image and magnitudes)
        
        redshift_analysis.append(predict_redshift.view(-1, 1))

        
    redshift_analysis = torch.cat(redshift_analysis, dim = 0)
    
    redshift_analysis = redshift_analysis.cpu().detach().numpy().reshape(real_redshift.shape[0],)
    
    plt_rdshft.plot_density(redshift_analysis, real_redshift, plot_to_save_path) #invoke for generating density scatter plot
    plt_rdshft.err_calculate(redshift_analysis, real_redshift, plot_to_save_path) #invoke for calculating statistical prediction evaluation metrics 

#This is the engine module for invoking and calling various modules
def engine(args):
    data = load_data(args.data_path)
    dataloader = data_loader(data, args.batch_size)
    model = load_model(args.model_path)
    resizing = image_resize(args.image_size)
    inference(model, dataloader, data[:][2], resizing, args.plot_path, device = args.device)

    
# Pathes and other inference hyperparameters can be adjusted below
if __name__ == '__main__':
    prj_dir = '/scratch/aww9gh/Cosmic_Cloud/Big_Data_Conference/' #adjust based on your system's directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--data_path', type = str, default = 'Inference.pt')
    parser.add_argument('--model_path', type = str, default  = prj_dir + 'Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt')
    parser.add_argument('--device', type = str, default = 'cuda')

    parser.add_argument('--plot_path', type = str, default = prj_dir + 'Plots/inference.png')
    args = parser.parse_args()
    
    engine(args)
    
