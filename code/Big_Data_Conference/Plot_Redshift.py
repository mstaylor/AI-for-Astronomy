import math
import numpy as np
import torch, json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

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
    
    plt.savefig(save_plot_path)
    
    
def err_calculate(prediction, z, save_plot_path):
    prediction = np.array(prediction)
    z = np.array(z)
    mae = np.sum(abs(prediction - z)) / z.shape[0]
    mse = np.sum((prediction - z)**2) / z.shape[0]
    
    deltaz = (prediction - z) / (1 + z)
    bias = np.sum(deltaz) / z.shape[0]
    nmad = 1.48 * np.median(abs(deltaz - np.median(deltaz)))
    r2 = r2_score(z, prediction)
    
    
    errs = {
    'MAE': mae,
    'MSE': mse,
    'Bias': bias,
    'Precision': nmad,
    'R2': r2
    }
    
    with open(save_plot_path + '_Results.json', 'w') as file:
        json.dump(errs, file, indent=5)
