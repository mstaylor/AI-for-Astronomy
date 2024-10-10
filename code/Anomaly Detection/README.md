
# AstroMAE: Redshift Prediction Using a Masked Autoencoder with a Novel Fine-Tuning Architecture

AstroMAE is a novel approach for redshift prediction, designed to address the limitations of traditional machine learning methods that rely heavily on labeled data and feature extraction. Redshift is a key concept in astronomy, referring to the stretching of light from distant galaxies as they move away from us due to the expansion of the universe. By measuring redshift, astronomers can determine the distance and velocity of celestial objects, providing valuable insights into the structure and evolution of the cosmos.

Utilizing a masked autoencoder, AstroMAE pretrains a vision transformer encoder on Sloan Digital Sky Survey (SDSS) images to capture general patterns without the need for labels. This pretrained encoder is then fine-tuned within a specialized architecture for redshift prediction, combining both global and local feature extraction. AstroMAE represents the first application of a masked autoencoder for astronomical data and outperforms other vision transformer and CNN-based models in accuracy, showcasing its potential in advancing our understanding of the cosmos.

In evaluating AstroMAE, several metrics were used to compare its performance with other models: Mean Absolute Error (MAE), Mean Square Error (MSE), Bias, Precision, and R² score. These metrics provide a comprehensive assessment of the model's prediction accuracy and reliability, highlighting the effectiveness of AstroMAE in redshift prediction tasks.

- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors between predicted and true values.
  
  $$
  MAE = \frac{1}{n} \sum_{i=1}^{n} |z_i - \hat{z}_i|
  $$

- **Mean Square Error (MSE)**: Measures the average of the squares of the errors between predicted and true values.
  
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (z_i - \hat{z}_i)^2
$$

- **Bias**: Measures the average residuals between predicted and true values, normalized by the true value.
  
$$
\text{Bias} = \left\langle \frac{\hat{z} - z}{1 + z} \right\rangle
$$

- **Precision**: Measures the expected scatter and is a robust measure of the spread of errors.
  
$$
\text{Precision} = 1.48 \times \text{median} \left( \left| \frac{\hat{z} - z}{1 + z} \right| \right)
$$

- **R² score**: Evaluates how well the model predicts, with a value closer to 1 indicating better performance.
 
  ![image](https://github.com/user-attachments/assets/9c1391d7-cc7c-4ee8-af7d-f0b7a5c4cdab)


In these formulas, $z$, $\hat{z}$, and $\bar{z}$ represent the ground-truth value, predicted value, and mean of the ground-truth values, respectively, and $\(n\)$ is the number of data samples.

# AI for Astronomy Inference Step-by-Step Guide

## Overview

This guide provides step-by-step instructions for running inference on the AI for Astronomy project. This process is intended for both Windows and Mac users, though the screenshots and terminal commands shown here are from a Windows computer.

## Prerequisites

- **Python** installed (3.x version recommended).
- Basic understanding of terminal usage.
- Ensure **Git** is installed to clone the repository or download the code.

### Step 1: Clone the Repository

You have two options to get the code:

### Option A: Clone via Terminal

1. Copy the GitHub repository URL: [UVA-MLSys/AI-for-Astronomy](https://github.com/UVA-MLSys/AI-for-Astronomy).
2. Open your terminal and navigate to the directory where you want to save the project.
3. Run the following command:
   ```sh
   git clone https://github.com/UVA-MLSys/AI-for-Astronomy.git
   ```
4. Follow the prompts to enter your GitHub username, password, or authentication token if required.

### Option B: Download as ZIP

1. From the GitHub page, click on "Download ZIP."
2. Extract the ZIP file by right-clicking on it in your file explorer and selecting "Extract All." Ensure that all files and their structure are maintained.

## Step 2: Set Up the Directory

- Save the extracted or cloned folder to the desired directory from which you will run the Python script.
- If you are using Rivanna or any other computing platform, ensure the folder structure remains intact and accessible by the Python environment or IDE you plan to use.

## Step 3: Update File Paths

1. Navigate to the following directory in your local project folder:
   ```
   AI-for-Astronomy-main\AI-for-Astronomy-main\code\Anomaly Detection\Inference
   ```
2. Locate the `inference.py` file in this directory.
3. Open `inference.py` and update the directory paths in the following lines:
   - **Line 3**: Update the path to point to the "Anomaly Detection" folder.
   - **Line 65**: Update the path as needed for your system.
   - **Line 69**: Update the path to point to the `Inference.pt` dataset.

## Step 4: Run the Inference Script

1. Open your terminal and navigate to the directory containing `inference.py`:
   ```sh
   cd C:\...\AI-for-Astronomy-main\AI-for-Astronomy-main\code\Anomaly Detection\Inference
   ```
2. Run the inference script using the following command:
   ```sh
   python inference.py
   ```
   - The script may take about one minute to complete.
   - If prompted for missing libraries, install them using `pip`. Ensure that the **timm** library version is **0.4.12**.

## Step 5: View Results

1. Once the script completes, navigate to the following directory:
   ```
   C:\...\AI-for-Astronomy-main\AI-for-Astronomy-main\code\Anomaly Detection\Plots
   ```
2. Open the following files to view the results:
   - **inference.png**: This contains a visual representation of the inference results.
   - **inference.png_Results.json**: This JSON file contains the detailed numerical results of the inference.

### Troubleshooting

- If you encounter issues with missing libraries, ensure you have installed all required packages by using `pip install`. The version of **timm** must be **0.4.12** to avoid compatibility issues.

### Notes

- Ensure that all directory paths are properly set according to your system's file structure.
- These instructions have been tested on both Windows and Mac systems, with only minor variations.

### Contact

For additional support, please open an issue on the GitHub repository or reach out to the project maintainers.


## Support

Don't hesitate to get in touch with us:

- aww9gh@virginia.edu
- ear3cg@virginia.edu
