Inference Step – by – Step Instructions 

Note: Screenshots and instructions are from a Windows computer. Not much should differ on a Mac. 

Clone this GitHub repository. UVA-MLSys/AI-for-Astronomy (github.com) Two options a and b below. 

In the terminal 

Copy the web url 

 

Navigate to the terminal and enter: git clone https://github.com/UVA-MLSys/AI-for-Astronomy.git 

 

Follow the prompted instructions regarding entering GitHub username, password, etc. 

If you have issues with option a, download ZIP file 

From the GitHub page, click “Download ZIP” 

From your file explorer, “Extract all” It is important that all files and their structure are maintained. 

 

Save the elements of the folder to the desired directory that you will be running the python script. Add the unzipped folder to Rivanna directory or make sure they can be properly accessed by whichever IDE you will be using. Again, it is important that all files and their structure are maintained. 

Navigate to the directory path on your machine: C:\....AI-for-Astronomy-main\AI-for-Astronomy-main\code\Big_Data_Conference\Inference 

Notice the inference.py file. There are three places in the file that you will have to update the directory based on the path on your machine: lines 3, 65, and 69. See below that lines 3 and 65 navigate to the “Big_Data_Conference” folder and line 69 navigates to the “Inference.pt” dataset. 

 

 

In the terminal, run the inference using: python inference.py  

This may take about one minute to run completely.  

 

You may receive errors prompting you to pip install libraries. Note that the timm library version must be 0.4.12 

 

Once the run is complete, navigate to the directory path on your machine: C:\...AI-for-Astronomy-main\AI-for-Astronomy-main\code\Big_Data_Conference\Plots 

Open the “inference.png” and “inference.png_Results.json” files to view the results. 
