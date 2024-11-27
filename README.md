# BiofilmOCT
OCT images

OCT Biofilm analysis:

It is recommended to use a high-performance computer for this program such as university HPC cluster specially when processing large number of images at the same time. 

Please download the code into your preferred directory and enter the directory using $cd 

add the following weight to the Models_pretrained folder using the command:
 $wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


HPC Cluster:
For UoE HPC cluster guide please visit:  Cluster, or request an account 
For using HPC cluster, you can use MoboXterm software or your command prompt. To login to your cluster account, run:
	$ ssh -l <username> -X ceres.essec.ac.uk
Then, inter your password and Authenticator code. After login, use the command to open an interactive session:
	$ qrsh
Once you logged in to an interactive session, you can access python to run your code.
To access to your file and codes, you must direct to the folder which your files and code are located.
To install requirements, on the directory which requirement.txt file is located, run:
	$ pip3.9 install -r requirements.txt
For running the code, direct to the path with main.py file and run:
$ python39 main.py -pathimage <image directory> -pathsave <save directory> -model YOLO -imagetype oct
The program has been tested on python 3.9. 

See an example below, after login to the cluster, by considering that the project folder is in /projects/oct_project/ directory. You can change the paths for images and save directory to any directory that your images are located or you want to save the results:

qrsh
cd /projects/oct_project/oct_biofilm
pip3.9 install -r requirements.txt
python39 main.py -pathimage /projects/oct_project/oct_images -pathsave /projects/oct_projects/oct_result -model YOLO -imagetype oct



CONDA:
Installation Instructions
If you want to use it on your computer, make sure anaconda is install on your computer. For installing it, you must follow steps below:
1-	Download Anaconda
2-	In installation process, make sure you tick the option for add your python to the system path.
3-	After completing installation process, you can search anaconda navigator using windows search and run it to access to all python IDE and modules.
For Linux, download the proper anaconda version. The setup process is similar to Windows. For more information, please see anaconda guide.

Conda Environment Setup 
If you prefer using Conda, you can create a Python 3.9 environment for Windows and Linux by following these steps:
1.	Open your terminal and run:
$ conda create -n <environment name> python=3.9
2.	Then you need to activate your environment. For this purpose, run:
$ conda activate <environment name>
3.	Then you access to your environment and you can install packages using pip or conda command: 
$ pip install <package name> 
or 
$ conda install <package name>
4.	You can run your code on your environment by running command: 
$ python <your code name>.py
5.	To exit from your environment run: 
$ conda deactivate

To install all the required dependencies, on your command prompt:
1.	Navigate to the directory where requirements.txt file is located.
2.	On the code folder, run the following command in your command prompt:
$ pip install -r requirements.txt
3.	After installing requirements, run the code:
$ python main.py -pathimage <image directory> -pathsave <directory to save folder> -model YOLO -imagetype oct

See an example below, by considering that the project folder is in /projects/oct_project/ directory. You can change the paths for images and save directory to any directory that your images are located or you want to save the results:

For Linux:
conda create -n oct_env python=3.9
After installing your environment:
conda activate oct_env
cd /projects/oct_project/oct_biofilm
pip install -r requirements.txt
python main.py -pathimage ../oct_images -pathsave ../oct_result -model YOLO -imagetype oct

For Windows:
conda create -n oct_env python=3.9
After installing your environment:
conda activate oct_env
cd C:\users\Download\projects\oct_project\oct_biofilm
pip install -r requirements.txt
python main.py -pathimage ../oct_images -pathsave ../oct_result -model YOLO -imagetype oct

Conda Environment Setup using yml file
The following steps will create an environment called BiofilmOCT with the required packages.
1.	Open your terminal and run:
$ conda env create -f environment.yml


Using python:
We recommend using python 3.9. If you do not have python installed on your system, please follow the official guide for installation: Download python 



To install all the requirements:
1.	Open command prompt.
2.	To install packages in python use:
$ pip install <package name>
3.	Redirect to the directory where requirements.txt is located.
4.	On the code folder, run the following command in your command prompt:
$ pip install -r requirements.txt
5.	After installing requirements, run the code:
$ python main.py -pathimage <image directory> -pathsave <directory to save folder> -model YOLO -imagetype oct

See an example below, by considering that the project folder is in /projects/oct_project/ directory. You can change the paths for images and save directory to any directory that your images are located or you want to save the results:

For Linux:
cd /projects/oct_project/oct_biofilm
pip install -r requirements.txt
python main.py -pathimage ../oct_images -pathsave ../oct_result -model YOLO -imagetype oct

For Windows:
cd C:\users\Download\projects\oct_project\oct_biofilm
pip install -r requirements.txt
python main.py -pathimage ../oct_images -pathsave ../oct_result -model YOLO -imagetype oct


Main code arguments description
The arguments list in table below.
Argument	Description	Default value
-h, --help	show this help message and exit.	None
-pathimage	path which the OCT image Experiments are located.	None
-pathsave	path which segments OCT images will be saved.	None
-model	YOLO, SAM, or SEGFORMER	YOLO
-imagetype	oct or img	oct

Result analysis
All results from different runs are saved in the pathsave directory specified by the user when they run the code. Within this directory, each model has its own folder, named according to the model at the end of its name. The results for each model are saved in their respective folders.
The models—YOLO, SAM, and SEGFORMER—each generate three images for each OCT or image file. These image files are named with the suffixes Raw, Mask, and Line. Raw images are visualized images extracted from the OCT file, mask images are segmentation outputs from the model, and line images are images related to the measurement calculations.

In Line images blue line shows the substrate(which has been rotated based on the slope), red line is the surface(topmost pixel on vertical axis), and yellow line is the average height of the surface.

Additionally, there is a CSV file which contains the measurement calculations. The columns are: 

•	FilePath
•	Plate
•	Volume
•	Thickness
•	Rq
•	Ra
•	Density Distribution
•	Pixel-based Density



Notes:

•	*This code does not support 3-dimensional images at the moment. *

•	The recommended image type is OCT

•	It is recommended to use this program on cluster in a conda environment. create conda environments on the CERES Cluster.

Troubleshot:

•	SAM weights:
It needs to be downloaded if you got the code from github.
There might be errors related to sam_vit_h_4b8939.pth The weight is located in the zip folder, ocationally, it gets corrupted in the download/upload/extracting process.
If there is any error related just delete the weight and download the new weight directry to the directory using:
 $wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

•	In case of any error related to installation of SAM(Segment anything). It can be installed using:
$pip install 'git+https://github.com/facebookresearch/segment-anything.git'

•	While creating environment on cluster for the first time, with running “conda activate” you might encounter an error: CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run:
$ conda init bash
Once it is done, you would need to log out from the interactive session and then use $qrsh for a new session. Then you would see a (base) at the beginning of your prompt.
Example:
Before conda: [pa19618@compute-1-28 OCTBiofilm]$
After using conda: (base) [pa19618@compute-1-28 OCTBiofilm]$
Conda environment (biofilm) active: (biofilm) [pa19618@compute-1-28 OCTBiofilm]$
•	Extra / or \ at the end of the paths, would likely cause errors in saving and prevents the program from running.

