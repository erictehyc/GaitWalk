# Only Tested on Ubuntu for now
## Requirement
1. Needs ffmpeg installed.
2. Ensure Gstreamer is installed on Ubuntu / Klite codec is installed on Windows
3. Requires Cuda 10.1 and Cudnn Installed to run tensorflow with GPU. Follow installation shown in tensorflow-gpu installation guide.

To use this module, 
1. Git clone this project.
2. Navigate to base folder and run installation with: 
	ipython installation.py
This might take a while as it downloads the ZQPei/deep_sort_pytorch module from https://github.com/ZQPei/deep_sort_pytorch.git to the base directory, and download the neede pretrained weights for yolov3.
3. Create a python virtual environment and install the requirements needed listed in "requirements.txt" with the following code depending on the version:

#### before virtualenv version 15.1.0
	virtualenv --no-site-packages --distribute .env &&\
	    source .env/bin/activate &&\
	    pip install -r requirements.txt

#### after deprecation of some arguments in 15.1.0
	virtualenv .env && source .env/bin/activate && pip install -r requirements.txt
You can check your virtualenv version with the following line on command prompt:

	virtualenv --version
To perform prediction on input videos, you can choose to open the UI by navigating to UI folder and running:

	python 0_mainUI.py
or choose to run the prediction file manually after setting appropriate variables with:

	python run_prediction.py
	
	




