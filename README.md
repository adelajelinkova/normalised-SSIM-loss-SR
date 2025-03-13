# normalised-SSIM-loss-SR
Code for super-resolution algorithm training, inference for observing SSIM_loss in minimising composite loss component function

run this to open virtual environment (1st time settings below):
.\tf1_gpu_env\Scripts\activate

to train:

1) edit train_config_tf.yml file - set experiment folder and starting weights (empty if starting from zero, ckpt file if continuing)
2) python Train_final_combined.py 

to infer (not yet implemented):
1) edit infer_config.yml
2) python Infer_final_combined.py

cls to clear screen

When you're done working inside the virtual environment, deactivate it by running:
deactivate

Before first run:
1. Install Python 3.6.5 and CUDA 10.0 for the Virtual Environment
You'll first need to install Python 3.6.5, then CUDA 10.0. If you have multiple versions installed, you can keep other CUDA as the default for your general usage and use CUDA 10.0 only within your virtual environment for TensorFlow 1.15.

2. Install Python Virtual Environment
If you haven't already installed virtualenv, install it using pip:
pip install virtualenv

3. Create a New Virtual Environment
Create a virtual environment that will be dedicated to your project using CUDA 10.0 and TensorFlow 1.15.

Run the following command in your project directory:
virtualenv tf1_gpu_env

This will create a virtual environment named tf1_gpu_env in the current directory.

4. Activate the Virtual Environment
run this:
.\tf1_gpu_env\Scripts\activate
Once activated, your terminal prompt will change to indicate that you are working inside the tf1_gpu_env virtual environment.

5. Set Environment Variables for CUDA 10.0
While in the virtual environment, set the CUDA 10.0 environment variables:

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%

You can add these commands to your activate.bat file located in tf1_gpu_env\Scripts\activate.bat to automate the setting of these environment variables every time the virtual environment is activated.

6. Install Project Requirements in the Virtual Environment
With the virtual environment activated, you can now install all the project dependencies, including TensorFlow 1.15 with GPU support, using a requirements.txt file.

If you have a requirements.txt file with your project’s dependencies, you can install everything at once. Save this:

tensorflow-gpu==1.15.0
Pillow==8.4.0
numpy==1.19.5
pyyaml==6.0
scikit-image==0.17.2
scipy==1.5.4

to requirements.txt

To install it:

pip install -r requirements.txt
7. Verify GPU Availability
To ensure that TensorFlow is using CUDA 10.0 and your GPU is available, run the following Python script, named gpu_checker.py:


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU is available:", tf.test.is_gpu_available())
This will confirm that TensorFlow 1.15 is using the correct CUDA 10.0 version inside the virtual environment.

run as:
python gpu_checker.py

8. Deactivating the Virtual Environment
When you're done working inside the virtual environment, deactivate it by running:

deactivate

or just simply close the console
