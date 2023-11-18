# COMP6321 FALL 2023 Course Project
Machine Learning Project (Fall 2023) using pytorch
# Feature Extraction through CNN model-training and Transfer Learning for Classification using SVM, KNN and RF

This github repository is the Course Project Submission for Fall 2023 COMP 6321 – Machine Learning course. 

In this project, we aim to investigate the impact of model selection and initialization methods on classification performance, evaluate feature extraction efficiency in pre-trained models, and assess their adaptability across different domains.

This project has been implemented in a series of two tasks. In task 1, we train a CNN model from scratch using random weight initialization on the Colorectal Cancer (CRC) dataset for the task of image classification. 

Task 2 builds on Task 1 using the encoder from the CNN model trained in Task 1 to extract features for Dataset 2 ( Prostate Cancer ) and Dataset 3 ( Animal Faces Dataset ). Finally, the extracted features are employed for classification using Support Vector Machine (SVM) and Random Forest. The process is repeated for a pre-trained PyTorch model to compare classification performance for features extracted from first model and the second one. 

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


### Built With


* [![Python][Python-img]][Python-url]
* [![PyTorch][Pytorch-img]][Pytorch-url]
* [![Colab][Colab-img]][Colab-url]
* [![Matplotlib][Matplotlib-img]](https://matplotlib.org/stable/users/installing/index.html)
* [![NumPy][Numpy-img]](https://numpy.org/install/)
* [![SciPy](https://img.shields.io/badge/SciPy-1.2-green)](https://scipy.org/install/)
* [![scikit-learn](https://img.shields.io/badge/scikit--learn-0.21-green)](https://scikit-learn.org/stable/install.html)
* [![PIL](https://img.shields.io/badge/PIL-6.0-orange)](https://pillow.readthedocs.io/en/stable/installation.html)
* [![Pickle](https://img.shields.io/badge/Pickle-4.0-lightgrey)](https://docs.python.org/3/library/pickle.html)
* [![THOP](https://img.shields.io/badge/THOP-0.0.31-blue)](https://pypi.org/project/thop/)


## Overview
## Repository Structure

![image](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/assets/8848193/91a97c8b-efb7-443c-9f4f-cc6dadf1b50b)


## Installation
Python version used for this project is 3.10.2 which can be downloaded and installed using the installer on https://www.python.org/downloads/release/python-3102/

You can install the needed python libraries using `pip`. Make sure to install specific versions for compatibility:

```bash
pip install pandas==2.0.3
pip install torch==2.0.0
pip install torchvision==0.15.1
pip install tqdm==4.66.1
pip install Pillow==9.5.0
pip install json==2.0.9
pip install scikit-learn==1.2.2
pip install numpy==1.23.5
pip install matplotlib==3.7.2
```
To run the code, you can use one of three options - 
### Option 1: Install Anaconda
Installing Anaconda allows to run all Jupyter Notebook files on local computer. If you haven't installed Anaconda, go here: https://store.continuum.io/cshop/anaconda/ This will install everything that you need.

### Option 2: Run on Google Colab
Running on Google Colab without local computer setup, which requires a Google Colab account. If you haven't register Colab, go here: https://colab.research.google.com/signup

### Option 3: Install jupyter notebook seperately on your system  
Jupyter notebbok can be installed independently from anaconda using the following cmd command -
>>> python -m pip install jupyter

To run the jupter notebook, open cmd and enter the following command - 
>>> jupyter notebook

## Usage

# Dataset Download Links

- **Colorectal Cancer Classification**: [Download Link](https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp)
- **Prostate Cancer Classification**: [Download Link](https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp)
- **Animal Faces Classification**: [Download Link](https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp)

# Task 1 Instructions

To run Task 1 Jupyter Notebook files, follow these steps:

1. Open each notebook in the Task1 folder.
2. Update the following variables in each notebook:

   a. `path`: Path to the folder where you downloaded the Colorectal Cancer dataset.
   b. `saveFilePath`: Path to save the hyperparameters (include the file name with extension).
   c. `saveModelPath`: Path to save the trained ResNet18 model.
   d. `imageInputChange`: Path of the file in the last code cell of each file (Cell titled "Model FLOPS").

   Note: Provide different paths for `saveFilePath` in each notebook to get different hyperparameter files.

3. Save the changes and run the notebooks.

# Task 2 Instructions

For Task 2, update variables in each Jupyter Notebook as follows:

1. In `Task2_FeatureExtraction_KNN` and `Task2_FeatureExtraction_SVM`:

   a. `path`: Point to the folder where you downloaded the Prostate Cancer Dataset in the 3rd code cell.
   b. `path1`: Point to the folder where you downloaded the Animal Face Dataset in the 17th code cell.
   c. `imageInput`: Point to an image file inside the Colorectal Cancer dataset.

2. In other Task 2 files:

   a. `pcancer_path`: Points to the folder where Prostate Cancer Dataset is downloaded.
   b. `ccancer_path`: Points to the folder where Colorectal Cancer Dataset is downloaded.
   c. `afaces_path`: Points to the folder where Animal Faces Dataset is downloaded.
   d. `imageInput`: Points to an image in the Colorectal Cancer dataset.

3. Inside Task 2 files, change the path for the variable `model1` to point to the path where you saved the models trained in Task 1.

4. Save the changes and run the notebooks.

Feel free to reach out if you encounter any issues or have questions!


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[Python-img]: https://img.shields.io/badge/Python-3.6%2B-blue
[Python-url]: https://www.python.org/
[Pytorch-img]: https://img.shields.io/badge/PyTorch-1.0%2B-orange
[Pytorch-url]: https://pytorch.org/
[Colab-img]: https://img.shields.io/badge/Colab-Notebook-yellow
[Colab-url]: https://colab.research.google.com/
[Matplotlib-img]: https://img.shields.io/badge/Matplotlib-v3.0-blue
[Numpy-img]: https://img.shields.io/badge/NumPy-1.16-yellow
[Scipy-img]: https://img.shields.io/badge/SciPy-1.2-green
[scikit-learn-img]: https://img.shields.io/badge/scikit--learn-0.21-green
[scikitLearn-url]: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/install.html)
[Pickle-img]: https://img.shields.io/badge/Pickle-4.0-lightgrey
[THOP-img]: https://img.shields.io/badge/THOP-0.0.31-blue
