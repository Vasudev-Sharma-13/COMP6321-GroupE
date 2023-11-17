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
* [![Python][Python]][Python-url]
* [![Pytorch][Pytorch]][Pytorch-url]
* [![Colab][Colab]][Colab-url]
* ![Matplotlib][Matplotlib]
* ![Numpy][Numpy]
* ![Scipy][Scipy]
* ![scikit-learn][scikit-learn]
* ![PIL][PIL]
* ![Pickle][Pickle]
* ![THOP][THOP]

## Overview
## Repository Structure

![image](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/assets/8848193/91a97c8b-efb7-443c-9f4f-cc6dadf1b50b)


## Installation
Python version used for this project is 3.10.2 which can be downloaded and installed using the installer on https://www.python.org/downloads/release/python-3102/

You can install the needed python libraries using `pip`. Make sure to install specific versions for compatibility:

[Pandas] (Version 2.0.3) ---> pip install pandas==2.0.3<br>
[PyTorch] (Version2.0.0) ---> pip install torch==2.0.0 <br>
[TorchVision] (Version 0.15.1 ) ---> pip install torchvision==0.15.1 <br>
[tqdm] (Version 4.66.1) ---> pip install tqdm==4.66.1 <br>
[PIL] (Version 9.5.0) ---> pip install PIL==9.5.0 <br>
[json] (Version 2.0.9) ---> pip install json==2.0.9 <br>
[SciKit Learn] (Version 1.2.2) ---> pip install sklearn==1.2.2 <br>
[NumPy] (Version 1.23.5) ---> pip install numpy==1.23.5 <br>
[MatPlotLib] (Version 3.7.2) ---> pip install matplotlib==3.7.2 <br>

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

##Usage

1. Download the three datasets involved using the following links :- <br>
   a.) Colorectal Cancer Classification - https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp <br>
   b.) Prostate Cancer Classification - https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp <br>
   c.) Animal Faces Classification - https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp <br>

2. To run task 1 ipynb files, first change the following variables in each of the ipynb files inside folder Task1 :- <br>
   a.) path --> give the path of the folder where you downloaded the Colorectal Cancer dataset <br>
   b.) saveFilePath --> path where you want to save the hyperparameters. i.e give path for the HyperParameter.pkl pickle file ( note - give the path along with the file name ) <br>
   c.) saveModelPath --> path where you want to save the trained ResNet18 model
   d.) imageInputChange path of the file in the last code cell of each file ( Cell titled Model FLOPS )

Variables a, b, and c can be found in the second cell of each file ( cell just below the heading - Setting Path ), in the 12th code cell ( under heading - Loading the loss,training accuracy,validation accuracy in a file ) and in the next cell ( 13th cell titled 'Saving the Models' ). Please note that you may give a different pickle save file path in each of the notebooks to get a different file for hyperparameters for all the four scenarios, but it is not a requirement.

3. For Task 2 - In each ipynb file inside subfolders Task2_FeatureExtraction_KNN, Task2_FeatureExtraction_SVM and the file Task2_FeatureExtraction.ipynb in the Task2 folder, you will have to change the following variables -
   a.) 'path' to point to the folder where you downloaded the Prostate Cancer Dataset in the 3rd code cell of each ipynb file ( cell titled 'Setting path' ) in Task2_FeatureExtraction_D2_KNN/SVM.ipynb and Task2_FeatureExtraction_D2_TL_KNN/SVM.ipynb. IN the other two files, make 'path' variable point to Animal Faces dataset.
   b.) 'path1' to point to the folder where you downloaded the Animal Face Dataset in the 17th code cell ( titled 'Dataset-3 Path' ) -- ( only in Task2_FeatureExtraction.ipynb )
   c.) imageInput to point to an image file inside the Colorectal Cancer dataset ( ( only in Task2_FeatureExtraction.ipynb )

4. For the rest of the files in Task 2, change following variables -
   a.) pcancer_path ---> points to the folder where Prostate Cancer Dataset has been downloaded
   b.) ccancer_path ---> points to the folder where Colorectal Cancer Dataset has been downloaded
   c.) afaces_path ---> points to the folder where Animal Faces Dataset has been downloaded
   d.) imageInput ---> points to an image in Colorectal Cancer dataset

5.) Inside files for task 2, you would have to change the path for varaible 'model1' to point to the path you gave to save the models trained in Task 1. 

Once you have changed the path variables in a jupyter notebook file, you can run it to see the results.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python]: https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Pytorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Colab]:https://colab.research.google.com/assets/colab-badge.svg
[Colab-url]: https://colab.research.google.com/notebooks/intro.ipynb
[Matplotlib]: https://matplotlib.org/
[Numpy]: https://numpy.org/
[Scipy]: https://img.shields.io/badge/Scipy-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[scikit-learn]: https://img.shields.io/badge/scikit-learn-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[PIL]: https://img.shields.io/badge/PIL-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Pickle]: https://docs.python.org/3/library/pickle.html
[THOP]: https://img.shields.io/badge/THOP-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
