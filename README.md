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
- [Examples](#examples)
- [Contributing](#contributing)
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
Parent
 |--Models		# Holds the various trained models 
 |   |--Task1_No_Pretraining
 |   |--Task1_Pretrained
 |   |--hyperparameters.pkl                              #pickle file for hyperparameter values
 |   |--hyperparameters_TransferLearning.pkl             #pickle file 
 |--Task1
 |   |--Task1.ipynb               	 	         #Training of ResNet-18 model and hyperparameter tuning
 |   |--Task1_LossFunction_Tuning.ipynb	                 #Plotting performance for selected hyperparameters
 |   |--Task1_TransferLearning.ipynb	#Training using weights from Pytorch’s IMAGENET1K_V1
 |   |--Task1_TransferLearning_NLLLoss.ipynb       # Plotting performance for selected                                                                                                                                                                           hyperperparameters
 |--Task2
 |   |--Task2_FeatureExtraction_KNN
 |   |	|--Task2_FeatureExtraction_D2_KNN (1).ipynb    #SVM on features extracted using  ResNet 18 on Prostate Cancer dataset 
 |   |	|--Task2_FeatureExtraction_D2_TL_KNN.ipynb     #SVM on features extracted using encoder trained on weights of IMAGENET1K_V1 on Prostate Cancer dataset
 |   |	|--Task2_FeatureExtraction_D3_KNN.ipynb	       #SVM on features extracted using  ResNet 18 on Animal Faces dataset
 |   |	|--Task2_FeatureExtraction_D3_TL_KNN.ipynb     #SVM on features extracted using encoder trained on weights of IMAGENET1K_V1 on Animal Faces dataset
 |   |--Task2_FeatureExtraction_SVM
 |   |	|--Task2_FeatureExtraction_D2_SVMipynb	#SVM on features extracted using  ResNet 18 on Prostate Cancer dataset
 |   |	|--Task2_FeatureExtraction_D2_TL_SVM.ipynb	#SVM on features extracted using encoder trained on weights of IMAGENET1K_V1 on Prostate Cancer dataset
 |   |	|--Task2_FeatureExtraction_D3_SVM.ipynb	#SVM on features extracted using  ResNet 18 on Animal Faces dataset
 |   |	|--Task2_FeatureExtraction_D3_TL_SVM.ipynb	#SVM on features extracted using encoder trained on weights of IMAGENET1K_V1 on Animal Faces dataset
 |   |--Random Forest Implementation With Pytorch Model Update.ipy  #RF on Features extracted from both ResNET18 and IMAGENET1K_V1 on Prostate Cancer and Animal Faces dataset
 |   |--Task2_FeatureExtraction.ipynb
 |   |--Task2_FeatureExtraction_Final_SVM.ipynb
 |--readme.txt

![image](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/assets/8848193/f81a6e7a-a5b2-4e13-997a-45317ec89bbe)

## Installation

You can install these libraries using `pip`. Make sure to install specific versions for compatibility:

- [PyTorch](https://pytorch.org/) (Version X.X.X)
  ```bash
  pip install torch==X.X.X


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
