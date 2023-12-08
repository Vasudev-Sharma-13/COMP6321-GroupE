# Feature Extraction through CNN model-training and Transfer Learning for Classification using SVM, KNN and RF

<div align="justify">

This study delves into the realm of transfer learning within deep learning (DL) to address the issue of prolonged training time. Two convolutional neural network models are trained on a medical-related dataset, one trained from scratch with randomly initialized weights and another with pre-trained ImageNet weights. Moreover, transfer learning is utilized to extract features from diverse datasets by pre-trained DL models and then Machine Learning classifiers are employed for feature classification. The study explores how transfer learning success depends on the similarity between the pre-trained and target datasets. Results demonstrate that employing ImageNet weights for training yields superior performance, with faster convergence. Furthermore, when comparing pre-trained models, one trained on a limited-size medical dataset and the other on a large, diverse image dataset, the latter excels at classifying new diverse datasets, while the specialized pre-trained model excels only on similar dataset to its pre-trained dataset.

This github repository is the Course Project Submission for Fall 2023 COMP 6321 – Machine Learning course. 
In this project, we aim to investigate the impact of model selection and initialization methods on classification performance, evaluate feature extraction efficiency in pre-trained models, and assess their adaptability across different domains.This project has been implemented in a series of two tasks. In task 1, we train a CNN model from scratch using random weight initialization on the Colorectal Cancer (CRC) dataset for the task of image classification. Task 2 builds on Task 1 using the encoder from the CNN model trained in Task 1 to extract features for Dataset 2 ( Prostate Cancer ) and Dataset 3 ( Animal Faces Dataset ). Finally, the extracted features are employed for classification using Support Vector Machine (SVM) and Random Forest. The process is repeated for a pre-trained PyTorch model to compare classification performance for features extracted from first model and the second one. 
</div>

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset-download-links)
- [Task 1 Instructions](#task-1-instructions)
- [Task 2 Instructions](#task-2-instructions)
   - [Support Vector Machines](#svm)
   - [K Nearest Neighbor](#knn)
   - [Random Forest](#random-forest)
- [Test Dataset](test-dataset-creation-code)
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

## Installation
Python version used for this project is 3.10.2 which can be downloaded and installed using the [installer](https://www.python.org/downloads/release/python-3102/)

You can install the needed python libraries using `pip`. Make sure to install specific versions for compatibility:

```bash
pip install pandas==2.0.3 torch==2.0.0 torchvision==0.15.1 tqdm==4.66.1 Pillow==9.5.0 json==2.0.9 scikit-learn==1.2.2 numpy==1.23.5 matplotlib==3.7.2
```
To run the code, you can use one of three options - 
### Option 1: Install Anaconda
Installing Anaconda allows to run all Jupyter Notebook files on local computer. If you haven't installed Anaconda, go [here](https://store.continuum.io/cshop/anaconda/) This will install everything that you need.

### Option 2: Run on Google Colab
Running on Google Colab without local computer setup, which requires a Google Colab account. If you haven't register Colab, go [here](https://colab.research.google.com/signup)

### Option 3: Install jupyter notebook seperately on your system  
Jupyter notebbok can be installed independently from anaconda using the following cmd command -
>>> python -m pip install jupyter

To run the jupter notebook, open cmd and enter the following command - 
>>> jupyter notebook


# Dataset Download Links

- Colorectal Cancer Classification [[Project Dataset](https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp)|[Original Dataset](https://zenodo.org/record/1214456)]
- Prostate Cancer Classification [[Project Dataset](https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp)|[Original Dataset](https://zenodo.org/records/4789576)]
- Animal Faces Classification[[Project Dataset](https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp)|[Original Dataset](https://www.kaggle.com/datasets/andrewmvd/animal-faces)]

# Task 1 Instructions

To run Task 1 Jupyter Notebook files, follow these steps:

1. There are four files for ResNet18 for solving Task 1 as follows

   a. **[Task1.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task1/Task1.ipynb)**: Implementation of the scratch model called CRC-Enc with grid search.

   b. **[Task1_LossFunction_Tuning.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task1/Task1_LossFunction_Tuning.ipynb)**: CRC-Enc with NLLLoss tuning for tuning purposes.

   c. **[Task1_TransferLearning.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task1/Task1_TransferLearning.ipynb)**: Model using ImageNet weights, referred to as Imag-Enc.

   d. **[Task1_TransferLearning_NLLLoss.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task1/Task1_TransferLearning_NLLLoss.ipynb)**: Imag-Enc with NLLLoss as the loss function for tuning purposes.

2. Load the desired file(s) using a editor(Jupyter Notebook,Google Colab, Kaggle Notebooks, etc.) compatible with .ipynb extension.

3. Download the dataset from [Download Dataset Section](#dataset-download-links).
   
4. Update the following variables in the jupyter notebook:

   a. `path`: Path to the folder where you downloaded the Colorectal Cancer dataset (under section "Setting Path").
   
      ```python
      # give path of the input dataset folder
      path="C:/Users/vshar/Downloads/Dataset 1/Dataset 1/Colorectal Cancer"
      ```

   b. `saveFilePath`: Path to save the hyperparameters (include the file name with extension)(under section "Setting Path").

      ```python 
      # give path to save the plot results(Example training vs epoch,loss vs steps,etc)
      saveFilePath="C:/Users/vshar/Documents/hyperparameters.pkl"
      ```

   c. `saveModelPath`: Path to save the trained ResNet18 model(under section "Setting Path").

      ```python
      # give path to save the trained model
      saveModelPath="C:/Users/vshar/Documents"
      ```

   d. `imageInputChange`: Path of the image under section "Model FLOPS".

      ```python
      # give path to image for calculatintg the FLOPS
      imageInputChange="C:/Users/vshar/Downloads/Dataset 1/Dataset 1/Colorectal Cancer/NORM/NORM-ADQNLKLS.tif"
      ```

5. Change the following variables under the "HyperParameter tuning" subsection in the notebook(only if tuning is required,otherwise skip this step)

    ```python
   #change input dimensions of the image fed to the CNN
   inputDimension=(256,256)
   #Setting different batch sizes
   batch_sizes=[128,64,32]
   #Setting different learning rates
   learning_rates=[0.00001,0.00005,0.0001,0.0005,0.001,.005,0.01,0.05]
   #Setting the number of epochs
   epochs=10
   #setting the loss function
   criterion=nn.CrossEntropyLoss()
   ```

6. Run the notebook, and see the results. The subheadings and comments in the respective ResNet18 notebooks explain each cell and it's functioning.

**Note**: The models generated from these jupyter files have been saved under the Models folder. For the scratch models refer [TASK1_No_Pretraining](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/tree/main/Models/TASK1_No_Pretraining) and for the ImageNet model refer [TASK1_Pretrained](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/tree/main/Models/TASK1_Pretrained). The accuracy, loss, etc have been stored in [hyperparameters.pkl](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Models/hyperparameters.pkl) for scratch models and for pretrained models refer [hyperparameters_TransferLearning.pkl](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Models/hyperparameters_TransferLearning.pkl). To load the content in these files use the following code:

   ```python
      saveFilePath="C:/Users/vshar/Documents/hyperparameters.pkl"
      with open(saveFilePath, 'rb') as f:
          data = pickle.load(f)
          hyper_parameters = data['hyper_parameters']
          train_acc_hyper_paramaters = data['train_acc_hyper_paramaters']
          train_acc_valid_hyper_paramaters = data['train_acc_valid_hyper_paramaters']
          train_loss_hyper_paramaters = data['train_loss_hyper_paramaters']
          train_acc_hyper_paramaters_step = data['train_acc_hyper_paramaters_step']
          train_acc_valid_hyper_paramaters_step = data['train_acc_valid_hyper_paramaters_step']
          train_loss_hyper_paramaters_step = data['train_loss_hyper_paramaters_step']
          test_loss=data['test_loss']
   ```

# Task 2 Instructions

## SVM

To run Task 2 Jupyter Notebook files, follow these steps:

There are four files for SVM for solving Task 2 using Support Vector Machines as follows

1. There are four files for the task as follows

   a. **[Task2_FeatureExtraction_D2_SVM.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_SVM/Task2_FeatureExtraction_D2_SVM.ipynb)**: Feature Extraction for Dataset2 using CRC_ENC followed by SVM predictions
   
   b. **[Task2_FeatureExtraction_D2_TL_SVM.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_SVM/Task2_FeatureExtraction_D2_TL_SVM.ipynb)**: Feature Extraction for Dataset2 using Pre_IMg followed by SVM predictions
   
   c. **[Task2_FeatureExtraction_D3_SVM.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_SVM/Task2_FeatureExtraction_D3_SVM.ipynb)**: Feature Extraction for Dataset3 using CRC_ENC followed by SVM predictions
   
   d. **[Task2_FeatureExtraction_D3_TL_SVM.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_SVM/Task2_FeatureExtraction_D3_TL_SVM.ipynb)**: Feature Extraction for Dataset3 using Pre_IMg followed by SVM predictions

2. Load the desired file(s) using a editor(Jupyter Notebook,Google Colab, Kaggle Notebooks, etc.) compatible with .ipynb extension.

3. Download the dataset from [Download Dataset Section](#dataset-download-links).

4. Update the following variables in the jupyter notebook:

   a. `path`: Path to the folder where you downloaded the Dataset2 and Dataset3
   
      ```python
      # give the path of the input dataset folder
      path = "/kaggle/input/comp6321-project-datasets/Dataset 2/Dataset 2/Prostate Cancer"
   
      ```
   
   b. `model1`: Path of the model to be loaded, either the one trained from task1 or the PreTrained model from PyTorch Using ImgNET weights
      
      ```python
      # For CRC_ENC use the following model
      model1 = torch.load('/kaggle/input/models/COMP6321_ResNet_Task1_CancerDataset_Model_Final_HyperParamaterTuning8.pth')
      ```
      
      ```python
      # For Pre_Img use the following model
      model1 = model1 = models.resnet18(weights="IMAGENET1K_V1").to(device)
      ```

5. The generated files by the notebook are:

   a. `extracted_features.csv`: Extracted Features using the encoder are saved in the file `extracted_features.csv`. Update the path for      `extracted_features.csv` to make `final_features.csv`
         
      ```
      '/kaggle/working/extracted_features.csv'
      ```
      
   b.  `final_features.csv`: Contains 512 extracted features along with the labels. 
        Update path for `final_features.csv` which will be used for SVM Predictions
         
      ```
      '/kaggle/working/final_features.csv'
      ```


6. Run the notebook, and see the results. The subheadings and comments in the respective notebooks explain each cell and its functioning.


## KNN

To run Task 2 Jupyter Notebook files, follow these steps:

There are four files for KNN for solving Task 2 using K-Nearest Neighbors as follows

1. There are four files for the task as follows

   a. **[Task2_FeatureExtraction_D2_KNN.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_KNN/Task2_FeatureExtraction_D2_KNN.ipynb)**: Feature Extraction for Dataset2 using CRC_ENC followed by KNN predictions
   
   b. **[Task2_FeatureExtraction_D2_TL_KNN.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_KNN/Task2_FeatureExtraction_D2_TL_KNN.ipynb)**: Feature Extraction for Dataset2 using Pre_IMg followed by KNN predictions
   
   c. **[Task2_FeatureExtraction_D3_KNN.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_KNN/Task2_FeatureExtraction_D3_KNN.ipynb)**: Feature Extraction for Dataset3 using CRC_ENC followed by KNN predictions
   
   d. **[Task2_FeatureExtraction_D3_TL_KNN.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task2_FeatureExtraction_KNN/Task2_FeatureExtraction_D3_TL_KNN.ipynb)**: Feature Extraction for Dataset3 using Pre_IMg followed by KNN predictions

2. Load the desired file(s) using a editor(Jupyter Notebook,Google Colab, Kaggle Notebooks, etc.) compatible with .ipynb extension.

3. Download the dataset from [Download Dataset Section](#dataset-download-links).

4. Update the following variables in the jupyter notebook:

   a. `path`: Path to the folder where you downloaded the Dataset2 and Dataset3
   
      ```python
      # give the path of the input dataset folder
      path = "/kaggle/input/comp6321-project-datasets/Dataset 3/Dataset 3/Animal Faces"
   
      ```
   
   b. `model1`: Path of the model to be loaded, either the one trained from task1 or the PreTrained model from PyTorch Using ImgNET weights
      
      ```python
      # For CRC_ENC use the following model
      model1 = torch.load('/kaggle/input/models/COMP6321_ResNet_Task1_CancerDataset_Model_Final_HyperParamaterTuning8.pth')
      ```
      
      ```python
      # For Pre_Img use the following model
      model1 = model1 = models.resnet18(weights="IMAGENET1K_V1").to(device)
      ```
      
5. The generated files by the notebook are:

   a. `extracted_features.csv`: Extracted Features using the encoder are saved in the file `extracted_features.csv`.
       Update the path for `extracted_features.csv` to make `final_features.csv`
         
      ```
      '/kaggle/working/extracted_features.csv'
      ```
      
   b.  `final_features.csv`: Contains 512 extracted features along with the labels.
        Update path for `final_features.csv` which will be used for KNN Predictions
         
      ```
      '/kaggle/working/final_features.csv'
      ```
   
6. Run the notebook, and see the results. The subheadings and comments in the respective notebooks explain each cell and its functioning.


## Random Forest: 
To run the Task 2 files pertaining to the training of Random Forest on the features extracted from the models trained in Task 1 and using Transfer Learning, follow these steps:

1. There is only one file that needs to be run. The file is named - **[Task 2 Random Forest.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Task2/Task%202%20Random%20Forest.ipynb)** - This file can be accessed thorugh the given https link or it can be located inside the Task 2 folder of the github repository for this project.

2. Load the desired file(s) using a editor(Jupyter Notebook,Google Colab, Kaggle Notebooks, etc.) compatible with .ipynb extension.

3. Download the dataset from [Download Dataset Section](#dataset-download-links).

4. Update the following variables in the jupyter notebook: ( These variable can be found in the 4th cell of the jupyter notebook. )

   a. `pcancer_path`: Path to the folder where you downloaded the Dataset2 - Prostate Cancer Dataset
   
      ```python
      # give the path of the input dataset folder
      pcancer_path = "/kaggle/input/comp6321-project-datasets/Dataset 2/Dataset 2/Prostate Cancer"
   
      ```
   
   b. `ccancer_path`: Points to the folder where the Colorectal Cancer Dataset is downloaded.

      ```python
      # give the path of the input dataset folder
      ccancer_path = "/kaggle/input/comp6321-project-datasets/Dataset 1/Dataset 1/Colorectal Cancer "
   
      ```

   c. `afaces_path`: Points to the folder where the Animal Faces Dataset is downloaded.

      ```python
      # give the path of the input dataset folder
      afaces_path = "/kaggle/input/comp6321-project-datasets/Dataset 3/Dataset 3/Animal Faces"
   
      ```

   d. `model1_path`: path where the model trained for Task 1 is stored
   
      ```python
      model1_path = "/kaggle/input/models/COMP6321_ResNet_Task1_CancerDataset_Model_Final_HyperParamaterTuning8.pth"
   
      ```

   e. `plot_img_path`: path to one of the images stored in Prostate Cancer dataset
   
      ```python
      plot_img_path = '/kaggle/input/comp6321-project-datasets/Dataset 2/Dataset 2/Prostate Cancer/tumor/tu.1001.jpg'
   
      ```

   f. `output_path`: path to the folder where you want to store the various intermediate output files like the files which store the extracted features. ( Note - give the folder path and not the file path ) 
   
      ```python
      output_path = "/kaggle/working"
   
      ```
      The generate files in this notebook are -
         a.) extracted_features_pm1.csv and final_features_pm1.csv
         b.) extracted_features_pm2.csv and final_features_pm2.csv
         c.) extracted_features_am1.csv and final_features_am1.csv
         d.) extracted_features_am2.csv and final_features_am2.csv
         e.) feature_maps_pm1.jpg
         f.) feature_maps_pm2.jpg
   
5. Run each cell in the notebook in a sequential order and see the results.

## Test Dataset Creation code:

1. There is only one file that needs to be run. The file is named - **[Test Dataset Creation.ipynb](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/Test%20Dataset%20Creation.ipynb)** - This file can be accessed thorugh the given https link or it can be located inside the parent folder of the github repository for this project.

2. Load the desired file(s) using a editor(Jupyter Notebook,Google Colab, Kaggle Notebooks, etc.) compatible with .ipynb extension.

3. Download the datasets from [Download Dataset Section](#dataset-download-links). Please store all 3 datasets inside one folder ( with sub folders for each dataset ).

4. Update the following variables in the jupyter notebook: ( These variable can be found in the 4th cell of the jupyter notebook. )

   a. `data_dir`: Path to the folder where you downloaded all the 3 datasets
   
      ```python
      # give the path of the folder containing all 3 datasets
      data_dir = '/kaggle/input/comp6321-project-datasets/'
   
      ```
   
   b. `test_dir`: Output directory where you want to store the test dataset containg 100 images ( 11 images per class of each of the 3 datasets )

      ```python
      # give the path of the output folder
      test_dir = '/kaggle/Test100/'

        ```
5. Now that the test dataset folder is created, you can use the paths inside these test dataset in the Task 2 Jupyter files as path of your datasets to test their performance. Alternatively, you can simply download the provided dataset zip folder provided in the project submission directly after decompressing it.
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
