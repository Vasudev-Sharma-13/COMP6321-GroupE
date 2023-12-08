# Feature Extraction through CNN model-training and Transfer Learning for Classification using SVM, KNN and RF

<div align="justify">

This study delves into the realm of transfer learning
within deep learning (DL) to address the issue of prolonged
training time. Two convolutional neural network models
are trained on a medical-related dataset, one trained from
scratch with randomly initialized weights and another with
pre-trained ImageNet weights. Moreover, transfer learn-
ing is utilized to extract features from diverse datasets by
pre-trained DL models and then Machine Learning classi-
fiers are employed for feature classification. The study ex-
plores how transfer learning success depends on the simi-
larity between the pre-trained and target datasets. Results
demonstrate that employing ImageNet weights for training
yields superior performance, with faster convergence. Fur-
thermore, when comparing pre-trained models, one trained
on a limited-size medical dataset and the other on a large,
diverse image dataset, the latter excels at classifying new di-
verse datasets, while the specialized pre-trained model ex-
cels only on similar dataset to its pre-trained dataset

This github repository is the Course Project Submission for Fall 2023 COMP 6321 – Machine Learning course. 
In this project, we aim to investigate the impact of model selection and initialization methods on classification performance, evaluate feature extraction efficiency in pre-trained models, and assess their adaptability across different domains.This project has been implemented in a series of two tasks. In task 1, we train a CNN model from scratch using random weight initialization on the Colorectal Cancer (CRC) dataset for the task of image classification. Task 2 builds on Task 1 using the encoder from the CNN model trained in Task 1 to extract features for Dataset 2 ( Prostate Cancer ) and Dataset 3 ( Animal Faces Dataset ). Finally, the extracted features are employed for classification using Support Vector Machine (SVM) and Random Forest. The process is repeated for a pre-trained PyTorch model to compare classification performance for features extracted from first model and the second one. 
</div>
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

3. Download the dataset from [Dataset download section](https://github.com/Vasudev-Sharma-13/COMP6321-GroupE/blob/main/README.md#dataset-download-links).
   
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

6. Run notebook, and see the results. The subheadings and comments in the respective ResNet18 notebooks explain each cell and functioning.


# Task 2 Instructions

For Task 2, update variables in each Jupyter Notebook as follows:

1. In `Task2_FeatureExtraction_KNN` and `Task2_FeatureExtraction_SVM`:

   a. `path`: Point to the folder where you downloaded the Prostate Cancer Dataset in the 3rd code cell.

   b. `path1`: Point to the folder where you downloaded the Animal Face Dataset in the 17th code cell.

   c. `imageInput`: Point to an image file inside the Colorectal Cancer dataset.

3. In other Task 2 files:

   a. `pcancer_path`: Points to the folder where Prostate Cancer Dataset is downloaded.

   b. `ccancer_path`: Points to the folder where Colorectal Cancer Dataset is downloaded.

   c. `afaces_path`: Points to the folder where Animal Faces Dataset is downloaded.

   d. `imageInput`: Points to an image in the Colorectal Cancer dataset.

5. Inside Task 2 files, change the path for the variable `model1` to point to the path where you saved the models trained in Task 1.

6. Save the changes and run the notebooks.

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
