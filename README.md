# Ad-Creative-Recognition-with-Computer-Vision
Classify images as 'ad creative' or 'non-ad creative' using a CNN-based model on ResNet50 architecture, pre-trained on ImageNet. Ideal for marketers, advertisers, and data scientists to automate ad creative identification in large image collections.

# Overview
The goal of this project is to develop a model that can accurately classify images into two categories: "ad creative" and "non-ad creative". The project leverages a pre-trained ResNet50 model and fine-tunes it on a custom dataset.

# Dataset
The dataset consists of two directories: ad-creatives and non_ad-creatives.
Each directory contains images in .jpg or .png format.
The images are resized to 224x224 pixels and normalized for training.

# Installation
1.Clone the repository: 
      git clone https://github.com/your-username/ad-creative-classification.git
      
      cd ad-creative-classification
      
2.Install the required dependencies:

      pip install -r requirements.txt
      
# Usage
1.Ensure your dataset is organized in the following structure:

ad-creative-classification/
├── ad-creatives/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── non_ad-creatives/
    ├── image1.jpg
    ├── image2.jpg
    └── ...


2.Run the training script:
    python train_model.py
3.The trained model will be saved as ad_deploy.h5 in the specified directory.

# Model Architecture
 The base model is a ResNet50 pre-trained on ImageNet.
Additional layers include a global average pooling layer, a dense layer with 128 units and ReLU activation, and a final dense layer with sigmoid activation for binary classification.
Evaluation
The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.
A sample of test images along with predicted labels and probability scores are displayed for manual inspection.

# Results
After training the model, you can evaluate its performance on the test set. Here are some key evaluation metrics:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
