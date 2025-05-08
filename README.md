** Facial Recognition with ViT on LFW Dataset
This project demonstrates facial recognition using a Vision Transformer (ViT) model on the LFW (Labeled Faces in the Wild) dataset. The dataset is provided as a zip archive (archive.zip), which will be uploaded and processed at runtime.

** Overview
The objective of this project is to classify faces into different individuals using a pre-trained Vision Transformer (ViT) model. The model is fine-tuned to perform facial recognition on the LFW dataset.

** Features
Pre-trained Vision Transformer (ViT): Uses a pre-trained ViT model fine-tuned for facial recognition.

LFW Dataset: Labeled Faces in the Wild dataset is used for training and testing the model.

Data Augmentation: The dataset is processed with necessary transformations for training, including resizing, normalization, and augmentation.

Model Evaluation: The model’s performance is evaluated using accuracy and loss metrics

** Dataset
The LFW dataset is included in the project as archive.zip. Upon running the script, the zip file is extracted, and the dataset is preprocessed into appropriate train-test splits.

The dataset contains 13,000 labeled images of 5,749 individuals, used for training and testing.

** Model Architecture
This project uses the Vision Transformer (ViT) model from the torchvision library. We fine-tune the pre-trained ViT model with the following modifications:

Replace the final classification layer to match the number of classes in the LFW dataset.

Use CrossEntropy loss and AdamW optimizer for training.

The learning rate is dynamically adjusted with ReduceLROnPlateau.
