# Facial Recognition with ViT and FaceNet on LFW Dataset
This project demonstrates facial recognition using two deep learning models: a Vision Transformer (ViT) and a FaceNet model built from scratch, applied to the LFW (Labeled Faces in the Wild) dataset. The dataset is provided as a zip archive (archive.zip), which is uploaded and processed at runtime.

# Overview 
The objective of this project is to classify and verify faces using two approaches:

A pre-trained Vision Transformer (ViT) fine-tuned on the LFW dataset.

A FaceNet model, implemented from scratch, that learns facial embeddings for identity verification.

The goal is to evaluate and compare both models' performance in facial recognition tasks.

# Features
* Pre-trained Vision Transformer (ViT)
Fine-tuned on LFW for multi-class facial classification (5,749 individuals).

Uses a transformer-based attention mechanism for global feature representation.

* FaceNet (Custom-Built)
CNN-based architecture trained from scratch using Triplet Loss.

Learns 128-dimensional face embeddings for identity verification.

Classification/verification is performed using cosine similarity or KNN on embeddings.

#  Model Evaluation
ViT: Evaluated using accuracy and confusion matrix.

FaceNet: Evaluated using verification accuracy, ROC-AUC, and distance-based matching.

# Dataset 
13,000 labeled facial images of 5,749 unique individuals.

Preprocessed with resizing, normalization, and augmentation.

Automatically extracted from archive.zip during runtime.

The LFW dataset is included in the project as archive.zip. On execution:

The zip is extracted.

Data is split into train and test sets.

Images are resized to 224x224 and augmented during training.

This dataset features real-world variations in facial images, making it a suitable benchmark for evaluating facial recognition models.

# Model Architecture
* Vision Transformer (ViT)
Imported from torchvision.models, pre-trained on ImageNet.

Final classification layer replaced to match the number of identities in the LFW dataset.

Trained using CrossEntropyLoss, AdamW optimizer, and ReduceLROnPlateau scheduler.

* FaceNet (From Scratch)
Built using convolutional layers with batch normalization and ReLU activations.

Outputs 128-dimensional embeddings.

Trained with Triplet Loss on anchor-positive-negative triplets.

Final classification is based on similarity of embeddings rather than direct softmax output.
