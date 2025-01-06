# Facial Expression Recognition using MobileNet

This project implements a deep learning model for facial expression recognition using transfer learning with MobileNet as the base architecture.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohamedIKenedy/Computer-vision-based-Emotion-tracker/blob/main/YOUR_NOTEBOOK_NAME.ipynb)
## Dataset

The model uses the FER2013 dataset, which contains:
- 48x48 pixel grayscale images of faces
- 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Training set: ~28,000 images
- Test set: ~3,500 images

## Model Architecture

The model employs transfer learning with the following structure:

1. Base Model: MobileNet (pretrained on ImageNet)
   - Input shape: 128x128x3
   - Weights: ImageNet
   - Top layers removed

2. Custom layers added:
   - Conv2D layer (32 filters, 1x1 kernel)
   - Conv2D layer (64 filters, 3x3 kernel)
   - Conv2D layer (128 filters, 3x3 kernel)
   - Conv2D layer (256 filters, 3x3 kernel)
   - Conv2D layer (512 filters, 3x3 kernel)
   - Global Average Pooling
   - Flatten
   - Dense layer (256 units) with dropout (0.3)
   - Dense layer (512 units) with dropout (0.3)
   - Output layer (7 units, softmax activation)

## Training Details

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy
- Callbacks:
  - ModelCheckpoint: Saves best model based on validation accuracy
  - EarlyStopping: Stops training if no improvement after 10 epochs
  - ReduceLROnPlateau: Reduces learning rate when loss plateaus

## Performance

Based on the training graphs:
- Training accuracy reaches ~73%
- Validation accuracy stabilizes around 68%
- Model shows good convergence with minimal overfitting
- Loss decreases steadily throughout training

## Usage

```python
# Load and preprocess image
img = load_img('path_to_image', target_size=(128, 128))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Load model and predict
model = load_model('best_model.h5')
prediction = model.predict(x)
emotion = emotions[np.argmax(prediction)]
```

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Kaggle API (for dataset download)
- Python 3.x
