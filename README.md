# Decision Fusion on Otoscopic Image Analysis and Tympanometry

Decision fusion system for detecting **ear-drum abnormalities** using **VGG19 for otoscopic image analysis** and **Random Forest for tympanometry data classification**.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/decision-fusion-ear-drum-detection.git
cd decision-fusion-ear-drum-detection
```
Install dependencies:

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib opencv-python
```
---
## Import Modules

```python
from tensorflow.keras.applications import VGG19
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cv2
```
---
## Configuration

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
RF_TREES = 100
```
---
## Initialize Models

```python
# VGG19 model for image analysis
base_model = VGG19(weights='imagenet', include_top=False)

# Random Forest for tympanometry classification
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```
---
## Image Prediction

Example for predicting ear infection using otoscopic images:

```python
def predict_image(image_path, model):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224))
    image = image / 255.0

    prediction = model.predict(np.expand_dims(image, axis=0))

    return prediction
```
---
## Tympanometry Prediction

```python
def predict_tympanometry(data, model):

    prediction = model.predict(data)

    return prediction
```
---
## Decision Fusion

Combine both predictions to generate final diagnosis.

``` python
def decision_fusion(image_pred, tymp_pred):

    if image_pred == 1 or tymp_pred == 1:
        return "Ear Infection Detected"

    return "Normal Ear"
```
---
## Results

Accuracy achieved by the system:

```
Decision Fusion Model Accuracy: 93.3%
```
---
## Project Structure

```
decision-fusion-ear-drum-detection
│
├── project.ipynb
├── README.md
└── dataset
```

