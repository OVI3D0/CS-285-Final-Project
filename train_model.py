# dataset:
# https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images

import os
import joblib
from sklearn.svm import SVC
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np
import PIL

# Set the path to the 'Data' folder
base_path = 'Data'

def load_data(data_type):
    image_paths = []
    labels = []

    for class_folder in ['Fire', 'Non_Fire']:
        folder_path = os.path.join(base_path, data_type, class_folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(class_folder)

    # Convert labels to binary (0 for 'Non_Fire', 1 for 'Fire')
    label_map = {'Non_Fire': 0, 'Fire': 1}
    labels = [label_map[label] for label in labels]

    return image_paths, labels

# Load training and testing data
X_train, y_train = load_data('Train_Data')
X_test, y_test = load_data('Test_Data')

# Load pre-trained VGG16 model without the top layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = vgg_model.predict(img_array)
        return img_array.flatten()
    except PIL.UnidentifiedImageError:
        print(f"Error loading image: {image_path}")
        return None

# Extract features from training and testing images
X_train_features = []
for path in X_train:
    features = extract_features(path)
    if features is not None:
        X_train_features.append(features)
X_train_features = np.array(X_train_features)

X_test_features = []
for path in X_test:
    features = extract_features(path)
    if features is not None:
        X_test_features.append(features)
X_test_features = np.array(X_test_features)

# Remove samples with missing features
X_train_valid = [x for x in X_train_features if x is not None]
y_train_valid = [y for x, y in zip(X_train_features, y_train) if x is not None]
X_test_valid = [x for x in X_test_features if x is not None]
y_test_valid = [y for x, y in zip(X_test_features, y_test) if x is not None]

# Train SVM classifier
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_valid, y_train_valid)

joblib.dump(svm_model, 'fire_detection_model.pkl')

# Evaluate the model
accuracy = svm_model.score(X_test_valid, y_test_valid)
print(f"Model accuracy: {accuracy}")