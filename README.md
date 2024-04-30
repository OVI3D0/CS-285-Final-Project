# Fire Detection with DJI Tello Drone

This project demonstrates real-time fire detection using a DJI Tello drone and a machine learning model trained on a dataset of fire and non-fire images. The model is based on transfer learning using the VGG16 convolutional neural network and an SVM classifier.

## Prerequisites

Before running the project, make sure that you have the following:

- Python 3.x installed on your system
- A DJI Tello drone (charged and connected to your computer via Wi-Fi)

## Setup Instructions

### Windows

1. Download the project zip file and extract it to your desired location on your computer.

2. Open a command prompt or PowerShell and navigate to the project directory.

3. Create a new Python virtual environment by running the following command:
   >>> python -m venv myenv

4. Activate the virtual environment:
   >>> myenv\Scripts\activate

5. Install the required libraries by running the following command:
   >>> pip install -r requirements.txt

6. Place your dataset images in the 'Data' folder, following the required directory structure:
   Data/
     Train_Data/
       Fire/
         F_0.jpg
         F_1.jpg
         ...
       Non_Fire/
         NF_0.jpg
         NF_1.jpg
         ...
     Test_Data/
       Fire/
         F_0.jpg
         F_1.jpg
         ...
       Non_Fire/
         NF_0.jpg
         NF_1.jpg
         ...

<br>Download the data from:<br>
https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images

7. Run the `train_model.py` script to train the fire detection model:
   >>> python train_model.py

8. Once the model is trained, connect your DJI Tello drone to your computer via Wi-Fi.

9. Run the `drone_fire_detection.py` script to perform real-time fire detection using the drone:
   >>> python drone_fire_detection.py

10. The drone will take off, and the live video feed will be displayed with fire detection results.

11. Press 'q' to quit the program, and the drone will automatically land.

### macOS

1. Download the project zip file and extract it to a desired location on your computer.

2. Open a terminal and navigate to the project directory.

3. Create a new Python virtual environment by running the following command:
   >>> python3 -m venv myenv

4. Activate the virtual environment:
   >>> source myenv/bin/activate

5. Install the required libraries by running the following command:
   >>> pip install -r requirements.txt

6. Follow steps 6-11 from the Windows instructions above.

## Notes

- Ensure that you comply with local laws and regulations regarding drone usage and obtain any necessary permissions before flying the drone.
- The accuracy of the fire detection model may vary depending on the quality and diversity of the training dataset.
- If you encounter any issues or have questions, please refer to the documentation of the respective libraries and the DJI Tello drone.

## Requirements

The `requirements.txt` file should contain the following libraries:

scikit-learn
keras
tensorflow
pillow
opencv-python
djitellopy