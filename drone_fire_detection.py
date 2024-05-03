import cv2
import numpy as np
from djitellopy import Tello
import joblib
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

# load the trained model and VGG16 model for feature extraction
svm_model = joblib.load('fire_detection_model.pkl')
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_frame(frame):
    """convert frame to format compatible with VGG16 model."""
    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    features = vgg_model.predict(img_array)
    return features.flatten()

def detect_fire(frame):
    """detect fire in the frame using the SVM model."""
    features = preprocess_frame(frame)
    prediction = svm_model.predict([features])
    return "Fire Detected!" if prediction[0] == 1 else "No Fire Detected."

def put_text_on_frame(frame, text, coord=(50, 50), color=(0, 0, 255), font_scale=1, thickness=2):
    """put text on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, coord, font, font_scale, color, thickness, cv2.LINE_AA)

def video():
    tello = Tello()
    tello.connect()
    print("Battery:", tello.get_battery())
    tello.streamon()

    # have the drone take off and hover
    tello.takeoff()
    tello.send_rc_control(0, 0, 0, 0)  # maintain position in the air

    frame_read = tello.get_frame_read()

    while True:
        frame = frame_read.frame

        # detect fire and get status
        fire_status = detect_fire(frame)

        # put status text on the frame
        put_text_on_frame(frame, fire_status, coord=(50, 50), color=(0, 255, 0), font_scale=1, thickness=2)

        cv2.imshow("Drone Camera", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    tello.land()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()
