import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D
from keras.optimizers import Adam
import os
import cv2
import numpy as np

base_dataset_path = r'C:\Users\anmol\BusProject'

train_path = f'{base_dataset_path}/train'
val_path = f'{base_dataset_path}/val'
test_path = f'{base_dataset_path}/test'

image_height, image_width = 64, 64

datagen_train_val = ImageDataGenerator(rescale=1./255)

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


datagen_test = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train_val.flow_from_directory(
    train_path,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb'
)

validation_generator = datagen_train_val.flow_from_directory(
    val_path,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb'
)

test_generator = datagen_test.flow_from_directory(
    test_path,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)


model = Sequential([
    Flatten(input_shape=(image_height, image_width, 3)),
    Dense(256, activation='relu'),
    Dropout(0.3),  # Adjust dropout rate
    Dense(128, activation='relu'),
    Dropout(0.3),  # Adjust dropout rate
    Dense(1, activation='sigmoid')
])



model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history=model.fit(train_generator, epochs= 50, validation_data=validation_generator, verbose=0)

loss, accuracy = model.evaluate(test_generator)
vr = model.evaluate(validation_generator)
print(f"Test accuracy: {accuracy}")
print(f"Validation Loss:", vr[0])
print(f"Validation Accuracy:", vr[1])



class_indices = train_generator.class_indices
label_map = {v: k for k, v in class_indices.items()}
print(class_indices)

def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Capture Image (Press "c" to capture)', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid,pclass):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    font = cv2.FONT_HERSHEY_SIMPLEX  
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(vid,pclass,(x, y-10),font, 1,(0, 255, 255),2,cv2.LINE_AA)
    return faces

def show_video(New_Ans2):
    while True:
        result, video_frame = video_capture.read()  # read frames from the video
        preprocessed_image = preprocess_image(video_frame)
        prediction = model.predict(preprocessed_image)
        rounded_prediction = int(np.round(prediction / 0.5) * 0.5)
        pclass = label_map[rounded_prediction]
        if New_Ans2 == 'yes' or New_Ans2 == 'Yes':
            save_image(video_frame, pclass, New_Ans2)
        else:
            if result is False:
                break  # terminate the loop if the frame is not read successfully
            
            faces = detect_bounding_box(video_frame, pclass) 
            cv2.imshow("Live Webcam", video_frame) 

            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == ord("Q"):
                break
    video_capture.release()
    cv2.destroyAllWindows()

def preprocess_image(image):
    image = cv2.resize(image, (image_height, image_width))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def save_image(image, predicted_class, Ans2): #used the code to save the photos according to the classes (FOR MODEL TRAINING)
    class_folder = f"{Ans2}\{predicted_class}"
    os.makedirs(class_folder, exist_ok=True)
    image_path = os.path.join(class_folder, f"image_{len(os.listdir(class_folder)) + 1}.png")
    cv2.imwrite(image_path, image)
    print(f"Image saved in: {image_path}")

 # for Ans We will input yes to work with camera and no to work with input files
Ans = str(input('Do you want to capture a photo or input a photo (yes for Capturing or no for input and n for Live Video Capturing): '))

# for Ans1 we ask if the user wants to save image or not
Ans1 = str(input('Do you also want to save your images for more traning ? yes or no: '))

New_Ans2 = ''

if Ans1 == 'yes' or Ans1 == 'Yes':
    Ans2 = str(input('where do you want to store it: '))
    New_Ans2 = Ans2.replace('"','')
    print(New_Ans2)
else:
    print('Nothing will be stored')

if Ans == 'Yes' or Ans == 'yes':
    new_image = capture_image()
    if new_image is not None:
        preprocessed_image = preprocess_image(new_image)
        prediction = model.predict(preprocessed_image)
        rounded_prediction = int(np.round(prediction / 0.5) * 0.5)
        predicted_class = label_map[rounded_prediction]
        save_image(new_image, predicted_class,New_Ans2)
        print(f"Predicted class: {predicted_class}")
    else:
        print("No image was captured.")    
elif Ans == 'No' or Ans == 'no':
    
    photo_location = input('Input the photo location: ')
    New_photo = photo_location.replace('"','')
    
    new_image_rd = cv2.imread(New_photo)
    if new_image_rd is not None:
        preprocessed_image = preprocess_image(new_image_rd)
        prediction = model.predict(preprocessed_image)
        rounded_prediction = int(np.round(prediction / 0.5) * 0.5)
        predicted_class = label_map[rounded_prediction]
        if New_Ans2 == 'Yes' or New_Ans2 == 'yes':
            save_image(new_image_rd, predicted_class, New_Ans2)
        else:    
            print(f"Predicted class: {predicted_class}")
    else:
        print("No image was captured.")
else:
    show_video(New_Ans2)


