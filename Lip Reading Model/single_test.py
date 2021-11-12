import cv2
import numpy as np
from PIL import Image
from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import sys 
from create_embeddings import video_to_npy_array

#test video path
video_path=<add video path here>

if os.path.exists(video_path):
    print("path exists")
else:
    sys.exit("video path does not exist")

models_dir = 'lip_reading_models/'

model_sve_name = "test_model"

# to create feature embedding from video will have to call the function with width, height and depth as set in preprocessign file
video_npy = video_to_npy_array(video_path)

def get_model(model_path, model_weights_path):

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_weights_path)
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print("Loaded Model Weights from disk")

    return loaded_model


model = get_model(models_dir + model_sve_name + '.json', models_dir + model_sve_name + '.h5')

# test if model loaded correctly
model.summary()

# load the test video and create embedding 
def load_data(video_path):
    
    X = []        
    sample = np.load(video_path,allow_pickle=True)
    print(sample.shape)
    sample = (sample.astype("float16") - 128) / 128  # normalize to 0 - 1
    X.append(sample)
    
    X = np.array(X)
    X = X.reshape(X.shape + (1,))

    return X


X_test = load_data(video_npy)

# Predictions
predictions = model.predict(X_test)
print(predictions)
y_classes = predictions.argmax(axis=-1)
print(y_classes)
