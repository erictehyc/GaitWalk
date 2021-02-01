import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from fyp_prepare_img_data import img_seq_generator
import numpy as np
import pickle, os, sys
import random, math
from pathlib import Path
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
print("Clearing session")
tf.keras.backend.clear_session()
BASE_DIR = os.getcwd()
called_dir = BASE_DIR
while os.path.basename(BASE_DIR) != "fyp_team4c":
    path = Path(BASE_DIR)
    BASE_DIR = str(path.parent)
    if BASE_DIR == '/':
        print("Please call this script in the fyp_team4c directory")
        break
sys.path.append(BASE_DIR)
from utils import *

def get_f1(y_true, y_pred): #taken from old keras source code
    print(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
def evaluate_model(model, pickle_dir, batch_size, im_h, im_w, target_names):
    #get true label
    y_true = []
    for fn in sorted(os.listdir(pickle_dir)):
        label = fn[-5]
        y_true.append(int(label))

    num_test = len(os.listdir(pickle_dir))
    test_gen = img_seq_generator(pickle_dir, batch_size, im_h, im_w, mode="eval")
    y_pred = model.predict_generator(test_gen, steps=math.ceil(num_test/batch_size))
    pred_class = []

    print("Predicted value")
    for i,p in enumerate(y_pred):
        val = 0 if p< 0.5 else 1
        pred_class.append(val)
        print(f"Predicted prob:{p}.\tPredicted val:{val}\tTrue val: {y_true[i]}")

    # print(f"Predicted prob: {val}.\tTrue val: {y_test[i]}")


    # print(f"Predicted prob: {val}.\tTrue val: {y_test[i]}")

    #Confution Matrix and Classification Report
    print('Confusion Matrix')
    print(confusion_matrix(y_true, pred_class))
    print('Classification Report')
    print(classification_report(y_true, pred_class, target_names=target_names))

def main():
    #set variables here
    custom_metric=True
    model_name = "seq30_128dense_lost0.58_f1-0.7_100x700.h5"
    pickle_base="pickled_seq_images"
    batch_size = 3
    im_h = 700
    im_w = 100
    target_names = ['Depressed', 'Happy']

    #Run evaluation
    pickle_validation = os.path.join(pickle_base, "test")
    pickle_train = os.path.join(pickle_base, "train")
    model_fp = os.path.join("model", model_name)
    # model_fp = os.path.join(TRAINING_DIR, model_fp)
    print(model_fp)
    if os.path.exists(model_fp):
        if custom_metric:
            model = tf.keras.models.load_model(
            model_fp, custom_objects={"get_f1":get_f1}, compile=True
            )
        else:
            model = tf.keras.models.load_model(
            model_fp, custom_objects=None, compile=True
            )
    else:
        print("Missing model for prediction!")
        raise Exception("Missing model")
    print("Evaluating validation set")
    evaluate_model(model, pickle_validation, batch_size, im_h, im_w, target_names)
    print("Evaluating training set")
    evaluate_model(model, pickle_train, batch_size, im_h, im_w, target_names)

if __name__ == "__main__":
    main()