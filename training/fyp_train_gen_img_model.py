import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ConvLSTM2D, AveragePooling3D, Reshape, BatchNormalization, MaxPooling3D, Activation, Flatten, Dropout, LSTM, TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from fyp_prepare_img_data import save_data_as_pickle, img_seq_generator
import numpy as np
import pickle, os
import random, math, sys
from pathlib import Path
K.set_image_data_format('channels_last')
print("Clearing session")
tf.keras.backend.clear_session()

# Uncomment this section when tensorboard, early stopping and model checkpoint is needed
# Importing the Callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
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
TRAINING_DIR = os.path.join(BASE_DIR, 'training')
OUTPUT_DIR = os.path.join(TRAINING_DIR, 'output')
FRAME_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'frames', 'train')
FRAME_DIR_TEST = os.path.join(OUTPUT_DIR, 'frames', 'test')

def build_model():
    print("Creating model")
    model = Sequential()
    model.add(ConvLSTM2D(filters = filter_size, kernel_size = (kernel_size, kernel_size), input_shape = (seq_size, im_h, im_w, 1), data_format="channels_last", activation="tanh", return_sequences=True)) #shape: num_seq, image_size, channel_num (1)
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1,2,2), data_format="channels_last", padding="same" ))
    
    model.add(ConvLSTM2D(filters = filter_size//2, kernel_size = (kernel_size, kernel_size), data_format="channels_last", activation="tanh", return_sequences=True)) #shape: num_seq, image_size, channel_num (1)
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1,2,2), data_format="channels_last", padding="same" ))

    model.add(ConvLSTM2D(filters = filter_size//4, kernel_size = (kernel_size, kernel_size), data_format="channels_last", stateful=False, kernel_initializer="random_uniform",padding="same", activation="tanh", return_sequences=True)) #shape: num_seq, image_size, channel_num (1)
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1,2,2), data_format="channels_last", padding="same" ))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128,)))
    model.add(TimeDistributed(Dense(128,)))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model

# def build_model2():
    # print("Creating model")

    # model = Sequential()
    # model.add(ConvLSTM2D(filters = filter_size, kernel_size = (kernel_size, kernel_size), input_shape = (seq_size, im_h, im_w, 1),  return_sequences=True)) #shape: num_seq, image_size, channel_num (1)
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling3D(pool_size=(2,2,2)))
    # model.add(Dropout(0.5))
    # model.add(Flatten())

    # model.add(Dense(1, activation='sigmoid'))
    #return model

def train_model(im_w, im_h, pickle_base_dir, lr=1e-6, seq_size=30, filter_size=32, kernel_size=3, batch_size=1, epochs=5, logs=None, weights_dir=None, early_stop=False, save_model=False, debug=False):
#~~~~~~~~~~~~~~~~~~~~|| SET UP CALLBACKS IF NEEDED || ~~~~~~~~~~~~~~~~~~~~~~

    callbacks = []
    if logs:
        # Saving logs
        LOG_DIR = os.path.join(os.getcwd(), 'logs')
        tb = TensorBoard(LOG_DIR)
        callbacks.append(tb)
    if weights_dir:
        # Saving weights
        weights_dir = f'{weights_dir}/convlstm'+ '-{epoch:02d}-{loss:.2f}'+f'-{im_h}x{im_w}.h5'
        chkpt = ModelCheckpoint(filepath=weights_dir, monitor='loss', save_best_only=True, save_weights_only=True, mode='auto', period=1)
        callbacks.append(chkpt)
    if early_stop:
        # Stop training when val_acc is not improving after 5 epochs
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
        callbacks.append(early_stop)

#~~~~~~~~~~~~~~~~~~~~|| PREPROCESS VIDEO TO OBTAIN SEQUENCES OF TRAINING IMAGES TO BE LOADED TO MODEL LATER || ~~~~~~~~~~~~~~~~~~~~~~
    pickle_dir_test = os.path.join(TRAINING_DIR, pickle_base_dir, 'test')

    pickle_dir_train = os.path.join(TRAINING_DIR, pickle_base_dir, 'train')




    #prepare img sequences data
    if not (os.path.exists(pickle_dir_train) and os.path.exists(pickle_dir_test)):
        print("Preparing data... ", FRAME_DIR_TEST, FRAME_DIR_TRAIN)

        if not os.path.exists(pickle_dir_test): 
            os.makedirs(pickle_dir_test)
        if not os.path.exists(pickle_dir_train): 
            os.makedirs(pickle_dir_train)
        num_train, h, w= save_data_as_pickle(FRAME_DIR_TRAIN, pickle_dir_train, seq_size=seq_size, debug=debug)
        num_test, _, _ =save_data_as_pickle(FRAME_DIR_TEST, pickle_dir_test, seq_size=seq_size, debug=debug)
    else:
        num_train = len(os.listdir(pickle_dir_train))
        num_test = len(os.listdir(pickle_dir_test))
    print(num_train, num_test, "HALOOOOOOOOOOOOO")
    #get the test labels
    y_test = []
    for fn in sorted(os.listdir(pickle_dir_test)):
        label = fn[-5]
        y_test.append(int(label))



#~~~~~~~~~~~~~~~~~~~~|| CREATE MODEL || ~~~~~~~~~~~~~~~~~~~~~~

    model = build_model()



#~~~~~~~~~~~~~~~~~~~~|| COMPILE AND FIT MODEL || ~~~~~~~~~~~~~~~~~~~~~~

    print("Compiling model")
    opt = tf.keras.optimizers.Adam(lr=lr, decay=1e-7)

    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = [get_f1])

    # then train it
    print("Fitting model with batch size: ", batch_size)
    train_generator = img_seq_generator(pickle_dir_train, batch_size, im_h, im_w, mode='train', shuffle=True)
    test_generator = img_seq_generator(pickle_dir_test, batch_size, im_h, im_w, mode='train', shuffle=True)

    class_weight={1:0.35, 0:0.65}
    if callbacks:
        model.fit_generator(train_generator, # shape (300, 200, 256, 256, 3)
                steps_per_epoch=num_train // batch_size,
                validation_data=test_generator,
                validation_steps=num_test // batch_size,
                epochs=epochs,
                class_weight=class_weight,
                callbacks=callbacks)
    else:
        model.fit_generator(train_generator, # shape (300, 200, 256, 256, 3)
        steps_per_epoch=num_train // batch_size,
        validation_data=test_generator,
        validation_steps=num_test // batch_size,
        epochs=epochs,
        class_weight=class_weight)

    #save model
    if save_model:
        model_name = f"model-{filter_size}-{kernel_size}-{im_h}x{im_w}-{str(random.random())[-3:]}.h5"
        model_path = f'model/{model_name}'
        tf.keras.models.save_model(
            model = model,
            filepath = model_path,
            overwrite = True,
            include_optimizer=True,
            save_format=None,
            signatures=None
        )

#~~~~~~~~~~~~~~~~~~~~|| RUNNING EXAMPLE PREDICTION ON TEST SET || ~~~~~~~~~~~~~~~~~~~~~~
    test_gen = img_seq_generator(pickle_dir_test, batch_size, im_h, im_w, mode="eval")
    y_pred = model.predict_generator(test_gen, steps=math.ceil(num_test/batch_size))
    pred_class = []

    print("Predicted value in validation set")
    for i,p in enumerate(y_pred):
        val = 0 if p< 0.5 else 1
        pred_class.append(val)
        print(f"Predicted prob:{p}.\tPredicted val:{val}\tTrue val: {y_test[i]}")

    # print(f"Predicted prob: {val}.\tTrue val: {y_test[i]}")


    # print(f"Predicted prob: {val}.\tTrue val: {y_test[i]}")

    #Confution Matrix and Classification Report
    print('Confusion Matrix')
    print(confusion_matrix(y_test, pred_class))
    print('Classification Report')
    target_names = ['Sad', 'Happy']
    print(classification_report(y_test, pred_class, target_names=target_names))


if __name__ == "__main__":
    train_model(100, 700, 'pickled_seq_images', epochs=19, seq_size=30, filter_size=20, kernel_size=3, batch_size=3, save_model=True)