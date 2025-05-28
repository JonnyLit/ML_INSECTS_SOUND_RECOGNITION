import numpy as np
import matplotlib.pyplot as plt
import os
import re
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from datetime import datetime 
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import my_tools as mt

def make_model(X_shape_1, X_shape_2, n_outputs):
    model=Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(X_shape_1, X_shape_2, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.25))
    model.add(Dense(32 , activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy', 'Precision', 'Recall'])
    return model



def train_evaluate(X_train, Y_train, X_test, Y_test, n_outputs, num_batch_size, num_epochs, class_names, target_names):
    print('Training...')
    all_history = []
    model = make_model(X_train.shape[1], X_train.shape[2], n_outputs )
    print("X_train.shape")
    print(X_train.shape[0], X_train.shape[1])
    X_train = X_train.reshape(-1,  X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2],  1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    print("X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    le = LabelEncoder()
    print(f"le: {le}")
    print("after LabelEncoder()")
    print("X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    Y_train = to_categorical(le.fit_transform(Y_train))
    print("after Y_train = to_categorical(le.fit_transform(Y_train))")
    print("X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    Y_test = to_categorical(le.fit_transform(Y_test))
    #Calculate pre-training accuracy
    print("before model.evaluate and after Y_test = to_categorical(le.fit_transform(Y_test))")
    print("X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("after model.evaluate")
    print("X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    accuracy = 100*score[1]
    print("Predicted accuracy: ", accuracy)
    #Training the network
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    start = datetime.now()
    history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs,
                    validation_data=(X_test, Y_test), verbose=1)

    #########################################

    all_history.append(history)

    # Evaluate Model
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(x=X_test, y=Y_test, steps=len(Y_train), verbose=1)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")
    # Save the model
    import time
    import tensorflow as tf
    model_name = 'queen_bee_presence_prediction'
    model.save(model_name + '.h5')
    time.sleep(3)

    # load h5 module
    model_h5 = tf.keras.models.load_model('/home/zord/PycharmProjects/queen_bee_presence_prediction/' + model_name + '.h5')
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model_h5)

    time.sleep(3)

    # convert to tflite
    tflite_model = tflite_converter.convert()
    open(model_name + ".tflite", "wb").write(tflite_model)
    #########################################

    interpreter = tf.lite.Interpreter(model_path='//home/zord/PycharmProjects/queen_bee_presence_prediction/queen_bee_presence_prediction.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)
    #########################################



    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # Evaluating the model on the training and testing set
    score1 = model.evaluate(X_train, Y_train, verbose=1)
    print("Training Accuracy: ", score1[1])
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("Testing Accuracy: ", score[1])
     #predicting
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(np.round(Y_pred), axis=1)
    rounded_predictions = model.predict_classes(X_test, batch_size=128, verbose=0)
    print(rounded_predictions[1])
    rounded_labels=np.argmax(Y_test, axis=1)
    print(rounded_labels[1])
    #Confusion matrix
    cnf_matrix = confusion_matrix(rounded_labels, rounded_predictions)
    np.set_printoptions(precision=2)
    
    mt.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix:')
    plt.show()

    #################################################################

    # --- Plot Training and Validation Curves ---
    plt.figure(figsize=(13, 7))
    for i, hist in enumerate(all_history):
        plt.subplot(2, 2, 1)
        plt.plot(hist.history['loss'], label=f'Fold {i + 1} Train')
        plt.plot(hist.history['val_loss'], label=f'Fold {i + 1} Validation')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower center')

        plt.subplot(2, 2, 2)
        plt.plot(hist.history['accuracy'], label=f'Fold {i + 1} Train')
        plt.plot(hist.history['val_accuracy'], label=f'Fold {i + 1} Validation')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower center')

        plt.subplot(2, 2, 3)
        plt.plot(hist.history['precision'], label=f'Fold {i + 1} Train')
        plt.plot(hist.history['val_precision'], label=f'Fold {i + 1} Validation')
        plt.title('Training and Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend(loc='lower center')

        plt.subplot(2, 2, 4)
        plt.plot(hist.history['recall'], label=f'Fold {i + 1} Train')
        plt.plot(hist.history['val_recall'], label=f'Fold {i + 1} Validation')
        plt.title('Training and Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend(loc='lower center')

    plt.tight_layout()
    plt.show()

    #################################################################

    print ('\nClassification report for MfCCs + CNN for fold1:\n', classification_report(rounded_labels, rounded_predictions, target_names=target_names ))
    return rounded_predictions, rounded_labels