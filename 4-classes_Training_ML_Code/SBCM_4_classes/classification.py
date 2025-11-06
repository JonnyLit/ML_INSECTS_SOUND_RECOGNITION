import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import my_tools as mt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D, Dense, LeakyReLU




# Config_14
import tensorflow as tf

# tf.keras.mixed_precision.set_global_policy('mixed_float16') # supported only for Only Nvidia GPUs with compute capability of at least 7.0 run quickly with mixed_float16. If not, it goes very slow.

from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D, GlobalMaxPooling2D, Activation, Permute
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D


def se_block(input_tensor, ratio=8):
    channels = int(input_tensor.shape[-1])
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)
    return multiply([input_tensor, se])


def residual_block(input_tensor, filters, downsample=False):
    shortcut = input_tensor

    # Main path
    x = Conv2D(filters, (3, 3), strides=2 if downsample else 1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut path
    if downsample:
        # Adjust shortcut with 1x1 conv to match dimensions
        shortcut = Conv2D(filters, (1, 1), strides=2, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add
    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.1)(x)
    return x

def bam_block(input_tensor, reduction_ratio=16, dilation_rate=4):
    # Channel Attention
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)
    avg_pool = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    shared_mlp = tf.keras.Sequential([
        Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu'),
        Dense(input_tensor.shape[-1])
    ])
    max_out = shared_mlp(max_pool)
    avg_out = shared_mlp(avg_pool)
    channel_attention = Activation('sigmoid')(max_out + avg_out)
    x = multiply([input_tensor, channel_attention])

    # Spatial Attention
    avg_pool_spatial = tf.reduce_mean(x, axis=3, keepdims=True)
    max_pool_spatial = tf.reduce_max(x, axis=3, keepdims=True)
    concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=3)
    spatial_attention = Conv2D(1, kernel_size=7, dilation_rate=dilation_rate, padding='same', activation='sigmoid')(
        concat)
    x = multiply([x, spatial_attention])
    return x

def eca_block(input_tensor, gamma=2, b=1):
    channels = int(input_tensor.shape[-1])
    t = int(abs((tf.math.log(tf.cast(channels, tf.float32)) / tf.math.log(2.0)) + b) / gamma)
    k_size = t if t % 2 else t + 1  # Make kernel size odd

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(input_tensor)
    # 1D convolution
    gap = Reshape((-1, 1))(gap)
    conv1d = tf.keras.layers.Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)(gap)
    attention = tf.keras.layers.Activation('sigmoid')(conv1d)
    attention = Reshape((1, 1, channels))(attention)
    return multiply([input_tensor, attention])


def cbam_block(input_tensor, reduction_ratio=8, kernel_size=7):
    # Channel Attention
    channel = input_tensor.shape[-1]
    shared_dense_one = Dense(channel // reduction_ratio, activation='relu') # vedi pag.5 paper sul reduction_ratio
    shared_dense_two = Dense(channel)
    #print(f"shared_dense_one.shape={channel // reduction_ratio}")
    #print(f"shared_dense_two.shape={channel}")
    avg_pool = GlobalAveragePooling2D()(input_tensor) # vedi eq.2 paper
    max_pool = GlobalMaxPooling2D()(input_tensor) # vedi eq.2 paper
    #print(f"avg_pool.shape={avg_pool.shape}")
    #print(f"max_pool.shape={max_pool.shape}")
    avg_dense = shared_dense_two(shared_dense_one(avg_pool)) # vedi eq.2 paper
    max_dense = shared_dense_two(shared_dense_one(max_pool)) # vedi eq.2 paper
    #print(f"avg_dense.shape={avg_dense.shape}")
    #print(f"max_dense.shape={max_dense.shape}")
    channel_attention = tf.keras.layers.Add()([avg_dense, max_dense]) # vedi eq.2 paper
    #print(f"channel_attention.shape={channel_attention.shape}")
    channel_attention = Activation('sigmoid')(channel_attention) # vedi eq.2 paper
    #print(f"channel_attention.shape={channel_attention.shape}")
    channel_attention = Reshape((1, 1, channel))(channel_attention)
    #print(f"channel_attention.shape={channel_attention.shape}")
    x = multiply([input_tensor, channel_attention])
    #print(f"x.shape={x.shape}")
    # Spatial Attention
    avg_pool_spatial = tf.reduce_mean(x, axis=3, keepdims=True) # vedi eq.3 paper, pool sui canali
    max_pool_spatial = tf.reduce_max(x, axis=3, keepdims=True) # vedi eq.3 paper, pool sui canali
    #print(f"avg_pool_spatial.shape={avg_pool_spatial.shape}")
    #print(f"max_pool_spatial.shape={max_pool_spatial.shape}")
    concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=3)
    #print(f"concat.shape={concat.shape}")
    spatial_attention = Conv2D(1, kernel_size, padding='same', activation='sigmoid')(concat)
    #print(f"spatial_attention.shape={spatial_attention.shape}")
    x = multiply([x, spatial_attention])
    #print(f"x.shape={x.shape}")
    return x

def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    for filters in [32, 64, 128]:
        x = residual_block(x, filters, downsample=True)
        # Replace SE block with CBAM
        x = cbam_block(x, reduction_ratio=4, kernel_size=7) # val_loss=0.2024, val_accuracy/precision/recall = 98.2%
        # x = bam_block(x, reduction_ratio=8, dilation_rate=4) # val_loss=0.25, val_accuracy/precision/recall = 98%
        # x = eca_block(x, gamma=2, b=1) # val_loss=0.3, val_accuracy/precision/recall = 97.9%
        x = residual_block(x, filters)

    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model








def train_evaluate(X_train, Y_train, X_test, Y_test, n_outputs, num_batch_size, num_epochs, class_names, target_names):
    print('Training...')
    all_history = []
    model = make_model(X_train.shape[1], X_train.shape[2], n_outputs)




    """
    Purpose: Stops training early if the model's validation accuracy (val_accuracy) stops improving.
    How it works:
        It monitors val_accuracy.
        If val_accuracy doesn't improve for 10 consecutive epochs (patience=10), training stops.
        When stopping, it restores the model weights to the best epoch (restore_best_weights=True).

    """
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10, # it was 10
        restore_best_weights=True
    )



    """
    Purpose: Saves the model weights to a file (best_model-bee_presence_AI_Belha.h5) whenever the validation accuracy improves.
    How it works:
        It monitors val_accuracy.
        Only saves the model when val_accuracy reaches a new maximum (save_best_only=True).
        Ensures you keep the best version of your model based on validation accuracy.

    """
    model_name = 'best_model-bee_presence_SBCM'
    checkpoint = ModelCheckpoint(
        model_name+'.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )


    """
    Purpose: Reduces the learning rate when the model's performance plateaus.
    How it works:
        Monitors val_accuracy.
        If val_accuracy doesn't improve for 5 epochs (patience=5), it reduces the learning rate by a factor of 0.5.
        The learning rate won't go below 1e-6.
        verbose=1 prints messages when the LR is reduced.
        
    lr_reducer = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    """

    #Config14-15
    lr_reducer = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5, # it was 0.5
        patience=3, # it was 5
        min_lr=1e-6,
        verbose=1
    )

    """
    #Config16
    lr_reducer = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    """


    '''
    # Use a learning rate scheduler
    # NOT GOOD
    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    '''



    print("X_train.shape")
    print(X_train.shape[0], X_train.shape[1])
    X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], 1)
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
    # Calculate pre-training accuracy
    print("before model.evaluate and after Y_test = to_categorical(le.fit_transform(Y_test))")
    print("X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("after model.evaluate")
    print("X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    accuracy = 100 * score[1]
    print("Predicted accuracy: ", accuracy)
    # Training the network
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    start = datetime.now()
    #history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[early_stop, checkpoint, lr_reducer, lr_schedule], verbose=1) #LR_SCHEDULE
    history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs,validation_data=(X_test, Y_test), callbacks=[early_stop, checkpoint, lr_reducer], verbose=1)
    #history = model.fit(train_datagen.flow(X_train, Y_train, batch_size=num_batch_size),epochs=num_epochs,validation_data=(X_test, Y_test))
    #########################################

    # After training, load best weights
    model.load_weights('best_model-bee_presence_SBCM.h5')


    all_history.append(history)

    # Evaluate Model
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(x=X_test, y=Y_test, steps=len(Y_train), verbose=1)

    print(
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")
    # Save the model
    import time
    import tensorflow as tf

    model.save(model_name + '.h5')
    time.sleep(3)

    # load h5 module
    model_h5 = tf.keras.models.load_model('/home/zord/PycharmProjects/SBCM_4_classes/' + model_name + '.h5')
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model_h5)

    time.sleep(3)

    # convert to tflite
    tflite_model = tflite_converter.convert()
    open(model_name + ".tflite", "wb").write(tflite_model)
    #########################################

    interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/SBCM_4_classes/' + model_name + '.tflite')
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
    # predicting
    Y_pred = model.predict(X_test)

    Y_pred = np.argmax(np.round(Y_pred), axis=1)

    # predictions = model.predict_classes(X_test, batch_size=128, verbose=0)
    predictions = model.predict(X_test, batch_size=num_batch_size, verbose=0)
    rounded_predictions = np.argmax(predictions, axis=1)
    print(rounded_predictions[1])
    rounded_labels = np.argmax(Y_test, axis=1)
    print(rounded_labels[1])
    # Confusion matrix
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

    print('\nClassification report for MfCCs + CNN for fold1:\n',
          classification_report(rounded_labels, rounded_predictions, target_names=target_names))
    return rounded_predictions, rounded_labels











