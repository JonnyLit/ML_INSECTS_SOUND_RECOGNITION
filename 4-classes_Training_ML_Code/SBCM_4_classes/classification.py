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

'''
# Original model
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
'''

'''
#My new model (good, 80% acc, prec, recall)
def make_model(X_shape_1, X_shape_2, n_outputs):
    model = Sequential()

    # First conv block
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(X_shape_1, X_shape_2, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    # Second conv block
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    # Third conv block
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    # Fourth conv block
    model.add(Conv2D(128, kernel_size=(3, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    # Global average pooling instead of flatten
    model.add(GlobalAveragePooling2D())

    # Dense layers for classification
    model.add(Dense(256, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(n_outputs, activation='softmax'))

    # Compile
    from tensorflow.keras.optimizers import RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0005), metrics=['accuracy', 'Precision', 'Recall'])
    return model
'''

'''
# VERY GOOD, SOLID 90+%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

def se_block(input_tensor, ratio=8):
    channels = int(input_tensor.shape[-1])
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.multiply([input_tensor, se])


def residual_block(x, filters, kernel_size=(3,3)):
    shortcut = x
    # If input filters != desired filters, adjust shortcut
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1,1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.1)(x)
    return x


def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))

    # First conv block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Residual block with attention
    for filters in [64, 128]:
        x = residual_block(x, filters)
        x = se_block(x)  # Add Squeeze-and-Excitation block
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)

    # Fourth conv layer similar to your last conv layer
    x = Conv2D(128, (3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.0005),
        metrics=['accuracy', 'Precision', 'Recall']
    )

    return model
'''

"""
# VERY GOOD, 95-96%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your SE block, residual block, and model as before
def se_block(input_tensor, ratio=8):
    channels = int(input_tensor.shape[-1])
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.multiply([input_tensor, se])

def residual_block(x, filters, kernel_size=(3,3)):
    shortcut = x
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1,1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.1)(x)
    return x

def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Residual + SE blocks
    for filters in [64, 128]:
        x = residual_block(x, filters)
        x = se_block(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)

    # Last conv layer
    x = Conv2D(128, (3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)


    #VERY GOOD: AROUND 94-95%
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.0005), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )



    '''
    from tensorflow.keras.optimizers import Adam

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    '''



    '''
    # NOT GOOD: Validation Loss: 0.2318, Validation Accuracy: 0.9160, Validation Precision: 0.9259, Validation Recall: 0.9057
    model.compile(
        loss='categorical_crossentropy',
        optimizer = tf.keras.optimizers.Adam(learning_rate=4e-4),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    '''

    return model
"""


"""
#VERY GOOD: 96-97 %
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define SE block
def se_block(input_tensor, ratio=8):
    channels = int(input_tensor.shape[-1])
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)
    return multiply([input_tensor, se])

# Define residual block with optional expansion
def residual_block(x, filters, kernel_size=(3,3), downsample=False):
    shortcut = x
    stride = 2 if downsample else 1
    if downsample:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Add se_block after residual convs for more representational power
    x = se_block(x)

    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.1)(x)
    return x

def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Increasing depth with more residual blocks and filters
    filter_sizes = [64, 128, 256]
    for filters in filter_sizes:
        # Add two residual blocks per filter size, with downsampling at start of each new filter size
        x = residual_block(x, filters, downsample=True)
        x = residual_block(x, filters)

        # Optional: Add SE block after each pair for attention
        x = se_block(x)

        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)

    # Additional convolutional layer for more abstraction
    x = Conv2D(256, (3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Dense layers with increased capacity
    x = Dense(512, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    # Final output layer
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compile with same optimizer and metrics
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
"""







"""
# VERY GOOD, 26, 97, 98, 96
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define SE block
def se_block(input_tensor, ratio=8):
    channels = int(input_tensor.shape[-1])
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)
    return multiply([input_tensor, se])

# Define residual block with regularization
def residual_block(x, filters, kernel_size=(3,3), downsample=False, weight_decay=1e-4):
    shortcut = x
    stride = 2 if downsample else 1
    reg = tf.keras.regularizers.l2(weight_decay)

    if downsample:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_regularizer=reg)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)

    # Add SE block for attention
    x = se_block(x)

    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.1)(x)
    return x

def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Increasing depth with residual blocks and filters
    filter_sizes = [64, 128, 256]
    for filters in filter_sizes:
        # Add two residual blocks per filter size, with downsampling at start
        x = residual_block(x, filters, downsample=True)
        x = residual_block(x, filters)

        # Add an SE attention block after each pair (optional)
        x = se_block(x)

        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)

    # Add more residual blocks at the deepest level for deeper capacity
    # You can add as many as needed, here's an example of 3 more
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Additional convolutional layer for abstraction
    x = Conv2D(256, (3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Dense layers with regularization and dropout
    x = Dense(512, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    # Final output layer
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
"""


"""
# BEST: 98.5% acc,prec,recall
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator




# Define SE block
def se_block(input_tensor, ratio=8):
    channels = int(input_tensor.shape[-1])
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)
    return multiply([input_tensor, se])

# Optional: CBAM-like spatial attention
def spatial_attention(input_tensor):
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_tensor)
    max_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(input_tensor)
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    attn = Conv2D(1, (7,7), padding='same', activation='sigmoid')(concat)
    return multiply([input_tensor, attn])

# Residual block with BatchNorm, SE, and optional Spatial Attention
def residual_block(x, filters, kernel_size=(3,3), downsample=False, weight_decay=1e-4):
    shortcut = x
    stride = 2 if downsample else 1
    reg = tf.keras.regularizers.l2(weight_decay)

    if downsample:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_regularizer=reg)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)

    # Add SE block
    x = se_block(x)

    # Optional: Add spatial attention
    x = spatial_attention(x)

    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.1)(x)
    return x

def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Progressive increase in filters with residual blocks
    filter_sizes = [64, 128, 256]
    for filters in filter_sizes:
        # Downsampling residual block
        x = residual_block(x, filters, downsample=True)
        x = residual_block(x, filters)

        # Optional: Add another SE or attention layer
        x = se_block(x)
        # Or add spatial attention:
        # x = spatial_attention(x)

        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)

    # Add more residual blocks at deepest level for more capacity
    for _ in range(3):
        x = residual_block(x, 256)

    # Additional convolution for feature abstraction
    x = Conv2D(256, (3, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    # Global pooling
    x = GlobalAveragePooling2D()(x)

    # Dense layers with regularization
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model
"""





"""
#Config_9 97.5% and low number of parameters
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D
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


def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    for filters in [32, 64]:
        x = residual_block(x, filters, downsample=True)
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
"""

"""
# 97.7%, small net
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D
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


def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    # Add an extra residual block
    for filters in [32, 64, 128]:
        x = residual_block(x, filters, downsample=True)
        x = residual_block(x, filters)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)  # increased units
    x = Dropout(0.4)(x)  # slightly increased dropout
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model
"""



"""
#97.67 but overfitting
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D
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


def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    # Add residual blocks with SE after each
    for filters in [32, 64, 128]:
        x = residual_block(x, filters, downsample=True)
        x = residual_block(x, filters)
        x = se_block(x)  # Insert SE block after each residual block

    # Optional: Add an extra residual block with more filters for capacity
    x = residual_block(x, 256, downsample=True)
    x = se_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)  # Slightly increased dropout for regularization
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
"""




"""
#Config 11 - 96.7%
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D
from tensorflow.keras import regularizers

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



def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    for filters in [32, 64]:
        x = residual_block(x, filters, downsample=True)
        x = residual_block(x, filters)

    # Add Dropout before final dense layers
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

"""






"""
#Config_12 97.5%
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D
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


def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    for filters in [32, 64]:
        x = residual_block(x, filters, downsample=True)
        x = se_block(x, ratio=4)  # Add SE after first residual block at each stage
        x = residual_block(x, filters)
        # Optionally add SE after the second residual block
        # x = se_block(x, ratio=4)

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
"""


"""
#Config_13 98 98 98
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D
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


def make_model(X_shape_1, X_shape_2, n_outputs):
    inputs = Input(shape=(X_shape_1, X_shape_2, 1))
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    for filters in [32, 64, 128]:
        x = residual_block(x, filters, downsample=True)
        x = se_block(x, ratio=4)  # Add SE after first residual block at each stage
        x = residual_block(x, filters)
        # Optionally add SE after the second residual block
        # x = se_block(x, ratio=4)

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
"""



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
    # implementation of channel attention that produces per-channel vectors by global pooling, then processes them with dense layers,
    # but keeping spatial dimensions intact (useful if you want to extend attention mechanisms to spatial attention as well).
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
    # common implementation of channel attention that produces per-channel vectors by global pooling, then processes them with dense layers.
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




"""
# Config_15 (SBCM)
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Reshape, multiply, SpatialDropout2D, GlobalMaxPooling2D, Activation, Permute
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D
from tensorflow.keras.regularizers import l2

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
    # implementation of channel attention that produces per-channel vectors by global pooling, then processes them with dense layers,
    # but keeping spatial dimensions intact (useful if you want to extend attention mechanisms to spatial attention as well).
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
    # common implementation of channel attention that produces per-channel vectors by global pooling, then processes them with dense layers.
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

    # Initial Conv layer
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    # Deeper residual + attention blocks
    for filters in [32, 64, 128, 256]:  # added one more layer
        x = residual_block(x, filters, downsample=True)
        # Optional: switch between CBAM, BAM, ECA
        x = cbam_block(x, reduction_ratio=4, kernel_size=7)
        # Uncomment if you want to experiment with other attention modules
        # x = bam_block(x, reduction_ratio=8, dilation_rate=4)
        # x = eca_block(x, gamma=2, b=1)

    # Global pooling
    x = GlobalAveragePooling2D()(x)
    # Fully connected layers for better feature learning
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)

    # Output layer
    outputs = Dense(n_outputs, activation='softmax')(x)

    # Compile model
    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0003),  # reduce LR for finer training
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model
"""







"""
#Config16
import tensorflow as tf
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
    # implementation of channel attention that produces per-channel vectors by global pooling, then processes them with dense layers,
    # but keeping spatial dimensions intact (useful if you want to extend attention mechanisms to spatial attention as well).
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
    # common implementation of channel attention that produces per-channel vectors by global pooling, then processes them with dense layers.
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
        x = cbam_block(x, reduction_ratio=4, kernel_size=7)
        x = residual_block(x, filters)

    # Add an extra residual block to deepen the model
    x = residual_block(x, 256, downsample=True)
    x = cbam_block(x, reduction_ratio=4, kernel_size=7)

    # Final classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model
"""







def train_evaluate(X_train, Y_train, X_test, Y_test, n_outputs, num_batch_size, num_epochs, class_names, target_names):
    print('Training...')
    all_history = []
    model = make_model(X_train.shape[1], X_train.shape[2], n_outputs)





    '''
    # Data augmentation setup
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )
    '''




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


















'''
#ORIGINAL
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
    model_name = 'bee_presence_AI_Belha'
    model.save(model_name + '.h5')
    time.sleep(3)

    # load h5 module
    model_h5 = tf.keras.models.load_model('/home/zord/PycharmProjects/AI-Belha/' + model_name + '.h5')
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model_h5)

    time.sleep(3)

    # convert to tflite
    tflite_model = tflite_converter.convert()
    open(model_name + ".tflite", "wb").write(tflite_model)
    #########################################

    interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/AI-Belha/'+model_name+'.tflite')
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



    
    #predictions = model.predict_classes(X_test, batch_size=128, verbose=0)
    predictions = model.predict(X_test, batch_size=128, verbose=0)
    rounded_predictions = np.argmax(predictions, axis=1)
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
'''