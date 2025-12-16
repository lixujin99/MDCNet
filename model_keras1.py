from tensorflow.keras.layers import Activation, Input, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, DepthwiseConv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import max_norm
from tensorflow import keras
import tensorflow as tf
from loss_keras import FocalLoss, focal_loss


def EEGmulti_MDC(input_time=1000, fs=128, ncha=64, filters_per_branch=8,
                 scales_time=(500, 250, 125, 62.5), dropout_rate=0.25,
                 activation='elu', n_classes=3, learning_rate=0.001):

    # ============================= CALCULATIONS ============================= #
    input_samples = int(input_time * fs / 1000)
    scales_time_first = (250, 250)
    scales_samples_first = [int(s * fs / 1000) for s in scales_time_first]
    scales_samples = [int(s * fs / 1000) for s in scales_time]

    # ================================ INPUT ================================= #
    input_layer = Input((input_samples, ncha, 1))

    # ========================== BLOCK 1: INCEPTION ========================== #
    b1_units = list()
    for i in range(len(scales_samples_first)):
        unit = Conv2D(filters=filters_per_branch,
                      kernel_size=(scales_samples_first[i], 1),
                      kernel_initializer='he_normal',
                      padding='same', dilation_rate=(2,1))(input_layer)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)
        unit = Dropout(dropout_rate)(unit)

        unit = DepthwiseConv2D((1, ncha),
                               use_bias=False,
                               depth_multiplier=2,
                               depthwise_constraint=max_norm(1.))(unit)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)
        unit = Dropout(dropout_rate)(unit)

        b1_units.append(unit)

    # Concatenation
    b1_out = keras.layers.concatenate(b1_units, axis=3)
    b1_out = AveragePooling2D((4, 1))(b1_out)

    # ========================== BLOCK 2: INCEPTION ========================== #
    b2_units = list()
    for i in range(len(scales_samples)):
        unit = Conv2D(filters=filters_per_branch,
                      kernel_size=(int(scales_samples[i]/4), 1),
                      kernel_initializer='he_normal',
                      use_bias=False,
                      padding='same', dilation_rate=(2,1))(b1_out)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)
        unit = Dropout(dropout_rate)(unit)

        b2_units.append(unit)
 
    # Concatenate + Average pooling
    b2_out = keras.layers.concatenate(b2_units, axis=3)
    b2_out = AveragePooling2D((2, 1))(b2_out)


    # ============================ BLOCK 3: OUTPUT_tri =========================== #
    b3_u1 = Conv2D(filters=int(filters_per_branch*len(scales_samples)/2),
                   kernel_size=(8, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same', dilation_rate=(2,1))(b2_out)
    b3_u1 = BatchNormalization()(b3_u1)
    b3_u1 = Activation(activation)(b3_u1)
    b3_u1 = AveragePooling2D((2, 1))(b3_u1)
    b3_u1 = Dropout(dropout_rate)(b3_u1)

    b3_u2 = Conv2D(filters=int(filters_per_branch*len(scales_samples)/4),
                   kernel_size=(4, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same', dilation_rate=(2,1))(b3_u1)
    b3_u2 = BatchNormalization()(b3_u2)
    b3_u2 = Activation(activation)(b3_u2)
    b3_u2 = AveragePooling2D((2, 1))(b3_u2)
    b3_out = Dropout(dropout_rate)(b3_u2)

    output_layer = Flatten()(b3_out)
    output_layer_2 = Dense(n_classes-1, activation='softmax', name='binary')(output_layer)
    
    output_layer_3 = Dense(n_classes)(output_layer)
    output_layer_3 = tf.multiply(output_layer_3, tf.concat([output_layer_2,output_layer_2[:,1:]],axis=-1))
    output_layer_3 = Activation('softmax', name='triplet')(output_layer_3)

    # ================================ MODEL ================================= #
    model = keras.models.Model(inputs=input_layer, outputs=[output_layer_3, output_layer_2])

    return model