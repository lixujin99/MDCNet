from tensorflow.keras.layers import Activation, Input, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, DepthwiseConv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import max_norm
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K



class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, reduction='auto', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = tf.keras.losses.CategoricalCrossentropy(reduction=reduction)

    def call(self, y_true, y_pred):
        logp = self.ce(y_true, y_pred)
        p = tf.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss
    

def focal_loss(gamma=5, alpha=0.25):
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)
    Cross_entropy = tf.keras.losses.CategoricalCrossentropy(reduction='auto')

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        #ce = -tf.log(y_t)
        ce = Cross_entropy(y_true,y_pred)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        #fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        fl = tf.multiply(weight, ce)
        #loss = tf.reduce_mean(fl)
        loss = fl
        return loss
    return multi_category_focal_loss2_fixed

def focal_loss(gamma=5):
    epsilon = 1.e-7
    gamma = float(gamma)
    Cross_entropy = tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        ce = Cross_entropy(y_true,y_pred)
        y_t = tf.exp(-ce)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(weight, ce)
        loss = fl
        return loss
    return multi_category_focal_loss2_fixed


def balanced_accuracy(y_true, y_pred):

    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_pred_labels = tf.one_hot(y_pred_labels, depth=tf.shape(y_true)[-1]) 


    correct_predictions = tf.equal(y_true, y_pred_labels)
    correct_predictions = tf.where(correct_predictions, tf.ones_like(y_true), tf.zeros_like(y_true))


    class_counts = tf.reduce_sum(y_true, axis=0)


    class_accuracies = tf.reduce_sum(tf.multiply(correct_predictions, y_true), axis=0) / (class_counts + 1e-10)


    balanced_acc = tf.reduce_mean(class_accuracies)

    return balanced_acc
