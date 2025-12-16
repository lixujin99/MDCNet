#!/usr/bin/env python
# encoding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='5'

import tensorflow as tf
import tensorflow
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import argparse
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import scipy.io as sio
import numpy as np
import gc
import copy
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import KFold
from scipy import signal
from model_keras1 import EEGmulti_MDC
from loss_keras import FocalLoss, focal_loss, balanced_accuracy 


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60, help="number of epochs of training") # todo
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")  # 30
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")  # 0.0002
parser.add_argument("--b1", type=float, default=0.9, help="adam: learning rate")  # 0.0002
parser.add_argument("--b2", type=float, default=0.999, help="adam: learning rate")  # 0.0002
parser.add_argument("--wd", type=float, default=0.0, help="weight_dec")  # 0.0002
parser.add_argument('--sp', default=1, type=int)
parser.add_argument('--chans', default=64, type=int)
parser.add_argument('--samples', default=128, type=int)
parser.add_argument('--cuda', default=2, type=int)
parser.add_argument('--gamma', default=5, type=int)
parser.add_argument('--resample', default=128, type=int)
parser.add_argument('--model', default='ablation_MDC', type=str)
parser.add_argument('--patience', default=100, type=int)
opt = parser.parse_args()
print(opt)

binary_cls = False
exclude_unseen = True

if binary_cls:
    cls_nb = 2
else:
    cls_nb = 3


model_save_folder = '../save_para/%s/' %(opt.model)
os.makedirs(model_save_folder, exist_ok=True)

Fold = 3

all_sub1 = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S21', 'S22', 'S23', 'S24', 'S25']
all_sub2 = ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20','S26', 'S27', 'S28', 'S29', 'S30']
perf1 = np.zeros((len(all_sub1), 2, Fold))
ba1 = np.zeros((len(all_sub1), 2, Fold))
conf1 = np.zeros((len(all_sub1), 2, Fold, cls_nb, cls_nb)) 
perf2 = np.zeros((len(all_sub2), 2, Fold))
ba2 = np.zeros((len(all_sub2), 2, Fold))
conf2 = np.zeros((len(all_sub2), 2, Fold, cls_nb, cls_nb))

for task in range(1, 5):
    if task in [1,2]:
        all_sub = all_sub1
    else:
        all_sub = all_sub2

    all_data = []
    all_label = []
    all_behave = []
    for isub, subject in enumerate(all_sub):
        name = '../%s.mat' % subject

        print(name)
        mat = sio.loadmat(name)

        print('task', task)
        if task in [1, 2]:
            ti = str(task)
        else:
            if task==3:
                ti = 'A'
            else:
                ti = 'B'
        # ti=task
        data = mat['eeg_task'+ti]
        label_all = mat['label_task'+ti][0]
        behave = mat['behavior_task'+ti][0]
        data = signal.resample(data, opt.resample, axis=-1)

        if exclude_unseen:
            data = data[behave!=0, :]
            label_all = label_all[behave!=0, ]
        
        label_bin = np.where(label_all==0, 0, 1)
        label_all = np.concatenate((np.expand_dims(label_all,axis=1),np.expand_dims(label_bin,axis=1)),axis=-1)
        data_t_train_mean = data.mean(-1)[:, :, np.newaxis]
        data_t_train_std = data.std(-1)[:, :, np.newaxis]
        data = (data - data_t_train_mean) / data_t_train_std

        data = np.transpose(data[:, :, :, np.newaxis], (0, 2, 1, 3))
        all_data.append(data)
        all_label.append(label_all)
        all_behave.append(behave)


    for isub, test_sub in enumerate(all_sub):

        test_data = all_data[isub]
        test_label = all_label[isub]

        adc = copy.deepcopy(all_data)
        alc = copy.deepcopy(all_label)

        adc.pop(isub)
        alc.pop(isub)
        print('all train sub: ', len(adc))
        
        train_data_ori = np.concatenate(adc, axis=0)
        train_label_ori = np.concatenate(alc, axis=0)

        for fd in range(Fold):

            train_data, val_data, train_label, val_label = train_test_split(train_data_ori, train_label_ori,
                                                                            test_size=0.2,
                                                                            stratify=train_label_ori, random_state=fd)


            class_weights_tri = class_weight.compute_class_weight('balanced', classes=np.unique(train_label[:,0]), y=train_label[:,0])
            class_weights_bin = class_weight.compute_class_weight('balanced', classes=np.unique(train_label[:,1]), y=train_label[:,1])
            class_weights_tri = dict(enumerate(class_weights_tri))
            class_weights_bin = dict(enumerate(class_weights_bin))

            print("\r", "Task: %d | test subject: [%d/%d//%s] | fold: [%d/%d]" % (task, isub + 1, len(all_sub), test_sub, fd+1, Fold))

            X_train = train_data
            Y_train_tri = np_utils.to_categorical(train_label[:,0])
            Y_train_bin = np_utils.to_categorical(train_label[:,1])

            X_validate = val_data
            Y_validate_tri = np_utils.to_categorical(val_label[:,0])
            Y_validate_bin = np_utils.to_categorical(val_label[:,1])

            model = EEGmulti_MDC(dropout_rate=0.5)

            # compile the model and set the optimizers
            Rd_LR = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_triplet_loss', factor=0.5, patience=5, verbose=0, mode='auto')

            optim = Adam(lr=opt.lr, beta_1=opt.b1, beta_2=opt.b2, epsilon=1e-8, decay=opt.wd, amsgrad=False)
            model.compile(loss={'triplet': 'categorical_crossentropy', 'binary': 'categorical_crossentropy'}, optimizer=optim, loss_weights={'triplet':1, 'binary':0.01},
                            metrics={'triplet':balanced_accuracy, 'binary':balanced_accuracy})

            tensorboard = TensorBoard(log_dir='./tsboard')
            if binary_cls:
                ckp_name = '../save_para/tmp_model_SI/checkpoint_bin_M%s_S%s_T%d_F%d.h5' % (opt.model, test_sub, task, fd)
            else:
                ckp_name = '../save_para/tmp_model_SI/checkpoint_M%s_S%s_T%d_F%d.h5' % (opt.model, test_sub, task, fd)


            checkpointer = ModelCheckpoint(filepath=ckp_name, verbose=0, save_best_only=True)

            early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)


            fittedModel = model.fit(X_train, {'triplet': Y_train_tri, 'binary':Y_train_bin}, batch_size=opt.batch_size, epochs=opt.epochs,
                                        verbose=0, validation_data=(X_validate, {'triplet':Y_validate_tri, 'binary':Y_validate_bin}),
                                        callbacks=[early_stopping, checkpointer, tensorboard, Rd_LR], class_weight={'triplet':class_weights_tri, 'binary':class_weights_bin})
            
            model.load_weights(ckp_name)

            probs = model.predict(test_data)
            probs = probs[0]
            clf_perf2 = accuracy_score(test_label[:,0], probs.argmax(-1))
            clf_ba = balanced_accuracy_score(test_label[:,0], probs.argmax(-1))
            clf_conf = confusion_matrix(test_label[:,0], probs.argmax(-1))
            if task in [1, 2]:
                perf1[isub, task - 1, fd] = clf_perf2
                ba1[isub, task - 1, fd] = clf_ba
                conf1[isub, task - 1, fd, :] = clf_conf
            else:
                perf2[isub, task - 3, fd] = clf_perf2
                ba2[isub, task - 3, fd] = clf_ba
                conf2[isub, task - 3, fd, :] = clf_conf

            print("\r", "Task: %d | test subject: [%d/%d//%s] | fold: [%d/%d]" % (
            task, isub + 1, len(all_sub), test_sub, fd + 1, Fold))
            print("\r", "Test : BA:%.4f" % (clf_ba))
            print("\r", clf_conf)

            K.clear_session()
            del model, fittedModel
            gc.collect()

        mean_sub = np.mean(ba1,axis=-1)
        mean_task = np.mean(mean_sub,axis=0)
        print(mean_sub)
        print(mean_task)
        mean_sub = np.mean(ba2,axis=-1)
        mean_task = np.mean(mean_sub,axis=0)
        print(mean_sub)
        print(mean_task)


if binary_cls:
    result_name = '../results_SI/%s_bin.mat'%(opt.model)
else:
    result_name = '../results_SI/%s_tri.mat'%(opt.model)
sio.savemat(result_name,
            {'perf123': perf1, 'perf45': perf2, 'ba123': ba1, 'ba45': ba2, 'conf123': conf1,
                'conf45': conf2})
