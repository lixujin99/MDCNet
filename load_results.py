import os 
import scipy.io as sio
import numpy as np

save_model = ' '
save_path = '../%s_tri.mat'%(save_model)

mat = sio.loadmat(save_path)


ba1 = mat['ba123']
ba2 = mat['ba45']

mean_fold = np.mean(ba1,axis=-1)
mean_sub = np.mean(mean_fold,axis=-1)
mean_task = np.mean(mean_sub,axis=0)
print(mean_sub)
print(mean_task)
mean_fold = np.mean(ba2,axis=-1)
mean_sub = np.mean(mean_fold,axis=-1)
mean_task = np.mean(mean_sub,axis=0)
print(mean_sub)
print(mean_task)