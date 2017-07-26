from dataLoader import DataSet
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model, Input
import cv2 
import keras
import tensorflow as tf
import pdb
from keras.models import Sequential
from utils import *


img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
dataFold = '/home/athira/AngiogramProject/'
field = 'area'
resize = False

if field is 'area':
  cutoff_percentage = 60
  output_fold = '/home/athira/Codes/AttentionModel/Output_reg_best/Output_reg_end2end/'
elif field == 'dia':
  cutoff_percentage = 40
  output_fold = '/home/athira/Codes/AttentionModel/Output_reg_best/'
field_onehot = field + '_onehot'

print("Loading data..")
data = DataSet(dataFold, preprocess = True, wavelet = False, histEq = False, tophat = False, 
    norm = 'm1to1', resize = resize, smooth = False, context = False, vgg = False)
print("Data loaded!")
train, val, test = data.splitData()
nTrain = data.nTrain; nVal = data.nVal; nTest = data.nTest
train = data.createOnehot(train, percentage = cutoff_percentage)
val = data.createOnehot(val, percentage = cutoff_percentage)
test = data.createOnehot(test, percentage = cutoff_percentage)
s = val['imgs'].shape[1]
nFeatures = 1

model_name = output_fold+'model.h5'

X = train['imgs'].reshape([-1,s,s,nFeatures])
valX = val['imgs'].reshape([-1,s,s,nFeatures])
testX = test['imgs'].reshape([-1,s,s,nFeatures])
input_shape = (s,s,nFeatures)

get_custom_objects().update({'sparsemax': sparsemax})
model_name = output_fold+'model.h5'
print('------Restoring model-------')
model = load_model(model_name, custom_objects = {"sparsemax":sparsemax})
model.summary()

pdb.set_trace()

test_pred = model.predict(testX)

sub_model_1 = model.layers[0]
sub_model_2 = model.layers[1]
alpha_layer = sub_model_2.get_layer('sparsemax')

get_alpha = Model(inputs = sub_model_2.get_input_at(0), outputs = alpha_layer.output)

small_model = Sequential()
small_model.add(sub_model_1)
small_model.add(get_alpha)

alphas = small_model.predict(testX)

test['predictions'] = test_pred
test['alphas'] = alphas

np.save('eval_test', test)







