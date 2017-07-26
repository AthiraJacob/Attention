import argparse 
from dataLoader import DataSet, DataSet_3D
from utils import *
import pdb
import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, merge
from keras.layers.merge import Multiply, Dot
from keras import backend as K
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint
from keras import regularizers

from vggnet import vgg_run
import numpy as np
import pdb
import argparse
import os

from sklearn.metrics import log_loss

CUDA_VISIBLE_DEVICES="0"

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--nEpochs',type = int, default = 100, help = 'Number of epochs to train for')
parser.add_argument('--restore',type = bool, default = True, help = 'Restore network')
parser.add_argument('--inspect',type = bool, default = False, help = 'Inspect results')
parser.add_argument('--augment',type = bool, default = True, help = 'Augment images')
parser.add_argument('--resize',type = bool, default = False, help = 'Resize images')
parser.add_argument('--type',type = str, default = 'area', help = 'area/dia')
parser.add_argument('--dataset',type = str, default = '2D', help = '2D/3D')
parser.add_argument('--use_all_val',type = bool, default = False, help = 'Use both val and test  for validation')
parser.add_argument('--use_all_train',type = bool, default = False, help = 'Use side patches for training')
FLAGS, unparsed = parser.parse_known_args()


img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
batch_size = 16 
batchSize_train = 64
nEpochs = FLAGS.nEpochs


# Load Cifar10 data. Please implement your own load_data() module for your own dataset
# X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
dataFold = '/home/athira/AngiogramProject/'
augment = True
field = 'area'

if field is 'area':
  cutoff_percentage = 60
  output_fold = '/home/athira/Codes/AttentionModel/Output_reg/'
elif field == 'dia':
  cutoff_percentage = 40
  output_fold = '/home/athira/Codes/AttentionModel/Output_reg/'
field_onehot = field + '_onehot'


print("Loading data..")
data = DataSet(dataFold, preprocess = True, wavelet = False, histEq = False, tophat = False, 
    norm = 'm1to1', resize = FLAGS.resize, smooth = False, context = False)

train, val, test = data.splitData()
nTrain = data.nTrain; nVal = data.nVal; nTest = data.nTest
train = data.createOnehot(train, percentage = cutoff_percentage)
val = data.createOnehot(val, percentage = cutoff_percentage)
test = data.createOnehot(test, percentage = cutoff_percentage)
nFeatures = 3

model_name = output_fold+'model.h5'
s = 224

X = train['imgs'].reshape([-1,s,s,nFeatures])
valX = val['imgs'].reshape([-1,s,s,nFeatures])
testX = test['imgs'].reshape([-1,s,s,nFeatures])
input_shape = (s,s,nFeatures)


if FLAGS.use_all_train is True:
    side_train, _, _ = data.get_side_patches(dataFold, preprocess = True, wavelet = False, 
        histEq = False, tophat = False, norm = 'm1to1', resize = FLAGS.resize)
    # pdb.set_trace()
    side_X = side_train['imgs'].reshape([-1,s,s,nFeatures])
    super_X = np.concatenate([X, side_X], axis = 0)
    super_X_labels = np.concatenate([train[field], side_train[field]], axis = 0)
else:
    super_X = X; super_X_labels = train[field]

if FLAGS.use_all_val is True:
    super_valX = np.concatenate([valX, testX], axis = 0)
    super_val_labels = np.concatenate([val[field], test[field]], axis = 0)
else:
    super_valX = valX
    super_val_labels = val[field]

print(super_valX.shape)


model_name = output_fold+'model.h5'
best_model = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True)

# pdb.set_trace()

if os.path.isfile(model_name) and FLAGS.restore is True:
    print('------Restoring model-------')
    model = load_model(model_name)
    model.summary()
    print(input_shape)
elif ~os.path.isfile(model_name) or FLAGS.restore is False:
    
    vgg_fold = '/home/athira/Codes/vgg/Output_reg/'
    vgg_mod_name = vgg_fold + 'model.h5'
    print('------Loading VGG model-------')
    vgg_model = load_model(vgg_mod_name)
    
    for layer in vgg_model.layers:
           layer.trainable = False
    print("Creating VGG-part")
    layer = vgg_model.layers[-10] #Orig: -10, Try -12, -8, -7 (7x7)
    part_vgg_model = Model(inputs=vgg_model.input,outputs=layer.output)


    print("Creating attention mechanism")
    input_shape = (14,14,512)
    L = input_shape[0]*input_shape[1];  D = input_shape[2]; 

    x = Input(shape=(input_shape))
    x_reshaped = Reshape((-1,D))(x)
    x_norm = BatchNormalization(momentum = 0.95, center = True, scale = True)(x_reshaped)  # N x L x D
    h_proj = Dense(D, activation='relu', kernel_initializer = 'glorot_uniform')(x_norm) #N x Lx D
    features = Dropout(0.5)(h_proj) #N x Lx D
    h_att = Dense(1, activation='linear', use_bias = False, kernel_initializer = 'glorot_uniform', activity_regularizer=regularizers.l1(0.001))(features) #N x Lx 1
    h_att = Reshape((L,))(h_att)
    alpha = Activation('softmax')(h_att)
    alpha_reshaped = Reshape((L, 1))(alpha) #N x L x 1
    # pdb.set_trace()
    context = Dot(axes = 1)([alpha_reshaped, x_norm]) # N x D
    context = Reshape((D,))(context)
    h1 = Dense(512, activation='relu', kernel_initializer = 'glorot_uniform')(context)
    h1 = Dropout(0.5)(h1)
    y = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform')(h1)

    att_model = Model(inputs=x, outputs=y)


    model = Sequential()
    model.add(part_vgg_model)
    model.add(att_model)
    model.compile(loss='mean_squared_error',
    		  optimizer=keras.optimizers.Adadelta())

if FLAGS.inspect is False:

    datagen = ImageDataGenerator(
        samplewise_center=False, samplewise_std_normalization=False,
        featurewise_center=False, featurewise_std_normalization=False,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0,
        fill_mode = 'nearest',
        preprocessing_function = None, 
        cval = 0)
    print('---Augmenting images!---')

    model.fit_generator(datagen.flow(super_X, super_X_labels/100.0, batch_size=batchSize_train),
        steps_per_epoch=len(super_X) / batchSize_train, epochs=nEpochs,
        validation_data=(super_valX, super_val_labels/100.0),callbacks=[best_model])

    print('Loading the best model...')
    model = load_model(model_name)
    print('Best Model loaded!')

    score = model.evaluate(valX, val[field]/100.0, verbose=0)
    print('Validation loss:', score)
    score = model.evaluate(testX, test[field]/100.0, verbose=0)
    print('Test loss:', score)

else:
    print('--Debug Mode--')
    pred_test = model.predict(testX)
    act_test = test[field]/100.0
    pred_val = model.predict(valX)
    act_val = val[field]/100.0
    # pdb.set_trace()
    # pred = {'pred_test' : pred_test, 'act_test': act_test, 'pred_val': pred_val, 'act_val': act_val}
    pred = [pred_test, act_test, pred_val, act_val]
    np.save('pred_'+field+'.npy', pred)

    visualize = 0
    if visualize == 1:
        #To visualize worst cases
        pdb.set_trace()
        pred_val = pred_val[:,0]; pred_test = pred_test[:,0]
        ind_val = (np.absolute(pred_val - act_val)*100)>30
        imgs_val = np.squeeze(valX[ind_val])
        pred = pred_val[ind_val]; act = act_val[ind_val];
        worst = {}; worst['imgs'] = imgs_val; worst['pred'] = pred; worst['act'] = act
        np.save('worst',worst)
        #Best cases 
        ind_val = (np.absolute(pred_val - act_val)*100)<1
        imgs_val = np.squeeze(valX[ind_val])
        pred = pred_val[ind_val]; act = act_val[ind_val];
        best = {}; best['imgs'] = imgs_val; best['pred'] = pred; best['act'] = act
        np.save('best',best)


    _, side_val, side_test = data.get_side_patches(dataFold, preprocess = True, wavelet = False, 
    histEq = False, tophat = False, norm = 'm1to1', resize = FLAGS.resize)

    side_valX = side_val['imgs'].reshape([-1,s,s,nFeatures])
    side_testX = side_test['imgs'].reshape([-1,s,s,nFeatures])

    pred_test = model.predict(side_testX)
    act_test = side_test[field]/100.0
    pred_val = model.predict(side_valX)
    act_val = side_val[field]/100.0
    pred = [pred_test, act_test, pred_val, act_val]
    np.save('side_pred_'+field+'.npy', pred)

    print('--Saved!')



