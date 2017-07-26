from dataLoader import DataSet
import numpy as np
import pdb
import argparse
import os

from keras.models import load_model
from keras.models import Model

CUDA_VISIBLE_DEVICES="0"

# FLAGS = None
# parser = argparse.ArgumentParser()
# parser.add_argument('--restore',type = bool, default = True, help = 'Restore network')
# parser.add_argument('--resize',type = bool, default = False, help = 'Resize images')
# parser.add_argument('--type',type = str, default = 'area', help = 'area/dia')
# parser.add_argument('--dataset',type = str, default = '2D', help = '2D/3D')
# FLAGS, unparsed = parser.parse_known_args()

def vgg_run():


    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3

    dataFold = '/home/athira/AngiogramProject/'
    augment = True
    field = 'area'
    resize = False

    if field is 'area':
      cutoff_percentage = 60
      output_fold = '/home/athira/Codes/vgg/Output_reg/'
    elif field == 'dia':
      cutoff_percentage = 40
      output_fold = '/home/athira/Codes/vgg/Output_reg/'
    field_onehot = field + '_onehot'

    print("Loading data..")
    data = DataSet(dataFold, preprocess = True, wavelet = False, histEq = False, tophat = False, 
        norm = 'm1to1', resize = resize, smooth = False, context = False)
    
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

    print('------Restoring model-------')
    model = load_model(model_name)
    model.summary()
    print(input_shape)

    # pdb.set_trace()
    # inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    layer = model.layers[-10]
    intermediate_layer_model = Model(inputs=model.input,outputs=layer.output)

    print("Predicting..")
    train['features'] = intermediate_layer_model.predict(X)
    val['features'] = intermediate_layer_model.predict(valX)
    test['features'] = intermediate_layer_model.predict(testX)

    L = 196; D = 512;

    train['features'] = train['features'].reshape([-1, L, D])
    val['features'] = val['features'].reshape([-1, L, D])
    test['features'] = test['features'].reshape([-1, L, D])

    print(train['features'].shape)

    output_fold = '/home/athira/Codes/ShowNTell/'

    side_train, side_val , side_test = data.get_side_patches(dataFold, preprocess = True, wavelet = False, 
            histEq = False, tophat = False, norm = 'm1to1', resize = resize)
    side_X = side_train['imgs'].reshape([-1,s,s,nFeatures])
    side_valX = side_val['imgs'].reshape([-1,s,s,nFeatures])
    side_testX = side_test['imgs'].reshape([-1,s,s,nFeatures])
    side_train = data.createOnehot(side_train, percentage = cutoff_percentage)
    side_val = data.createOnehot(side_val, percentage = cutoff_percentage)
    side_test = data.createOnehot(side_test, percentage = cutoff_percentage)

    print("Predicting side patches..")
    side_train['features'] = intermediate_layer_model.predict(side_X)
    side_val['features'] = intermediate_layer_model.predict(side_valX)
    side_test['features'] = intermediate_layer_model.predict(side_testX)

    side_train['features'] = side_train['features'].reshape([-1, L, D])
    side_val['features'] = side_val['features'].reshape([-1, L, D])
    side_test['features'] = side_test['features'].reshape([-1, L, D])
    print(side_train['features'].shape)

    # side_train = train; side_val = val; side_test = test; #TEMPERORY!


    return train, val, test, side_train, side_val, side_test








