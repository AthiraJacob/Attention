import argparse 
from dataLoader import DataSet, DataSet_3D
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
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from vggnet import vgg_run
from dataLoader import DataSet
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


train, val, test, side_train, side_val, side_test = vgg_run()

field = 'area'
field_onehot = field + '_onehot'

s = 224; nFeatures = 3

batch_size = 16 
batchSize_train = 64
nEpochs = FLAGS.nEpochs



if FLAGS.use_all_val is True:
    super_valX = np.concatenate([val['features'], test['features']], axis = 0)
    super_val_labels = np.concatenate([val[field_onehot], test[field_onehot]], axis = 0)
else:
    super_valX = val['features']
    super_val_labels = val[field_onehot]

if FLAGS.use_all_train is True:
    super_X = np.concatenate([train['features'], side_train['features']], axis = 0)
    super_X_labels = np.concatenate([train[field_onehot], side_train[field_onehot]], axis = 0)
else:
    super_X = train['features']; super_X_labels = train[field_onehot]

output_fold = '/home/athira/Codes/AttentionModel/Output_class/'
model_name = output_fold+'model.h5'
best_model = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True)

# pdb.set_trace()
L = 196; D = 512; 
input_shape = (L,D)
    
if os.path.isfile(model_name) and FLAGS.restore is True:
    print('------Restoring model-------')
    model = load_model(model_name)
    model.summary()
    print(input_shape)
elif ~os.path.isfile(model_name) or FLAGS.restore is False:
    
    x = Input(shape=(input_shape))
    x_norm = BatchNormalization(momentum = 0.95, center = True, scale = True)(x)  # N x L x D
    # x = Reshape((L, D))(x)

    h_proj = Dense(D, activation='linear', kernel_initializer = 'glorot_uniform')(x_norm) #N x Lx D
    # h_proj = Reshape((L, D))(h_proj)

    features = Activation('relu')(h_proj)
    features_reshaped = Reshape((-1, D))(features)  #N x Lx D
    h_att = Dense(1, activation='linear', use_bias = False, kernel_initializer = 'glorot_uniform')(features_reshaped) #N x Lx 1
    h_att = Reshape((L,))(h_att)
    alpha = Activation('softmax')(h_att)

    alpha_reshaped = Reshape((L, 1))(alpha) #N x L x 1
    # pdb.set_trace()
    context = Dot(axes = 1)([alpha_reshaped, x_norm]) # N x D
    
    context = Reshape((D,))(context)
    h1 = Dense(1024, activation='relu', kernel_initializer = 'glorot_uniform')(context)
    h1 = Dropout(0.5)(h1)
    y = Dense(2, activation='softmax', kernel_initializer = 'glorot_uniform')(h1)

    model = Model(inputs=x, outputs=y)


    model.compile(loss=keras.losses.categorical_crossentropy,
    		  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# datagen = ImageDataGenerator(
#     samplewise_center=False, samplewise_std_normalization=False,
#     featurewise_center=False, featurewise_std_normalization=False,
#     rotation_range=40,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     horizontal_flip=True,
#     vertical_flip=True,
#     zoom_range=0,
#     fill_mode = 'nearest',
#     cval = 0)
# print('---Augmenting images!---')
# datagen.fit(X)
# model.fit_generator(datagen.flow(X, train[field_onehot], batch_size=batchSize_train),
#     steps_per_epoch=len(X) / batchSize_train, epochs=nEpochs,
#     validation_data=(valX, val[field_onehot]),callbacks=[best_model])

model.fit(super_X, super_X_labels,
          batch_size=batchSize_train,
          epochs=FLAGS.nEpochs,
          verbose=1,
          validation_data=(super_valX, super_val_labels), callbacks=[best_model])    

    # model.save(output_fold+'model.h5')

print('Loading the best model...')
model = load_model(model_name)
print('Best Model loaded!')

score = model.evaluate(val['features'], val[field_onehot], verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
score = model.evaluate(test['features'], test[field_onehot], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('---------Saving model---------')

print('--Debug Mode--')
pdb.set_trace()
y_p = model.predict(test['features'])
y_p_class = np.argmax(y_p, axis = 1)
y_a = test[field_onehot][:,1]
acc_test = np.equal(y_p_class, y_a)*1
auc_test = roc_auc_score(y_a, y_p_class)
print(confusion_matrix(y_a, y_p_class))
print("AUC test = " + str(auc_test))
y_p = model.predict(val['features'])
y_p_class = np.argmax(y_p, axis = 1)
y_a = val[field_onehot][:,1]
acc_val = np.equal(y_p_class, y_a)*1
auc_val = roc_auc_score(y_a, y_p_class)
print(confusion_matrix(y_a, y_p_class))
print("AUC val = " + str(auc_val))









   


















