#====================================This part does the network training using cropped images================================
#===============================================in separate .py file===========================================================


import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from PIL import Image
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

train_images = glob("/home/nick/kg/train_cropped/*jpg")
test_images = glob("/home/nick/kg/test_cropped/*jpg")
INPUT_DIR_TRAIN = '/home/nick/kg/train_cropped'
INPUT_DIR_TEST = '/home/nick/kg/test_cropped'
data = pd.read_csv(f'/home/nick/kg/train.csv')
test = pd.read_csv(f'/home/nick/kg/sample_submission.csv')
train = data.copy()
SIZE = 128

data = pd.read_csv(f'/home/nick/kg/train.csv')
train = data.copy()
train["Image"] = train["Image"].map( lambda x : "/home/nick/kg/train_cropped/"+x)
ImageToLabelDict = dict( zip( train["Image"], train["Id"]))

def ImportImage( filename):
    img = Image.open(filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]
train_img = np.array([ImportImage( img) for img in train_images])
x = train_img


def import_images(filename):
    SIZE= 128
    img = Image.open(f'{filename}')
    img_arr = np.array(img)
    img_arr = np.resize(img_arr, [SIZE,SIZE])
    img_arr = img_arr.astype('float32')
    return img_arr
'''
print('Reading image data...')
X_train = [import_images(filename) for filename in list(train['Image'])]

X_train = np.array(X_train).reshape((-1,SIZE,SIZE,1))
'''
print('Image data read!')





class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)

y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)
Y_train_final = y_cat

WeightFunction = lambda x : 1./x**0.3
ClassLabel2Index = lambda x : lohe.le.inverse_tranform( [[x]])
CountDict = dict( train["Id"].value_counts())
class_weight_dic = { lohe.le.transform( [image_name])[0] : WeightFunction(count) for image_name, count in CountDict.items()}

x = x.reshape( (-1,SIZE,SIZE,1))
input_shape = x[0].shape
X_train = x.astype("float32")


image_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=30,
    width_shift_range=.55,
    height_shift_range=.55,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range = 0.5)

#training the image preprocessing
image_gen.fit(X_train, augment=True)



k_size = (4,4)
pool_size = (2,2)
batch_size = 500

epochs = 8
input_shape = (SIZE, SIZE, 1)


# NETWORK
model = Sequential()
model.add(Conv2D(48, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.33))
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(36, activation='relu'))
model.add(Dense(len(Y_train_final.toarray()[0]), activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

model.fit_generator(image_gen.flow( X_train, Y_train_final.toarray(), batch_size=batch_size), epochs=epochs, verbose=1)
model.save('Final_708.model')
print('Best model written to file!')
print('All complete!')
