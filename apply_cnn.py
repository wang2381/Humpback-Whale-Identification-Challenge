#===========================================This part does the model prediction using the trained model==================================
#============================================================in separate .py file===================================================================
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.models import load_model

INPUT_DIR_TEST = '/home/nick/kg/test_cropped'
SIZE = 128
from keras.models import load_model
model = load_model('/home/nick/kg/Final_708.model')


train_images = glob("/home/nick/kg/train_cropped/*jpg")
test_images = glob("/home/nick/kg/test_cropped/*jpg")

data = pd.read_csv(f'/home/nick/kg/train.csv')
train = data.copy()
train["Image"] = train["Image"].map( lambda x : "/home/nick/kg/train_cropped/"+x)
ImageToLabelDict = dict( zip( train["Image"], train["Id"]))

def ImportImage( filename):
    img = Image.open(filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]
train_img = np.array([ImportImage( img) for img in train_images])
x = train_img

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



def import_images(filename):
    SIZE = 128
    img = Image.open(f'{INPUT_DIR_TEST}/{filename}')
    img_arr = np.array(img)
    img_arr = np.resize(img_arr, [SIZE,SIZE])

    return img_arr







import warnings
from os.path import split

print('Writing prediction to file...')
test = pd.read_csv(f'/home/nick/kg/sample_submission.csv')
with open("/home/nick/kg/final_submission.csv","w") as f:
    with warnings.catch_warnings():
        f.write("Image,Id\n")
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        for image in test_images:
            #img = import_images(image)
            img = ImportImage(image)
            img = img.astype( "float32")
            #applying preprocessing to test images
            #img = np.array(img).reshape((1,SIZE,SIZE,1))

            img = image_gen.standardize( img.reshape(1,SIZE,SIZE))
            #y = model.predict_proba(img)
            y = model.predict_proba(img.reshape(1,SIZE,SIZE,1))
            predicted_args = np.argsort(y)[0][::-1][:5]
            predicted_tags = lohe.inverse_labels( predicted_args)
            image = split(image)[-1]
            predicted_tags = " ".join( predicted_tags)
            f.write("%s,%s\n" %(image, predicted_tags))
print('Final predicted labels written to file!')
print('All complete!')
