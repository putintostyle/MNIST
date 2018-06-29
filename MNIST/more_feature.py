import numpy as np
np.random.seed(123)

from matplotlib import pyplot as plt 
#import matplotlib
#matplotlib.use('Agg')
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPool2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import callbacks
from keras.callbacks import EarlyStopping
import cv2

(X_train, y_train), (X_test, y_test) = mnist.load_data()
roat_X_train=X_train

roat_90_X_train=X_train

gaussian_X_train=X_train

roat_X_test=X_test

roat_90_X_test=X_test

gaussian_X_test=X_test

temp_y_train=y_train

temp_y_test=y_test

#rotation
rows,cols = X_train[0].shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
for i in range(60000):
	roat_X_train[i] = cv2.warpAffine(roat_X_train[i],M,(cols,rows))
for i in range(10000):
	roat_X_test[i] = cv2.warpAffine(roat_X_test[i],M,(cols,rows))
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
for i in range(60000):
        roat_90_X_train[i] = cv2.warpAffine(roat_90_X_train[i],M,(cols,rows))
for i in range(10000):
        roat_90_X_test[i] = cv2.warpAffine(roat_90_X_test[i],M,(cols,rows))


#Gaussian filter
kernel = np.ones((2,2),np.uint8)
for i in range(60000):
	gaussian_X_train[i] = cv2.GaussianBlur(gaussian_X_train[i],(5,5),0.001)
	#X_train[i] = cv2.morphologyEx(X_train[i],cv2.MORPH_GRADIENT,kernel)
for i in range(10000):
	gaussian_X_test[i] = cv2.GaussianBlur(gaussian_X_test[i],(5,5),0.001)
	#X_test[i] = cv2.morphologyEx(X_test[i],cv2.MORPH_GRADIENT,kernel)

#convert to list
X_train=X_train.tolist()
X_test= X_test.tolist()


y_train=y_train.tolist()
y_test=y_test.tolist()
#append
for i in range(60000):
	X_train.append( roat_X_train[i])
		
	y_train.append( temp_y_train[i])
for i in range(10000):
	X_test.append( roat_X_test[i])
	y_test.append( temp_y_test[i])

for i in range(60000):
	X_train.append( gaussian_X_train[i])
	y_train.append( temp_y_train[i])
for i in range(10000):
	X_test.append( gaussian_X_test[i])
	y_test.append( temp_y_test[i])
for i in range(60000):
        X_train.append( roat_90_X_train[i])
        y_train.append( temp_y_train[i])
for i in range(10000):
        X_test.append( roat_90_X_test[i])
        y_test.append( temp_y_test[i])

#convert to array
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#X_train = X_train.reshape(X_train[0], 1, (28, 28))
#X_test = X_test.reshape(X_test[0], 1, (28, 28))
X_train = X_train.astype('float')
X_test = X_test.astype('float')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#model
def get_model():
    model = Sequential()

    model.add(Convolution2D(30,(5,5),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Dropout(0.1))
    model.add(Convolution2D(60,(5,5),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(120,(2,2),activation='relu'))
    model.add(MaxPool2D(pool_size=(3,3)))
    #model.add(Dropout(0.1))
    #model.add(Convolution2D(32,(3,3),strides=(2,2),activation='relu'))
    #model.add(MaxPool2D(pool_size=(2,2)))
    #model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))#100
    #model.add(Dropout(0.1))
    #model.add(Dense(64,activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = get_model()


#callbacks = EarlyStopping(monitor='val_loss',patience=2,verbose=1,mode='min')
history = model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=20, verbose=1,
          validation_split=0.1)

model.save('my_model_06_29.h5')

def plot_learning_curve(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss.png')
    plt.clf()
    
plot_learning_curve(history)


    
   
