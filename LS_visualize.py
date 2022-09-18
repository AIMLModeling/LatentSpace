import matplotlib.pyplot as plt
import numpy as np
import warnings as w
w.simplefilter(action = 'ignore')
import argparse
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
from PIL import Image
import cv2

#------------------------------------------------------------------------------#
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description='Autoencoder Latent Space Visualization')
parser.add_argument('--input',type=str,required=True,help='Directory containing images eg: data/')
parser.add_argument('--name',type=str,required=True,help='Model Name')
parser.add_argument('--epoch',type=int,default=500,help='No of training iterations')
parser.add_argument('--mode',type=str,required=True,help='Mode: train or plot')
args = parser.parse_args()

#------------------------------------------------------------------------------#

bottleneck_size = 2

input_img = Input(shape=(12288,))

encoded = Dense(1024, activation='relu')(input_img)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(bottleneck_size, activation='linear')(encoded)
encoder = Model(input_img, encoded)


encoded_input = Input(shape=(bottleneck_size,))
decoded = Dense(16, activation='relu')(encoded_input)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(12288, activation='sigmoid')(decoded)
decoder = Model(encoded_input, decoded)


full = decoder(encoder(input_img))
ae = Model(input_img, full)
ae.compile(optimizer='adam', loss='mean_squared_error')


#------------------------------------------------------------------------------#
import numpy as np

y_train = []
y_test =[]

for each in os.listdir(args.input):
    img = cv2.imread(os.path.join(args.input,each))
    np_im = np.array(img)
    y_train.append(img)
    y_test.append(img)

x_train=np.array(y_train)
x_test=np.array(y_test)
    
print(x_train.shape)
    
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0], 12288))
x_test = x_test.reshape((x_test.shape[0], 12288))

#------------------------------------------------------------------------------#
if "model_"+args.name+".h5" in os.listdir():
    ae = load_model('model_'+args.name+'.h5')
    encoder = load_model('encoder_'+args.name+'.h5')
    decoder = load_model('decoder_'+args.name+'.h5')

if args.mode in ['train', 'Train']:
    
    for i in range(1000):
        print("Run "+str(i)+": ")
        ae.fit(x_train, x_train, 
            epochs = args.epoch,
            batch_size=256,
            validation_data=(x_test, x_test))
        ae.save('model_'+args.name+'.h5')
        encoder.save('encoder_'+args.name+'.h5')
        decoder.save('decoder_'+args.name+'.h5')


#------------------------------------------------------------------------------#
        
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)



buckets = [0] * x_test.shape[0]
fig, ax = plt.subplots(1, 2, figsize=(128,128))
ax[0].scatter(encoded_imgs[:,0],encoded_imgs[:,1],
	c=buckets, s=8, cmap='tab10')


def onclick(event):
    global flag
    ix, iy = event.xdata, event.ydata
    latent_vector = np.array([[ix, iy]])
    
    decoded_img = decoder.predict(latent_vector.astype('float32'))
    decoded_img = decoded_img.reshape(64, 64, 3)
    decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    decoded_img = cv2.resize(decoded_img,(512,512))
    ax[1].imshow(decoded_img, cmap='gray')
    plt.draw()

cid = fig.canvas.mpl_connect('motion_notify_event', onclick)

plt.show()
#------------------------------------------------------------------------------#
