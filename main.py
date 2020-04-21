import numpy as np
import scipy.io as sio
import tensorflow as tf
# import matplotlib.pyplot as plt


class PhysionetDataset:
    
    def __init__(self, foldername="Data\\"):
        
        self.leads = dict()
        self.specs = dict()
        self.labels = dict()
        self.diseases = dict()
         
        self.diseases['AF'] = 0
        self.diseases['I-AVB'] = 1
        self.diseases['LBBB'] = 2
        self.diseases['Normal'] = 3
        self.diseases['PAC'] = 4
        self.diseases['PVC'] = 5
        self.diseases['RBBB'] = 6
        self.diseases['STD'] = 7
        self.diseases['STE'] = 8

        self.xtrain = None
        self.ytrain = None
        self.xvalid = None
        self.yvalid = None
        
        self.foldername = foldername       
        return
        
    
    @staticmethod
    def vstr(a):
        s = str(a)
        t = '0'*(4 - len(s))
        s = t + s
        return s
    
    
    def load_data(self, id1=4762, id2=5290):
        
        for i in range(id1, id2):
            if i % 100 == 0:
                print(i, ' of ', (id2 - id1 + 1))
                
            mat = sio.loadmat(self.foldername + 'A' + self.vstr(i) + '.mat')
            hea = open("Data\A" + self.vstr(i) + ".hea", 'r+')

            s = hea.readlines()
            d = s[15][:-1].split(' ')
            d = d[1:][0].split(',')
            
            self.leads[i] = mat['val']
            self.specs[i] = [item for item in d]
            
            label = np.zeros([1, 9])
            for item in d:
                label[0, self.diseases[item]] = 1
            self.labels[i] = label
            
        return
    
    def get_save_data(self, id1=4762, id2=5290, tsize=4000, valid_ratio=0.9):
        
        x0 = np.zeros([id2-id1, 12, tsize, 1])
        y0 = np.zeros([id2-id1, 9])
        
        for i in range(id1, id2):
            if i % 100 == 0:
                print(i, ' of ', (id2 - id1 + 1))
            x = self.leads[i]
            x0[i-id1, :, :, :] = np.reshape(x[:, :tsize], [1, 12, tsize, 1])
            y0[i-id1, :] = self.labels[i]
            
        m = int(x0.shape[0]*valid_ratio)

        np.save('xt.npy', x0[:m])
        np.save('yt.npy', y0[:m])
        np.save('xv.npy', x0[m:])
        np.save('yv.npy', y0[m:])
        
        return
    
    def get_train_data(self, id1=4762, id2=5290, valid_ratio=0.9, tsize=4000, output=False): 
        
        x0 = np.zeros([id2-id1, 12, tsize, 1])
        y0 = np.zeros([id2-id1, 9])
        
        for i in range(id1, id2):
            if i % 100 == 0:
                print(i, ' of ', (id2 - id1 + 1))
            x = self.leads[i].astype(np.float32)
            x0[i-id1, :, :, :] = np.reshape(x[:, :tsize], [1, 12, tsize, 1])
            y0[i-id1, :] = self.labels[i].astype(np.float32)
            
        m = int(x0.shape[0]*valid_ratio)
        if output:
            return x0[:m], y0[:m], x0[m:], y0[m:]
        
        self.xtrain = x0[:m]/1000
        self.ytrain = y0[:m]
        self.xvalid = x0[m:]/1000
        self.yvalid = y0[m:]
        
        return
    
    def load_train_data(self, output=False):
        
        x0 = np.load('xt.npy').astype(np.float32)
        y0 = np.load('yt.npy').astype(np.float32)
        x1 = np.load('xv.npy').astype(np.float32)
        y1 = np.load('yv.npy').astype(np.float32)
        
        if output:
            return x0, y0, x1, y1
        
        self.xtrain = x0/100
        self.ytrain = y0
        self.xvalid = x1/100
        self.yvalid = y1
        
        return


class Encoder(tf.keras.Model):
    def __init__(self, tsize=4000):
        super(Encoder, self).__init__()
        m = int(tsize/(1))
        # self.conv1 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 40), 
        #                                     strides=(1, 10), padding='same', activation='relu',
        #                                     input_shape=(12, tsize, 1))
        # self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 20), 
        #                                     strides=(1, 10), padding='same', activation='relu')
        # self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 20), 
        #                                     strides=(1, 4), padding='same', activation='relu')
        self.fltn1 = tf.keras.layers.Flatten(input_shape=(12, m, 1))
        
        self.dens1 = tf.keras.layers.Dense(input_shape=(None, m*12*1), units=100, activation='sigmoid')
        # self.drop1 = tf.keras.layers.Dropout(rate=0.0)
        
        self.dens2 = tf.keras.layers.Dense(input_shape=(None, 100), units=9, activation='sigmoid')
        # self.dens3 = tf.keras.layers.Dense(input_shape=(None, 50), units=9, activation='relu')
        return

    def call(self, inputs):
        x = inputs
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.fltn1(x)
        
        # x = self.drop1(x)
        x = self.dens1(x)
        x = self.dens2(x)
        # x = self.dens3(x)
        
        return x
     
      
id1 = 1
id2 = 3000
dataset = PhysionetDataset()
# dataset.load_data(id1=id1, id2=id2)
# dataset.get_save_data(id1=id1, id2=id2, valid_ratio=0.9)
dataset.load_train_data()

model = Encoder()

model.compile(
          loss=tf.keras.losses.mse,
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
          metrics=['accuracy'])

# model.compile(
#           loss=tf.keras.losses.mse,
#           optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#           metrics=['accuracy'])

model.fit(x=dataset.xtrain[0:2500, :, :, :], y=dataset.ytrain[0:2500, :],
          batch_size=100, epochs=200, workers=4)
a = dataset.xtrain[1:10, :, :, :]
b = dataset.ytrain[1:10, :]
print(b)
print(model(a))
