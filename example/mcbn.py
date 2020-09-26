import os

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import b4tf


def f(x):
    return 0.1 * (x ** 5) - 0.2*(x ** 4) - 0.1*(x ** 3) - 0.3*(x ** 2) - 0.4*x + 0.5

def normalize(value,mean,std):
    return (value - mean) /std

def de_normalize(value,mean,std):
    return value * std + mean


eps = 0.5
size = 300

id = (np.arange(801) / 100) - 4
x = np.random.uniform(-2.0,3.0,size)
y = f(x) + np.random.normal(loc=0.0,scale=eps,size=size)

x_mean = x.mean()
x_std = x.std()
y_mean = y.mean()
y_std = y.std()
y_min = y.min()
y_max = y.max()

id_ = normalize(id,x_mean,x_std)
x_  = normalize(x ,x_mean,x_std)
y_  = normalize(y ,y_mean,y_std)


mcbn = b4tf.models.MCBN(Sequential([Dense(30,use_bias=False,input_shape=(1,)),
                                    BatchNormalization(),
                                    Activation("relu"),
                                    Dense(30,use_bias=False),
                                    BatchNormalization(),
                                    Activation("relu"),
                                    Dense(30,use_bias=False),
                                    BatchNormalization(),
                                    Activation("relu"),
                                    Dense(1)]),
                        eps**2,
                        input_shape=(1,))
mcbn.compile("adam","mean_squared_error")

mcbn.fit(x_,y_,epochs=500,batch_size=64,
         validation_split=0.05,
         callbacks=[EarlyStopping(patience=20,restore_best_weights=True)])

m, cov = mcbn.predict(id_,n_batches=500)
m, cov = tf.squeeze(m), tf.squeeze(cov)


plt.figure(figsize=(15,15))
plt.plot(x,y,linestyle="",marker=".",label="ovservation")
plt.plot(id,f(id),linestyle=":",label="ground truth")
plt.plot(id,de_normalize(m,y_mean,y_std),label="predict mean")
plt.fill_between(id,
                 de_normalize(m+tf.sqrt(cov),y_mean,y_std),
                 de_normalize(m-tf.sqrt(cov),y_mean,y_std),
                 alpha=0.5,label="credible interval")
plt.legend()
plt.ylim(y_min-5,y_max+5)
plt.savefig(os.path.join(os.path.dirname(__file__),"mcbn_result.png"))
