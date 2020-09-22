# This example requires "matplotlib" and "scikit-learn", too.

import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import b4tf


X, y = datasets.load_boston(return_X_y=True)
# X, y = datasets.fetch_california_housing(return_X_y=True)



scale_x = StandardScaler()
scale_y = StandardScaler()

_X = scale_x.fit_transform(X)
_y = scale_y.fit_transform(y.reshape(-1,1))


pbp = b4tf.models.PBP([50,50,1],input_shape=X.shape[1])
pbp.fit(_X,_y,batch_size=8)


id = np.arange(X.shape[0])
m, v = pbp.predict(_X)
m, v = tf.squeeze(m), tf.squeeze(v)


plt.figure(figsize=(15,15))
plt.plot(id,y,linestyle="",marker=".",label="data")
plt.plot(id,scale_y.inverse_transform(m),alpha=0.5,label="predict mean")
plt.fill_between(id,
                 scale_y.inverse_transform(m+tf.sqrt(v)),
                 scale_y.inverse_transform(m-tf.sqrt(v)),
                 alpha=0.5,label="credible interval")
plt.xlabel("data id")
plt.ylabel("target")
plt.legend()

plt.savefig(os.path.join(os.path.dirname(__file__),"pbp_results.png"))

