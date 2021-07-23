#!/usr/bin/env python
# coding: utf-8

# In[44]:


import tensorflow
from tensorflow import keras


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


import matplotlib.pyplot as plt


# In[47]:


batch_size=128
num_class=10
epochs=20
(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()


# In[48]:


keras.backend.image_data_format()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')


# In[49]:


x_train, x_test = x_train / 255.0, x_test / 255.0
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])


# In[50]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[51]:


model.fit(x_train, y_train, epochs=5)


# In[52]:


x_train.shape


# In[53]:



x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)


# In[ ]:


for i in range(10):
    plt.subplot(2,5,i+1)
    plt.title("Label:"+str(i))
    plt.imshow(x_test[i].reshape(28,28),cmap=None)


# In[ ]:


import pandas as pd

#現在の最大表示行数の出力
pd.get_option("display.max_rows")

#最大表示行数の指定（ここでは50行を指定）
pd.set_option('display.max_rows', 50)
model.predict(x_test)


# In[ ]:


import tensorflow
from tensorflow import keras
batch_size=128
num_class=10
epochs=20
(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
model=keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10,activation="softmax")
])
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)


# In[ ]:


for i in range(10):
    plt.subplot(2,5,i+1)
    plt.title("Label:"+str(i))
    plt.imshow(x_train[i].reshape(28,28),cmap=None)


# In[ ]:


model.predict(x_test)


# In[ ]:


y_test


# In[ ]:




