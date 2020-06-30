#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
import seaborn as sns


# In[2]:


data=pd.read_csv(r'C:\Users\Amandeep\Downloads\Compressed\skin-cancer-mnist-ham10000\HAM10000_metadata.csv')


# In[3]:


data=data.drop(columns=['lesion_id'])


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data=data.dropna()


# In[7]:


data.isnull().sum()


# In[8]:


sns.countplot(data['dx'])


# In[9]:


PATH=r'C:\Users\Amandeep\Downloads\Compressed\skin-cancer-mnist-ham10000\HAM10000_images'

#plot the data
count=0
for row in data.iterrows():
    
    
    #print(row[1][0])
    img_name=row[1][0]
    i_name=img_name+'.jpg'
    PA=os.path.join(PATH,i_name)
    img_arr=cv.imread(PA,1)
    img_arr=cv.resize(img_arr,(100,100))
    img_arr=cv.cvtColor(img_arr,cv.COLOR_BGR2RGB)
    #image_value.append([img_arr])
    
    plt.imshow(img_arr)
    plt.show() 
    print(row[1][1])
    
    if count==10:
        break
    count+=1


# In[10]:


PATH=r'C:\Users\Amandeep\Downloads\Compressed\skin-cancer-mnist-ham10000\HAM10000_images'

image_value=[]
for row in data.iterrows():
    
    img_name=row[1][0]
    i_name=img_name+'.jpg'
    PA=os.path.join(PATH,i_name)
    img_arr=cv.imread(PA,1)
    img_arr=cv.resize(img_arr,(100,100))
    img_arr=cv.cvtColor(img_arr,cv.COLOR_BGR2RGB)
    image_value.append([img_arr,row[1][1]])
    #plt.imshow(img_arr)
    #plt.show() 
    


# In[11]:


len(image_value)


# In[12]:


import random
random.shuffle(image_value)


# In[13]:


X=[]
y=[]

for feature,label in image_value:
    X.append(feature)
    y.append(label)

len(X),len(y)


# In[14]:


y


# In[15]:


from sklearn.preprocessing import LabelEncoder

lbl=LabelEncoder()
y=lbl.fit_transform(y)
y.shape


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[18]:


from keras.utils import to_categorical

one_hot_train=to_categorical(train_y)
one_hot_train


# In[19]:


one_hot_test=to_categorical(test_y)
one_hot_test


# In[20]:


#array reshape
train_X=np.array(train_X).reshape(-1,100,100,3)
train_X=train_X/255.0
test_X=np.array(test_X).reshape(-1,100,100,3)
test_X=test_X/255.0

train_X.shape,test_X.shape,one_hot_train.shape,one_hot_test.shape


# In[21]:


from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[22]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(100,100,3),padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.40))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.20))

'''model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.40))'''

model.add(Flatten())

model.add(Dense(64, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(128, activation='linear'))
model.add(Dense(256, activation='linear'))
model.add(Dense(7, activation='softmax'))
model.summary()


# In[23]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_X,one_hot_train,batch_size=128,epochs=10,validation_split=0.2)


# In[24]:


test_loss,test_acc=model.evaluate(test_X,one_hot_test)
test_loss,test_acc


# In[25]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


# In[28]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


# In[ ]:




