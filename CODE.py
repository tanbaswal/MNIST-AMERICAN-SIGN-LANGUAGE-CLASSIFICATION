#--------------------------------------DSML ASSESSMENT 4 18291 ---------------------------------------------------------

# ------------------------------------ LINEAR REGRESSION MODEL ---------------------------------------------------------

#Importing Libraries:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import plot_confusion_matrix

#Loading Dataset
df=pd.read_csv('C:\\Users\\tanis\\Downloads\\archive (1)\\ASL_train.csv')
print(df.shape)
df.head()
X=df.values[0:,1:]
Y = df.values[0:,0]
sample = X[1]
plt.imshow(sample.reshape((28,28)))
Y.shape

#Dataset Splitting and Scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.30)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# Model and Prediction
lr = LinearRegression()
lr.fit(X_train_scaled,Y_train)
Y_pred=lr.predict(X_test_scaled)
Y_pred

#Accuracy of the Linear Regression Model:
accuracy =r2_score(Y_test, Y_pred)
print(accuracy)

print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_pred))










#----------------------------------------------- LOGISTIC REGRESSION -----------------------------------------------------

# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Loading Dataset
df=pd.read_csv('C:\\Users\\tanis\\Downloads\\archive (1)\\ASL_train.csv')
print(df.shape)
df.head()
X=df.values[0:,1:]
Y = df.values[0:,0]
sample = X[1]
plt.imshow(sample.reshape((28,28)))
Y.shape

#Dataset Splitting and Scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1,test_size=0.20)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0


#Model and Prediction
LR = LogisticRegression(penalty='none', tol=0.1, solver='saga',multi_class='multinomial').fit(X_train_scaled, Y_train)
Y_pred=LR.predict(X_test_scaled)
Y_test
Y_pred

# Confusion Matrix 
result = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(result)

# Classification report
result1 = classification_report(Y_test, Y_pred)
print("\nClassification Report:")
print (result1)

# Accuracy score
result2 = accuracy_score(Y_test, Y_pred)
print("\nAccuracy:",result2)

cm = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], normalize='index')
p = plt.figure(figsize=(10,10));
p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)









#------------------------------------------- SUPPORT VECTOR MACHINE------------------------------------------------------

#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

#Loading Dataset
df = pd.read_csv('C:\\Users\\tanis\\Downloads\\archive (1)\\ASL_train.csv')
print(df.shape)
df.head()
X = df.iloc[0:,1:].values
X = X/225
Y = df.iloc[0:,0].values

#Dataset Splitting and Scaling 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 50)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


#Model and Prediction
SV = svm.SVC(gamma = 0.001, C = 1000, decision_function_shape='ovo', random_state = 100)
SV.fit(X_train, y_train)
Y_pred = SV.predict(X_test)
Y_accuracy = accuracy_score(y_test, Y_pred)
print(Y_accuracy)

#Confusion Matrix:
plot_confusion_matrix(SV, X_test, y_test, cmap=plt.cm.CMRmap)
plt.figure(figsize=(24, 24))
plt.show()









#-------------------------------------------- RANDOM FOREST -------------------------------------------------------------

#Importing the Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

#Loading Dataset
data=pd.read_csv('C:\\Users\\tanis\\Downloads\\archive (1)\\ASL_train.csv');
df_x = data.iloc[:,1:];
df_y = data.iloc[:,0];

#Dataset Splitting 
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4);

#Model and Prediction
rf = RandomForestClassifier(n_estimators=100);
rf.fit(x_train,y_train);
pred=rf.predict(x_test);

count=0;
s = y_test.values;
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count + 1;

#Confusion Matrix:
plot_confusion_matrix(rf, x_test, y_test, cmap=plt.cm.CMRmap)
plt.figure(figsize=(48, 48))
plt.show()

#Accuracy of the Random Forest Model:        
print((count/len(pred)) *100);








#------------------------------------------- NEURAL NETWORK (ANN) -----------------------------------------------------------

#Importing the Libraries
import pandas as pd
import keras
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten

#Loading Dataset
df = pd.read_csv('C:\\Users\\tanis\\Downloads\\archive (1)\\ASL_train.csv')
df.head()
X = df.iloc[:,1:]
X = X/225
Y = df.iloc[:,0]
Y = Y.astype(int)

#Dataset Splitting and Scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state=0)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# Model and Prediction
model = Sequential(Flatten(input_shape = [28, 28]))
model.add(Dense(300, activation='relu'))
model.add(Dropout(rate= 0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(25, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam',
              metrics='accuracy')
h = model.fit(X_train, y_train, epochs=6, verbose=True)


#Plot for Model Accuracy:
plt.plot(h.history['accuracy'])
plt.title('Model accuracy')
plt.show()

#Acuuracy of the ANN Model:
['accuracy']









#-------------------------------------------------- CONVOLUTIONAL NEURAL NETWORK(CNN)----------------------------------

#Importing the Libraries
import os
import numpy as np 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Loading Dataset
train_df=pd.read_csv('C:\\Users\\tanis\\Downloads\\archive (1)\\ASL_train.csv')
test_df=pd.read_csv('C:\\Users\\tanis\\Downloads\\archive (1)\\ASL_test.csv')
train_df.info()
test_df.info()
train_df.describe()
train_df.head(6)
train_label=train_df['label']
train_label.head()
trainset=train_df.drop(['label'],axis=1)
trainset.head()
X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
print(X_train.shape)
test_label=test_df['label']
X_test=test_df.drop(['label'],axis=1)
print(X_test.shape)
X_test.head()


#Scaling
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)
y_train

X_test=X_test.values.reshape(-1,28,28,1)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

train_datagen = ImageDataGenerator(rescale = 1./255,rotation_range = 0,height_shift_range=0.2,width_shift_range=0.2,shear_range=0,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

X_test=X_test/255


#Dataset Visualization of Random Characters:
fig,axe=plt.subplots(2,2)
fig.suptitle('Preview of dataset')
axe[0,0].imshow(X_train[0].reshape(28,28),cmap='gray')
axe[0,0].set_title('label: 3  letter: C')
axe[0,1].imshow(X_train[1].reshape(28,28),cmap='gray')
axe[0,1].set_title('label: 6  letter: F')
axe[1,0].imshow(X_train[2].reshape(28,28),cmap='gray')
axe[1,0].set_title('label: 2  letter: B')
axe[1,1].imshow(X_train[4].reshape(28,28),cmap='gray')
axe[1,1].set_title('label: 13  letter: M')



sns.countplot(train_label)
plt.title("Frequency of each label")


#CNN Model:
# Conv Layer 1 -- UNITS - 128  KERNEL SIZE - 5 * 5   STRIDE LENGTH - 1   ACTIVATION - ReLu
# Conv Layer 2 -- UNITS - 64   KERNEL SIZE - 3 * 3   STRIDE LENGTH - 1   ACTIVATION - ReLu
# Conv Layer 3 -- UNITS - 32   KERNEL SIZE - 2 * 2   STRIDE LENGTH - 1   ACTIVATION - ReLu
# MaxPool Layer 1 -- MAX POOL WINDOW - 3 * 3   STRIDE - 2
# MaxPool Layer 2 -- MAX POOL WINDOW - 2 * 2   STRIDE - 2
# MaxPool Layer 3 -- MAX POOL WINDOW - 2 * 2   STRIDE - 2

model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))      
model.add(Flatten())



model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


#Model Training:
model.fit(train_datagen.flow(X_train,y_train,batch_size=200), epochs = 35,
          validation_data=(X_test,y_test), shuffle=1)



(ls,acc)=model.evaluate(x=X_test,y=y_test)

#Accuracy of the CNN Model:
print('Model Accuracy = {}%'.format(acc*100))
