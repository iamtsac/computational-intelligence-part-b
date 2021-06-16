import sys
import pandas as pd
import numpy as np
from numpy import std
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import KFold 
from matplotlib import pyplot 

#------------------------------------------------# 

# Loading data and splitting into inputs and labels, train and test #
def load_data(train_csv,test_csv):
    train_csv = pd.read_csv(train_csv)
    test_csv = pd.read_csv(test_csv)
    x_train = train_csv.loc[:, train_csv.columns != 'label'].to_numpy().reshape(train_csv.shape[0],28,28)
    y_train = train_csv['label'].to_numpy() 
    x_test = test_csv.loc[:, train_csv.columns != 'label'].to_numpy().reshape(test_csv.shape[0],28,28) 
    y_test = test_csv['label'].to_numpy() 


    return x_train, y_train, x_test, y_test 


def data_preprocessing(x_train, y_train, x_test, y_test, preprocessing_type='normalization'):
    
    # One hot on labels, because if the model predict a wrong class
    # we can not say that the input was wrong by an offset, because of different classes
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)

    ######## Normalization ############

    if preprocessing_type == 'normalization':
        x_normalized = tf.keras.utils.normalize(x_train)
        x_test_normalized = tf.keras.utils.normalize(x_test)

        return x_normalized, y_train, x_test_normalized, y_test

    ########## Centering ###################

    elif preprocessing_type == 'centering':

        x_centering = np.zeros((np.shape(x_train)[0], 28, 28))  
        x_test_centering =  np.zeros((np.shape(x_test)[0], 28, 28)) 

        for i,j in zip(range(0,np.shape(x_train)[0]),range(0,np.shape(x_test)[0])):

            scaler = preprocessing.StandardScaler(with_std=False).fit(x_train[i])
            scaler_test = preprocessing.StandardScaler(with_std=False).fit(x_train[j])
            x_centering[i] = scaler.transform(x_train[i])
            x_test_centering[j] = scaler.transform(x_test[j])

        return x_centering, y_train, x_test_centering, y_test 

    ########## Standardize ################

    elif preprocessing_type =='standardize':

        x_standardize = np.zeros((np.shape(x_train)[0], 28, 28))  
        x_test_standardize =  np.zeros((np.shape(x_test)[0], 28, 28)) 

        for i,j in zip(range(np.shape(x_train)[0]),range(np.shape(x_test)[0])): 
        
            scaler = preprocessing.StandardScaler(with_mean=False).fit(x_train[i])
            scaler_test = preprocessing.StandardScaler(with_std=False).fit(x_train[j])
            x_standardize[i] = scaler.transform(x_train[i])
            x_test_standardize[j] = scaler.transform(x_test[j])


        return x_standardize, y_train, x_test_standardize, y_test


def train_model(x_train,y_train,x_test,y_test,epochs=5,nodes=10,verbose=0,loss='categorical_crossentropy',metric=['accuracy'],learning_rate=0.001,momentum=0,plot='off',r=0.0):

    fold_number = 0
    scores ,histories = list(), list()
    sum_of_loss = sum_of_acc =  0
    kfold = KFold(n_splits=5, shuffle=False)


    # K-Fold iterations
    find_best = list()
    for train_index, test_index in kfold.split(x_train,y_train):  

        # Model configuration
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten()) # Change input from shape (,28,28) to (,784)
        model.add(tf.keras.layers.Dense(397, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum),
            loss=loss,
            metrics=metric
        )
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',min_delta=0,verbose=verbose,patience=5)

        history = model.fit(
            x_train[train_index],
            y_train[train_index],
            epochs=epochs,
            verbose=verbose,
            validation_data=(x_test, y_test),
            callbacks=[early_stop]
            ) 

        val_loss, val_acc = model.evaluate(x_train[test_index], y_train[test_index],verbose=verbose)
        find_best.append(val_loss) 

        if find_best[len(find_best) - 2] > val_loss and len(find_best)>1 : 
            model.save("model.h5")

        else:
            model.save("model.h5")


        print('|----------------------------------------------------------------------------|')
        fold_number += 1
        print("|  For fold ",(fold_number),"\n|  Loss: ", val_loss, " Accuracy: ",val_acc)
        sum_of_acc += val_acc
        sum_of_loss += val_loss 
        scores.append(val_acc)
        histories.append(history)

    print('|----------------------------------------------------------------------------|')
    print("\n \n The average of the Loss and Accuracy is: \n", "Loss: ",sum_of_loss/fold_number,"\n","Accuracy: ",sum_of_acc/fold_number," \n ") 





    return histories,plot


def prediction(model,x_test,y_test):
    predicts = model.predict(x_test) 
    classes = np.argmax(predicts, axis=1) 



def init_data(train_csv, test_csv,preprocessing_type='normalization'):

    x_train, y_train, x_test, y_test = load_data(train_csv,test_csv) 

    return  data_preprocessing(x_train,y_train, x_test,y_test,preprocessing_type)


x_train, y_train, x_test, y_test = init_data('data/mnist_train.csv','data/mnist_test.csv') # Path to CSVs

opt_learning_rate=0.05
default_learning_rate=0.001
default_momentum=0
opt_momentum=0.6
opt_nodes = 128
plot="on" #on or off
verbose=0
epochs=30
opt_r=0
loss_metrics = [ 'categorical_crossentropy','mse']
train_model(x_train,y_train,x_test,y_test,epochs=epochs,verbose=verbose,learning_rate=opt_learning_rate,plot="off",loss=loss_metrics[0],momentum=opt_momentum,r=opt_r)
