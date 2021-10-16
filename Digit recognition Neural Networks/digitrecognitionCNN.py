# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:10:26 2021

@author: SREENEHA
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import gzip
from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork:
    #Initializing weights, biases and setting up layers 
    def __init__(self, layers):
        self.layers = layers
        self.L = len(layers)
        self.num_features = layers[0]
        self.num_classes = layers[-1]
        
        self.W = {}
        self.b = {}
        
        self.dW = {}
        self.db = {}
        
        #fieldweight
        
        self.convshape=[5, 5, 1, 3]
        self.cw= tf.Variable(tf.random.normal(shape=self.convshape))
        
        for i in range(1, self.L):
            self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i],self.layers[i-1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i],1)))
        
     
    #feedforward function    
    def forward_pass(self, X):

        A = tf.convert_to_tensor(X, dtype=tf.float32)
        #print("A")
        #print(A.shape)
        A = tf.reshape(A, [-1,28,28,1])
        c1 = tf.nn.conv2d( A , self.cw , strides=[ 1 , 1 , 1 , 1 ], padding="SAME") 
        #( A , filters , strides=[ 1 , stride_size , stride_size , 1 ] , padding=padding )
        #print(c1.shape)
        p1 = tf.nn.max_pool2d( c1 , ksize=[ 1 , 2 , 2 , 1 ] , padding="SAME" , strides=[ 1 , 2 , 2 , 1 ] )
        #tf.nn.max_pool2d( inputs , ksize=[ 1 , 2 , 2 , 1 ] , padding='VALID' , strides=[ 1 , 2 , 2 , 1 ] )
        
        flatA = tf.reshape( p1 , shape=( tf.shape( p1 )[0], -1 ))
        #print("B")
        #print(flatA.shape)
        for i in range(1, self.L):
            Z = tf.matmul(flatA ,tf.transpose(self.W[i])) + tf.transpose(self.b[i])
            if i != self.L-1:
                flatA = tf.nn.sigmoid(Z)
            else:
                flatA = Z
        return flatA
    
    #computing loss / Cost fuction
    def compute_loss(self, A, Y):
        loss = tf.nn.softmax_cross_entropy_with_logits(Y,A)
        return tf.reduce_mean(loss)
    
    
    #updating weights and biases
    def update_params(self, lr):
        for i in range(1,self.L):
            self.W[i].assign_sub(lr * self.dW[i])
            self.W[i].assign_sub(lr*lmda*self.W[i]/n)
            self.b[i].assign_sub(lr * self.db[i])
            
    #Predicting the output using softmax and feedforward
    def predict(self, X):

        A = self.forward_pass(X)
        return tf.argmax(tf.nn.softmax(A), axis=1)
    
    
    #training batch wise    
    def train_on_batch(self, X, Y, lr):
         
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)
          
        with tf.GradientTape(persistent=True) as tape:
            A = self.forward_pass(X)
            loss = self.compute_loss(A, Y)
        for i in range(1, self.L):
            self.dW[i] = tape.gradient(loss, self.W[i])
            self.db[i] = tape.gradient(loss, self.b[i])
        del tape
        self.update_params(lr)
        return loss.numpy()
    
    #Starting to train the neural networks on the train and validation data
    def train(self, x_train, y_train, x_validation, y_validation, epochs, steps_per_epoch, batch_size, lr):

        history = {
            'val_loss':[],
            'train_loss':[],
            'val_acc':[]
        }
        
        for e in range(0, epochs):
            epoch_train_loss = 0.
            print('Epoch{}'.format(e), end='.')
            for i in range(0, steps_per_epoch):
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]
                
                batch_loss = self.train_on_batch(x_batch, y_batch,lr)
                epoch_train_loss += batch_loss
                
                if i%int(steps_per_epoch/10) == 0:
                    print(end='.')
                    
            history['train_loss'].append(epoch_train_loss/steps_per_epoch)
            print("Before Validation")
            print(x_validation.shape)
            val_A = self.forward_pass(x_validation)
            print("After Validation ")
            val_loss = self.compute_loss(val_A, y_validation).numpy()
            history['val_loss'].append(val_loss)
            val_preds = self.predict(x_validation)
            val_acc =    np.mean(np.argmax(y_validation, axis=1) == val_preds.numpy())
            history['val_acc'].append(val_acc)
            print('Validation accuracy:',val_acc)
        return history


#loading the data from pickle file
f = gzip.open('mnist.pkl.gz', 'rb')
data = cPickle.load(f, encoding='bytes')
f.close()
(x_train, y_train), (x_validation, y_validation), (x_test, y_test) = data

#one hot encoding the label to desires format
y_train = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
y_validation = OneHotEncoder().fit_transform(y_validation.reshape(-1, 1)).toarray()
y_test = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
    
n=len(y_train)
    
#Creating object of the class to design a neural network
net = NeuralNetwork([588,128,10])

#Defining hyperparameters
batch_size = 10
epochs = 5
lmda=0.1
steps_per_epoch = int(x_train.shape[0]/batch_size)
lr = 0.3

#training the neural net object we created
history = net.train(
    x_train,y_train,
    x_validation, y_validation,
    epochs, steps_per_epoch,
    batch_size, lr)


#plotting the output
plt.figure(figsize=(12, 4))
epochs = len(history['val_loss'])
plt.subplot(1, 2, 1)
plt.plot(range(epochs), history['val_loss'], label='Val Loss')
plt.plot(range(epochs), history['train_loss'], label='Train Loss')
plt.xticks(list(range(epochs)))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(epochs), history['val_acc'], label='Val Acc')
plt.xticks(list(range(epochs)))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#redicting on the test data
preds = net.predict(x_test)
final_acc = np.mean(np.argmax(y_test, axis=1) == preds.numpy())
print('Test accuracy:',final_acc)


