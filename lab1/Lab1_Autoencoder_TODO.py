# importing libraries

import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise, Lambda
from keras.models import Model  
#from keras.layers import ActivityRegularization
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD  

from keras import backend as K
import numpy as np 
import random as rd
import matplotlib.pyplot as plt 



# Initialization - Autoencoder (n,k)
# M: number of messages to encode
# k = log2(M)
# n real-value signals (otherwise. for complex-valued signals, 2*n real numbers)
k = 2

n = 2
M = 2**k
R = k/n

#print('M:',M,'\t','k:',k,'\t','n:',n)

# Channel parameters
Eb_No_dB = 7
noise = 1/(10**(Eb_No_dB/10))
noise_sigma = np.sqrt(noise)
beta = 1/(2*R*(10**(Eb_No_dB/10)))
beta_sqrt = np.sqrt(beta)


# Creating a training set and test set 
eye_matrix = np.eye(M)
x_train = np.tile(eye_matrix, (1000, 1))  
x_test = np.tile(eye_matrix, (100, 1)) 
rd.shuffle(x_train)
rd.shuffle(x_test)


# Printing the shape of x_train and x_test




#Bit Error Rate
def BER(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)  


# Autoencoder definition/construction
    
# Take the input and convert it to Keras tensors
input_signal = Input(shape=(M,))
encoded_input = Input(shape=(n,))

  
#Transmitter Neural Network (encoder)


######## YOUR CODE STARTS HERE #######
#### Create two dense layers, with input 'input_signal' and output 'encoded' ####
### The first layer has activation ReLU and the second linear

#layer_relu = tf.keras.layers.ReLU()
#encoded = tf.keras.activations.linear(layer_relu(input_signal))

x=tf.keras.layers.Dense(M, activation='linear')(input_signal)
encoded = tf.keras.layers.Dense(n, activation='linear')(x)


######## END OF YOUR CODE      #######



# If you want to use regularizers or perform batch normalization, uncomment the below 
#encoded = ActivityRegularization(l2=0.02)(encoded)
#encoded = BatchNormalization()(encoded)

# Signal constraints
#normalize by the l2 norm and multiply by sqrt(n) why?
#that's actually the average power norm

encoded = Lambda(lambda x: K.l2_normalize(x, axis=1))(encoded) #energy constraint


######## YOUR CODE STARTS HERE #######
#### The above command implement a fixed energy constraint. For the question on implementing an average power constraint ####
#### Comment the above line and write your own command for the power constraint ####

#encoded = Lambda(lambda x: x/np.sqrt(np.mean(np.square(x), axis=1)))(encoded) #energy constraint
def mean_norm(x):

    sqrt_mean=K.sqrt(K.mean(K.square(x), axis=1))
    print(sqrt_mean)
    print(x.shape)

    x[:][0], x[:][1] = x[:][0]/sqrt_mean[k], x[:][1]/sqrt_mean[k]
        
    x=tf.conver_to_tensor(x, dtype=tf.float32)
    return x

#encoded = Lambda(mean_norm)(encoded) #energy constraint


######## END OF YOUR CODE      #######


# Channel Layer
encoded_noise = GaussianNoise(beta_sqrt)(encoded)

# Receiver Neural Network (decoder)

######## YOUR CODE STARTS HERE #######
#### Create two dense layers, with input 'encoded_noise' and output 'decoded' ####
### The first layer has activation ReLU and the second softmax
######## END OF YOUR CODE      #######

#layer_relu_decode = tf.keras.layers.ReLU()
#decoded = tf.keras.activations.softmax(layer_relu_decode(encoded_noise))

y=tf.keras.layers.Dense(M, activation='relu')(encoded_noise)
decoded = tf.keras.layers.Dense(M, activation='softmax')(y)

# We create the autoencoder with input_signal as the input and output being the final decoder layer
autoencoder = Model(inputs=input_signal, outputs=decoded)  


# We extract the encoder that takes the input_signal as the input and the output of the encoder is the encoded signal
encoder = Model(inputs=input_signal, outputs=encoded)


# To show the structure of the deep autoencodder (layers, trainable parameters,...)
autoencoder.summary()



# We compile the autoencoder model with Adam optimizer and categorical cross entropy as loss function

# To change the learning rate of Adam optimizer and/or SGD, you can use the following commmands
#adam=Adam(lr=0.001)
#sgd=SGD(lr=0.02)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy',BER])  

# Training of the autoencoder with 200 epochs and batch size of 32 (default value) 
hist = autoencoder.fit(x_train, x_train, epochs=20, batch_size=32,validation_data=(x_test, x_test))



# Predicting the test set using the encoder to view the encoded signal

encoded_signal = encoder.predict(x_test) 
print(encoded_signal)

#print(np.sqrt(n)*K.l2_normalize(encoded_signal, axis=1))

# Predicting the test set using the autoencoder to view (obtain) the encoded (reconstructed) signal
decoded_signal = autoencoder.predict(x_test)
print(decoded_signal)
print(decoded_signal.shape)



# Plotting the constellation diagram
encoded_planisphere = encoder.predict(eye_matrix) 


plt.title('Constellation')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot(encoded_planisphere[:,0], encoded_planisphere[:,1], 'r.')
plt.grid(True)
plt.xlabel('I axis')
plt.ylabel('Q axis')

# Plotting the model loss vs. epochs
plt.figure()
plt.plot(hist.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')


# making decoder from full autoencoder
deco = autoencoder.layers[-2](encoded_input)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)


# Calculating BER from -2dB to 10dB Eb/No

# Generating data of size N
N=100000
test_label = np.random.randint(M,size=N)

# Creating one hot encoded vectors
test_data=[]
for i in test_label:
    temp=np.zeros(M)
    temp[i]=1
    test_data.append(temp)
# Checking data shape
test_data=np.array(test_data)

EbNodB_range = list(np.arange(-2.0,10+1,1))
ber = [None]*len(EbNodB_range)
for n_i in range(0,len(EbNodB_range)):
    EbNo=10.0**(EbNodB_range[n_i]/10.0)
    noise_std = np.sqrt(1/(2*R*EbNo))
    noise_mean = 0
    no_errors = 0
    nn = N
    noise = noise_std*np.random.randn(nn,n)
    encoded_signal = encoder.predict(test_data) 
    final_signal = encoded_signal+noise
    pred_final_signal =  decoder.predict(final_signal)
    pred_output = np.argmax(pred_final_signal,axis=1)
    no_errors = (pred_output != test_label)
    no_errors =  no_errors.astype(int).sum()
    ber[n_i] = no_errors / nn 
    print ('Eb/No:',EbNodB_range[n_i],'BLER:',ber[n_i])


# ploting BER curve
import matplotlib.pyplot as plt
plt.figure()
plt.plot(EbNodB_range, ber, 'bo',label='Autoencoder(2,2)')
plt.yscale('log')
plt.xlabel('Eb/No [dB]')
plt.ylabel('Block error rate')
plt.grid(True)
plt.legend(loc='upper right',ncol = 1)
plt.ylim(10**-5,1)
plt.xlim(-2,10)
plt.show()



