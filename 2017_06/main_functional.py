import keras
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.datasets import mnist
import numpy as np

# This returns a tensor
inputs = Input(shape=(28,28,1))

# a layer instance is callable on a tensor, and returns a tensor
x1 = Flatten()(inputs)
x1 = Dense(64, activation='relu')(x1)
x1 = Dense(64, activation='relu')(x1)
predictions1 = Dense(10, activation='softmax')(x1)

# a layer instance is callable on a tensor, and returns a tensor
x2 = Conv2D(64, (28,28), activation='relu')(inputs)
x2 = Conv2D(64, (1,1), activation='relu')(x2)
predictions2 = Conv2D(10, (1,1), activation='softmax')(x2)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions2)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, newshape=(-1, 28, 28, 1))
y_train = np.reshape(keras.utils.to_categorical(y_train, 10), newshape=(-1,1,1,10))

if 0:
    model.fit(x_train, y_train)  # starts training

from keras.layers import TimeDistributed
# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 28, 28, 1))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)

import csv
folder = '/scratch/lameeus/data/datasets/trainingandtestdata/'
list_tweets = []
with open(folder + 'training.1600000.processed.noemoticon.csv', 'r', encoding="ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

    sliceObj = slice(0, 5)
    for row in spamreader:
        string = row[5]
        
        row_short = [int(row[0]), string]

        list_tweets.append(row_short)

print(list_tweets[0:5])
        
# def ingest():
#     data = pd.read_csv('./tweets.csv')
#     data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
#     data = data[data.Sentiment.isnull() == False]
#     data['Sentiment'] = data['Sentiment'].map(int)
#     data = data[data['SentimentText'].isnull() == False]
#     data.reset_index(inplace=True)
#     data.drop('index', axis=1, inplace=True)
#     print 'dataset loaded with shape', data.shape
#     return data
