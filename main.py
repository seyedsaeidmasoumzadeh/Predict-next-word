import collections
import nltk
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


# Word embedding
def word_embedding(words):
    vocabulary = collections.Counter(words).most_common()
    vocabulary_dictionary = dict()
    for word, _ in vocabulary:
        # Assign a numerical unique value to each word inside vocabulary 
        vocabulary_dictionary[word] = len(vocabulary_dictionary)
    rev_vocabulary_dictionary = dict(zip(vocabulary_dictionary.values(), vocabulary_dictionary.keys()))
    return vocabulary_dictionary, rev_vocabulary_dictionary


# Build Training data. For example if X = ['long', 'ago', ','] then Y = ['the']
def sampling(words, vocabulary_dictionary, window):
    X = []
    Y = []
    sample = []
    for index in range(0, len(words) - window):
        for i in range(0, window):
            sample.append(vocabulary_dictionary[words[index + i]])
            if (i + 1) % window == 0:
                X.append(sample)
                Y.append(vocabulary_dictionary[words[index + i + 1]])
                sample = []
    return X,Y


with open("data.txt") as f:
    content = f.read()
words = nltk.tokenize.word_tokenize(content)
vocabulary_dictionary, reverse_vocabulary_dictionary = word_embedding(words)

window = 3
num_classes = len(vocabulary_dictionary)
timesteps = window
num_hidden = 512
num_input = 1
batch_size = 20
iteration = 200


training_data, label = sampling(words, vocabulary_dictionary, window)


# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# tf graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


def RNN(x, weights, biases):

    # Unstack to get a list of 'timesteps' tensors, each tensor has shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Build a LSTM cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get LSTM cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables with default values
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for i in range(iteration):
        last_batch = len(training_data) % batch_size
        training_steps = (len(training_data) / batch_size) + 1
        for step in range(training_steps):
            X_batch = training_data[(step * batch_size) :((step + 1) * batch_size)]
            Y_batch = label[(step * batch_size) :((step + 1) * batch_size)]
            Y_batch_encoded = []
            for x in Y_batch:
                on_hot_vector = np.zeros([num_classes], dtype=float)
                on_hot_vector[x] = 1.0
                Y_batch_encoded = np.concatenate((Y_batch_encoded,on_hot_vector))
            if len(X_batch) < batch_size:
                X_batch = np.array(X_batch)
                X_batch = X_batch.reshape(last_batch, timesteps, num_input)
                Y_batch_encoded = np.array(Y_batch_encoded)
                Y_batch_encoded = Y_batch_encoded.reshape(last_batch, num_classes)
            else:
                X_batch = np.array(X_batch)
                X_batch = X_batch.reshape(batch_size, timesteps, num_input)
                Y_batch_encoded = np.array(Y_batch_encoded)
                Y_batch_encoded = Y_batch_encoded.reshape(batch_size, num_classes)
            _, acc, loss, onehot_pred = sess.run([train_op, accuracy, loss_op, logits], feed_dict={X: X_batch, Y: Y_batch_encoded})
            print("Step " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.2f}".format(acc * 100))
