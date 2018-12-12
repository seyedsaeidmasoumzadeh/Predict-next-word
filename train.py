import collections
import nltk
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from utils import chunks
import json
import os
import shutil


def word_indexing(words):
    """

    :param words: a string
    :return: a vocabulary dictionary {word1: 1, word2: 2,  ...} and
     its reveres {1: word1, 2: word2, ...}
    """
    vocab = collections.Counter(words).most_common()
    vocab_dict = dict()
    for word, _ in vocab:
        vocab_dict[word] = len(vocab_dict)
    rev_vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    return vocab_dict, rev_vocab_dict


def data_sampling(content, window):
    """

    :param content: Text vocab as string
    :param window: Window size for sampling, the window moves on the text vocab to build the samples
    :return: Training vocab includes (input, label) pair and number of classes

    If the window includes "cats like to chase mice" X is "cats like to chase" and y is "mice"
    """
    words = nltk.tokenize.word_tokenize(content)
    vocab_dict, rev_vocab_dict = word_indexing(words)
    with open('vocab/rev_vocab.json', 'w') as fp:
        json.dump(rev_vocab_dict, fp)
    with open('vocab/vocab.json', 'w') as fp:
        json.dump(vocab_dict, fp)
    training_data = []
    samples = chunks(words, window, truncate=True)
    for sample in samples:
        training_data.append(([vocab_dict[z] for z in sample[:-1]], vocab_dict[sample[-1:][0]]))
    return training_data, len(words)


with open("data.txt") as f:
    content = f.read()

window = 6
time_steps = window - 1
num_hidden = 512
num_input = 1
batch_size = 100
iteration = 250

training_data, num_classes = data_sampling(content, window=window)
# Build the Batches:
batches = chunks(training_data, batch_size)

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# tf graph input
X = tf.placeholder("float", [None, time_steps, num_input], name='X')
Y = tf.placeholder("float", [None, num_classes])


def RNN(x, weights, biases):

    # Unstack to get a list of 'timesteps' tensors, each tensor has shape (batch_size, n_input)
    x = tf.unstack(x, time_steps, 1)

    # Build a LSTM cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden)

    # Get LSTM cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
y_pred = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')
y_true = tf.argmax(Y, 1)

# Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss_op)
correct_pred = tf.equal(y_pred, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables with default values
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for i in range(0, iteration):
        loss_list = []
        acc_list = []
        for batch in batches:
            X_batch = [x[0] for x in batch]
            Y_batch = [x[1] for x in batch]
            Y_batch_encoded = []
            for x in Y_batch:
                one_hot_vector = np.zeros([num_classes], dtype=float)
                one_hot_vector[x] = 1.0
                Y_batch_encoded.append(one_hot_vector)
            Y_batch_encoded = np.vstack(Y_batch_encoded)
            X_batch = np.vstack(X_batch)
            X_batch = X_batch.reshape(len(batch), time_steps, num_input)
            Y_batch_encoded = Y_batch_encoded.reshape(len(batch), num_classes)
            _, acc, loss, onehot_pred = sess.run(
                [train_op, accuracy, loss_op, logits], feed_dict={X: X_batch, Y: Y_batch_encoded})
            loss_list.append(loss)
            acc_list.append(acc)
        loss = sum(loss_list)/len(loss_list)
        acc = sum(acc_list)/len(acc_list)
        print("Iteration " + str(i) + ", Loss= " + "{:.4f}".format(loss)
              + ", Training Accuracy= " + "{:.2f}".format(acc * 100))
    inputs = {
        "X": X,
    }
    outputs = {"y_pred": y_pred}
    if os.path.isdir("model"):
        shutil.rmtree('model')
    tf.saved_model.simple_save(
        sess, 'model/', inputs, outputs
    )