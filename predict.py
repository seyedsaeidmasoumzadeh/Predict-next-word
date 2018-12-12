import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
import json
import nltk

# 5 words as input
text = "For nearly an hour the"


words = nltk.tokenize.word_tokenize(text)
with open('vocab/vocab.json') as handle:
    vocab = json.loads(handle.read())
with open('vocab/rev_vocab.json') as handle:
    rev_vocab = json.loads(handle.read())
input = []
for word in words:
    input.append(vocab[word])
X_batch = np.array(input)
X_batch = X_batch.reshape(1, 5, 1)
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            'model/',
        )
        X = graph.get_tensor_by_name('X:0')
        y_pred = graph.get_tensor_by_name('y_pred:0')
        result = sess.run(y_pred, feed_dict={X: X_batch})
        print("the next word is '{}'".format(str(rev_vocab[str(result[0])])))
