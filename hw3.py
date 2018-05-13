import os
import numpy as np
import pandas as pd
import tensorflow as tf
import model
from skimage import io,transform
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

INPUT_SIZE = [224, 224, 3]
N_CLASSES = 120
LEARNING_RATE = 2e-5
EPOCHS = 1
BATCH_SIZE = 10
LOAD_PRETRAIN = False


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def train_eval(sess, x_data, y_label, batch_size, train_phase, is_eval,  epoch=None):
    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    tmp_loss, tmp_acc = 0, 0
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size) 
        _, batch_loss, batch_acc = sess.run([train_op, loss, accuracy], 
                                            feed_dict={x: x_data[start:end], y: y_label[start:end], 
                                            is_training: train_phase})
        tmp_loss += batch_loss * (end - start)
        tmp_acc += batch_acc * (end - start)
    tmp_loss /= n_sample
    tmp_acc /= n_sample
    if train_phase:
        print('\nepoch: {0}, loss: {1:.4f}, acc: {2:.4f}'.format(epoch+1, tmp_loss, tmp_acc))
        
def test_eval(sess, x_data, train_phase):
    batch_size = 1
    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    tmp_pred=[]
    log=[]
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        tmp_logits = sess.run(logits, feed_dict={x: x_data[start:end], is_training: train_phase})
        tmp=softmax(np.squeeze(tmp_logits))
        tmp_pred.append(tmp)
    tmp_pred = np.array(tmp_pred)

    return tmp_pred


# data preprocess by yourself 


if __name__ == '__main__':

    x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, N_CLASSES), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='train_phase')

    logits = model.VGG16(x=x, is_training=is_training, n_classes=N_CLASSES)

    with tf.name_scope('LossLayer'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1)), tf.float32))

    init = tf.global_variables_initializer()
    
    restore_variable = [var for var in tf.global_variables() if var.name.startswith('')]
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        if LOAD_PRETRAIN:
            saver.restore(sess, 'model/model.ckpt')
        else:
            sess.run(init)

        for i in range(EPOCHS):
            train_eval(sess=sess, x_data=train_data, y_label=train_label, batch_size=BATCH_SIZE, 
                    train_phase=True, is_eval=False,epoch=i)
        #saver.save(sess, 'model/model.ckpt')
        ans = test_eval(sess=sess, x_data=test_data, train_phase=False)






