import os
import numpy as np
import pandas as pd
import tensorflow as tf
import model
import pandas as pd
import cv2
import multiprocessing as mp
from skimage import io,transform
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

INPUT_SIZE = [224, 224, 3]
N_CLASSES = 120
LEARNING_RATE = 2e-5
EPOCHS = 1
BATCH_SIZE = 10
LOAD_PRETRAIN = False
DATA_PATH = 'dog-breed-identification'
BREEDS = ['affenpinscher','afghan_hound','african_hunting_dog','airedale','american_staffordshire_terrier','appenzeller','australian_terrier','basenji','basset','beagle','bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound','bluetick','border_collie','border_terrier','borzoi','boston_bull','bouvier_des_flandres','boxer','brabancon_griffon','briard','brittany_spaniel','bull_mastiff','cairn','cardigan','chesapeake_bay_retriever','chihuahua','chow','clumber','cocker_spaniel','collie','curly-coated_retriever','dandie_dinmont','dhole','dingo','doberman','english_foxhound','english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog','german_shepherd','german_short-haired_pointer','giant_schnauzer','golden_retriever','gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter','irish_terrier','irish_water_spaniel','irish_wolfhound','italian_greyhound','japanese_spaniel','keeshond','kelpie','kerry_blue_terrier','komondor','kuvasz','labrador_retriever','lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog','mexican_hairless','miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland','norfolk_terrier','norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound','papillon','pekinese','pembroke','pomeranian','pug','redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed','schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier','shetland_sheepdog','shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier','standard_poodle','standard_schnauzer','sussex_spaniel','tibetan_mastiff','tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner','welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier','yorkshire_terrier']


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


def load_train_img(path):
    return load_img(f'{DATA_PATH}/train/{path}', target_size=(INPUT_SIZE[0], INPUT_SIZE[1]))

def load_test_img(path):
    return load_img(f'{DATA_PATH}/test/{path}', target_size=(INPUT_SIZE[0], INPUT_SIZE[1]))


if __name__ == '__main__':
    pool = mp.Pool()

    labels = pd.read_csv(f'{DATA_PATH}/labels.csv').set_index('id')['breed'].to_dict()
    train_files = os.listdir(f'{DATA_PATH}/train')
    test_files = os.listdir(f'{DATA_PATH}/test')

    train_raw_label = list(map(lambda x: BREEDS.index(labels[x.rstrip('.jpg')]), train_files))
    train_label = tf.Session().run(tf.one_hot(train_raw_label, N_CLASSES))

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
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        if LOAD_PRETRAIN:
            saver.restore(sess, 'model/model.ckpt')
        else:
            sess.run(init)

        print('輸入訓練影像中...')
        train_data = np.array([img_to_array(img) for img in pool.imap_unordered(load_train_img, tqdm(train_files))])
        print(train_data.shape)
        

        print('Training...')
        for i in tqdm(range(EPOCHS)):
            train_eval(sess=sess, x_data=train_data, y_label=train_label, batch_size=BATCH_SIZE, 
                    train_phase=True, is_eval=False,epoch=i)
        #saver.save(sess, 'model/model.ckpt')
        del train_data # 避免記憶體占用


        print('輸入測試影像中...')
        test_data = np.array([img_to_array(img) for img in pool.imap_unordered(load_test_img, tqdm(test_files))])
        ans = test_eval(sess=sess, x_data=test_data, train_phase=False)
        del test_data # 避免記憶體占用






