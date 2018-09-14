import tensorflow as tf
from BotTrainer import get_data, create_network, train_network


weights_file = 'weights/model.ckpt'


epoch = 1

def train_session():
    batches = get_data()
    sess = tf.InteractiveSession()
    q_s, s = create_network()
    for i in range(epoch):
        train_network(q_s, s, sess, batches, i)

    # Save the weights
    saver = tf.train.Saver()
    saver.save(sess, weights_file)
    print('Weights saved')

train_session()