import tensorflow as tf
import os.path
from BotTrainer import get_data, create_network, train_network


weights_file = 'weights/model.ckpt'
weights_dir = 'weights'

epoch = 1

def train_session():
    # Fetch data
    print('Fetching data...')
    batches = get_data()

    # Define Session
    sess = tf.InteractiveSession()

    # Create Network
    print('Creating network...')
    q_s, s = create_network()

    # Train Network
    print('Training...')
    for i in range(epoch):
        train_network(q_s, s, sess, batches, i)

    # Load weight
    if os.path.exists(weights_dir):
        saver = tf.train.Saver()
        saver.restore(sess, "weights/model.ckpt")
        print('Weights loaded')

    # Save the weights
    saver = tf.train.Saver()
    saver.save(sess, weights_file)
    print('Weights saved')

train_session()