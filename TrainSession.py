import tensorflow as tf
from BotTrainer import get_data, create_network, train_network
import os

weights_file = 'model.ckpt'
weights_dir = 'tools/weights'


def train_session():
    print(__file__)
    # Fetch data
    print('Fetching data...')
    batches = get_data()

    # Define Session
    sess = tf.InteractiveSession()

    # Create Network
    print('Creating network...')
    q_s, s, variables = create_network()


    w_conv1, w_conv2, w_conv3, b_conv1, b_conv2, b_conv3, w_full, b_full = variables

    # Train Network
    print('Training...')
    for i, ant_batches in enumerate(batches):
        print('Training %d th ant data: ' % i)
        if len(ant_batches) > 1:



            # Train
            train_network(q_s, s, sess, ant_batches, variables)

            # Save the weights
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
                print('directory created')
            saver = tf.train.Saver({'w_conv1': w_conv1,
                                    'w_conv2': w_conv2,
                                    'w_conv3': w_conv3,
                                    'b-conv1': b_conv1,
                                    'b_conv2': b_conv2,
                                    'b_conv3': b_conv3,
                                    'w_full': w_full,
                                    'b_full': b_full})
            path = os.path.join(weights_dir, weights_file)
            saver.save(sess, path)
            print('Weights saved')


train_session()