import tensorflow as tf
from BotTrainer import get_data, create_network, train_network
import os
import sys
from pickle import Pickler

weights_file = 'model.ckpt'
weights_dir = 'tools/weights'
pickle_dir = 'pickle_files'
pickle_file_loss = 'loss.p'


def train_session(session_mode):

    # Fetch data
    print('Fetching data...')
    batches = get_data(session_mode)

    # Define Session
    sess = tf.InteractiveSession()

    # Create Network
    print('Creating network...')
    q_s, s, keep_prob = create_network()

    # Train Network
    print('Training...')
    if len(batches) > 1:
        # Train
        last_loss = train_network(q_s, s, sess, batches, keep_prob, session_mode)

        # Save the weights
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            print('directory created')
        saver = tf.train.Saver()
        path = os.path.join(weights_dir, weights_file)
        saver.save(sess, path)
        print('weights saved')

    # save loss value
    loss_path = os.path.join(pickle_dir, pickle_file_loss)
    with open(loss_path, 'ab+') as f:
        Pickler(f).dump(last_loss)


def main():
    # get command line arguments
    if len(sys.argv) > 1:
        session_mode = sys.argv[1]
    else:
        # default is 'training'
        session_mode = 'training'

    # decide session mode
    if session_mode == 'observing':
        print('Session mode: ' + session_mode)
    elif session_mode == 'debug':
        print('Session mode: ' + session_mode)
    else:
        print('Session mode: training')

    # start training
    train_session(session_mode='training')


if __name__ == '__main__':
    main()
