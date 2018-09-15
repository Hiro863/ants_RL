import tensorflow as tf
from BotTrainer import get_data, create_network, train_network


weights_file = 'tools/weights/model.ckpt'
weights_dir = 'tools/weights'


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
    train_network(q_s, s, sess, batches)

    # Save the weights
    saver = tf.train.Saver()
    saver.save(sess, weights_file)
    print('Weights saved')


train_session()
