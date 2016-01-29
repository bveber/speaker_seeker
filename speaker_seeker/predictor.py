import tensorflow as tf
import speaker_seeker as ss
import os
import numpy as np
import time
from sklearn import preprocessing
import pickle

__author__ = 'Brandon'

batch_size = 128


def main(pickle_file):
#     training_dataset, training_labels, valid_dataset, valid_labels = build_training_set(path)
#     test_path = os.path.join(path, os.pardir, 'wav_files', 'Simpsons_1x02.wav')
#     test_dataset = ss.audio_processor.get_features_parallel(test_path)
    loaded_data = pickle.load(open(pickle_file, 'rb'))
    train_dataset = loaded_data['train_dataset']
    train_labels = loaded_data['train_labels']
    valid_dataset = loaded_data['valid_dataset']
    valid_labels = loaded_data['valid_labels']
    test_dataset = loaded_data['test_dataset']
    label_columns = loaded_data['label_columns']
    test_pred = fit(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset)
    test_pred_argmax = np.zeros((test_pred.shape[0], test_pred.shape[1]))
    for i in range(len(test_pred)):
        argmax_index = np.argmax(test_pred[i])
        test_pred_argmax[i, argmax_index] = 1
    return test_pred, test_pred_argmax, label_columns


def save_datasets(path):
    training_dataset, training_labels, valid_dataset, valid_labels, label_columns = build_training_set(path)
    test_path = os.path.join(path, os.pardir, 'wav_files', 'Simpsons_1x13.wav')
    test_dataset = ss.audio_processor.get_features_parallel(test_path)
    print(training_dataset.shape, valid_dataset.shape, test_dataset.shape)
    pickle_file = os.path.join(path, 'processed_dataset.pickle')
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': training_dataset,
        'train_labels': training_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'label_columns': label_columns
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size: ', statinfo.st_size)


def build_training_set(path):
    training_files = os.listdir(path)
    label_columns = []
    model_dataset_exists = False
    all_characters_dataset_exists = False
    for exclude_file in ['.DS_Store', 'processed_dataset.pickle']:
        if exclude_file in training_files:
            training_files.remove(exclude_file)
    for i, character_filename in enumerate(training_files):
        label_columns.append(character_filename.split('.')[0])
        if 'x' in character_filename:
            model = True
            model_dataset = ss.audio_processor.get_features_parallel(os.path.join(path, character_filename), model=model)
            model_labels = np.zeros((len(model_dataset), len(training_files) + 1))
            model_labels[:, i] = 1
        else:
            model = False
            character_dataset = ss.audio_processor.get_features_parallel(os.path.join(path, character_filename), model=model)
            character_labels = np.zeros((len(character_dataset), len(training_files) + 1))
            character_labels[:, i] = 1
        if not all_characters_dataset_exists:
            all_characters_dataset = character_dataset
            all_characters_labels = character_labels
            all_characters_dataset_exists = True
        else:
            all_characters_dataset = np.vstack((all_characters_dataset, character_dataset))
            all_characters_labels = np.vstack((all_characters_labels, character_labels))
    characters_mean = all_characters_dataset.mean(axis=0)
    characters_std = all_characters_dataset.std(axis=0)
    average_character = np.random.normal(characters_mean, characters_std,
                                         (int(np.floor(len(all_characters_dataset)/len(label_columns))),
                                          all_characters_dataset.shape[1]))
    average_labels = np.zeros((len(average_character), len(training_files) + 1))
    average_labels[:, -1] = 1
    label_columns.append('Average')
    training_dataset = np.vstack((all_characters_dataset, model_dataset, average_character))
    training_labels = np.vstack((all_characters_labels, model_labels, average_labels))
    training_dataset, training_labels = randomize(training_dataset, training_labels)
    train_index = np.floor(len(training_dataset) * .8)
    valid_dataset = training_dataset[train_index:]
    valid_labels = training_labels[train_index:]
    training_dataset = training_dataset[:train_index]
    training_labels = training_labels[:train_index]
    return training_dataset, training_labels, valid_dataset, valid_labels, label_columns


def randomize(dataset, labels):
    np.random.seed(42)
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def fit(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset):
    scaler = preprocessing.StandardScaler()
    train_dataset = scaler.fit_transform(train_dataset)
    valid_dataset = scaler.transform(valid_dataset)
    test_dataset = scaler.transform(test_dataset)
    num_labels = train_labels.shape[1]
    num_features = train_dataset.shape[1]
    hidden_units_1 = 1024
    hidden_units_2 = 300
    hidden_units_3 = 50

    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, num_features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
        tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)

        # Variables.
        weights = {
        'h1': tf.Variable(tf.truncated_normal([num_features, hidden_units_1], stddev=.1)),
        'h2': tf.Variable(tf.truncated_normal([hidden_units_1, hidden_units_2], stddev=.1)),
        'h3': tf.Variable(tf.truncated_normal([hidden_units_2, hidden_units_3], stddev=.1)),
        'out': tf.Variable(tf.truncated_normal([hidden_units_3, num_labels], stddev=.25))
        }
        biases = {
        'b1': tf.Variable(tf.zeros([hidden_units_1])),
        'b2': tf.Variable(tf.zeros([hidden_units_2])),
        'b3': tf.Variable(tf.zeros([hidden_units_3])),
        'out': tf.Variable(tf.zeros([num_labels]))
            }
        beta = tf.constant(.001)
        keep_prob = tf.constant(1.0)

        # Training computation.
        layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights['h1']) + biases['b1'])
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
        layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])
        layer_3 = tf.nn.dropout(layer_3, keep_prob)
        logits = tf.matmul(layer_3, weights['out']) + biases['out']
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
        beta * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
                tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out']))

        # Optimizer.
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(0.5, global_step, len(train_dataset), .95)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        val_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights['h1']) + biases['b1'])
        val_layer_2 = tf.nn.relu(tf.matmul(val_layer_1, weights['h2']) + biases['b2'])
        val_layer_3 = tf.nn.relu(tf.matmul(val_layer_2, weights['h3']) + biases['b3'])
        valid_prediction = tf.nn.softmax(tf.matmul(val_layer_3, weights['out']) + biases['out'])
        test_layer_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights['h1']) + biases['b1'])
        test_layer_2 = tf.nn.relu(tf.matmul(test_layer_1, weights['h2']) + biases['b2'])
        test_layer_3 = tf.nn.relu(tf.matmul(test_layer_2, weights['h3']) + biases['b3'])
        test_prediction = tf.nn.softmax(tf.matmul(test_layer_3, weights['out']) + biases['out'])

    num_steps = 10001
    t0 = time.time()

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step", step, ":", l)
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))

        test_predictions = test_prediction.eval()
    print(time.time() - t0)
    return test_predictions
