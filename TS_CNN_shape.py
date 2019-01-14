from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import cv2
import random



def cnn_model_fn(features, labels, mode):

  # setup layers
  input_layer = tf.reshape(features["x"], [-1, 56, 56, 3])
  conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  flat = tf.reshape(pool3, [-1, 7 * 7 * 128])
  dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  logits = tf.layers.dense(inputs=dropout, units=8)

  # Generate predictions (for PREDICT and EVAL mode)
  prediction = tf.argmax(input=logits, axis=1)
  predictions = {
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    "classes": prediction
  }

  # Return predictions when in PREDICT mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def parse_dataset(train_dataset_size):

    # edit these values to change the sizes of the train and test datasets
    max_train_dataset_size = 4000
    max_test_dataset_size = 2000
    test_dataset_size = 500

    # parse training dataset
    max_train_images = []
    max_train_labels = []
    train_filenames = []
    while (len(max_train_images) < max_train_dataset_size):
        for root, dirs, files in os.walk('Training'):
            for directory in dirs:
                file_appended = False
                label = []
                textfile = open('Training/'+directory+'/class_annotation.txt', 'r')
                for line in textfile:
                    if 'Class' in line: label.append(int(line[7:]))
                    elif 'Shape' in line: label.append(int(line[7:]))
                    elif 'Main colour' in line: label.append(int(line[13:]))
                    elif 'Border colour' in line: label.append(int(line[15:]))
                    elif 'Background image' in line: label.append(int(line[18:]))
                    elif 'Symbol' in line: label.append(int(line[8:]))
                    elif 'Secondary symbol' in line: label.append(int(line[18:]))
                    elif 'Cross' in line: label.append(int(line[7:]))
                for subroot, subdirs, subfiles in os.walk('Training/'+directory):
                    for filename in subfiles:
                        if ('ppm' in filename) and not (filename in train_filenames):	
                            max_train_images.append(cv2.imread('Training/'+directory+'/'+filename))
                            max_train_labels.append(label)
                            train_filenames.append(filename)
                            file_appended = True
                        if file_appended: break
                textfile.close()


    # parse testing dataset
    max_test_images = []
    max_test_labels = []
    test_filenames = []
    while (len(max_test_images) < max_test_dataset_size):
        for root, dirs, files in os.walk('Testing'):
            for directory in dirs:
                file_appended = False
                label = []
                textfile = open('Testing/'+directory+'/class_annotation.txt', 'r')
                for line in textfile:
                    if 'Class' in line: label.append(int(line[7:]))
                    elif 'Shape' in line: label.append(int(line[7:]))
                    elif 'Main colour' in line: label.append(int(line[13:]))
                    elif 'Border colour' in line: label.append(int(line[15:]))
                    elif 'Background image' in line: label.append(int(line[18:]))
                    elif 'Symbol' in line: label.append(int(line[8:]))
                    elif 'Secondary symbol' in line: label.append(int(line[18:]))
                    elif 'Cross' in line: label.append(int(line[7:]))
                for subroot, subdirs, subfiles in os.walk('Testing/'+directory):
                    for filename in subfiles:
                        if ('ppm' in filename) and not (filename in test_filenames):	
                            max_test_images.append(cv2.imread('Testing/'+directory+'/'+filename))
                            max_test_labels.append(label)
                            test_filenames.append(filename)
                            file_appended = True
                        if file_appended: break
                textfile.close()

    # random sampling of training dataset
    train_images = []
    train_labels = []
    while len(train_images) < train_dataset_size:
        i = random.randint(0,len(max_train_images)-1)
        train_images.append(max_train_images[i])
        max_train_images.pop(i)
        train_labels.append(max_train_labels[i])
        max_train_labels.pop(i)

    # random sampling of testing dataset
    test_images = []
    test_labels = []
    while len(test_images) < test_dataset_size:
        i = random.randint(0,len(max_test_images)-1)
        test_images.append(max_test_images[i])
        max_test_images.pop(i)
        test_labels.append(max_test_labels[i])
        max_test_labels.pop(i)

    # resize images to be 56x56
    train_images_resized = []
    for image in train_images:
        resized_image = cv2.resize(image, (56, 56)) 
        train_images_resized.append(resized_image)
    test_images_resized = []
    for image in test_images:
        resized_image = cv2.resize(image, (56, 56)) 
        test_images_resized.append(resized_image)

    # convert dataset into numpy arrays
    train_images = np.asarray(train_images_resized, np.float32)
    train_labels = np.asarray(train_labels, np.int32)
    test_images = np.asarray(test_images_resized, np.float32)
    test_labels = np.asarray(test_labels, np.int32)

    return train_images, train_labels, test_images, test_labels





def train(train_dataset_size):

    tf.logging.set_verbosity(tf.logging.INFO)

    # get training and evaluation data
    train_data, train_labels, eval_data, eval_labels = parse_dataset(train_dataset_size)

    # Create the Estimator
    cnn = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="models/shape_cnn")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels[:,1],
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    cnn.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels[:,1],
        num_epochs=1,
        shuffle=False)
    eval_results = cnn.evaluate(input_fn=eval_input_fn)
    print ('\nAccuracy:')
    print(eval_results)

