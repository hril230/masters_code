from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import cv2



def cnn_model_fn(features, labels, mode):

  # Sort labels
  if not (mode == tf.estimator.ModeKeys.PREDICT):
    blocknum_labels = labels[:,0]
    narrow_labels = labels[:,1]
    lean_labels = labels[:,2]
    displaced_labels = labels[:,3]

  # layers
  input_layer = tf.reshape(features["x"], [-1, 56, 56, 3])
  blocknum_conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  blocknum_pool1 = tf.layers.max_pooling2d(inputs=blocknum_conv1, pool_size=[2, 2], strides=2)
  blocknum_conv2 = tf.layers.conv2d(inputs=blocknum_pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  blocknum_pool2 = tf.layers.max_pooling2d(inputs=blocknum_conv2, pool_size=[2, 2], strides=2)
  blocknum_conv3 = tf.layers.conv2d(inputs=blocknum_pool2, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  blocknum_pool3 = tf.layers.max_pooling2d(inputs=blocknum_conv3, pool_size=[2, 2], strides=2)
  blocknum_pool_flat = tf.reshape(blocknum_pool3, [-1, 7 * 7 * 128])
  blocknum_dense = tf.layers.dense(inputs=blocknum_pool_flat, units=1024, activation=tf.nn.relu)
  blocknum_dropout = tf.layers.dropout(inputs=blocknum_dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  blocknum_logits = tf.layers.dense(inputs=blocknum_dropout, units=4)

  # Generate predictions (for PREDICT and EVAL mode)
  blocknum_prediction = tf.argmax(input=blocknum_logits, axis=1)
  predictions = {
    "probabilities": tf.nn.softmax(blocknum_logits, name="softmax_tensor"),
    "blocknum_classes": blocknum_prediction
  }

  # Return predictions when in PREDICT mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=blocknum_labels, logits=blocknum_logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {"blocknum_accuracy": tf.metrics.accuracy(labels=labels[:,0], predictions=predictions["blocknum_classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def parse_dataset(train_dataset_size):
    image_files = []
    text_files = []
    images = []
    labels = []
    images_resized = []

    # load files from folder
    for root, dirs, files in os.walk("images"):  
        for filename in files:
            if 'before' in filename:
                image_files.append(filename)
            elif 'text' in filename:
                text_files.append(filename)

    # for each pair of files, append relevant data to image and label lists
    for imagename in image_files:
        images.append(cv2.imread('images/'+imagename))
        num = imagename[7:len(imagename)-4]
        for textname in text_files:
            if ('_'+num+'.') in textname:
                textfile = open('images/'+textname, 'r')
                label = []
                for line in textfile:
                    if 'Number of blocks' in line:
                        nblocks = int(line[18:].strip('\n'))
                        if nblocks == 2: label.append(0)
                        elif nblocks == 3: label.append(1)
                        elif nblocks == 4: label.append(2)
                        elif nblocks == 5: label.append(3)
                    elif 'Narrow base' in line:
                        if 'true' in line: label.append(1)
                        else: label.append(0)
                    elif 'On a lean' in line:
                        if 'true' in line: label.append(1)
                        else: label.append(0)
                    elif 'Block displaced' in line:
                        if 'true' in line: label.append(1)
                        else: label.append(0)
                labels.append(label)

    # resize images to be 28x28 instead of 200x200
    for image in images:
        resized_image = cv2.resize(image, (56, 56))
        images_resized.append(resized_image) 

    # separate images and labels into train and test sets
    train_images = images_resized[0:train_dataset_size]
    training_labels = labels[0:train_dataset_size]
    test_images = images_resized[2000:]
    test_labels = labels[2000:]

    # convert dataset into numpy arrays
    train_data = np.asarray(train_images, np.float32)
    train_labels = np.asarray(training_labels, np.int32)
    eval_data = np.asarray(test_images, np.float32)
    eval_labels = np.asarray(test_labels, np.int32)

    return train_data, train_labels, eval_data, eval_labels



# MAIN
def train(train_dataset_size):

    tf.logging.set_verbosity(tf.logging.INFO)

    # get training and evaluation data
    train_data, train_labels, eval_data, eval_labels = parse_dataset(train_dataset_size)

    # Create the Estimator
    cnn = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="models/blocknum_cnn")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
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
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cnn.evaluate(input_fn=eval_input_fn)
    print ('\nAccuracy:')
    print(eval_results)
