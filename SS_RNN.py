import numpy as np
import tensorflow as tf
import os
import cv2
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder
import re



def parse_dataset(predictions, offset):

    text_files = []
    image_files = []
    dataset = []
    labels = []

    # Get training dataset
    for prediction in predictions:
        if prediction[5] == 0: # use only predictions that come from the decision tree

            # create dataset for question zero
            data = [[1], [2], [3], [4], [0], [0], [0], [0], [0], [0], [0]] # this word encoding means that the question is 'Is this structure stable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))

            # create dataset for question one
            data = [[5], [1], [6], [2], [3], [7], [0], [0], [0], [0], [0]] # this word encoding means that the question is 'What is making this structure unstable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))

            # create dataset for question two
            data = [[5], [1], [6], [2], [3], [4], [0], [0], [0], [0], [0]] # this word encoding means that the question is 'What is making this structure stable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))

            # create dataset for question three
            data = [[5], [8], [9], [10], [11], [12], [10], [13], [2], [3], [4]] # this word encoding means 'What would need to be changed to make this structure stable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))

    # To avoid errors during testing, if all training samples were classified by logic, then the decision tree must be trained with those samples
    if len(dataset) == 0: using_all_training_data = True
    else: using_all_training_data = False

    if using_all_training_data:
        for prediction in predictions:
            # create dataset for question zero
            data = [[1], [2], [3], [4], [0], [0], [0], [0], [0], [0], [0]] # this word encoding means that the question is 'Is this structure stable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))
            # create dataset for question one
            data = [[5], [1], [6], [2], [3], [7], [0], [0], [0], [0], [0]] # this word encoding means that the question is 'What is making this structure unstable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))
            # create dataset for question two
            data = [[5], [1], [6], [2], [3], [4], [0], [0], [0], [0], [0]] # this word encoding means that the question is 'What is making this structure stable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))
            # create dataset for question three
            data = [[5], [8], [9], [10], [11], [12], [10], [13], [2], [3], [4]] # this word encoding means 'What would need to be changed to make this structure stable?'
            for feature in prediction[0:len(prediction)-1]:
                data.append([feature])
            dataset.append(np.array(data))


    # Get training labels
    for root, dirs, files in os.walk("images"):  
        for filename in files:
            if 'before' in filename:
                image_files.append(filename)
            elif 'text' in filename:
                text_files.append(filename)

    for n in range(len(predictions)):
        i = n + offset
        imagename = image_files[i]
        num = imagename[7:len(imagename)-4]
        for textname in text_files:
            if ('_'+num+'.') in textname:
                if (predictions[n][5] == 0) or (using_all_training_data): # use only labels for predictions that come from the decision tree
                    textfile = open('images/'+textname, 'r')

                    # parse data from textfile
                    data = []
                    for line in textfile:
                        if 'Number of blocks' in line:
                            nblocks = int(line[18:].strip('\n'))
                            if nblocks == 2: data.append([0])
                            elif nblocks == 3: data.append([1])
                            elif nblocks == 4: data.append([2])
                            elif nblocks == 5: data.append([3])
                        elif 'Narrow base' in line:
                            if 'true' in line:
                                data.append([1])
                            else:
                                data.append([0])
                        elif 'On a lean' in line:
                            if 'true' in line:
                                data.append([1])
                            else:
                                data.append([0])
                        elif 'Block displaced' in line:
                            if 'true' in line:
                                data.append([1])
                            else:
                                data.append([0])
                        elif 'Stable' in line:
                            data.append([1])
                            labels.append([1])
                        elif 'Unstable' in line:
                            data.append([0])
                            labels.append([0])

                    # create labels for question one
                    if data[0] == [1]: labels.append([2])
                    elif (data == [[0], [2], [0], [0], [0]]) or (data == [[0], [3], [0], [0], [0]]): labels.append([3])
                    elif (data == [[0], [0], [1], [0], [0]]) or (data == [[0], [1], [1], [0], [0]]): labels.append([4])
                    elif (data == [[0], [0], [0], [1], [0]]) or (data == [[0], [1], [0], [1], [0]]): labels.append([5])
                    elif (data == [[0], [0], [0], [0], [1]]) or (data == [[0], [1], [0], [0], [1]]): labels.append([6])
                    elif (data == [[0], [2], [1], [0], [0]]) or (data == [[0], [3], [1], [0], [0]]): labels.append([7])
                    elif (data == [[0], [2], [0], [1], [0]]) or (data == [[0], [3], [0], [1], [0]]): labels.append([8])
                    elif (data == [[0], [2], [0], [0], [1]]) or (data == [[0], [3], [0], [0], [1]]): labels.append([9])
                    elif (data == [[0], [0], [1], [1], [0]]) or (data == [[0], [1], [1], [1], [0]]): labels.append([10])
                    elif (data == [[0], [0], [1], [0], [1]]) or (data == [[0], [1], [1], [0], [1]]): labels.append([11])
                    elif (data == [[0], [0], [0], [1], [1]]) or (data == [[0], [1], [0], [1], [1]]): labels.append([12])
                    elif (data == [[0], [2], [1], [1], [0]]) or (data == [[0], [3], [1], [1], [0]]): labels.append([13])
                    elif (data == [[0], [2], [1], [0], [1]]) or (data == [[0], [3], [1], [0], [1]]): labels.append([14])
                    elif (data == [[0], [2], [0], [1], [1]]) or (data == [[0], [3], [0], [1], [1]]): labels.append([15])
                    elif (data == [[0], [0], [1], [1], [1]]) or (data == [[0], [1], [1], [1], [1]]): labels.append([16])
                    elif (data == [[0], [2], [1], [1], [1]]) or (data == [[0], [3], [1], [1], [1]]): labels.append([17])
                    elif (data == [[0], [0], [0], [0], [0]]) or (data == [[0], [1], [0], [0], [0]]): labels.append([18])

                    # create labels for question two
                    if data[0] == [0]: labels.append([19])
                    elif (data == [[1], [0], [1], [1], [1]]) or (data == [[1], [1], [1], [1], [1]]): labels.append([20])
                    elif (data == [[1], [2], [0], [1], [1]]) or (data == [[1], [3], [0], [1], [1]]): labels.append([21])
                    elif (data == [[1], [2], [1], [0], [1]]) or (data == [[1], [3], [1], [0], [1]]): labels.append([22])
                    elif (data == [[1], [2], [1], [1], [0]]) or (data == [[1], [3], [1], [1], [0]]): labels.append([23])
                    elif (data == [[1], [0], [0], [1], [1]]) or (data == [[1], [1], [0], [1], [1]]): labels.append([24])
                    elif (data == [[1], [0], [1], [0], [1]]) or (data == [[1], [1], [1], [0], [1]]): labels.append([25])
                    elif (data == [[1], [0], [1], [1], [0]]) or (data == [[1], [1], [1], [1], [0]]): labels.append([26])
                    elif (data == [[1], [2], [0], [0], [1]]) or (data == [[1], [3], [0], [0], [1]]): labels.append([27])
                    elif (data == [[1], [2], [0], [1], [0]]) or (data == [[1], [3], [0], [1], [0]]): labels.append([28])
                    elif (data == [[1], [2], [1], [0], [0]]) or (data == [[1], [3], [1], [0], [0]]): labels.append([29])
                    elif (data == [[1], [0], [0], [0], [1]]) or (data == [[1], [1], [0], [0], [1]]): labels.append([30])
                    elif (data == [[1], [0], [0], [1], [0]]) or (data == [[1], [1], [0], [1], [0]]): labels.append([31])
                    elif (data == [[1], [0], [1], [0], [0]]) or (data == [[1], [1], [1], [0], [0]]): labels.append([32])
                    elif (data == [[1], [2], [0], [0], [0]]) or (data == [[1], [3], [0], [0], [0]]): labels.append([33])
                    elif (data == [[1], [0], [0], [0], [0]]) or (data == [[1], [1], [0], [0], [0]]): labels.append([34])
                    elif (data == [[1], [2], [1], [1], [1]]) or (data == [[1], [3], [1], [1], [1]]): labels.append([18])

                    # create labels for question three
                    if data[0] == [1]: labels.append([2])
                    elif (data == [[0], [2], [0], [0], [0]]) or (data == [[0], [3], [0], [0], [0]]): labels.append([35])
                    elif (data == [[0], [0], [1], [0], [0]]) or (data == [[0], [1], [1], [0], [0]]): labels.append([36])
                    elif (data == [[0], [0], [0], [1], [0]]) or (data == [[0], [1], [0], [1], [0]]): labels.append([37])
                    elif (data == [[0], [0], [0], [0], [1]]) or (data == [[0], [1], [0], [0], [1]]): labels.append([38])
                    elif (data == [[0], [2], [1], [0], [0]]) or (data == [[0], [3], [1], [0], [0]]): labels.append([39])
                    elif (data == [[0], [2], [0], [1], [0]]) or (data == [[0], [3], [0], [1], [0]]): labels.append([40])
                    elif (data == [[0], [2], [0], [0], [1]]) or (data == [[0], [3], [0], [0], [1]]): labels.append([41])
                    elif (data == [[0], [0], [1], [1], [0]]) or (data == [[0], [1], [1], [1], [0]]): labels.append([42])
                    elif (data == [[0], [0], [1], [0], [1]]) or (data == [[0], [1], [1], [0], [1]]): labels.append([43])
                    elif (data == [[0], [0], [0], [1], [1]]) or (data == [[0], [1], [0], [1], [1]]): labels.append([44])
                    elif (data == [[0], [2], [1], [1], [0]]) or (data == [[0], [3], [1], [1], [0]]): labels.append([45])
                    elif (data == [[0], [2], [1], [0], [1]]) or (data == [[0], [3], [1], [0], [1]]): labels.append([46])
                    elif (data == [[0], [2], [0], [1], [1]]) or (data == [[0], [3], [0], [1], [1]]): labels.append([47])
                    elif (data == [[0], [0], [1], [1], [1]]) or (data == [[0], [1], [1], [1], [1]]): labels.append([48])
                    elif (data == [[0], [2], [1], [1], [1]]) or (data == [[0], [3], [1], [1], [1]]): labels.append([49])
                    elif (data == [[0], [0], [0], [0], [0]]) or (data == [[0], [1], [0], [0], [0]]): labels.append([18])

                    textfile.close()

    # Convert labels into one-hot-encoding form
    enc = OneHotEncoder(n_values=50, sparse=False, dtype=np.int32)
    labels = enc.fit_transform(labels)

    return dataset, labels



def answer_question(features):

    # Restore model from saved checkpoint
    tf.reset_default_graph()
    num_hidden = 12
    input_data = tf.placeholder(tf.float32, [None, 16, 1])
    output_target = tf.placeholder(tf.float32, [None, 50])
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(output_target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[output_target.get_shape()[1]]))
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "models/rnn/rnn_model.ckpt")

        # Get the user to ask a question
        question = raw_input('Please type in a question. For the best performance, use one of the questions from the training set: \'is this structure stable\', \'what is making this structure unstable\', \'what is making this structure stable\' or \'what would need to be changed to make this structure stable\'.\n')

        # Convert the question into a word embedding
        vocabulary = {'is':1, 'this':2, 'structure':3, 'stable':4, 'what':5, 'making':6, 'unstable':7, 'would':8, 'need':9, 'to':10, 'be':11, 'changed':12, 'make':13}
        wordlist = re.sub("[^\w]", " ",  question).split()
        for i in range(len(wordlist)): wordlist[i] = wordlist[i].lower()
        input_vector = []
        for word in wordlist: input_vector.append([vocabulary[word]])
        while len(input_vector) < 11: input_vector.append([0]) # zero padding to length 11 (longest question in training set) 

        # Add the rest of the inputs to input_vector and pass through RNN
        input_vector.append([features[0]])
        input_vector.append([features[1]])
        input_vector.append([features[2]])
        input_vector.append([features[3]])
        input_vector.append([features[4]])
        feed_dict = {input_data: [input_vector]}
        prediction_result = prediction.eval(feed_dict, session=sess)
        probabilities = prediction_result[0]
        index = np.unravel_index(np.argmax(probabilities, axis=None), probabilities.shape)
        answer = index[0]

        # Print answer
        print ('\nAnswer:')
        if (answer == 0): print ('No')
        elif (answer == 1): print ('Yes')
        elif (answer == 2): print ('Nothing; it is stable')
        elif (answer == 3): print ('It has a large number of blocks')
        elif (answer == 4): print ('It has a narrow base')
        elif (answer == 5): print ('It is on a lean')
        elif (answer == 6): print ('One of the blocks is displaced')
        elif (answer == 7): print ('It has a large number of blocks and a narrow base')
        elif (answer == 8): print ('It has a large number of blocks and is on a lean')
        elif (answer == 9): print ('It has a large number of blocks and one of them is displaced')
        elif (answer == 10): print ('It has a narrow base and is on a lean')
        elif (answer == 11): print ('It has a narrow base and one of the blocks is displaced')
        elif (answer == 12): print ('It is on a lean and one of the blocks is displaced')
        elif (answer == 13): print ('It has a large number of blocks, a narrow base and is on a lean')
        elif (answer == 14): print ('It has a large number of blocks, a narrow base and one of the blocks is displaced')
        elif (answer == 15): print ('It has a large number of blocks, is on a lean and one of the blocks is displaced')
        elif (answer == 16): print ('It has a narrow base, is on a lean and one of the blocks is displaced')
        elif (answer == 17): print ('It has a large number of blocks, a narrow base, is on a lean and one of the blocks is displaced')
        elif (answer == 18): print ('Unknown')
        elif (answer == 19): print ('Nothing; it is unstable')
        elif (answer == 20): print ('It has a small number of blocks')
        elif (answer == 21): print ('It has a wide base')
        elif (answer == 22): print ('It is standing straight')
        elif (answer == 23): print ('The blocks are well balanced on top of one another')
        elif (answer == 24): print ('It has a small number of blocks and a wide base')
        elif (answer == 25): print ('It has a small number of blocks and is standing straight')
        elif (answer == 26): print ('It has a small number of blocks and they are well balanced on top of one another')
        elif (answer == 27): print ('It has a wide base and is standing straight')
        elif (answer == 28): print ('It has a wide base and they are well balanced on top of one another')
        elif (answer == 29): print ('It is standing straight and the blocks are well balanced on top of one another')
        elif (answer == 30): print ('It has a small number of blocks, a wide base and is standing straight')
        elif (answer == 31): print ('It has a small number of blocks, a wide base and the blocks are well balanced on top of one another')
        elif (answer == 32): print ('It has a small number of blocks, is standing straight and the blocks are well balanced on top of one another')
        elif (answer == 33): print ('It has a wide base, is standing straight and the blocks are well balanced on top of one another')
        elif (answer == 34): print ('It has a small number of blocks, a wide base, is standing straight and the blocks are well balanced on top of one another')
        elif (answer == 35): print ('The number of blocks needs to be reduced')
        elif (answer == 36): print ('A wider block needs to be used for the base')
        elif (answer == 37): print ('The structure needs to be straightened')
        elif (answer == 38): print ('The blocks need to be placed more directly above one another')
        elif (answer == 39): print ('The number of blocks needs to be reduced and a wider block needs to be used for the base')
        elif (answer == 40): print ('The number of blocks needs to be reduced and the structure needs to be straightened')
        elif (answer == 41): print ('The number of blocks needs to be reduced and the blocks need to be placed more directly above one another')
        elif (answer == 42): print ('A wider block needs to be used for the base and the structure needs to be straightened')
        elif (answer == 43): print ('A wider block needs to be used for the base and the blocks need to be placed more directly above one another')
        elif (answer == 44): print ('The structure needs to be straightened and the blocks need to be placed more directly above one another')
        elif (answer == 45): print ('The number of blocks needs to be reduced, a wider block needs to be used for the base and the structure needs to be straightened')
        elif (answer == 46): print ('The number of blocks needs to be reduced, a wider block needs to be used for the base and the blocks need to be placed more directly above one another')
        elif (answer == 47): print ('The number of blocks needs to be reduced, the structure needs to be straightened and the blocks need to be placed more directly above one another')
        elif (answer == 48): print ('A wider block needs to be used for the base, the structure needs to be straightened and the blocks need to be placed more directly above one another')
        elif (answer == 49): print ('The number of blocks needs to be reduced, a wider block needs to be used for the base, the structure needs to be straightened and the blocks need to be placed more directly above one another')

        # Close tensorflow sesstion
        sess.close()




def train_RNN(train_data, train_labels):

    tf.reset_default_graph()

    # Placeholders
    input_data = tf.placeholder(tf.float32, [None, 16, 1])
    output_target = tf.placeholder(tf.float32, [None, 50])

    # Set up LSTM cell
    num_hidden = 12 # Number of hidden layers (this can be changed)
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1) # last is now the output from the LSTM after it has processed all inputs

    # Set up the LSTM's weights and biases
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(output_target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[output_target.get_shape()[1]]))

    # The LSTM's final output
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    # Loss and optimization functions
    cross_entropy = -tf.reduce_sum(output_target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    # Calculating test data performance
    mistakes = tf.not_equal(tf.argmax(output_target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    # Run a session to train the LSTM
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    batch_size = 1 # this can be changed
    no_of_batches = int(len(train_data)/batch_size)
    incorrect = len(train_data)
    previous_incorrect = len(train_data)+10
    i = 0
    while ((incorrect+5) < previous_incorrect): # keep training until accuracy improves less than 5% per epoch (this limit prevents overfitting)
        i = i+1
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_data[ptr:ptr+batch_size], train_labels[ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(minimize,{input_data: inp, output_target: out})
        print "Epoch - ",str(i)

        # Test the LSTM on the test dataset
        previous_incorrect = incorrect
        incorrect = sess.run(error,{input_data: train_data, output_target: train_labels})
        print('Epoch {:2d} accuracy {:3.1f}%'.format(i + 1, (100 - 100 * incorrect)))

    # Save the model
    save_path = saver.save(sess, "models/rnn/rnn_model.ckpt")

    sess.close()

    accuracy = 100 - 100*incorrect

    return accuracy

