import numpy as np
import tensorflow as tf
import os
import cv2
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder
import re
import random
import sys



def parse_dataset(train_predictions, train_labels):
    # predictions format: [class_prediction, shape, main colour, border colour, background image, symbol, secondary symbol, cross, classifier]
    # labels format: [class_label, feature_labels]

    # Create dictionaries of correct answers
    question_zero_answers = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24, 25:25, 26:26, 27:27, 28:28, 29:29, 30:30, 31:31, 32:32, 33:33, 34:34, 35:35, 36:36, 37:37, 38:38, 39:39, 40:40, 41:41, 42:42, 43:42, 44:43, 45:44, 46:45, 47:46, 48:47, 49:48, 50:44, 51:49, 52:50, 53:34, 54:51, 55:52, 56:53, 57:54, 58:55, 59:56, 60:57, 61:58}
    question_one_answers = {0:59, 1:60, 2:61, 3:62, 4:63, 5:64, 6:65, 7:66, 8:67, 9:68, 10:69, 11:70, 12:71, 13:72, 14:73, 15:74, 16:75, 17:76, 18:77, 19:78, 20:79, 21:80, 22:22, 23:81, 24:82, 25:83, 26:84, 27:85, 28:86, 29:87, 30:88, 31:89, 32:90, 33:91, 34:92, 35:93, 36:94, 37:95, 38:96, 39:97, 40:98, 41:99, 42:42, 43:42, 44:100, 45:101, 46:102, 47:103, 48:104, 49:105, 50:101, 51:106, 52:107, 53:92, 54:108, 55:109, 56:110, 57:111, 58:112, 59:113, 60:57, 61:58}
    question_two_answers = {0:114, 1:115, 2:114, 3:116, 4:117, 5:116, 6:117, 7:114, 8:114, 9:114, 10:114, 11:118, 12:119, 13:114, 14:114, 15:116, 16:117, 17:120, 18:114, 19:121, 20:122, 21:123, 22:124, 23:125, 24:126, 25:127, 26:128, 27:129, 28:130, 29:94, 30:131, 31:132, 32:133, 33:134, 34:135, 35:136, 36:94, 37:137, 38:138, 39:139, 40:140, 41:141, 42:42, 43:42, 44:142, 45:143, 46:144, 47:145, 48:146, 49:147, 50:143, 51:114, 52:148, 53:135, 54:149, 55:148, 56:150, 57:151, 58:152, 59:153, 60:154, 61:148}

    question_zero_train_dataset = []
    question_one_train_dataset = []
    question_two_train_dataset = []
    question_three_train_dataset = []
    question_zero_train_labels = []
    question_one_train_labels = []
    question_two_train_labels = []
    question_three_train_labels = []
    for i in range(len(train_predictions)):
        if train_predictions[i][8] == 0: # use only predictions that come from the decision tree
            # create dataset for question zero
            data = [[62], [63], [64], [65], [66], [67], [0], [0]] # this word encoding means that the question is 'What type of sign is this?'
            for feature in train_predictions[i][0:len(train_predictions[i])-1]:
                data.append([feature])
            question_zero_train_dataset.append(np.array(data))
            # create labels for question zero
            question_zero_train_labels.append([question_zero_answers[train_labels[i][0]]])
            # create dataset for question one
            data = [[62], [66], [68], [69], [70], [0], [0], [0]] # this word encoding means that the question is 'What is the sign's message?'
            for feature in train_predictions[i][0:len(train_predictions[i])-1]:
                data.append([feature])
            question_one_train_dataset.append(np.array(data))
            # create labels for question one
            question_one_train_labels.append([question_one_answers[train_labels[i][0]]])
            # create dataset for question two
            data = [[71], [72], [68], [73], [74], [75], [67], [65]] # this word encoding means that the question is 'How should the driver react to this sign?'
            for feature in train_predictions[i][0:len(train_predictions[i])-1]:
                data.append([feature])
            question_two_train_dataset.append(np.array(data))
            # create labels for question two
            question_two_train_labels.append([question_two_answers[train_labels[i][0]]])
            # create dataset for question three
            data = [[66], [67], [76], [random.randint(0,61)], [65], [0], [0], [0]] # this word encoding means 'Is this a [insert sign type here] sign?'
            for feature in train_predictions[i][0:len(train_predictions[i])-1]:
                data.append([feature])
            question_three_train_dataset.append(np.array(data))
            # create labels for question three
            if (data[3] == train_labels[i][0]): question_three_train_labels.append([155])
            else: question_three_train_labels.append([156])
	
    # Convert labels into one-hot-encoding form
    enc = OneHotEncoder(n_values=157, sparse=False, dtype=np.int32)
    question_zero_train_labels = enc.fit_transform(question_zero_train_labels)
    question_one_train_labels = enc.fit_transform(question_one_train_labels)
    question_two_train_labels = enc.fit_transform(question_two_train_labels)
    question_three_train_labels = enc.fit_transform(question_three_train_labels)

    # combine datasets and labels into train dataset
    train_data = question_zero_train_dataset + question_one_train_dataset + question_two_train_dataset + question_three_train_dataset
    train_labels = np.concatenate((question_zero_train_labels, question_one_train_labels, question_two_train_labels, question_three_train_labels), axis=0)

    return train_data, train_labels



def answer_question(features):
# features format: [class_prediction, shape, main colour, border colour, background image, symbol, secondary symbol, cross, classifier]

    # Restore model from saved checkpoint
    tf.reset_default_graph()
    num_hidden = 24
    input_data = tf.placeholder(tf.float32, [None, 17, 1])
    output_target = tf.placeholder(tf.float32, [None, 157])
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(output_target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[output_target.get_shape()[1]]))
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        reuse=True
        saver.restore(sess, "models/rnn/rnn_model.ckpt")

        # Get the user to ask a question
        question = raw_input('Please type in a question. For the best performance, use one of the questions from the training set: \'what type of sign is this\', \'what is the signs message\', \'how should the driver react to this sign\' or \'is this a [insert class number here] sign\'.\n')

        # Convert the question into a word embedding
        vocabulary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, '23':23, '24':24, '25':25, '26':26, '27':27, '28':28, '29':29, '30':30, '31':31, '32':32, '33':33, '34':34, '35':35, '36':36, '37':37, '38':38, '39':39, '40':40, '41':41, '42':42, '43':43, '44':44, '45':45, '46':46, '47':47, '48':48, '49':49, '50':50, '51':51, '52':52, '53':53, '54':54, '55':55, '56':56, '57':57, '58':58, '59':59, '60':60, '61':61, 'what':62, 'type':63, 'of':64, 'sign':65, 'is':66, 'this':67, 'the':68, 'signs':69, 'message':70, 'how':71, 'should':72, 'driver':73, 'react':74, 'to':75, 'a':76}
        wordlist = re.sub("[^\w]", " ",  question).split()
        for i in range(len(wordlist)): wordlist[i] = wordlist[i].lower()
        input_vector = []
        for word in wordlist: input_vector.append([vocabulary[word]])
        while len(input_vector) < 9: input_vector.append([0]) # zero padding to length 9 (longest question in training set) 

        # Add the rest of the inputs to input_vector and pass through RNN
        for feature in features[:len(features)-1]:
            input_vector.append([feature])
        feed_dict = {input_data: [input_vector]}
        prediction_result = prediction.eval(feed_dict, session=sess)
        probabilities = prediction_result[0]
        index = np.unravel_index(np.argmax(probabilities, axis=None), probabilities.shape)
        answer = index[0]

        # Print answer
        answers = {0:'bumpy road', 1:'speedbump', 2:'slippery road', 3:'left turn', 4:'right turn', 5:'left then right turn', 6:'right then left turn', 7:'children possible', 8:'bicycles possible', 9:'cattle possible', 10:'road workers possible', 11:'traffic signals', 12:'gated crossing', 13:'danger point', 14:'road narrows', 15:'road narrows on left', 16:'road narrows on right', 17:'priority to through-traffic', 18:'uncontrolled intersection', 19:'yield', 20:'oncoming traffic', 21:'stop', 22:'do not enter', 23:'no bicycles permitted', 24:'total vehicle weight limit', 25:'no trucks permitted', 26:'width limit', 27:'height limit', 28:'no motorized vehicles', 29:'no left turn', 30:'no right turn', 31:'no passing', 32:'speed limit', 33:'shared use path', 34:'one-way traffic', 35:'turn here', 36:'straight or right', 37:'roundabout', 38:'bicycle lane', 39:'bicycle and pedestrian path', 40:'no parking', 41:'no stopping', 42:'unknown', 43:'priority over oncoming vehicles', 44:'parking', 45:'disabled parking', 46:'car parking', 47:'truck parking', 48:'bus parking', 49:'traffic calming zone', 50:'end of traffic calming zone', 51:'no through road', 52:'road worker zone ends', 53:'pedestrian crossing', 54:'bicycle crossing', 55:'parking to the right', 56:'speedbump crossing', 57:'priority road starts', 58:'priority road ends', 59:'uneven surfaces ahead', 60:'speedbump ahead', 61:'water, ice or oil ahead', 62:'dangerous curve to the left', 63:'dangerous curve to the right', 64:'double curves, first to the left', 65:'double curves, first to the right', 66:'there may be children on this road', 67:'there may be cyclists on this road', 68:'there may be cattle on this road', 69:'there may be road workers on this road', 70:'traffic lights ahead', 71:'level crossing with barrier or gate ahead', 72:'pay close attention to the following sign', 73:'narrow road ahead', 74:'the left side of the road narrows ahead', 75:'the right side of the road narrows ahead', 76:'give through-traffic priority at the next intersection', 77:'uncontrolled intersection ahead, priority not assigned', 78:'yield to cross traffic', 79:'oncoming traffic, red direction must yield', 80:'come to a complete stop at this intersection', 81:'bicycles are not permitted on this road', 82:'no vehicles over this weight permitted', 83:'no trucks permitted on this road', 84:'no vehicles over this width permitted', 85:'no vehicles over this height permitted', 86:'no motorized vehicles of any type permitted', 87:'no left turn from this intersection', 88:'no right turn from this intersection', 89:'no passing on this road', 90:'do not drive faster than this speed limit', 91:'both pedestrians and bicycles may use this path', 92:'this road is one-way only', 93:'turn in the indicated direction', 94:'go straight or turn right', 95:'roundabout ahead', 96:'this lane is for bicycles only', 97:'this path has separate lanes for bicycles and pedestrians', 98:'no parking, waiting is allowed', 99:'no parking or waiting is allowed', 100:'vehicles on the right have priority', 101:'parking available here', 102:'only disabled drivers may park here', 103:'only cars may be parked here', 104:'trucks may be parked here', 105:'busses may be parked here', 106:'in this zone there may be pedestrians and children', 107:'traffic calming zone ends', 108:'no through road', 109:'road worker zone ends', 110:'yield to pedestrians', 111:'yield to bicycles', 112:'there are parking spots on the right', 113:'speedbump crossing ahead', 114:'drive with caution', 115:'slow down for the speedbump', 116:'slow and stay to the right', 117:'slow and stay to the left', 118:'obey traffic signals', 119:'stop for gate', 120:'give through-traffic priority at the next intersection', 121:'yield to cross traffic', 122:'if on left, continue, if on right, yield', 123:'stop', 124:'avoid that entrance', 125:'avoid if riding a bicycle', 126:'avoid if vehicle weight exceeds limit', 127:'avoid if driving a truck', 128:'avoid if vehicle width exceeds limit', 129:'avoid if vehicle height exceeds limit', 130:'avoid this road', 131:'go straight or turn left', 132:'do not overtake the car in front', 133:'stay under the speed limit', 134:'avoid pedestrians', 135:'avoid this road if not traveling in the right direction', 136:'turn in the indicated direction', 137:'yield to traffic in circle, signal on exit', 138:'avoid if not riding a bicycle', 139:'stay in the bicycle lane if riding a bicycle', 140:'do not park here', 141:'do not stop here', 142:'if on the right, continue, if on the left, yield', 143:'the driver may park here', 144:'the driver may park here if disabled', 145:'car drivers may park here', 146:'truck drivers may park here', 147:'bus drivers may park here', 148:'resume normal driving', 149:'avoid this road if looking for a way through', 150:'yield to pedestrians', 151:'yield to bicycles', 152:'the driver may park on the right', 153:'slow down for the speedbump and watch out for pedestrians', 154:'yield to all pedestrians or bicycles, signal all turns', 155:'yes', 156:'no'}
        print ('\nAnswer:')
        print (answers[answer])

        # Close tensorflow sesstion
        sess.close()




def train_RNN(train_data, train_labels):

    tf.reset_default_graph()

    # Placeholders
    input_data = tf.placeholder(tf.float32, [None, 17, 1])
    output_target = tf.placeholder(tf.float32, [None, 157])

    # Set up LSTM cell
    num_hidden = 24 # Number of hidden layers (this can be changed)
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

