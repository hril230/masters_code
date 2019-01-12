from __future__ import print_function
import os
import numpy as np



# This function finds the values for a column in the dataset
def unique_vals(rows, col):
    return set([row[col] for row in rows])



# This function counts the number of each type of example in the dataset
def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in this dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts



# This function tests if a value is numeric
def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)



# This class records a 'column number' (e.g., 0 for Color) and a 'column value' (e.g., Green).
class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # Print the question in a readable format
        header = ["class", "shape", "main colour", "border colour", "background image", "symbol", "secondary symbol", "cross"]
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


# For each row in the dataset, check if it matches the question. If so, add it to 'true rows', otherwise, add it to 'false rows'.
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows



# Calculate the Gini Impurity for a list of rows
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity



# Information Gain = the uncertainty of the starting node, minus the weighted impurity of two child nodes
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)



# Find the best question to ask by iterating over every feature / value and calculating the information gain
def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # Try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question



# This holds a dictionary of class (e.g., "Apple") -> number of times it appears in the rows from the training data that reach this leaf.
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)



# This holds a reference to the question, and to the two child nodes.
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch



# Builds the tree.
def build_tree(rows):

    # Try partitioing the dataset on each of the unique attribute, calculate the information gain, and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions, we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow depending on the answer.
    return Decision_Node(question, true_branch, false_branch)



def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")



def classify(row, node, branches_used):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions, branches_used

    if node.question.match(row):
        branches_used.append(str(node.question) + '  -  true')
        return classify(row, node.true_branch, branches_used)
    else:
        branches_used.append(str(node.question) + '  -  false')
        return classify(row, node.false_branch, branches_used)



def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



# MAIN
def decision_tree(train_data):

    # Train with the samples that the logic classifier couldn't predict
    training_data = []
    for row in train_data:
        if row[0] == -1:
            training_data.append(row[1:len(row)])

    # To avoid errors during testing, if all training samples were classified by logic, then the decision tree must be trained with those samples
    if len(training_data) == 0:
        for row in train_data:
            training_data.append(row[1:len(row)])

    # Train tree
    my_tree = build_tree(training_data)

    # Return stability predictions for training data
    train_predictions = [] # format of predictions: [class, features, classifier] (1=stable, ..., 1=logic)
    for i in range(len(train_data)):
        row = train_data[i]
        features = []
        if row[0] == -1:
            classifier = 0
            counts, branches_used = classify(row[1:], my_tree, [])
            prediction = print_leaf(counts)
            try:
                prediction_zero = int(prediction[0][0:len(prediction[0])-1])
            except KeyError:
                prediction_zero = 0
            try:
                prediction_one = int(prediction[1][0:len(prediction[1])-1])
            except KeyError:
                prediction_one = 0
            if prediction_one < prediction_zero: numprediction = 0
            else: numprediction = 1
        else:
            numprediction = row[0]
            classifier = 1
        features.append(numprediction)
        for i in range(1,len(row)): features.append(row[i]) # create a list containing the features of the structure
        features.append(classifier)
        train_predictions.append(features)

    return my_tree, train_predictions # return a list of lists, with each sub-list containing the stability prediction and features of a structure
