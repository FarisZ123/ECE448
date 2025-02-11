# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    print(trainingdir)
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""


def naive_bayes(train_labels, train_data, dev_data, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace, pos_prior)

    poscount = Counter()
    negcount = Counter()
    postotal = 0
    negtotal = 0

    for i in range(len(train_data)):
        if train_labels[i] == 1:
            poscount.update(train_data[i])
            postotal = postotal + len(train_data[i])
        else:
            negcount.update(train_data[i])
            negtotal = negtotal + len(train_data[i])


    vocabulary = set(poscount.keys()).union(set(negcount.keys()))
    vocab_size = len(vocabulary)


    yhats = []


    for doc in tqdm(dev_data, disable=silently):
        log_prob_pos = math.log(pos_prior)
        log_prob_neg = math.log(1 - pos_prior)


        for word in doc:
            word_count_pos = poscount[word] + laplace
            log_prob_pos += math.log(word_count_pos / (postotal + laplace * vocab_size))

            word_count_neg = negcount[word] + laplace
            log_prob_neg += math.log(word_count_neg / (negtotal + laplace * vocab_size))

        if log_prob_pos > log_prob_neg:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats