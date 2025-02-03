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

# def naive_bayes(train_labels, train_data, dev_data, laplace=1.0, pos_prior=0.5, silently=False):
#     print_values(laplace, pos_prior)

#     # Calculate the prior probabilities
#     neg_prior = 1.0 - pos_prior

#     # Split the training data into positive and negative documents
#     pos_docs = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 1]
#     neg_docs = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 0]

#     # Count the frequency of each word in positive and negative documents
#     poscount = Counter(word for doc in pos_docs for word in doc)
#     negcount = Counter(word for doc in neg_docs for word in doc)

#     # Calculate the total number of words in positive and negative documents
#     total_pos_words = sum(poscount.values())
#     total_neg_words = sum(negcount.values())

#     # Calculate the vocabulary size
#     vocabulary = set(poscount.keys()).union(set(negcount.keys()))
#     vocab_size = len(vocabulary)

#     # Calculate the log likelihoods for each word
#     pos_log_likelihoods = {}
#     neg_log_likelihoods = {}

#     for word in vocabulary:
#         pos_count = poscount.get(word, 0)
#         neg_count = negcount.get(word, 0)

#         pos_log_likelihoods[word] = math.log((pos_count + laplace) / (total_pos_words + laplace * vocab_size))
#         neg_log_likelihoods[word] = math.log((neg_count + laplace) / (total_neg_words + laplace * vocab_size))

#     # Predict the class for each document in the development set
#     yhats = []
#     for doc in tqdm(dev_data, disable=silently):
#         pos_score = math.log(pos_prior)
#         neg_score = math.log(neg_prior)

#         for word in doc:
#             if word in pos_log_likelihoods:
#                 pos_score += pos_log_likelihoods[word]
#             if word in neg_log_likelihoods:
#                 neg_score += neg_log_likelihoods[word]

#         if pos_score > neg_score:
#             yhats.append(1)
#         else:
#             yhats.append(0)

#     return yhats

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
        # Initialize log probabilities for positive and negative classes
        log_prob_pos = math.log(pos_prior)
        log_prob_neg = math.log(1 - pos_prior)

        # Calculate the log probability for each word in the document
        for word in doc:
            # Positive class
            word_count_pos = poscount[word] + laplace
            log_prob_pos += math.log(word_count_pos / (postotal + laplace * vocab_size))

            # Negative class
            word_count_neg = negcount[word] + laplace
            log_prob_neg += math.log(word_count_neg / (negtotal + laplace * vocab_size))

        # Assign the class with the higher log probability
        if log_prob_pos > log_prob_neg:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats