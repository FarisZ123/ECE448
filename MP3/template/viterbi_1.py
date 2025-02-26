# """
# Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
# but it should do better than the baseline at words with multiple tags (because now you're using context
# to predict the tag).
# """

# import math
# from collections import defaultdict, Counter
# from math import log

# # Note: remember to use these two elements when you find a probability is 0 in the training data.
# epsilon_for_pt = 1e-5
# emit_epsilon = 1e-5   # exact setting seems to have little or no effect


# def training(sentences):
#     """
#     Computes initial tags, emission words and transition tag-to-tag probabilities
#     :param sentences:
#     :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
#     """
#     init_prob = defaultdict(lambda: 0) # {init tag: #}
#     emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
#     trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
#     # TODO: (I)
#     # Input the training set, output the formatted probabilities according to data statistics.
#     tag_count = defaultdict(lambda: 0)
#     for sentence in sentences:
#         prev_tag = None
#         for i, (word, tag) in enumerate(sentence):
#             if i == 0:
#                 init_prob[tag] += 1  # Count initial tags
            
#             emit_prob[tag][word] += 1  # Count word occurrences given tag
#             tag_count[tag] += 1  # Count occurrences of tag
            
#             if prev_tag is not None:
#                 trans_prob[prev_tag][tag] += 1  # Count transitions
            
#             prev_tag = tag  # Update previous tag for next iteration

#     # Convert counts to probabilities
#     total_init = sum(init_prob.values())
#     for tag in init_prob:
#         init_prob[tag] /= total_init

#     for tag in emit_prob:
#         for word in emit_prob[tag]:
#             emit_prob[tag][word] /= tag_count[tag]

#     for prev_tag in trans_prob:
#         total_trans = sum(trans_prob[prev_tag].values())
#         for tag in trans_prob[prev_tag]:
#             trans_prob[prev_tag][tag] /= total_trans

#     return init_prob, emit_prob, trans_prob

# def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
#     """
#     Does one step of the viterbi function
#     :param i: The i'th column of the lattice/MDP (0-indexing)
#     :param word: The i'th observed word
#     :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
#     previous column of the lattice
#     :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
#     of the lattice for each tag in the previous column
#     :param emit_prob: Emission probabilities
#     :param trans_prob: Transition probabilities
#     :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
#     """
#     log_prob = {} # This should store the log_prob for all the tags at current column (i)
#     predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

#     # TODO: (II)
#     # implement one step of trellis computation at column (i)
#     # You should pay attention to the i=0 special case.
#     for curr_tag in emit_prob:
#         max_prob = float('-inf')
#         best_prev_tag = None

#         for prev_tag in prev_prob:
#             # Compute transition and emission probabilities (use epsilon if missing)
#             trans_p = trans_prob[prev_tag].get(curr_tag, epsilon_for_pt)
#             emit_p = emit_prob[curr_tag].get(word, emit_epsilon)

#             # Compute log probability
#             prob = prev_prob[prev_tag] + log(trans_p) + log(emit_p)

#             # Find the best previous tag for this current tag
#             if prob > max_prob:
#                 max_prob = prob
#                 best_prev_tag = prev_tag

#         log_prob[curr_tag] = max_prob
#         predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [best_prev_tag] if best_prev_tag else []
#     return log_prob, predict_tag_seq

# def viterbi_1(train, test, get_probs=training):
#     '''
#     input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#             test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
#     output: list of sentences, each sentence is a list of (word,tag) pairs.
#             E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#     '''
#     init_prob, emit_prob, trans_prob = get_probs(train)
    
#     predicts = []
    
#     for sen in range(len(test)):
#         sentence=test[sen]
#         length = len(sentence)
#         log_prob = {}
#         predict_tag_seq = {}
#         # init log prob
#         for t in emit_prob:
#             if t in init_prob:
#                 log_prob[t] = log(init_prob[t])
#             else:
#                 log_prob[t] = log(epsilon_for_pt)
#             predict_tag_seq[t] = []

#         # forward steps to calculate log probs for sentence
#         for i in range(length):
#             log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
#         # TODO:(III) 
#         # according to the storage of probabilities and sequences, get the final prediction.
#         # Find the best final tag


#     return predicts


import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect

def training(sentences):
    init_prob = defaultdict(lambda: 0)
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))
    tag_counts = defaultdict(int)

    # Count occurrences
    for sentence in sentences:
        prev_tag = None
        first_tag = sentence[0][1]
        init_prob[first_tag] += 1  # Count initial tag

        for word, tag in sentence:
            tag_counts[tag] += 1
            emit_prob[tag][word] += 1
            if prev_tag is not None:
                trans_prob[prev_tag][tag] += 1  # Count tag transitions
            prev_tag = tag

    # Normalize probabilities
    total_sentences = len(sentences)
    for tag in tag_counts:
        init_prob[tag] = (init_prob[tag] + epsilon_for_pt) / (total_sentences + len(tag_counts) * epsilon_for_pt)

        for word in emit_prob[tag]:
            emit_prob[tag][word] = (emit_prob[tag][word] + emit_epsilon) / (tag_counts[tag] + emit_epsilon * len(emit_prob[tag]))

        for next_tag in trans_prob[tag]:
            trans_prob[tag][next_tag] = (trans_prob[tag][next_tag] + epsilon_for_pt) / (tag_counts[tag] + epsilon_for_pt * len(tag_counts))

    return init_prob, emit_prob, trans_prob



def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    log_prob = {}
    predict_tag_seq = {}

    for curr_tag in emit_prob:
        max_prob = float('-inf')
        best_prev_tag = None

        for prev_tag in prev_prob:
            transition_prob = trans_prob[prev_tag].get(curr_tag, epsilon_for_pt)
            emission_prob = emit_prob[curr_tag].get(word, emit_epsilon)

            prob = prev_prob[prev_tag] + log(transition_prob) + log(emission_prob)

            if prob > max_prob:
                max_prob = prob
                best_prev_tag = prev_tag

        log_prob[curr_tag] = max_prob
        predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [curr_tag]  # Append current tag

    return log_prob, predict_tag_seq


def viterbi_1(train, test, get_probs=training):
    init_prob, emit_prob, trans_prob = get_probs(train)

    predicts = []

    for sentence in test:
        length = len(sentence)

        log_prob = {}
        predict_tag_seq = {}

        # Initialization step
        for tag in emit_prob:
            log_prob[tag] = log(init_prob.get(tag, epsilon_for_pt))
            predict_tag_seq[tag] = []

        # Forward pass
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        # Find the best final tag
        best_final_tag = max(log_prob, key=log_prob.get)
        best_sequence = predict_tag_seq[best_final_tag] + [best_final_tag]  # Append last tag

        predicts.append(list(zip(sentence, best_sequence)))

    return predicts


