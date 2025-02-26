"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

'''
input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
from collections import defaultdict

def baseline(train, test):
        word_tag_counts = defaultdict(lambda: defaultdict(int))
        tag_counts = defaultdict(int)
        for line in train:                                                                              ## gets the total tag count for each word and totals them up
                for word, tag in line:
                        word_tag_counts[word][tag] += 1
                        tag_counts[tag] += 1

        most_frequent_tag = {}
        for word, tag_freq in word_tag_counts.items():                                          
                most_frequent_tag[word] = max(tag_freq, key=tag_freq.get)                               ## gets the most frequent tag for each word

        total_most_tag = max(tag_counts, key=tag_counts.get)                                            ## gets overall most frequent tag (for unseen words)

        tagged_test = []

        for line in test:
                tagged_line = []
                for word in line:
                        if word in most_frequent_tag:
                                tagged_line.append((word, most_frequent_tag[word]))                     ## tag each word with its most frequent tag (tag total_most_tag for unseen words)
                        else:
                                tagged_line.append((word, total_most_tag))
                tagged_test.append(tagged_line)

        return tagged_test      
