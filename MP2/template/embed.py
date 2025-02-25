import numpy as np

def initialize(data, dim):
    '''
    Initialize embeddings for all distinct words in the input data.
    Most of the dimensions will be zero-mean unit-variance Gaussian random variables.
    In order to make debugging easier, however, we will assign special geometric values
    to the first two dimensions of the embedding:

    (1) Find out how many distinct words there are.
    (2) Choose that many locations uniformly spaced on a unit circle in the first two dimensions.
    (3) Put the words into those spots in the same order that they occur in the data.

    Thus if data[0] and data[1] are different words, you should have

    embedding[data[0]] = np.array([np.cos(0), np.sin(0), random, random, random, ...])
    embedding[data[1]] = np.array([np.cos(2*np.pi/N), np.sin(2*np.pi/N), random, random, random, ...])

    ... and so on, where N is the number of distinct words, and each random element is
    a Gaussian random variable with mean=0 and standard deviation=1.

    @param:
    data (list) - list of words in the input text, split on whitespace
    dim (int) - dimension of the learned embeddings

    @return:
    embedding - dict mapping from words (strings) to numpy arrays of dimension=dim.
    '''
    words = list(set(data))
    num_of_words = len(words)
    angles = np.linspace(0, 2 * np.pi, num_of_words, endpoint=False)
    circle_points = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
    
    embedding = {}
    for i, word in enumerate(words):
        first_two_dims = circle_points[i]
        remaining_dims = np.random.normal(0, 1, dim - 2)
        embedding[word] = np.concatenate([first_two_dims, remaining_dims])

    return embedding

def gradient(embedding, data, t, d, k):
    '''
    Calculate gradient of the skipgram NCE loss with respect to the embedding of data[t]

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    t (int) - data index of word with respect to which you want the gradient
    d (int) - choose context words from t-d through t+d, not including t
    k (int) - compare each context word to k words chosen uniformly at random from the data

    @return:
    g (numpy array) - loss gradients with respect to embedding of data[t]
    '''
    
    target_word = data[t]
    target_vec = embedding[target_word]
    # context_indices = [i for i in range(max(0, t - d), min(len(data), t + d + 1)) if i != t]
    
    g = np.zeros_like(target_vec)
    #---------------------------------------
    for i in range (-d, d+1):
        if (i == 0  or (0 >t+i)  or  t+i >= len(data)):
            continue
        
        context_word = data[t + i]
        context_vec = embedding[context_word]

        score = np.dot(target_vec, context_vec)
        prob = 1 / (1 + np.exp(-score))
        g += (1 - prob) * context_vec

        negative = 0
        for j in range(k):
            context_vec = embedding[np.random.choice(list(embedding))]
            n_similarity = np.dot(target_vec, context_vec)
            expected_value = 1/ (1 + np.exp(-1 * n_similarity))
            negative += expected_value * context_vec

        negative /= k
        g -= negative
            
    return (g)

           
def sgd(embedding, data, learning_rate, num_iters, d, k):
    '''
    Perform num_iters steps of stochastic gradient descent.

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    learning_rate (scalar) - scale the negative gradient by this amount at each step
    num_iters (int) - the number of iterations to perform
    d (int) - context width hyperparameter for gradient computation
    k (int) - noise sample size hyperparameter for gradient computation

    @return:
    embedding - the updated embeddings
    '''   
            
    for _ in range(num_iters):
        t = np.random.randint(0, len(data))
        grad = gradient(embedding, data, t, d, k)
        embedding[data[t]] -= learning_rate * grad
    
    return embedding