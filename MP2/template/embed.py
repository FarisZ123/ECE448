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
    context_indices = [i for i in range(max(0, t - d), min(len(data), t + d + 1)) if i != t]
    
    g = np.zeros_like(target_vec)
    
    for c in context_indices:
        context_word = data[c]
        context_vec = embedding[context_word]

        score = np.dot(target_vec, context_vec)
        prob = 1 / (1 + np.exp(-score))
        g += (prob - 1) * context_vec

        negative_samples = np.random.choice(data, k, replace=True)
        for neg_word in negative_samples:
            neg_vec = embedding[neg_word]
            neg_score = np.dot(target_vec, neg_vec)
            neg_prob = 1 / (1 + np.exp(-neg_score))
            g += neg_prob * neg_vec 
    
    return g


           
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
    vocab = list(embedding.keys())
    
    for iteration in range(num_iters):
        for i, target_word in enumerate(data):
            context_words = data[max(0, i - d):i] + data[i + 1:min(len(data), i + d + 1)]
            noise_samples = np.random.choice(vocab, size=k, replace=False)
            target_embedding = embedding[target_word]

            gradient = np.zeros_like(target_embedding)
            

            for context_word in context_words:
                context_embedding = embedding[context_word]
                context_sigmoid = 1 / (1 + np.exp(-np.dot(target_embedding, context_embedding)))
                gradient += (1 - context_sigmoid) * context_embedding

            for noise_word in noise_samples:
                noise_embedding = embedding[noise_word]
                noise_sigmoid = 1 / (1 + np.exp(-np.dot(target_embedding, noise_embedding)))
                gradient -= noise_sigmoid * noise_embedding

            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 1.0:
                gradient = gradient / gradient_norm 
            

            embedding[target_word] += learning_rate * gradient
            
    return embedding
    

