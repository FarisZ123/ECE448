import numpy as np

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
        context_word = data[t + c]
        context_vec = embedding[context_word]

        score = np.dot(target_vec, context_vec)
        prob = 1 / (1 + np.exp(-score))
        g += (1 - prob) * context_vec

    neg = 0
    for i in range(k):
        context_vec = embedding[np.random.choice(list(embedding))]
        n_similarity = np.dot(target_vec, context_vec)
        expected_value = 1/ (1 + np.exp(-1 * n_similarity))
        neg += expected_value * context_vec

    neg /= k
    g -= neg
    return (g)

# Test case for single word in vocabulary
def test_gradient_single_word():
    embedding = {'word': np.array([1.0, 2.0])}
    data = ['word']
    t = 0
    d = 1
    k = 1
    g = gradient(embedding, data, t, d, k)
    z = np.dot(embedding['word'], embedding['word'])
    sigma_z = 1 / (1 + np.exp(-z))
    # 2*d*(2*sigma(v*v)-1)*v
    expected_g = 2*d*(2*(sigma_z)-1) * embedding['word']
    print(f"Calculated gradient: {g}")
    print(f"Expected gradient: {expected_g}")
    assert np.allclose(g, expected_g, atol=1e-2), f"Test Failed: {g} != {expected_g}"

test_gradient_single_word()