import numpy as np
from mp1 import *

  
def test_create_linear_data(num_samples = 100, slope = 2, intercept = 1, x_range = [-10,10], noise = 0):
    x, y = create_linear_data(num_samples=num_samples, slope=slope, 
                              intercept=intercept, x_range=x_range, noise=noise)
    assert type(x) == np.ndarray and type(y) == np.ndarray
    print(x.shape)
    assert x.shape == (num_samples,1)
    assert y.shape == (num_samples,1)



test_create_linear_data(num_samples = 100, slope = 2, intercept = 1, x_range = [-10,10], noise = 0)