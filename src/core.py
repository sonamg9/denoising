
import numpy as np
from scipy.stats import norm

#######################################################

def generate_random_1d_vector( x_vector):
    '''
    Function to generate 1D numpy array with three gaussians
    '''

    size=x_vector.shape[0]

    # Define three Gaussian functions
    gaussian1 = norm.pdf(x_vector, loc=50, scale=5.0)
    gaussian2 = norm.pdf(x_vector, loc=100, scale=1)
    gaussian3 = norm.pdf(x_vector, loc=150, scale=2)

    # Combine Gaussians with white noise
    random_vector = (
        5 * gaussian1 +
        7.5 * gaussian2 +
        2 * gaussian3 +
        0.025 * np.random.normal(size=size)
    )

    return random_vector
#######################################################

def generate_2D_sample(xdim, xmin, xmax, ydim):
    '''
    Function to generate 2D numpy array resembling dataset from measurments
    '''

    x_vector = np.linspace( xmin, xmax, xdim)

    data=np.zeros((xdim, ydim))

    for i in range(ydim):
        data[:,i]= generate_random_1d_vector( x_vector )

    return x_vector,data

#######################################################
