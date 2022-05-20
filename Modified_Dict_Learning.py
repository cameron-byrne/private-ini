import numpy as np
from autograd import grad
import scipy.io as sio
from Learning_multiprocessing import Timer
from Learning_autograd import print_confusion_matrix
from numba import njit, prange

def main():
    data_matrix = get_data_matrices()
    print("Data loaded, beginning modified dictionary learning.")

    dict_learning_custom_matrix(data_matrix, 500)

def dict_learning_custom_matrix(data, target_dimension):
    '''
    This is the method that can run dictionary learning on the ini dataset. Currently, some optimization has been done,
    but the gradient descent itself needs to be modified to change the step size over time for better convergence.
    '''
    timer = Timer()
    alpha = .020  # step size for grad descent, .001 seems to work well
    steps_between_probings = 500
    probe_multiplier = 2
    lamb = 0


    numrows = data.shape[0]
    numcols = data.shape[1]

    dict = np.random.rand(numrows, target_dimension)

    # representations also initialized to randoms
    representation = np.random.rand(target_dimension, numcols)

    m, n, c = 1000, 1500, 1200
    A = np.random.randint(1, 50, size=(m, n))
    B = np.random.randint(1, 50, size=(n, c))

    a = []
    print("numpys matrix multiplication:")
    for i in range(10):
        timer.start()
        a.append(A @ B)
        timer.stop()

    print("numba's matrix multiplication:")
    for i in range(10):
        timer.start()
        a.append(mat_mult(A, B))
        timer.stop()

def get_data_matrices():
    data_ra1 = turn_scipy_matrix_to_numpy_matrix(sio.loadmat('dataset1.mat', struct_as_record=True)['data_sa'].squeeze())
    data_ra2 = turn_scipy_matrix_to_numpy_matrix(sio.loadmat('sin_dataset1.mat', struct_as_record=True)['data_sa'].squeeze())
    total_ra = np.hstack((data_ra1, data_ra2))
    return total_ra

def turn_scipy_matrix_to_numpy_matrix(matrix):
    # This turns the non-gradient-descent-tracked numpy array into something that can be used with autograd
    return np.array(matrix.tolist()).transpose()


@njit(parallel=True)
def mat_mult(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res


if __name__ == "__main__":
    main()
