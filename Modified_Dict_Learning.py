import numpy as np
from autograd import grad
import scipy.io as sio
from Learning_multiprocessing import Timer
from Learning_autograd import print_confusion_matrix
from Learning_autograd import loss_function_no_lasso
from Learning_autograd import loss_function

from numba import njit, prange

def main():
    data_matrix = get_data_matrices()
    print("Data loaded, beginning modified dictionary learning.")

    # do_loss_comparison(data_matrix)
    dict_learning_custom_matrix(data_matrix, 140)

def do_loss_comparison(data):
    dict = np.load("dictionary.npy")
    representation = np.load("representation.npy")
    print("loss =", loss_function_no_lasso(data,dict,representation))

    print(dict)
    epsilon = .0001
    average_total = 0
    for col in range(dict.shape[1]):
        tot = 0
        for row in range(dict.shape[0]):
            if abs(dict[row, col]) > epsilon:
                tot -= -1   # if only python had the "++" operator
        average_total += tot
    average = average_total / dict.shape[1]  # divide by number of columns to get avg number of non-zeros in each col
    print("\naverage used in column:", average)
    print("total in column:", dict.shape[0])
    print("sparsity percent:", round(100 * average / dict.shape[1], 4))



    reconstructed_matrix = dict @ representation
    reconstructed_matrix[reconstructed_matrix >= 0.50] = 1
    reconstructed_matrix[reconstructed_matrix < 0.50] = 0
    print_confusion_matrix(data, reconstructed_matrix)

def dict_learning_custom_matrix(data, target_dimension):
    '''
    This is the method that can run dictionary learning on the ini dataset. Currently, some optimization has been done,
    but the gradient descent itself needs to be modified to change the step size over time for better convergence.
    '''
    print("enter lambda")
    lamb = float(input())
    timer = Timer()
    alpha = .010  # step size for grad descent, .001 seems to work well
    steps_between_probings = 100
    probe_multiplier = 2

    numrows = data.shape[0]
    numcols = data.shape[1]

    dict = np.random.rand(numrows, target_dimension)

    # representations are found quickly via least squares
    representation = np.linalg.lstsq(dict, data)[0]
    print(representation.shape)

    # This tells us how often to recompute the representation matrix (using least squares)
    dictionary_gradient_steps = 1
    is_done = False
    max_iterations = 2000
    # the try block is for ctrl C to terminate the training process while still printing results
    try:
        for iteration in range(1, 100000000):

            # input handling to make life easier in perseus terminal, only happens after specified number of iterations
            while max_iterations <= iteration:
                try:
                    print("enter num iters to do, enter 0 if you're done:")
                    input_iters = input()
                    max_iterations = int(input_iters)
                    if max_iterations == 0:
                        is_done = True
                        break
                    print("enter alpha, current alpha is ", round(alpha,4))
                    alpha = float(input())
                except:
                    print("invalid num iters or alpha")
            if is_done:
                break



            dict *= dict.shape[1] / np.linalg.norm(dict, ord='fro')

            if iteration % dictionary_gradient_steps == 0:
                representation = np.linalg.lstsq(dict, data)[0]
            if iteration == 10:
                dictionary_gradient_steps = 50
            dict_gradient = compute_dictionary_gradient(dict, representation, data, lamb=lamb)

            dict -= alpha * dict_gradient
            #print(dict)


            if iteration % steps_between_probings == 1:

                # display input to impatient user
                print("\niteration:", iteration, "\nloss =", loss_function_no_lasso(data, dict, representation))
                print("lasso loss:", loss_function(data,dict,representation))
                if loss_function_no_lasso(data, dict, representation) < 20:
                    break
                if loss_function_no_lasso(data, dict, representation) > 150:
                    print("dict =", dict)
                    print("dict fro norm:", np.linalg.norm(dict, ord='fro'))
                    print("repr fro norm:", np.linalg.norm(representation, ord='fro'))
                    break

                '''
                # probing step, try a few gradient descent steps with different alpha sizes
                dict_big_alpha = dict + np.zeros(dict.shape)
                dict_small_alpha = dict + np.zeros(dict.shape)
                dict_same_alpha = dict + np.zeros(dict.shape)
    
                for i in range(10):
                    dict_same_alpha -= alpha * compute_dictionary_gradient(dict_same_alpha, representation, data)
                    dict_small_alpha -= (alpha / probe_multiplier) * compute_dictionary_gradient(dict_small_alpha, representation, data)
                    dict_big_alpha -= (alpha * probe_multiplier) * compute_dictionary_gradient(dict_big_alpha, representation, data)
                loss_big = loss_function_no_lasso(data, dict_big_alpha, representation)
                loss_small = loss_function_no_lasso(data, dict_small_alpha, representation)
                loss_same = loss_function_no_lasso(data, dict_same_alpha, representation)
    
                # update alpha based on result of probes
                if loss_big < loss_same and loss_big < loss_small:
                    alpha *= probe_multiplier
                    print(f"Probe complete. Alpha grows to {round(alpha, 5)}")
                elif loss_small < loss_same and loss_small < loss_big:
                    alpha /= probe_multiplier
                    print(f"Probe complete. Alpha shrinks to {round(alpha, 5)}")
                else:
                    print(f"Probe complete. Alpha stays at {round(alpha, 5)}")
                '''
    finally:
        reconstructed_matrix = dict @ representation
        reconstructed_matrix[reconstructed_matrix >= 0.50] = 1
        reconstructed_matrix[reconstructed_matrix < 0.50] = 0


        #prints out the whole dictionary instead of abbreviated
        np.set_printoptions(threshold=np.inf)
        print(dict)
        np.set_printoptions(threshold=1000)


        print_confusion_matrix(data, reconstructed_matrix)
        np.save("dictionary.npy", dict)
        np.save("representation.npy", representation)

        # sparsity examination time
        epsilon = .001
        average_total = 0
        for col in range(dict.shape[1]):
            tot = 0
            for row in range(dict.shape[0]):
                if abs(dict[row, col]) > epsilon:
                    tot -= -1  # if only python had the "++" operator
            average_total += tot
        average = average_total / dict.shape[1]  # divide by number of columns to get avg number of non-zeros in each col
        print("\naverage used in column:", average)
        print("total in column:", dict.shape[0])
        print("sparsity percent:", round(100 * average / dict.shape[1], 4))

def compute_dictionary_gradient(dict, representation, data, lamb=0):
    '''
    Anyways, this computes the gradient that the dictionary should follow.
    lamb is the coefficient for the wacky shit we're adding on at the end [the 1 norm to get sparsity of dict cols]
    '''

    # This is kind of disgusting but it lets me swap between np's matmul and a custom numba matmul based on
    #    whichever is more efficient
    error_term = (dict @ representation - data) @ representation.transpose()
    lasso_term = np.zeros(dict.shape) + lamb  # broadcasts lasso gradient to all terms, will change later for other term
    lasso_term = np.multiply(lasso_term, np.sign(dict))
    total_error = error_term + lasso_term
    return total_error * total_error.shape[1] / np.linalg.norm(total_error, ord='fro')

def mat_mul2(A, B):
    return A @ B


def get_data_matrices():
    data_sa1 = turn_scipy_matrix_to_numpy_matrix(sio.loadmat('dataset1.mat', struct_as_record=True)['data_sa'].squeeze())
    data_sa2 = turn_scipy_matrix_to_numpy_matrix(sio.loadmat('sin_dataset1.mat', struct_as_record=True)['data_sa'].squeeze())
    total_sa = np.hstack((data_sa1, data_sa2))
    print(total_sa.shape)
    return total_sa


def turn_scipy_matrix_to_numpy_matrix(matrix):
    # This turns the non-gradient-descent-tracked numpy array into something that can be used with autograd
    return np.array(matrix.tolist()).transpose()


@njit(parallel=True)
def mat_multa2(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res


if __name__ == "__main__":
    main()
