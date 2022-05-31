import numpy as np
from autograd import grad
import scipy.io as sio
from Learning_multiprocessing import Timer
from Learning_autograd import print_confusion_matrix
from Learning_autograd import loss_function_no_lasso
from Learning_autograd import loss_function

from numba import njit, prange
import matplotlib.pyplot as plt

def main():
    receptor_type = "RA"  # options are SA (562 neurons), RA (948), PC (196)

    # this can be swapped around later to try to get more or less out of it (it's all about 1/4 dimension right now)
    if receptor_type == "PC":
        target_dimension = 49
    elif receptor_type == "RA":
        target_dimension = 237
    elif receptor_type == "SA":
        target_dimension = 170

    data_matrix = get_data_matrices(receptor_type, is_test=False)
    test_matrix = get_data_matrices(receptor_type, is_test=True)

    print("Data loaded, beginning modified dictionary learning.")

    dict_learning_custom_matrix(data_matrix, target_dimension, receptor_type)
    do_loss_comparison(test_matrix, receptor_type)

def do_loss_comparison(data, receptor_type):
    dict = np.load("ALTdictionary" + receptor_type + "BIG.npy")

    # don't actually load representation, needs to be remade anyways
    # representation = np.load("ALTrepresentation" + receptor_type + ".npy")

    print(dict)
    if receptor_type == "PC":
        epsilon = .04
    elif receptor_type == "SA":
        epsilon = .005
    elif receptor_type == "RA":
        epsilon = .01
    else:
        raise Exception("you specified an invalid receptor type lol")


    average_total = 0
    dictionary_column_totals = []

    # loop through all columns, get the average number of
    for col in range(dict.shape[1]):
        tot = 0
        for row in range(dict.shape[0]):
            if abs(dict[row, col]) > epsilon:
                tot -= -1   # if only python had the "++" operator
            else:
                dict[row,col] = 0
        average_total += tot
        dictionary_column_totals.append(tot)
    
    '''
    # use only the top n most influential of each row
    n = 17
    for col in range(dict.shape[1]):
        # idea: store top n indices, sorted, every time checking new, see if abs(dict[index]) is greater than lowest
        # in top_n. If so, sort it into top_n, removing the lowest
        # after top n indices are gotten, set all dict elements to zero that aren't those top n
        top_n = []
        for i in range(n):
            top_n.append(i)

        print("new col")
        print(top_n)


        def get_item(row):
            return dict[row, col]
        top_n.sort(key=get_item)

        for row in range(n, dict.shape[0]):
            # if this dict[row,col] element is more influencial than the others, remove smallest of top_n, put new index in
            if abs(dict[row, col]) > top_n[0]:
                top_n.pop(0)
                # sort new guy into top_n
                top_n.append(row)
                top_n.sort(key=get_item)

            print(top_n)

        for row in range(dict.shape[0]):
            if row not in top_n:
                dict[row,col] = 0

        # now, top_n in column have been found
    '''

    for col in range(dict.shape[1]):
        total = 0
        for row in range(dict.shape[0]):
            if dict[row,col] != 0:
                total -= -1
        print(total)


    plt.matshow(dict)
    plt.show()


    representation = np.linalg.lstsq(dict,data)[0]
    average = average_total / dict.shape[1]  # divide by number of columns to get avg number of non-zeros in each col
    print("\naverage used in column:", average)
    print("total in column:", dict.shape[0])
    print("sparsity percent:", round(100 * average / dict.shape[0], 4))

    # print("loss =", loss_function_no_lasso(data,dict,representation))
    reconstructed_matrix = dict @ representation

    if receptor_type == "SA":
        cutoff = .36
    elif receptor_type == "PC":
        cutoff = .40
    else:
        cutoff = .36
    reconstructed_matrix[reconstructed_matrix >= cutoff] = 1
    reconstructed_matrix[reconstructed_matrix < cutoff] = 0
    print_confusion_matrix(data, reconstructed_matrix)

    accuracy_list = []
    precision_list = []
    recall_list = []
    print(range(reconstructed_matrix.shape[1] // 1000))
    for group in range(reconstructed_matrix.shape[1] // 1000):
        reconstructed_column_list = []
        actual_column_list = []
        for col in range(group * 1000, (group + 1) * 1000):
            reconstructed_column_list.append(reconstructed_matrix[:, col])
            actual_column_list.append(data[:, col].transpose())
        reconstructed_group = np.vstack(tuple(reconstructed_column_list)).transpose()
        actual_group = np.vstack(tuple(actual_column_list)).transpose()
        print("\n group:", group)

        # data analysis for each one
        acc, prec, recall = print_confusion_matrix(actual_group, reconstructed_group)
        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(recall)

        if group == 54:
            plt.matshow(actual_group)
            plt.show()
            plt.matshow(reconstructed_group)
            plt.show()

    # we'll look at min, max, and average
    print("\nAccuracy")
    print_metrics(accuracy_list)

    print("\nPrecision")
    print_metrics(precision_list)

    print("\nRecall")
    print_metrics(recall_list)

    print("\nDictionary Column Sparsity")
    for i, value in enumerate(dictionary_column_totals):
        dictionary_column_totals[i] = value / dict.shape[0]
    print_metrics(dictionary_column_totals)




def print_metrics(lis):
    print("min, average, and max")
    print(min(lis))
    print(round(sum(lis) / len(lis), 4))
    print(round(max(lis), 4))


def dict_learning_custom_matrix(data, target_dimension, receptor_type, dict=None):
    '''
    This method runs the dictionary learning algorithm, but with a lasso penalty term on
    '''
    print("enter lambda")
    lamb = float(input())
    timer = Timer()
    alpha = .010  # step size for grad descent, .01 seems to work well
    steps_between_probings = 100
    probe_multiplier = 2

    numrows = data.shape[0]
    numcols = data.shape[1]

    if dict is None:
        dict = np.random.rand(numrows, target_dimension)

    # representations are found quickly via least squares
    timer.start()
    print("\nTiming first least squares computation: ")
    representation = np.linalg.lstsq(dict, data)[0]
    timer.stop()

    print(representation.shape)

    # This tells us how often to recompute the representation matrix (using least squares)
    dictionary_gradient_steps = 1
    is_done = False
    max_iterations = 1

    timer.start()
    # the try block is for ctrl C to terminate the training process while still printing results [and saving matrices]
    try:
        for iteration in range(1, 100000000):

            # input handling to make life easier in perseus terminal, only happens after specified number of iterations
            while max_iterations <= iteration:
                try:
                    timer.stop()
                except:
                    pass
                try:
                    print("enter num iters to do, enter 0 if you're done:")
                    input_iters = input()
                    max_iterations = int(input_iters)
                    if max_iterations == 0:
                        is_done = True
                        break
                    print("enter alpha, current alpha is ", round(alpha,4))
                    alpha = float(input())
                    timer.start()
                except:
                    print("invalid num iters or alpha")
            if is_done:
                break



            dict *= dict.shape[1] / np.linalg.norm(dict, ord='fro')

            if iteration % dictionary_gradient_steps == 0:
                representation = np.linalg.lstsq(dict, data)[0]
            if iteration == 10:
                dictionary_gradient_steps = 50
            dict_gradient = compute_dictionary_gradient(dict, representation, data, lamb=lamb, using_alt_penalty=True)

            dict -= alpha * dict_gradient
            #print(dict)


            if iteration % steps_between_probings == 1:

                # display input to impatient user
                print("\niteration:", iteration, "\nloss =", loss_function_no_lasso(data, dict, representation))
                print("lasso loss:", loss_function(data,dict,representation,lamb))

                if loss_function_no_lasso(data, dict, representation) > 30000:
                    # this is just for debugging numerical instability, typical loss values start at 300 or so
                    print("dict =", dict)
                    print("dict fro norm:", np.linalg.norm(dict, ord='fro'))
                    print("repr fro norm:", np.linalg.norm(representation, ord='fro'))
                    break

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

    finally:
        reconstructed_matrix = dict @ representation
        reconstructed_matrix[reconstructed_matrix >= 0.50] = 1
        reconstructed_matrix[reconstructed_matrix < 0.50] = 0


        #prints out the whole dictionary instead of abbreviated
        #np.set_printoptions(threshold=np.inf)
        #print(dict)
        #np.set_printoptions(threshold=1000)


        print_confusion_matrix(data, reconstructed_matrix)
        np.save("ALTdictionary" + receptor_type + "BIG.npy", dict)
        np.save("ALTrepresentation" + receptor_type + "BIG.npy", representation)

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
        print("sparsity percent:", round(100 * average / dict.shape[0], 4))

def compute_dictionary_gradient(dict, representation, data, lamb=0, using_alt_penalty=False):
    '''
    This computes the gradient that the dictionary should follow.
    lamb is the coefficient for the wacky shit we're adding on at the end [the 1 norm to get sparsity of dict cols]
        [or the 1/2 norm that isn't actually a norm]
    '''

    # This is kind of disgusting but it lets me swap between np's matmul and a custom numba matmul based on
    #    whichever is more efficient
    error_term = (dict @ representation - data) @ representation.transpose()
    if not using_alt_penalty:
        lasso_term = np.zeros(dict.shape) + lamb  # broadcasts lasso gradient to all terms, will change later for other term
        lasso_term = np.multiply(lasso_term, np.sign(dict))
        total_error = error_term + lasso_term
    else:
        alt_penalty = lamb * np.sign(dict) * compute_alt_penalty(dict)
        total_error = error_term + alt_penalty
    return total_error * total_error.shape[1] / np.linalg.norm(total_error, ord='fro')

@njit(parallel=True)
def compute_alt_penalty(dict):
    alt_penalty = np.zeros(dict.shape)
    for col in prange(dict.shape[1]):
        # first compute sum term
        tot_sum = 0
        for row in prange(dict.shape[0]):
            tot_sum += np.sqrt(abs(dict[row, col]) + 1)

        # now can compute gradient for this column in particular
        for row in prange(dict.shape[0]):
            alt_penalty[row, col] = tot_sum / np.sqrt(abs(dict[row, col]) + 1)

    return alt_penalty

def mat_mul2(A, B):
    return A @ B


def get_data_matrices(receptor_type, is_test):
    matrices_to_stack_horizontally = []

    if receptor_type == "SA":
        index = 'data_sa'
    if receptor_type == "RA":
        index = 'data_ra'
    if receptor_type == "PC":
        index = 'data_pc'
    if is_test:
        lin_matrix = sio.loadmat('total_lin_dataset_test4.mat', struct_as_record=True)[index]
        sin_matrix = sio.loadmat('total_sin_dataset_test16.mat', struct_as_record=True)[index]
    else:
        lin_matrix = sio.loadmat('total_lin_dataset_27.mat', struct_as_record=True)[index]
        sin_matrix = sio.loadmat('total_sin_dataset_162.mat', struct_as_record=True)[index]

    for i in range(lin_matrix.shape[0]):
        minimatrix = lin_matrix[i,:,:].squeeze()
        matrices_to_stack_horizontally.append(turn_scipy_matrix_to_numpy_matrix(minimatrix))

    for i in range(sin_matrix.shape[0]):
        minimatrix = sin_matrix[i, :, :].squeeze()
        matrices_to_stack_horizontally.append(turn_scipy_matrix_to_numpy_matrix(minimatrix))

    final_data = np.hstack(tuple(matrices_to_stack_horizontally))
    print("final data shape =", final_data.shape)
    return final_data


def turn_scipy_matrix_to_numpy_matrix(matrix):
    # This turns the non-gradient-descent-tracked scipy array into something that can be used with autograd
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
