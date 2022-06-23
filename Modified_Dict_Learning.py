import numpy as np
from autograd import grad
import scipy.io as sio
from Learning_multiprocessing import Timer
from Learning_autograd import print_confusion_matrix
from Learning_autograd import loss_function_no_lasso
import condensed

from numba import njit, prange
import matplotlib.pyplot as plt

def eval_old_dict():
    v_ra = sio.loadmat('feature_spaces_old/feature_spaces.mat', struct_as_record=True)["v_ra"]
    v_pc = sio.loadmat('feature_spaces_old/feature_spaces.mat', struct_as_record=True)["v_pc"]
    v_sa = sio.loadmat('feature_spaces_old/feature_spaces.mat', struct_as_record=True)["v_sa"]

    print(v_ra.shape)
    print(v_pc.shape)
    print(v_sa.shape)

    test_sa = get_data_matrices("SA", is_test=True)
    test_ra = get_data_matrices("RA", is_test=True)
    test_pc = get_data_matrices("PC", is_test=True)
    evaluate_old_dictionary(data_ra=test_ra, data_sa=test_sa, data_pc=test_pc)

def main():

    receptor_type = input("enter receptor type")  # options are SA (562 neurons), RA (948), PC (196)

    # this can be swapped around later to try to get more or less out of it (it's all about 1/4 dimension right now)
    if receptor_type == "PC":
        target_dimension = 49
    elif receptor_type == "RA":
        target_dimension = 237
    elif receptor_type == "SA":
        target_dimension = 170
    else:
        raise Exception("you specified an invalid receptor type lol")

    input_string = ""
    while input_string != "train" and input_string != "test":
        input_string = input("Are you training or testing? Enter one: (train/test)\n")
    if input_string == "train":
        is_training = True
    else:
        is_training = False


    # if is_training:
    data_matrix = get_data_matrices(receptor_type, is_test=False)
    test_matrix = get_data_matrices(receptor_type, is_test=True)

    print("Data loaded.")

    if is_training:
        dict_learning_custom_matrix(data_matrix, target_dimension, receptor_type)
    do_loss_comparison(test_matrix, receptor_type)


def get_orthonormality(dict):
    return np.linalg.norm((dict.transpose() @ dict - np.identity(dict.shape[1])), ord='fro')

def get_locality(dict, neuron_type, col=0, show_graph=True):
    if neuron_type == "RA":
        index = "dist_ra"
    elif neuron_type == "SA":
        index = "dist_sa"
    elif neuron_type == "PC":
        index = "dist_pc"
    else:
        raise Exception("you gave an invalid receptor type lol")

    # shape = (# neurons, 2 [location dimension])
    # first index # is the particular neuron in question
    # second index 0 is x
    # second index 1 is y
    locations = sio.loadmat('NeuronLocations.mat', struct_as_record=True)[index]
    used_x = []
    used_y = []
    unused_x = []
    unused_y = []
    for neuron_index in range(locations.shape[0]):
        if dict[neuron_index, col] != 0:
            used_x.append(locations[neuron_index, 0])
            used_y.append(locations[neuron_index, 1])
        else:
            unused_x.append(locations[neuron_index, 0])
            unused_y.append(locations[neuron_index, 1])

    x_avg = sum(used_x) / len(used_x)
    x_avg_plot = [x_avg]
    y_avg = sum(used_y) / len(used_y)
    y_avg_plot = [y_avg]

    if show_graph:
        plt.scatter(unused_x, unused_y, label="Unused Neurons")
        plt.scatter(used_x, used_y, label="Used Neurons")
        plt.scatter(x_avg_plot, y_avg_plot, label="\"Center\" of Feature")
        plt.title("Neurons Used in Feature " + str(col))
        plt.legend()
        # plt.show()

    average_distance_from_mean = 0
    for i in range(len(used_x)):
        distance = np.sqrt((used_x[i] - x_avg) ** 2 + (used_y[i] - y_avg) ** 2)
        average_distance_from_mean += distance
    average_distance_from_mean /= len(used_x)
    print("average distance from mean for feature", col, "=", average_distance_from_mean)
    return average_distance_from_mean

def compute_loss(data, dict, representation, lamb, using_alt_penalty=False, using_balanced_formulation=False, beta=None, using_sparsity_penalty=True):
    if using_alt_penalty:
        sparsity_penalty = lamb * compute_alt_penalty(dict)
    else:
        sparsity_penalty = lamb * compute_L1_penalty(dict)
    if using_balanced_formulation:
        error_term = np.linalg.norm((beta * data + 1) * (data - dict@representation))
    else:
        error_term = np.linalg.norm(data - dict @ representation)

    if not using_sparsity_penalty:
        sparsity_penalty = 0

    return sparsity_penalty + error_term


@njit
def compute_L1_penalty(dict):
    total = 0
    for row in prange(dict.shape[0]):
        for col in range(dict.shape[1]):
            total += np.abs(dict[row, col])
    return total



@njit
def compute_alt_penalty(dict):
    total = 0
    for col in range(dict.shape[1]):
        thing_to_square = 0
        for row in range(dict.shape[0]):
            thing_to_square += np.power(abs(dict[row,col]), .5)
        total += thing_to_square * thing_to_square
    return total


def do_loss_comparison(data, receptor_type):
    input_string = ""  # used to ask user if they want to use the balanced error version
    while input_string != "y" and input_string != "n":
        input_string = input("do you want to use the balanced formulation? (y/n)")
    if input_string == "y":
        file_string_1 = "balanced"
    else:
        file_string_1 = ""

    input_string = ""  # used to ask user if they want to use the balanced error version
    while input_string != "y" and input_string != "n":
        input_string = input("do you want to use the alt norm formulation? (y/n)")
    if input_string == "y":
        file_string_2 = "ALT"
    else:
        file_string_2 = ""

    print("loading dictionary:", file_string_1 + file_string_2 + "dictionary" + receptor_type + "BIG.npy")
    dict = np.load(file_string_1 + file_string_2 + "dictionary" + receptor_type + "BIG.npy")



    # don't actually load representation, needs to be remade anyways
    # representation = np.load("ALTrepresentation" + receptor_type + ".npy")

    print(dict)
    if receptor_type == "PC":
        epsilon = .10
    elif receptor_type == "SA":
        epsilon = .01
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
    
    #get_locality(dict, receptor_type, col=0)

    totals = []
    for col in range(dict.shape[1]):
        total = 0
        for row in range(dict.shape[0]):
            if dict[row,col] != 0:
                total -= -1
        totals.append(total)
    print("column totals:", totals)

    # plt.matshow(dict)
    # plt.show()


    representation = np.linalg.lstsq(dict,data)[0]

    print("loss on set before rounding: ", compute_loss(data, dict, representation, lamb=0, using_sparsity_penalty=False))

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

        if group == 3:
            plt.matshow(actual_group)
            plt.show()
            plt.matshow(reconstructed_group)
            plt.show()
            #get_locality(actual_group, receptor_type, col=11)
            #get_locality(reconstructed_group, receptor_type, col=11)


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

def evaluate_old_dictionary(data_ra, data_sa, data_pc):
    v_ra = sio.loadmat('feature_spaces_old/feature_spaces.mat', struct_as_record=True)["v_ra"]
    v_pc = sio.loadmat('feature_spaces_old/feature_spaces.mat', struct_as_record=True)["v_pc"]
    v_sa = sio.loadmat('feature_spaces_old/feature_spaces.mat', struct_as_record=True)["v_sa"]

    reconstructed_ra, reconstructed_sa, reconstructed_pc = condensed.calc(data_ra, data_sa, data_pc, v_ra, v_sa, v_pc)



    print("traditional dictionary RA test data: ")
    print_confusion_matrix(data_ra, reconstructed_ra)

    print("\ntraditional dictionary SA test data: ")
    print_confusion_matrix(data_sa, reconstructed_sa)

    print("\ntraditional dictionary PC test data: ")
    print_confusion_matrix(data_pc, reconstructed_pc)



def print_metrics(lis):
    print("min, average, and max")
    print(min(lis))
    print(round(sum(lis) / len(lis), 4))
    print(round(max(lis), 4))


def dict_learning_custom_matrix(data, target_dimension, receptor_type, dict=None):
    '''
    This method runs the dictionary learning algorithm, but with a lasso penalty term on
    '''
    print("enter lambda:")
    lamb = float(input())
    timer = Timer()
    alpha = .010  # step size for grad descent, .01 seems to work well
    steps_between_probings = 100
    probe_multiplier = 2

    numrows = data.shape[0]
    numcols = data.shape[1]

    input_string = ""   # used to ask user if they want to use the balanced error version
    while input_string != "y" and input_string != "n":
        input_string = input("do you want to use the balanced formulation? (y/n)")
    if input_string == "y":
        is_using_balanced_error = True
    else:
        is_using_balanced_error = False

    input_string = ""  # used to ask user if they want to use the alt norm version
    while input_string != "y" and input_string != "n":
        input_string = input("do you want to use the alt norm formulation? (y/n)")
    if input_string == "y":
        using_alt_penalty = True
    else:
        using_alt_penalty = False

    if is_using_balanced_error:
        beta = compute_beta(data)
        u = (beta * data + 1) * (beta * data + 1)
    else:
        u = None
        beta = None
    print("beta computed, beta = ", beta)

    if dict is None:
        dict = np.random.rand(numrows, target_dimension)

    # representations are found quickly via least squares
    timer.start()
    print("\nTiming first least squares computation: ")
    representation = np.linalg.lstsq(dict, data, rcond=None)[0]
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
                representation = np.linalg.lstsq(dict, data, rcond=None)[0]
            if iteration == 10:
                dictionary_gradient_steps = 50
            dict_gradient = compute_dictionary_gradient(dict, representation, data, lamb=lamb, using_alt_penalty=using_alt_penalty,
                                                        using_balanced_formulation=is_using_balanced_error, u=u)

            dict -= alpha * dict_gradient
            #print(dict)


            if iteration % steps_between_probings == 1:

                # display input to impatient user
                prior_loss_to_display = compute_loss(data, dict, representation, lamb=lamb,
                                                                          using_alt_penalty=using_alt_penalty,
                                                                          using_balanced_formulation=is_using_balanced_error,
                                                                          beta=beta,
                                                                          using_sparsity_penalty=True)
                prior_loss_no_sparsity_penalty = compute_loss(data, dict, representation, lamb=lamb,
                                                     using_alt_penalty=using_alt_penalty,
                                                     using_balanced_formulation=is_using_balanced_error,
                                                     beta=beta,
                                                     using_sparsity_penalty=True)
                print("\niteration:", iteration, "\nloss =", prior_loss_to_display)
                if using_alt_penalty:
                    sparsity_penalty = lamb * compute_alt_penalty(dict)
                else:
                    sparsity_penalty = lamb * compute_L1_penalty(dict)
                print("sparsity penalty loss:", sparsity_penalty)

                if loss_function_no_lasso(data, dict, representation) > 30000:
                    # this is just for debugging numerical instability, typical loss values start at 1,000 or so
                    print("dict =", dict)
                    print("dict fro norm:", np.linalg.norm(dict, ord='fro'))
                    print("repr fro norm:", np.linalg.norm(representation, ord='fro'))
                    break

                # probing step, try a few gradient descent steps with different alpha sizes
                dict_big_alpha = dict + np.zeros(dict.shape)
                dict_small_alpha = dict + np.zeros(dict.shape)
                dict_same_alpha = dict + np.zeros(dict.shape)

                for i in range(10):
                    dict_same_alpha -= alpha * compute_dictionary_gradient(dict_same_alpha, representation, data, lamb=lamb, using_alt_penalty=using_alt_penalty,
                                                        using_balanced_formulation=is_using_balanced_error, u=u)
                    dict_small_alpha -= (alpha / probe_multiplier) * compute_dictionary_gradient(dict_small_alpha, representation, data, lamb=lamb, using_alt_penalty=using_alt_penalty,
                                                        using_balanced_formulation=is_using_balanced_error, u=u)
                    dict_big_alpha -= (alpha * probe_multiplier) * compute_dictionary_gradient(dict_big_alpha, representation, data, lamb=lamb, using_alt_penalty=using_alt_penalty,
                                                        using_balanced_formulation=is_using_balanced_error, u=u)
                loss_big = compute_loss(data, dict_big_alpha, representation, lamb=lamb,
                                        using_alt_penalty=using_alt_penalty,
                                        using_balanced_formulation=is_using_balanced_error,
                                        beta=beta,
                                        using_sparsity_penalty=True)
                loss_small = compute_loss(data, dict_small_alpha, representation, lamb=lamb,
                                          using_alt_penalty=using_alt_penalty,
                                          using_balanced_formulation=is_using_balanced_error,
                                          beta=beta,
                                          using_sparsity_penalty=True)
                loss_same = compute_loss(data, dict_same_alpha, representation, lamb=lamb,
                                         using_alt_penalty=using_alt_penalty,
                                         using_balanced_formulation=is_using_balanced_error,
                                         beta=beta,
                                         using_sparsity_penalty=True)


                # update alpha based on result of probes
                if prior_loss_no_sparsity_penalty < loss_small:
                    alpha /= probe_multiplier
                    print(f"All probes were worse, decreasing alpha to {round(alpha, 6)}")
                elif loss_big < loss_same and loss_big < loss_small:
                    alpha *= probe_multiplier
                    print(f"Probe complete. Alpha grows to {round(alpha, 6)}")
                elif loss_small < loss_same and loss_small < loss_big:
                    alpha /= probe_multiplier
                    print(f"Probe complete. Alpha shrinks to {round(alpha, 6)}")
                else:
                    print(f"Probe complete. Alpha stays at {round(alpha, 6)}")

    finally:
        print("Learning finished, Saving Dictionary")
        #reconstructed_matrix = dict @ representation
        #reconstructed_matrix[reconstructed_matrix >= 0.50] = 1
        #reconstructed_matrix[reconstructed_matrix < 0.50] = 0


        #prints out the whole dictionary instead of abbreviated
        #np.set_printoptions(threshold=np.inf)
        #print(dict)
        #np.set_printoptions(threshold=1000)


        # print_confusion_matrix(data, reconstructed_matrix)

        # extra_string is just used to make saved dictionary file names unique
        if is_using_balanced_error:
            extra_string = "balanced"
        else:
            extra_string = ""

        if using_alt_penalty:
            file_string2 = "ALT"
        else:
            file_string2 = ""

        # if lamb == 0 (floating point), call the matrix VANILLA, for no Sparsity Penalty Function
        if lamb <= .00000001:
            file_string2 = "VANILLA"
        np.save(extra_string + file_string2 + "dictionary" + receptor_type + "BIG.npy", dict)

        print("Dictionary saved, Beginning testing")

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

def compute_dictionary_gradient(dict, representation, data, lamb=0, using_alt_penalty=False, using_balanced_formulation=False, u=None):
    '''
    This computes the gradient that the dictionary should follow.
    lamb is the coefficient for the wacky shit we're adding on at the end [the 1 norm to get sparsity of dict cols]
        [or the 1/2 norm that isn't actually a norm]

    the gradient is two things being added together: an error term and a sparsity penalty term
    ------------------------------------------------------------------------------------------------
    standard error term: ||x - Dr||^2
    balanced formulation error term: ||(Bx + 1) {hadamard product} (x - Dr)||^2

    standard penalty term: sum over all 'i' of ||d_i||_1  [1 norm of dictionary columns]
    alt penalty term: sum over all i of g(d_i), where g is given by the equation for the Lp norm, p=1/2
        *note, this means g is neither convex nor a norm, as it violates the triangle inequality
    '''


    if using_balanced_formulation:
        error_term = (u * (dict @ representation) - data * u) @ np.transpose(representation)
    else:
        error_term = (dict @ representation - data) @ representation.transpose()

    if lamb < .00000001:
        alt_penalty = 0
    elif not using_alt_penalty:
        lasso_term = np.zeros(dict.shape) + lamb  # broadcasts lasso gradient to all terms, will change later for other term
        lasso_term = np.multiply(lasso_term, np.sign(dict))
        total_error = error_term + lasso_term
    else:
        alt_penalty = lamb * np.sign(dict) * compute_alt_penalty_gradient(dict)
        total_error = error_term + alt_penalty
    return total_error * total_error.shape[1] / np.linalg.norm(total_error, ord='fro')


@njit
def compute_beta(data):
    '''
    This method computes the beta constant used in the balanced formulation
    Only needs to be computed once
    '''
    num_ones = 0
    for row in range(data.shape[0]):
        for col in prange(data.shape[1]):
            if data[row,col] == 1:
                num_ones += 1

    total = data.shape[0] * data.shape[1]
    print("total = ", total)
    print("num_ones = ", num_ones)
    return total / num_ones - 2



@njit(parallel=True)
def compute_alt_penalty_gradient(dict):
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
        lin_matrix = sio.loadmat('total_lin_dataset_test27.mat', struct_as_record=True)[index]
        sin_matrix = sio.loadmat('total_sin_dataset_test135.mat', struct_as_record=True)[index]
    else:
        lin_matrix = sio.loadmat('total_lin_dataset_27.mat', struct_as_record=True)[index]
        sin_matrix = sio.loadmat('total_sin_dataset_162.mat', struct_as_record=True)[index]

    for i in range(lin_matrix.shape[0]):
        minimatrix = lin_matrix[i, :, :].squeeze()
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
