import autograd.numpy as np
from autograd import grad
import scipy.io as sio
from Learning_multiprocessing import Timer


def print_confusion_matrix(act, pred):
    '''
    This function prints the confusion matrix from machine learning, as well as the precision and accuracy metrics
    given the actual and predicted matrices.
    '''
    total_firing = np.sum(act)
    total_zeroes = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total_correct = 0
    total_incorrect = 0

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if act[i, j] == 1:
                # ACTUAL POSITIVE
                if pred[i, j] == 1:
                    # TRUE POSITIVE
                    TP += 1
                else:
                    # FALSE NEGATIVE
                    # IE: PREDICTED IS 0 BUT ACTUAL IS 1
                    FN += 1
            else:
                # ACTUAL NEGATIVE
                total_zeroes += 1
                if pred[i, j] == 0:
                    # TRUE NEGATIVE
                    TN += 1
                else:
                    # FALSE POSITIVE
                    # IE: PREDICTED IS 1 BUT ACTUAL IS 0
                    FP += 1

    if TP != 0 or FN != 0:
        tpr = round(TP / (TP + FN), 4)
    else:
        tpr = 0

    if TN != 0 or FP != 0:
        tnr = round(TN / (TN + FP), 4)
    else:
        tnr = 0

    if TN != 0 or FP != 0:
        fpr = round(FP / (TN + FP), 4)
    else:
        fpr = 0

    if TP != 0 or FN != 0:
        fnr = round(FN / (FN + TP), 4)
    else:
        fnr = 0

    if TP != 0 or FP != 0:
        prec = round(TP / (TP + FP), 4)
    else:
        prec = 0

    print("accuracy =", round((TP + TN) / (TP + TN + FP + FN), 4))
    print("precision = ", prec)
    print("false positive rate =", fpr, "\n"
          "false negative rate =", fnr, "\n"
          "true negative rate =", tnr, "\n"
          "true positive rate =", tpr, "\n")


def Func2(v1, v2):
    '''
    multi-input test
    '''
    total = np.array([0.0])
    for i in range(v1.shape[0]):
        total += v1[i] * v2[i]
    return total

def Norm2(vector):
    '''
    This is a quick 2-norm function for testing the autograd on
    Seems to work great for 1-input functions that end in scalars
    '''
    sum = np.array([0.0])
    for dim1 in range(vector.shape[0]):
        for dim2 in range(vector.shape[1]):
            sum += 2 * vector[dim1, dim2]
    return sum


def example1():
    autograd_fct = grad(Norm2)

    v = np.array([1.0, 2.0])
    v2 = np.array([3.0, 4.0])

    grad_parameter_0 = grad(Func2, 0)
    grad_parameter_1 = grad(Func2, 1)

    print(Func2(v, v2))
    print(grad_parameter_0(v, v2))
    print(grad_parameter_1(v, v2))

    # output = Norm2(v)
    # print(v)
    # print(output)
    # print(autograd_fct(v))

    # print(output)


'''
This function calculates the loss function in dictionary learning, with a little extra sauce:
Here, instead of making the feature vectors sparse [only use a few features to represent each piece of data]
We add  'max{1-norm of all dict column vectors}' as a loss term to incentivize only using fewer data elts per feature
This function is mainly going to be used to compute a gradient from using autograd へuへ 
    
    Params:
    data - the data matrix (each column is a single sample to learn a representation for 
    dict - the dictionary
    representation - the representation matrix [ideally lower dimentional than the data matrix
    power - the power to raise the lasso term by. [try pluggin in 1/2 !] 
'''

def loss_function_half_norm(data, dict, representation, inner_norm=1.0):
    #define the power for more efficiency in a sec
    inner_norm = np.array([inner_norm])

    # sparsity weight term
    lamb = .5

    # First, we'll compute the loss term: [2 norm of the 1 norms of the columns]
    lasso_term = np.array([0.0])

    for col in range(dict.shape[1]):
        thing_to_square = np.array([0.0])
        for row in range(dict.shape[0]):
            thing_to_square += np.power(np.absolute(dict[row, col]), inner_norm)

        lasso_term += np.power(thing_to_square, 1/inner_norm)
    lasso_term = lamb * lasso_term


    reconstruction_term = np.linalg.norm(data - (dict @ representation))
    return reconstruction_term + lasso_term

def loss_function(data, dict, representation):

    # sparsity weight term
    lamb = 10

    # First, we'll compute the loss term: [2 norm of the 1 norms of the columns]
    lasso_term = np.array([0.0])

    for col in range(dict.shape[1]):
        thing_to_square = np.array([0.0])
        for row in range(dict.shape[0]):
            if dict[row, col] >= 0:
                thing_to_square += dict[row, col]
            else:
                thing_to_square -= dict[row, col]
        lasso_term += thing_to_square * thing_to_square
    lasso_term = lamb * lasso_term

    reconstruction_term = np.linalg.norm(data - (dict @ representation))
    return reconstruction_term + lasso_term

def loss_function_no_lasso(data, dict, representation):
    '''
    This is used for a gradient calculation, but does not include the lasso loss term
    Instead, that is calculated separately for efficiency
    '''

    reconstruction_term = np.linalg.norm(data - (dict @ representation))
    return reconstruction_term


def toy_test_dict_learning():
    # goal: encode this shit into a two-dimensional-feature matrix using a custom dictionary learning algo
    # dimension = 5x4, same as how the code following would imply
    data = np.array([[1, 1, 2, 2],
                     [2, 2, 1, 1],
                     [3, 3, 1, 1],
                     [4, 4, 3, 3],
                     [2, 2, 4, 4]])

    # dictionary initialized to randoms
    dict = np.random.rand(5, 2)

    # representations also initialized to randoms
    representation = np.random.rand(2, 4)

    # this is a function that gets gradient of dict given parameters for loss_function
    calc_dict_gradient = grad(loss_function_half_norm, 1)
    calc_representation_gradient = grad(loss_function_half_norm, 2)

    alpha = .002  # step size for grad descent, .001 seems to work well

    # This time, just alternate between the two things being optimized, maybe converge->converge->converge... later
    optimizing_dict_now = False
    for i in range(10000):

        #if i % 5 == 0:
        optimizing_dict_now = not optimizing_dict_now


        if optimizing_dict_now:
            dict_grad = calc_dict_gradient(data, dict, representation, 2.0)
            dict -= alpha * dict_grad
        else:
            repr_grad = calc_representation_gradient(data, dict, representation, 2.0)
            representation -= alpha * repr_grad
        if i % 1000 == 0:
            print("\nreconstructed matrix =", dict @ representation)

    print("\ndict =", dict)
    print("\nrepresentation =", representation)
    print("\nreconstructed matrix =", dict @ representation)

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

    # this is a function that gets gradient of dict given parameters for loss_function
    calc_dict_gradient = grad(loss_function_no_lasso, 1)
    calc_representation_gradient = grad(loss_function_no_lasso, 2)


    # This time, just alternate between the two things being optimized, maybe converge->converge->converge... later
    optimizing_dict_now = False

    # alpha  (alpha * 2) ( alpha / 2)
    #
    timer.start()
    # we're going to use a gradient descent with a changing step size
    # first, it probes a few steps of current alpha, then a larger alpha, then a smaller alpha. (think 5-10 steps)
    # then, alpha becomes whichever of those options works best for several iterations (think 100 or so)
    # probing periodically allows for arbitrarily large or small steps over time.
    for i in range(100000):
        if i % 20 == 0:
            optimizing_dict_now = not optimizing_dict_now
        if i % steps_between_probings == 0:
            timer.stop()
            timer.start()
            # we'll use these variables for probing
            initial_dict, initial_representation = dict, representation
            initial_optimizing_dict_now = optimizing_dict_now
            loss_current = None
            loss_larger = None
            loss_smaller = None

            # ------
            # PROBE 1: CURRENT STEP SIZE
            # ------
            for k in range(10):
                if optimizing_dict_now:
                    dict_grad = calc_dict_gradient(data, dict, representation)

                    # calculate the lasso loss gradient manually for efficiency [because it's easy]
                    for row in range(dict.shape[0]):
                        for col in range(dict.shape[1]):
                            dict_grad[row, col] += lamb * dict[row, col]

                    dict -= alpha * dict_grad
                else:
                    repr_grad = calc_representation_gradient(data, dict, representation)
                    representation -= alpha * repr_grad
                optimizing_dict_now = not optimizing_dict_now

            loss_current = loss_function(data, dict, representation)

            # ------
            # PROBE 2: LARGER STEP SIZE
            # ------
            dict = initial_dict
            representation = initial_representation
            optimizing_dict_now = initial_optimizing_dict_now
            for k in range(10):
                if optimizing_dict_now:
                    dict_grad = calc_dict_gradient(data, dict, representation)

                    # calculate the lasso loss gradient manually for efficiency [because it's easy]
                    for row in range(dict.shape[0]):
                        for col in range(dict.shape[1]):
                            dict_grad[row, col] += lamb * dict[row, col]

                    dict -= (alpha / probe_multiplier) * dict_grad
                else:
                    repr_grad = calc_representation_gradient(data, dict, representation)
                    representation -= (alpha / probe_multiplier) * repr_grad
                optimizing_dict_now = not optimizing_dict_now
            loss_smaller = loss_function(data, dict, representation)
            # ------
            # PROBE 3: LARGER STEP SIZE
            # ------
            dict = initial_dict
            representation = initial_representation
            optimizing_dict_now = initial_optimizing_dict_now
            for k in range(10):
                if optimizing_dict_now:
                    dict_grad = calc_dict_gradient(data, dict, representation)

                    # calculate the lasso loss gradient manually for efficiency [because it's easy]
                    for row in range(dict.shape[0]):
                        for col in range(dict.shape[1]):
                            dict_grad[row, col] += lamb * dict[row, col]

                    dict -= (alpha * probe_multiplier) * dict_grad
                else:
                    repr_grad = calc_representation_gradient(data, dict, representation)
                    representation -= (alpha * probe_multiplier) * repr_grad
                optimizing_dict_now = not optimizing_dict_now
            loss_larger = loss_function(data, dict, representation)

            # reset all values for actual gradient descent
            dict = initial_dict
            representation = initial_representation
            optimizing_dict_now = initial_optimizing_dict_now

            # modify step size based on results of the probing
            if loss_larger <= loss_smaller and loss_larger <= loss_current:
                alpha = alpha * probe_multiplier
                print("alpha grows to:", round(alpha, 7))
            elif loss_smaller <= loss_larger and loss_smaller <= loss_current:
                alpha = alpha / probe_multiplier
                print("alpha shrinks to:", round(alpha, 7))
            else:
                print("alpha stays the same:", round(alpha, 7))






        if optimizing_dict_now:
            dict_grad = calc_dict_gradient(data, dict, representation)

            # calculate the lasso loss gradient manually for efficiency [because it's easy]
            for row in range(dict.shape[0]):
                for col in range(dict.shape[1]):
                    dict_grad[row, col] += lamb * dict[row, col]

            dict -= alpha * dict_grad
        else:
            repr_grad = calc_representation_gradient(data, dict, representation)
            representation -= alpha * repr_grad


        if i % steps_between_probings == 0:
            print("iteration", i)
            #print("reconstructed =", (dict @ representation)[1, 1034])
            #print("actual =", data[1, 1034])
            #print_confusion_matrix(data, dict @ representation)
            print(loss_function_no_lasso(data, dict, representation), "\n")


    # when all learning is completed, get reconstruction and do the confusion matrix on it
    reconstructed_matrix = dict @ representation
    reconstructed_matrix[reconstructed_matrix >= 0.50] = 1
    reconstructed_matrix[reconstructed_matrix < 0.50] = 0

    print_confusion_matrix(data, reconstructed_matrix)

    column_nums = []
    for col in range(dict.shape[0]):
        column_nums.append(0)
    # get number of elements in each column
    for col in range(dict.shape[1]):
        num_in_col = 0
        for row in range(dict.shape[0]):
            if abs(dict[row, col]) < .01:
                num_in_col += 1
        column_nums[num_in_col] += 1
    print("lambda = ", lamb)
    print("number of columns with [index] number of elements:", column_nums)
    np.save("saved_dict.npy")


def turn_scipy_matrix_to_numpy_matrix(matrix):
    # This turns the non-gradient-descent-tracked numpy array into something that can be used with autograd
    return np.array(matrix.tolist()).transpose()



def test_speeds():
    timer = Timer()
    m, n, c = 1000, 1500, 1200
    A = 50 * np.random.rand(m, n)
    B = 50 * np.random.rand(n, c)
    print("A's shape is ", A.shape)
    print(A)
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

def main():

    # we merge the linear and sin datasets here (1 second of each)
    data_ra1 = turn_scipy_matrix_to_numpy_matrix(sio.loadmat('dataset1.mat', struct_as_record=True)['data_sa'].squeeze())
    data_ra2 = turn_scipy_matrix_to_numpy_matrix(sio.loadmat('sin_dataset1.mat', struct_as_record=True)['data_sa'].squeeze())
    total_ra = np.hstack((data_ra1, data_ra2))
    print(data_ra1.shape)
    print(data_ra2.shape)
    print(total_ra.shape)
    dict_learning_custom_matrix(total_ra, target_dimension=125)

if __name__ == "__main__":
    main()
