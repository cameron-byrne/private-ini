import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import sklearn.decomposition as skd


def calc(data_ra, data_sa, data_pc, v_ra, v_sa, v_pc):
    """
    returns reconstructed data matrices, given original data matrix and feature space.
    """
    # POPULATION EXPERIMENT WHERE WE SHUFFLE THE NEURONS OF THE DATA
    # UNCOMMENT BELOW FOR RANDOMIZING THE NEURONS

    # np.random.shuffle(data_ra)
    # np.random.shuffle(data_sa)
    # np.random.shuffle(data_pc)

    # SOLVE FOR (V x U) = DATA
    # ==> U = PINV(V) x DATA
    # MOORE-PENROSE PSEUDO INVERSE

    u_ra = np.linalg.pinv(v_ra) @ data_ra
    u_sa = np.linalg.pinv(v_sa) @ data_sa
    u_pc = np.linalg.pinv(v_pc) @ data_pc

    data_ra_rec = v_ra @ u_ra
    data_sa_rec = v_sa @ u_sa
    data_pc_rec = v_pc @ u_pc

    data_ra_rec[data_ra_rec >= 0.38] = 1
    data_ra_rec[data_ra_rec < 0.38] = 0
    data_sa_rec[data_sa_rec >= 0.38] = 1
    data_sa_rec[data_sa_rec < 0.38] = 0
    data_pc_rec[data_pc_rec >= 0.38] = 1
    data_pc_rec[data_pc_rec < 0.38] = 0
    print(v_ra.shape)
    return data_ra_rec, data_sa_rec, data_pc_rec
    

def confusion_matrix(act, pred, quite=False):
    # THE CONFUSION MATRIX FROM MACHINE LEARNING

    total_firing = np.sum(act)
    total_zeroes = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if act[i,j]==1:
                # ACTUAL POSITIVE
                if pred[i, j]==1:
                    # TRUE POSITIVE
                    TP+=1
                else:
                    # FALSE NEGATIVE
                    # IE: PREDICTED IS 0 BUT ACTUAL IS 1
                    FN+=1
            else:
                # ACTUAL NEGATIVE
                total_zeroes+=1
                if pred[i, j]==0:
                    # TRUE NEGATIVE
                    TN+=1
                else:
                    # FALSE POSITIVE
                    # IE: PREDICTED IS 1 BUT ACTUAL IS 0
                    FP+=1



    if TP!=0 or FN!=0:
        tpr = round(TP / (TP + FN), 4)
    else:
        tpr = 0

    if TN!=0 or FP!=0:
        tnr = round(TN / (TN + FP),4)
    else:
        tnr = 0

    if TN!=0 or FP!=0:
        fpr = round(FP / (TN + FP),4)
    else:
        fpr = 0

    if TP!=0 or FN!=0:
        fnr = round(FN / (FN + TP),4)
    else:
        fnr = 0

    if TP!=0 or FP!=0:
        prec = round(TP / (TP + FP), 4)
    else:
        prec = 0


def extract(data_ra,data_sa,data_pc):
    """
    THIS IS HOW THE FEATURE EXTRACTION WAS DONE TRADIIONALLY.
    NEEDS TO BE UPDATED, COEFFICIENTS FOR THE DICTIONARY LEARNING NEEDS TO BE CHANGED FOR A GIVEN EXPERIMENT.

    THE SET OF PARAMETERS FOR THE DICTIONARY LEARNING PROCESS GIVEN IN THE FUNCTION BELOW ARE DUMMY VALUES.
    VALUES NEED TO BE CHANGED FOR ACTUAL RUNS.
    """
    dict_learner_ra = skd.DictionaryLearning(664,1,800,1e-7,transform_algorithm='lasso_cd', transform_n_nonzero_coefs=1, n_jobs=-1, verbose=True)
    dict_learner_sa = skd.DictionaryLearning(364,1,800,1e-7,transform_algorithm='lasso_cd', transform_n_nonzero_coefs=1, n_jobs=-1, verbose=True)
    dict_learner_pc = skd.DictionaryLearning(135,1,800,1e-7,transform_algorithm='lasso_cd', transform_n_nonzero_coefs=1, n_jobs=-1, verbose=True)

    time_start = time.time()
    u_ra = dict_learner_ra.fit_transform(data_ra)
    v_ra = dict_learner_ra.components_
    v_ra = np.asarray(v_ra).T
    # NOW V IS IN NxF FORMAT
    time_stop = time.time()
    print((time_stop - time_start) // 3600, " :TIME hrs")

    time_start = time.time()
    u_sa = dict_learner_sa.fit_transform(data_sa)
    v_sa = dict_learner_sa.components_
    v_sa = np.asarray(v_sa).T
    time_stop = time.time()
    print((time_stop - time_start) // 3600, " :TIME hrs")

    time_start = time.time()
    u_pc = dict_learner_pc.fit_transform(data_pc)
    v_pc = dict_learner_pc.components_
    v_pc = np.asarray(v_pc).T
    time_stop = time.time()
    print((time_stop - time_start) // 3600, " :TIME hrs")

    sio.savemat('results_70_percent_M.mat', {'v_ra':v_ra, 'v_sa':v_sa, 'v_pc':v_pc, 'u_ra':u_ra, 'u_sa':u_sa, 'u_pc':u_pc}, do_compression=True)



