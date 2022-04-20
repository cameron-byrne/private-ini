
"""
FILE TO CALCULATE ENTROPY OF A DATASET AND ITS RECONSTRUCTION USING MECHANICAL FEATURES
"""


import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import scipy.stats as sst




def entropy_total():
    # EXP: ENTROPY OF SA: 8/19
    # DATA RA IS ACTUALLY DATA SA RIGHT NOW

    # data_ra, data_sa, data_pc = parse(save=False)
    data_ra = sio.loadmat('data_train.mat', struct_as_record=True)['data_ra']

    # data_sa = sio.loadmat('Data_M_test.mat', struct_as_record=True)['data_sa']
    # data_pc = sio.loadmat('Data_M_test.mat', struct_as_record=True)['data_pc']
    data_ra[data_ra == 1] = True
    data_ra[data_ra == 0] = False
    pparr = []

    print("HASHMAP TEST: HASHMAP:", hash_map_counter(data_ra))

    print("Shape after the 0s columns have been removed", data_ra.shape)

    #loop over all times
    for i in range(data_ra.shape[1]):
        pp = 1
        dele = []
        for k in range(data_ra.shape[1]):
            if (not np.any(np.logical_xor(data_ra[:, i], data_ra[:, k]))) and i != k:
                # IF EQUAL
                pp += 1
                # STORE THE INDICES OF THE COLS WHICH ARE THE SAME
                dele.append(k)

        if pp > 1:
            # IF THERE IS ANYTHING TO DELETE
            mask = np.ones(data_ra.shape[1], dtype=bool)
            # DELETE THE ONES WHICH ARE THE SAME
            mask[dele] = False
            data_ra = data_ra[:, mask]

        # STORE THE NUMBER OF TIMES VECTOR 'i' HAS BEEN REPEATED
        pparr.append(pp)
        #print(i, pp)
        #print(data_ra.shape)
        # THE SIZE OF THE DATASET WILL CHANGE SINCE WE ARE DELETING ALL COLUMNS THAT ARE SIMILAR
        # THEREFORE, WE HAVE TO STOP IF THE INDEX NUMBER EXCEEDS OR IS EQUAL TO THE SIZE OF THE ARRAY
        # SINCE ALL THE ONES BEHIND THE INDEX ARE NECESSARILY UNIQUE, BECAUSE THE ONES IN FRONT
        # WILL SIMPLY BE DELETED IF THEY WERE THE SAME AS ANY OF THE ONES ALREADY CONSIDERED
        if i >= data_ra.shape[1] - 1:
            # ABOVE MENTIONED REASON
            break

    sio.savemat('counts.mat', {'counts': pparr}, do_compression=True)
    print("HASHMAP TEST: CONTROL:", pparr)
    # plt.hist(pparr)
    # plt.show()


def entropy_recon():
    # stuff = sio.loadmat('../experiment_disjoint/results.mat', struct_as_record=True)
    stuff = sio.loadmat('../results_70_percent_M.mat', struct_as_record=True)
    #
    v_ra = stuff['v_ra']
    v_sa = stuff['v_sa']
    v_pc = stuff['v_pc']
    # print(v_ra.shape)

    # ve = stuff['v_ra_e']
    # vm = stuff['v_ra_m']
    # v_ra = np.hstack((vm, ve))
    # v_ra = vm

    # ve = stuff['v_sa_e']
    # vm = stuff['v_sa_m']
    # v_sa = np.hstack((vm, ve))
    # v_sa = vm

    # ve = stuff['v_pc_e']
    # vm = stuff['v_pc_m']
    # v_pc = np.hstack((vm, ve))
    # v_pc = vm

    data_ra, data_sa, data_pc = parse(save=False)

    u_ra = np.linalg.pinv(v_ra) @ data_ra
    u_sa = np.linalg.pinv(v_sa) @ data_sa
    u_pc = np.linalg.pinv(v_pc) @ data_pc
    data_ra = v_ra @ u_ra
    data_sa = v_sa @ u_sa
    data_pc = v_pc @ u_pc
    data_ra[data_ra >= 0.38] = 1
    data_ra[data_ra < 0.38] = 0
    data_sa[data_sa >= 0.38] = 1
    data_sa[data_sa < 0.38] = 0
    data_pc[data_pc >= 0.38] = 1
    data_pc[data_pc < 0.38] = 0
    data_ra = data_sa

    data_ra[data_ra == 1] = True
    data_ra[data_ra == 0] = False
    pparr = []

    print("Shape after the 0s columns have been removed", data_ra.shape)

    for i in range(data_ra.shape[1]):
        pp = 1
        dele = []
        for k in range(data_ra.shape[1]):
            if (not np.any(np.logical_xor(data_ra[:, i], data_ra[:, k]))) and i != k:
                # IF EQUAL
                pp += 1
                # STORE THE INDICES OF THE COLS WHICH ARE THE SAME
                dele.append(k)

        if pp > 1:
            # IF THERE IS ANYTHING TO DELETE
            mask = np.ones(data_ra.shape[1], dtype=bool)
            # DELETE THE ONES WHICH ARE THE SAME
            mask[dele] = False
            data_ra = data_ra[:, mask]

        # STORE THE NUMBER OF TIMES VECTOR 'i' HAS BEEN REPEATED
        pparr.append(pp)
        print(i, pp)
        print(data_ra.shape)
        # THE SIZE OF THE DATASET WILL CHANGE SINCE WE ARE DELETING ALL COLUMNS THAT ARE SIMILAR
        # THEREFORE, WE HAVE TO STOP IF THE INDEX NUMBER EXCEEDS OR IS EQUAL TO THE SIZE OF THE ARRAY
        # SINCE ALL THE ONES BEHIND THE INDEX ARE NECESSARILY UNIQUE, BECAUSE THE ONES IN FRONT
        # WILL SIMPLY BE DELETED IF THEY WERE THE SAME AS ANY OF THE ONES ALREADY CONSIDERED
        if i >= data_ra.shape[1] - 1:
            # ABOVE MENTIONED REASON
            break

    sio.savemat('counts_recon_VM_70%M.mat', {'counts': pparr}, do_compression=True)
    # plt.hist(pparr)
    # plt.show()



def calc_entropy():
    a = np.ndarray.flatten(sio.loadmat('counts.mat')['counts'])
    print(a.shape)
    a = a / sum(a)
    ent1 = 0
    for i in a:
        if i != 0:
            ent1 += -1 * (i * np.log(np.abs(i)))
    print(ent1)

    b = np.ndarray.flatten(sio.loadmat('counts_recon_VM_70%M.mat')['counts'])
    b = b / sum(b)
    ent2 = 0
    for i in b:
        if i != 0:
            ent2 += -1 * (i * np.log(np.abs(i)))

    print(ent2)


def hash_map_counter(data_matrix):
    """
    This method takes in a data matrix and returns 'pparr'
    The hashmap holds the column indices of columns that get mapped to a particular place
    """

    #
    hash_map = []
    for i in range(8192):
        hash_map.append([])

    # ppColumns is a dictionary from columns ppCounts index, keeping runtime nice and linear
    ppColumns = {}
    ppCounts = []

    for col in range(data_matrix.shape[1] - 1):
        hash = hash_function(data_matrix, col)
        is_duplicate = False

        # loop over all columns already in that hash of the hashmap
        for prior_column in hash_map[hash]:

            # Checks if the two vectors are exactly equal [not just having the same hash]
            if not np.any(np.logical_xor(data_matrix[:, col], data_matrix[:, prior_column])):
                is_duplicate = True
                ppCounts[ppColumns[prior_column]] += 1

        if not is_duplicate:
            hash_map[hash].append(col)
            ppCounts.append(1)
            ppColumns[col] = len(ppCounts) - 1

    return ppCounts







    # this checks for if two cols are the same
    # if (not np.any(np.logical_xor(data_ra[:, i], data_ra[:, k])))



def hash_function(data_matrix, col):
    """
    This computes a hash index for the hash_map, 0 <= hash <= 8191 (2^13 - 1)
    from the given column of the data matrix

    Does this by adding each subsequent by an increasing power of two, maxing out at 13
    and returning to 0. The row is also added to prevent things from being too cyclic
    """
    total = 0
    for row in range (data_matrix.shape[0] - 1):
        total += data_matrix[0, row, col] * (2 ** (row % 14)) + row
    return total % (2 ** 13)


if __name__ == "__main__":
    entropy_total()
    #entropy_recon()
    #calc_entropy()