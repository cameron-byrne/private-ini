import numpy as np
import scipy.io as sio
import sklearn.decomposition as skd
from condensed import confusion_matrix
import time


def calc(data_ra, data_sa, data_pc, v_ra, v_sa, v_pc):
    # POPULATION EXPERIMENT WHERE WE SHUFFLE THE NEURONS OF THE DATA
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


def hyperparams_opt():
    a = sio.loadmat('data_train.mat')

    data_ra = a['data_ra'][5,:,:]
    data_sa = a['data_sa'][5,:,:]
    data_pc = a['data_pc'][5,:,:]
    # sra = np.repeat(np.expand_dims(a['stim_ra'][5,:],1),948,1)
    # ssa = np.repeat(np.expand_dims(a['stim_sa'][5,:],1),562,1)
    # spc = np.repeat(np.expand_dims(a['stim_pc'][5,:],1),196,1)
    for i in [8,11,13,52,83,40,31,110,122,135,150]:
        data_ra = np.vstack((data_ra,a['data_ra'][i,:,:],np.repeat(np.expand_dims(a['stim_ra'][i,:],1),948,1)))
        data_sa = np.vstack((data_sa,a['data_sa'][i,:,:],np.repeat(np.expand_dims(a['stim_sa'][i,:],1),562,1)))
        data_pc = np.vstack((data_pc,a['data_pc'][i,:,:],np.repeat(np.expand_dims(a['stim_pc'][i,:],1),196,1)))



    # data_ra_test = a['data_ra'][10,:,:]
    # data_sa_test = a['data_sa'][10,:,:]
    # data_pc_test = a['data_pc'][10,:,:]
    # for i in [14,17,22,43,55,33,100,120,140,170]:
    #     data_ra_test = np.vstack((data_ra_test,a['data_ra'][i,:,:]))
    #     data_sa_test = np.vstack((data_sa_test,a['data_sa'][i,:,:]))
    #     data_pc_test = np.vstack((data_pc_test,a['data_pc'][i,:,:]))
    #
    # data_ra_test=data_ra_test.T
    # data_sa_test=data_sa_test.T
    # data_pc_test=data_pc_test.T


    tprs,alps,n_features,precs = [],[],[],[]

    cc=0
    number_features = np.random.uniform(0.25,0.75,7)
    for i in range(-5,5):
        ii = 10**i
        for j in number_features:
            cc += 1
            alps.append(ii)
            n_features.append(j)
            print("Iteration",cc, "Alpha value:",ii, "Feature Fraction:",j)

            M_ra = skd.DictionaryLearning(int(j*data_ra.shape[1]),ii,800,1e-7,transform_algorithm='lasso_lars', transform_n_nonzero_coefs=int(j*data_ra.shape[1]), n_jobs=-1, verbose=False)
            M_sa = skd.DictionaryLearning(int(j*data_sa.shape[1]),ii,800,1e-7,transform_algorithm='lasso_lars', transform_n_nonzero_coefs=int(j*data_sa.shape[1]), n_jobs=-1, verbose=False)
            M_pc = skd.DictionaryLearning(int(j*data_pc.shape[1]),ii,800,1e-7,transform_algorithm='lasso_lars', transform_n_nonzero_coefs=int(j*data_pc.shape[1]), n_jobs=-1, verbose=False)


            # DICTIONARY LEARNING DECOMPOSITION STARTS HERE
            time_start = time.time()
            u_ra = M_ra.fit_transform(data_ra)
            v_ra=(M_ra.components_)
            v_ra=np.asarray(v_ra).T

            # NOW V IS IN NxF FORMAT
            u_sa = M_sa.fit_transform(data_sa)
            v_sa=(M_sa.components_)
            v_sa=np.asarray(v_sa).T

            u_pc = M_pc.fit_transform(data_pc)
            v_pc=(M_pc.components_)
            v_pc=np.asarray(v_pc).T

            time_stop = time.time()
            print((time_stop - time_start) // 3600, " :TIME hrs")
            tpr_this_ra, prec_this_ra= [],[]
            tpr_this_sa, prec_this_sa= [],[]
            tpr_this_pc, prec_this_pc= [],[]

            for i in [10, 14, 17, 22, 43, 55, 33, 100, 120, 140, 170]:
                data_ra_test = a['data_ra'][i, :, :].T
                data_sa_test = a['data_sa'][i, :, :].T
                data_pc_test = a['data_pc'][i, :, :].T
                dra_rec,dsa_rec,dpc_rec = calc(data_ra_test,data_sa_test,data_pc_test,v_ra,v_sa,v_pc)
                tpr = [confusion_matrix(data_ra_test,dra_rec,True)[0],confusion_matrix(data_sa_test,dsa_rec,True)[0],
                       confusion_matrix(data_pc_test,dpc_rec,True)[0]]
                tpr_this_ra.append(tpr[0])
                tpr_this_sa.append(tpr[1])
                tpr_this_pc.append(tpr[2])

                prc = [confusion_matrix(data_ra_test,dra_rec,True)[4],confusion_matrix(data_sa_test,dsa_rec,True)[4],
                       confusion_matrix(data_pc_test,dpc_rec,True)[4]]
                prec_this_ra.append(prc[0])
                prec_this_sa.append(prc[1])
                prec_this_pc.append(prc[2])

            tpr_ra = np.average(tpr_this_ra)
            prc_ra = np.average(prec_this_ra)
            tpr_sa = np.average(tpr_this_sa)
            prc_sa = np.average(prec_this_sa)
            tpr_pc = np.average(tpr_this_pc)
            prc_pc = np.average(prec_this_pc)
            tpr = [tpr_ra,tpr_sa,tpr_pc]
            prc = [prc_ra,prc_sa,prc_pc]
            print(tpr,prc)
            tprs.append(tpr)
            precs.append(prc)

    sio.savemat('HyperP_results.mat',{'tprs':tprs,'precs':precs,'alphas':alps,'N_features':n_features})

def exp1():
    # EXPERIMENT TO EXTRACT FEATURE SPACE OF LOWEST ACCURACY FROM PREVIOS HYPERPARAMETER EXP RESUTLS
    # ALSO NEED TO OPTIMIZE THE TOLERANCE VALUE
    a = sio.loadmat('data_train.mat')

    data_ra = a['data_ra'][5, :, :]
    data_sa = a['data_sa'][5, :, :]
    data_pc = a['data_pc'][5, :, :]
    # sra = np.repeat(np.expand_dims(a['stim_ra'][5,:],1),948,1)
    # ssa = np.repeat(np.expand_dims(a['stim_sa'][5,:],1),562,1)
    # spc = np.repeat(np.expand_dims(a['stim_pc'][5,:],1),196,1)
    for i in [8, 11, 13, 52, 83, 40, 31, 110, 122, 135, 150]:
        data_ra = np.vstack((data_ra, a['data_ra'][i, :, :], np.repeat(np.expand_dims(a['stim_ra'][i, :], 1), 948, 1)))
        data_sa = np.vstack((data_sa, a['data_sa'][i, :, :], np.repeat(np.expand_dims(a['stim_sa'][i, :], 1), 562, 1)))
        data_pc = np.vstack((data_pc, a['data_pc'][i, :, :], np.repeat(np.expand_dims(a['stim_pc'][i, :], 1), 196, 1)))

    data_ra = data_ra.T
    data_sa = data_sa.T
    data_pc = data_pc.T

    print(data_ra.shape)

    M_ra = skd.DictionaryLearning(int(0.323 * data_ra.shape[1]), 0.01, 800, 1e-5, transform_algorithm='lasso_lars',
                                  transform_n_nonzero_coefs=int(data_ra.shape[1]), n_jobs=-1, verbose=False)
    M_sa = skd.DictionaryLearning(int(0.323 * data_sa.shape[1]), 0.01, 800, 1e-5, transform_algorithm='lasso_lars',
                                  transform_n_nonzero_coefs=int(0.323 * data_sa.shape[1]), n_jobs=-1, verbose=False)
    M_pc = skd.DictionaryLearning(int(0.323 * data_pc.shape[1]), 0.01, 800, 1e-5, transform_algorithm='lasso_lars',
                                  transform_n_nonzero_coefs=int(data_pc.shape[1]), n_jobs=-1, verbose=False)

    # DICTIONARY LEARNING DECOMPOSITION STARTS HERE
    u_ra = M_ra.fit_transform(data_ra)
    v_ra = (M_ra.components_)
    v_ra = np.asarray(v_ra).T
    err_ra = M_ra.error_
    print(v_ra.shape)

    # NOW V IS IN NxF FORMAT
    u_sa = M_sa.fit_transform(data_sa)
    v_sa = (M_sa.components_)
    v_sa = np.asarray(v_sa).T
    err_sa = M_sa.error_

    u_pc = M_pc.fit_transform(data_pc)
    v_pc = (M_pc.components_)
    v_pc = np.asarray(v_pc).T
    err_pc = M_pc.error_

    sio.savemat('HYP_EXP1.mat',{'v_ra':v_ra,'v_sa':v_sa,'v_pc':v_pc,'err_ra':err_ra,'err_sa':err_sa,'err_pc':err_pc})

if __name__=='__main__':
    exp1()