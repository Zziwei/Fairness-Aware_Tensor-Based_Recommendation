import numpy
from pyten.tools import khatrirao
import pyten.tenclass
import pandas as pd
import numpy as np
import pyten.method
import copy


def FATR_GD(y, features, feature_d, feature_n, r=20, omega=None, omega_groups=None
                                  , learning_rate=0.001, reg_para=0.1, Freg_para=0.1, tol=1e-4, maxiter=500,
                                  init='random', printitn=100):
    """ CP_ALS Compute a CP decomposition of a Tensor (and recover it).
    ---------
     :param  'y' - Tensor with Missing data
     :param 'features' - The sensitive features vector
     :param 'feature_d' - The mode of the tensor that the sensitie features belong to
     :param 'feature_n' - The size of the sensitive feature vector
     :param  'r' - Rank of the tensor
     :param 'omega' - Missing data Index Tensor
     :param 'omega_groups' - A list of missing data indicator tensors for different groups
     :param 'tol' - Tolerance on difference in fit
     :param 'maxiters' - Maximum number of iterations
     :param 'init' - Initial guess ['random'|'nvecs'|'eigs']
     :param 'printitn' - Print fit every n iterations; 0 for no printing
    ---------
     :return
        'U' - Factorized matrices
        'Xf' - Recovered fair tensor
        'X' - Recovered tensor.
    ---------
    """
    lamb = np.ones(r)

    feature_vn = features.shape[1]

    group_n = len(omega_groups)  # number of sensitive groups
    group_nonempty_list = []  # list of number of data points in each group
    for i in range(group_n):
        group_nonempty_list.append(np.sum(omega_groups[i]))
    Xf = y.data.copy()  # recovered fair tensor
    Xf = pyten.tenclass.Tensor(Xf)
    X = y.data.copy()
    X = pyten.tenclass.Tensor(X)

    # Setting Parameters
    # Extract number of dimensions and norm of X.
    ndimsX = X.ndims
    normX = X.norm()
    dimorder = range(ndimsX)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = tol
    maxiters = maxiter

    # Set up and error checking on initial guess for U.
    if init == 'random':
        Uinit = range(ndimsX)
        for n in dimorder[:]:
            Uinit[n] = numpy.random.random([X.shape[n], r])
    elif init == 'nvecs' or init == 'eigs':
        Uinit = range(ndimsX)
        Uinit[0] = []
        for n in dimorder[1:]:
            Uinit[n] = X.nvecs(n, r)  # first r leading eigenvecters
    else:
        raise TypeError('The selected initialization method is not supported')

    # Set up for iterations - initializing U and the fit.
    U = Uinit[:]
    fit = 0  # the training error

    if printitn > 0:
        print('\nCP_ALS:\n')

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = X.data * 1.0

        #################################################################
        # first, deal with the mode of feature_d
        # convert the tensor to matrix at mode feature_d
        # Save hadamard product of each U[n].T*U[n]

        # Iterate over all N modes of the Tensor
        for n in range(ndimsX):
            temp1 = [n]
            temp2 = range(n)
            temp3 = range(n + 1, ndimsX)
            temp2.reverse()
            temp3.reverse()
            temp1[len(temp1):len(temp1)] = temp3
            temp1[len(temp1):len(temp1)] = temp2
            Xn = X.permute(temp1)
            Xn = Xn.tondarray()
            Xn = Xn.reshape([Xn.shape[0], Xn.size / Xn.shape[0]])
            # Calculate Unew = X_(n) * khatrirao(all U except n, 'r').

            # delete the U of mode n
            tempU = U[:]
            A = copy.copy(U[n])
            tempU.pop(n)

            # Calculate Unew = X_(n) * khatrirao(all reversed U except mode n).
            tempU.reverse()
            CkB = khatrirao(tempU)

            if n == feature_d:
                CkB_part1 = CkB[:, :len(CkB[0]) - feature_vn]
                CkB_part2 = CkB[:, len(CkB[0]) - feature_vn:len(CkB[0])]
                Xnew = Xn - np.dot(features, CkB_part2.T)
                A_old_part1 = A[:, :len(A[0]) - feature_vn]
                Xnew = np.dot(Xnew, CkB_part1)
                y = Freg_para * np.eye(A_old_part1.shape[1]) \
                    + np.dot(CkB_part1.T, CkB_part1)
                A_new_part1 = Xnew.dot(numpy.linalg.inv(y))
                gradient = reg_para * np.dot(np.dot(features,
                                                    features.T), A_new_part1)
                Anew_part1 = A_new_part1 - learning_rate * gradient
                A = np.concatenate((Anew_part1, features), axis=1)
            else:
                Xnew = Xn.dot(CkB)
                y = Freg_para * np.eye(A.shape[1]) + np.dot(CkB.T, CkB)
                A = Xnew.dot(numpy.linalg.inv(y))
            U[n] = A

        # Reconstructed fitted Ktensor
        P = pyten.tenclass.Ktensor(lamb, U)
        temp = P.tondarray()
        error = numpy.linalg.norm(temp * omega - X.data * omega)
        n_all = temp.size
        fairness = numpy.absolute(numpy.mean(temp * omega_groups[0]) * n_all
                                  / group_nonempty_list[0]
                                  - numpy.mean(temp * omega_groups[1]) * n_all
                                  / group_nonempty_list[1])
        B2tB1 = np.dot(features.T, Anew_part1)
        B2B1Fnorm = numpy.linalg.norm(B2tB1)

        X.data = temp * (1 - omega) + X.data * omega
        fitchange = numpy.linalg.norm(X.data - oldX)

        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            print 'CP_ALS: iters={0}, B1B2F={1}, err={2}, fair={3}' \
                .format(iter, B2B1Fnorm, error, fairness)

            # Check for convergence
        if flag == 0:
            break

    # Reconstructed fitted Ktensor without sensitive features
    Uf = copy.copy(U)
    for u in range(len(Uf)):
        tmpU = Uf[u]
        Uf[u] = tmpU[:, :len(tmpU[0, :]) - feature_vn]
    Pf = pyten.tenclass.Ktensor(lamb[:len(lamb) - feature_vn], Uf)
    tempf = Pf.tondarray()
    Xf.data = tempf * (1 - omega) + Xf.data * omega
    return U, Xf, X