import os, sys
os.chdir('/gstore/home/sorensc1/MSR/CSorensen_290')
curr_dir = os.path.split(os.getcwd())[0]
print(curr_dir)
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

import numpy as np
import scipy.fftpack as sf
import scipy.linalg as lp
import pickle
import pywt
import matplotlib.pyplot as plt

#code source: https://github.com/liubenyuan/pyBSBL/blob/master/bsbl.py
# vector to column (M,1) vector
def v2m(v):
    return v.reshape((v.shape[0],1))

# M = A*B*C
def dot3(A,B,C):
    return np.dot(np.dot(A, B), C)

# ravel list of 'unequal arrays' into a row vector
def ravel_list(d):
    r = np.array([], dtype='int')
    for i in range(d.shape[0]):
        r = np.r_[r,d[i]]
    return r

# extract block spacing information
def block_parse(blk_start_loc, N):
    blk_len_list = np.r_[blk_start_loc[1:], N] - blk_start_loc
    is_equal_block = (np.sum(np.abs(blk_len_list - blk_len_list.mean())) == 0)
    return blk_len_list, is_equal_block

# exploit AR(1) correlation in Covariance matrices
#   r_scale : scale the estimated coefficient
#   r_init : initial guess of r when no-basis is included
#   r_thd : the threshold of r to make the covariance matrix p.s.d
#           the larger the block, the smaller the value
def coeff_r(Cov, gamma, index, r_scale=1.1, r_init=0.90, r_thd=0.999):
    r0 = 0.
    r1 = 0.
    for i in index:
        temp = Cov[i] / gamma[i]
        r0 += temp.trace()
        r1 += temp.trace(offset=1)
    # this method tend to under estimate the correlation
    if np.size(index) == 0:
        r = r_init
    else:
        r = r_scale * r1/(r0 + 1e-8)
    # constrain the Toeplitz matrix to be p.s.d
    if (np.abs(r) >= r_thd):
        r = r_thd * np.sign(r)
    return r

# generate toeplitz matrix
def gen_toeplitz(r,l):
    jup = np.arange(l)
    bs = r**jup
    B = lp.toeplitz(bs)
    return B

#
class bo:
    """
    BSBL-BO : Bound Optimization Algos of BSBL framework
    Recover block sparse signal (1D) exploiting intra-block correlation, 
    given the block partition.
    The algorithm solves the inverse problem for the block sparse
                model with known block partition:
                         y = X * w + v
    Variables
    ---------
    X : array, shape = (n_samples, n_features)
          Training vectors.
    y : array, shape = (n_samples)
        Target values for training vectors
    w : array, shape = (n_features)
        sparse/block sparse weight vector
    Parameters
    ----------
    'learn_lambda' : (1) if (SNR<10dB), learn_lambda=1
                     (2) if (SNR>10dB), learn_lambda=2
                     (3) if noiseless, learn_lambda=0
                     [ Default value: learn_lambda=2 ]
    'lambda_init'  : initial guess of the noise variance
                     [ Default value: lambda_init=1e-2 ]
    'r_init'       : initial value for correlation coefficient
                     [ Default value: 0.90 ]
    'epsilon'      : convergence criterion
    'max_iters'    : Maximum number of iterations.
                     [ Default value: max_iters = 500 ]
    'verbose'      : print debuging information
    'prune_gamma'  : threshold to prune out small gamma_i
                     (generally, 10^{-3} or 10^{-2})
    'learn_type'   : learn_type = 0: Ignore intra-block correlation
                     learn_type = 1: Exploit intra-block correlation
                     [ Default: learn_type = 1 ]
    """

    # constructor
    def __init__(self, learn_lambda=2, lambda_init=1e-2, r_init=0.90,
                  epsilon=1e-8, max_iters=500, verbose=0,
                  learn_type=1, prune_gamma=1e-2):
        self.learn_lambda = learn_lambda
        self.lamb = lambda_init
        self.r_init = r_init
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.verbose = verbose
        self.learn_type = learn_type
        self.prune_gamma = prune_gamma

    # fit y
    def fit_transform(self, X, y, blk_start_loc=None):
        #
        self.scale = y.std()
        y = y / self.scale
        M, N = X.shape
        # automatically set block partition
        if blk_start_loc==None:
            blkLen = int(N/16.)
            blk_start_loc = np.arange(0,N,blkLen)
        blk_len_list, self.is_equal_block = block_parse(blk_start_loc, N)
        # init variables
        nblock      = blk_start_loc.shape[0]
        self.nblock = nblock
        w           = np.zeros(N,dtype='float')
        Sigma0      = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Sigma_w     = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Cov_x       = [np.identity(blk_len_list[i]) for i in range(nblock)]
        B           = [np.identity(blk_len_list[i]) for i in range(nblock)]
        invB        = [np.identity(blk_len_list[i]) for i in range(nblock)]
        block_slice = np.array([blk_start_loc[i] + np.arange(blk_len_list[i]) for i in range(nblock)])
        gamma       = np.ones(nblock, dtype='float')
        HX          = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Hy          = [np.zeros(blk_len_list[i]) for i in range(nblock)]
        # loops
        for count in range(self.max_iters):
            # prune weights as their hyperparameter goes to zero
            # index -- 0:unused, 1:used
            index = np.argwhere(gamma > self.prune_gamma).ravel()
            # calculate XBX^T
            XBX = np.zeros((M,M), dtype=float)
            for i in index:
                Xi = X[:, block_slice[i]]
                XBX += np.dot(np.dot(Xi, Sigma0[i]), Xi.T)
            invXBX = lp.inv(XBX + self.lamb * np.identity(M))
            #
            for i in index:
                Xi = X[:, block_slice[i]]
                Hi = np.dot(Xi.T, invXBX)
                Hy[i] = np.dot(Hi, y)
                HX[i] = np.dot(Hi, Xi)
            # now we update basis
            w_old = w.copy()
            for i in index:
                seg = block_slice[i]
                w[seg] = np.dot(Sigma0[i], Hy[i])
                Sigma_w[i] = Sigma0[i] - np.dot(np.dot(Sigma0[i], HX[i]), Sigma0[i])
                mu_v = v2m(w[seg])
                Cov_x[i] = Sigma_w[i] + np.dot(mu_v, mu_v.T)

            #=========== Learn correlation structure in blocks ===========
            # 0: do not consider correlation structure in each block
            # 1: constrain all the blocks have the same correlation structure
            if self.learn_type == 1:
                r = coeff_r(Cov_x, gamma, index, r_init=self.r_init)
                if self.is_equal_block:
                    jup = np.arange(Cov_x[0].shape[0])
                    bs = r**jup
                    B0 = lp.toeplitz(bs)
                    invB0 = lp.inv(B0)
                    for i in index:
                        B[i] = B0
                        invB[i] = invB0
                else:
                    for i in index:
                        jup = np.arange(B[i].shape[0])
                        bs = r**jup
                        B[i] = lp.toeplitz(bs)
                        invB[i] = lp.inv(B[i])

            # estimate gammas
            gamma_old = gamma.copy()
            for i in index:
                denom = np.sqrt(np.dot(HX[i], B[i]).trace())
                gamma[i] = gamma_old[i] * lp.norm(np.dot(lp.sqrtm(B[i]), Hy[i])) / denom
                Sigma0[i] = B[i] * gamma[i]
            # estimate lambda
            if self.learn_lambda == 1:
                lambComp = 0.
                for i in index:
                    Xi = X[:,block_slice[i]];
                    lambComp += np.dot(np.dot(Xi, Sigma_w[i]), Xi.T).trace()
                self.lamb = lp.norm(y - np.dot(X, w))**2./N + lambComp/N;
            elif self.learn_lambda == 2:
                lambComp = 0.
                for i in index:
                    lambComp += np.dot(Sigma_w[i], invB[i]).trace() / gamma_old[i]
                self.lamb = lp.norm(y - np.dot(X, w))**2./N + self.lamb * (w.size - lambComp)/N

            #================= Check stopping conditions, eyc. ==============
            dmu = (np.abs(w_old - w)).max(0); # only SMV currently
            if (dmu < self.epsilon):
                break
            if (count >= self.max_iters):
                break
        # exit
        self.count = count + 1
        self.gamma = gamma
        self.index = index
        # let's convert the backyard:
        w_ret = np.zeros(N)
        relevant_slice = ravel_list(block_slice[index])
        w_ret[relevant_slice] = w[relevant_slice]
        return w_ret * self.scale

def reconstructIndividual(indID):
    #os.chdir('/gstore/home/sorensc1/MSR/CSorensen_290')
    datapath = os.path.join(os.getcwd(),'compressedData','ind'+str(indID)+'.npy')
    compressedData = np.load(datapath)

    Apath = os.path.join(os.getcwd(),'matrixA.npy')
    A = np.load(Apath) #shape: (192, 768)

    n_vid, n_sensor, n_recording, _ = compressedData.shape
    clf = bo(verbose=1, learn_type=1, learn_lambda=2, prune_gamma=-1, epsilon=1e-8, max_iters=16)

    x_reconstructed = np.zeros((n_vid, n_sensor, n_recording, A.shape[1]))

    for vid in range(n_vid): #40
        print('beginning video '+str(vid))
        for sensor in range(n_sensor): #32
            print('beginning sensor '+str(sensor))
            for recording in range(n_recording): #10
                print('beginning recording '+str(recording))
                y = compressedData[vid,sensor,recording,:]
                rev_dct_coeff = clf.fit_transform(A, y)
                #step 2: recover the signal using the DCT ceofficients and the DCT basis
                x_reconstructed[vid,sensor,recording] = sf.idct(rev_dct_coeff, norm='ortho')

    print('saving to numpy file')
    filename = 'reconstructedData/ind'+str(indID)+'.npy'
    np.save(filename, x_reconstructed)

def main():
    indID = int(sys.argv[2])
    print('begin reconstruction for individual '+str(indID))
    reconstructIndividual(indID)

if __name__ == "__main__":
    main()