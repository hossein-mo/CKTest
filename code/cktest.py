import os
import sys
from cktest_misc import misc

if misc.is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Class for performing CK test on a single process.
class CKTest():

    # The constructor function.
    def __init__(self, x, dx=1, bins_mode='std', bins=6, device="auto", dtype=torch.float):
        """
        x: Array of data points. If x is a one dimensional array, CK test will perform
        with respect to time lag τ. If x is two dimensional array, CK test will perform with
        respect to t_i and t_j. If x is two dimensional its format should be x[time, samples].

        dx: Determine the bin width of probability distributions.

        bins_mode: 'std' or 'real'. If set to 'std' the width of bins will be
        standard deviation of x multiplied by dx. If set to 'real' width of bins will be
        equal to dx.

        bins: 0 , any positive integer or array like object: If equals to 0 bins include
        all data points. If bins > 0 bins range from -bins × dx to bins × dx with width
        equals to dx. If bins is an array like object it should be the list of left edges
        of the bins.

        device: 'auto' or name of torch device: If set to 'auto' and a cuda device is
        avaliable to torch it will use gpu for computations otherwise it will use cpu.

        dtype: torch data type. Data type of torch arrays, better to leave unchanged.
        """
        self.x = x.copy()

        # Probability distributions will be saved in this list.
        self.pdfs = []

        # This will be used for CK test values.
        self.ck = 0
        self.dtype = dtype

        # Markov length will be saved in this variable.
        self.lm = -1

        # If a cuda device is avaliable torch will use it for its computations.
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Checking if x includes more than one sample or not.
        if np.ndim(x) == 2 and self.x.shape[0] > 1 and self.x.shape[1] > 1:
            self.is_sample = True
        else:
            self.is_sample = False
            self.x = self.x.reshape(-1)

        # Generating left edges of the bins.
        if np.ndim(bins) == 0:
            self.xbins = S_Functions.find_bins(self.x, dx, bins_mode, max_bin=bins)
        else:
            self.xbins = bins.copy()

        self.pdf_shape = (len(self.xbins), len(self.xbins))

        # Finding data points corresponding index in xbins.
        self.xdigit = S_Functions.digitize(self.x, self.xbins, dtype=self.dtype, device=self.device)

    # After creating a CKTest() object this function should be called for performing the test.
    def run_test(self, lag, progress_bar=True):
        '''
        lag: positive integer. In the case if x includes only one sample, Ck test will continue until time lag τ=0, 
        otherwise in the case x has more than one samples Ck test will continue until t_i - t_j = 0.

        progress_bar: bool, default is True. If true a tqdm progress bar shows the progress of the test.
        '''

        if self.is_sample:
            # if there is more than one sample the condition is true and this part will run.

            # number of data points
            n = self.xdigit.shape[1]
            # crating torch arrays for CK test and its error.
            self.ck = torch.zeros((lag, lag), device=self.device, dtype=self.dtype)
            self.ck_error = torch.zeros((lag, lag), device=self.device, dtype=self.dtype)
            for t3 in tqdm(range(lag)):
                for t1 in range(lag):
                    t2 = int((t3 - t1) / 2)
                    t2 += t1

                    # calculating p(t3,t1)
                    xx = torch.cat((self.xdigit[t3], self.xdigit[t1]), 0)
                    xx = torch.reshape(xx, (2, n))
                    p31 = S_Functions.pdf(xx,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    # calculating p(t3,t2)
                    xx = torch.cat((self.xdigit[t3], self.xdigit[t2]), 0)
                    xx = torch.reshape(xx, (2, n))
                    p32 = S_Functions.pdf(xx,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    # calculating p(t2,t1)
                    xx = torch.cat((self.xdigit[t2], self.xdigit[t1]), 0)
                    xx = torch.reshape(xx, (2, n))
                    p21 = S_Functions.pdf(xx,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    # performing the test using single_test() function from S_Functions class.
                    self.ck[t1, t3], self.ck_error[t1, t3] = S_Functions.single_test(p31, p32, p21, n)
        else:
            
            # if there is only one sample the condition is false and this part will run.
            if lag > len(self.pdfs) + 1:
                if len(self.pdfs) == 0:
                    self.ck = torch.zeros(lag - 1, device=self.device, dtype=self.dtype)
                    self.ck_error = torch.zeros(lag - 1, device=self.device, dtype=self.dtype)
                    n = len(self.xdigit) - 1
                    xx = torch.cat((self.xdigit[1:], self.xdigit[:-1]), 0)
                    xx = torch.reshape(xx, (2, n))
                    self.pdfs.append(
                        S_Functions.pdf(xx, n, self.pdf_shape, dtype=self.dtype,
                                       device=self.device))
                else:
                    self.ck = torch.cat((self.ck,
                                         torch.zeros(lag - len(self.pdfs),
                                                     device=self.device,
                                                     dtype=self.dtype)))
                    self.ck_error = torch.cat((self.ck_error,
                                               torch.zeros(lag - len(self.pdfs),
                                                           device=self.device,
                                                           dtype=self.dtype)))

                for t3 in tqdm(range(len(self.pdfs) + 1, lag + 1), disable=not progress_bar):
                    n = len(self.xdigit) - t3
                    xx = torch.cat((self.xdigit[t3:], self.xdigit[:n]), 0)
                    xx = torch.reshape(xx, (2, n))
                    self.pdfs.append(
                        S_Functions.pdf(xx, n, self.pdf_shape, dtype=self.dtype,
                                       device=self.device))

                    t2 = int(t3 / 2)
                    t1 = t3 - t2
                    self.ck[t3 - 2], self.ck_error[t3 - 2] = S_Functions.single_test(
                        self.pdfs[t3 - 1].to_dense(), self.pdfs[t2 - 1].to_dense(),
                        self.pdfs[t1 - 1].to_dense(), n)
        ck_array = np.array(self.ck.tolist())
        ck_error_array = np.array(self.ck_error.tolist())
        self.lm = S_Functions.markov_length(ck_array, ck_error_array)
        return ck_array, ck_error_array

    def plot(self, figsize=(12, 6), dpi=80, file=None, grid=True, linewidth=0.3, s=50):
        ck_array = np.array(self.ck.tolist())
        if self.is_sample:
            cbaru = np.max(ck_array)
            cbard = np.min(ck_array)
            lm_mask = self.lm > 0
            lm_x = self.lm[lm_mask]
            lm_y = np.arange(0, len(self.lm))
            lm_y = lm_y[lm_mask]
            misc.sample_plot(figsize, dpi, ck_array, (lm_x, lm_y), cbaru, cbard, s=s, file=file)
        else:
            ck_error_array = np.array(self.ck_error.tolist())
            misc.single_plot(ck_array,
                             ck_error_array,
                             self.lm,
                             figsize=figsize,
                             dpi=dpi,
                             file=file,
                             grid=grid,
                             linewidth=linewidth,
                             s=s)

class S_Functions():

    @staticmethod
    def single_test(p31, p32, p21, n):
        '''
        p31, p32, p21: two dimensional arrays representing their 
        respective probability distributions.

        n: number of data points.
        '''
        # calculating the variance (square of error) of PDFs.
        p31_var = S_Functions.pdf_var(p31, n)
        p21_var = S_Functions.pdf_var(p21, n)

        # calculating p(x2)
        p2 = p21.sum(1)
        p2[p2 == 0] = float('inf')
        #calculating p(x3|x2) from p(x3,x2)
        p3c2 = p32 / p2

        p3c2_var = S_Functions.pdf_var(p3c2, n) / p2

        # performing the test with given PDFs.
        ck = p31 - torch.matmul(p3c2, p21)
        ck = torch.abs(ck)

        # calculating the error.
        ck_var = p31_var + torch.matmul(p3c2_var, p21**2) + torch.matmul(p3c2**2, p21_var)

        return ck.sum(), torch.sqrt(ck_var).sum()


    @staticmethod
    def find_bins(x, dx, bins_mode, max_bin=0):
        if bins_mode == 'std':
            std = np.std(x)
            x_std = x / std
        elif bins_mode == 'real':
            std = 1
            x_std = x.copy()
        if max_bin == 0:

            xdlimit = int(np.amin(x_std) * (1 / dx) - 1)
            xulimit = int(np.amax(x_std) * (1 / dx) + 1)
            xdlimit *= dx
            xulimit *= dx
            xbins = np.arange(xdlimit, xulimit + 2 * dx, dx) - dx / 2
        else:
            xdlimit = -max_bin * dx
            xulimit = max_bin * dx
            xbins = np.arange(xdlimit, xulimit + 2 * dx, dx) - dx / 2
        xbins *= std
        return xbins

    @staticmethod
    def digitize(x, bins, dtype=None, device=None):
        x[x > bins[-1]] = bins[-1]
        x[x < bins[0]] = bins[0]
        xdig = np.digitize(x, bins) - 1
        xdig = torch.tensor((xdig,), dtype=dtype, device=device)
        return torch.reshape(xdig, x.shape)

    @staticmethod
    def pdf(x_digitize, n, shape, dtype=None, device=None):
        v = torch.ones_like(x_digitize[0], dtype=dtype, device=device)
        if device.type == 'cpu' or device == None:
            p = torch.sparse.FloatTensor(x_digitize.long(), v, torch.Size(shape)).to_dense() / n
            return p.to_sparse()
        else:
            p = torch.cuda.sparse.FloatTensor(x_digitize.long(), v,
                                              torch.Size(shape)).to_dense() / n
            return p.to_sparse()

    @staticmethod
    def pdf_var(p, n):
        return p * (1 - p) / n

    @staticmethod
    def markov_length(ck, ck_error, is_coupled=False):
        diff = ck - ck_error

        if is_coupled:
            if len(diff.shape) == 3:
                b = np.ones_like(diff)
                diff = np.triu(diff, k=1) + np.tril(b, k=1)
                lm = np.argmax(diff <= 0, axis=2)
            else:
                lm = np.argmax(diff <= 0, axis=1) + 1

        else:
            if np.ndim(ck) == 2 and ck.shape[0] > 1:
                b = np.ones_like(diff)
                diff = np.triu(diff, k=1) + np.tril(b, k=1)
                lm = np.argmax(diff <= 0, axis=1)
            else:
                lm = np.argmax(diff <= 0) + 1
        return lm
