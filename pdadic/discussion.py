import numpy as np
import matplotlib.pyplot as plt
import math


def discuss_error(qmap, qrecomp):
    _K = len(qmap.values)

    cols = min(4,_K)
    rows = math.ceil(_K/4)

    plt.figure(figsize=(6*cols, 4*rows))

    for i,a,b in list(zip(range(_K), qmap.values, qrecomp.values)):

        _N = len(a)
        diff = np.array(a) - np.array(b)
        ax = plt.subplot(rows, cols , i+1)
        ax.bar(np.arange(_N), np.real(diff))
        ax.bar(np.arange(_N), np.imag(diff))

        # ax.set_title(f'Real and imaginary residuals by {_N} values')
        ax.set_xticks([0, _N-1])
        ax.set_xticklabels(['0', '1'])

    plt.show()

def discuss_imaginary_parts(qdecomp):
    _K = len(qdecomp.contributions)

    plt.figure(figsize=(4*(_K), 4))

    for i,a in enumerate(qdecomp.contributions):

        _N = len(a)
        ax = plt.subplot(1, _K, i+1)
        ax.bar(np.arange(_N), np.imag(a))

        # ax.set_title(f'Real and imaginary residuals by {_N} values')
        ax.set_xticks([0, _N-1])
        ax.set_xticklabels(['0', '1'])

    plt.show()

def discuss_approximations(qmap):
    _K = len(qmap.values)

    plt.figure(figsize=(4*(_K), 4))

    for i,a in enumerate(qmap.values):

        _N = len(a)
        ax = plt.subplot(1, _K, i+1)
        ax.bar(np.arange(_N), a)

        # ax.set_title(f'Real and imaginary residuals by {_N} values')
        ax.set_xticks([0, _N-1])
        ax.set_xticklabels(['0', '1'])

    plt.show()