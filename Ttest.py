import numpy as np
from scipy.stats import ttest_ind
import h5py
import matplotlib.pyplot as plt

def ttest(a, b, axis=0, equal_var=True, nan_policy='propagate',
          alternative='two.sided'):
    tval, pval = ttest_ind(a=a, b=b, axis=axis, equal_var=equal_var,
                           nan_policy=nan_policy)
    if alternative == 'greater':
        if tval < 0:
            pval = 1 - pval / 2
        else:
            pval = pval / 2
    elif alternative == 'less':
        if tval < 0:
            pval /= 2
        else:
            pval = 1 - pval / 2
    else:
        assert alternative == 'two.sided'
    return tval, pval


def loaddata():
    data_ca = h5py.File('canary .mat', 'r')
    canary = np.array(data_ca['canary ']).reshape(5000, 1)
    data_ex = h5py.File('extra.mat', 'r')
    extra = np.array(data_ex['extra']).reshape(5000, 1)
    data_in = h5py.File('independent.mat', 'r')
    independent = np.array(data_in['independent']).reshape(5000, 1)
    data_sh = h5py.File('shared.mat', 'r')
    shared = np.array(data_sh['shared']).reshape(5000, 1)
    return canary, extra, independent, shared



if __name__ == "__main__":
    canary, extra, independent, shared = loaddata()
    mu_canary = np.mean(canary)
    print('mean_$S_C$: ', mu_canary)
    mu_shared = np.mean(shared)
    print('mean_$S_S$: ', mu_shared)
    mu_extra = np.mean(extra)
    print('mean_$S_E$: ', mu_extra)
    mu_independent = np.mean(independent)
    print('mean_$S_I$: ', mu_independent)
    tval, pval = ttest(shared, canary, alternative="greater")
    print('tval 0 hypothesis S_S <= S_C: ', tval, ' pval: ', pval)
    tval, pval = ttest(shared, canary, alternative="two.sided")
    print('tval 0 hypothesis S_S == S_C: ', tval, ' pval: ', pval)
    tval, pval = ttest(shared, extra, alternative="greater")
    print('tval 0 hypothesis S_S <= S_E: ', tval, ' pval: ', pval)
    tval, pval = ttest(shared, extra, alternative="two.sided")
    print('tval 0 hypothesis S_S == S_E: ', tval, ' pval: ', pval)
    tval, pval = ttest(shared, independent, alternative="greater")
    print('tval 0 hypothesis S_S <= S_I: ', tval, ' pval: ', pval)
    tval, pval = ttest(shared, independent, alternative="two.sided")
    print('tval 0 hypothesis S_S == S_I: ', tval, ' pval: ', pval)