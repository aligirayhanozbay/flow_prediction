import h5py
import numpy as np
import argparse
import json
import itertools
import multiprocessing as mp
from multiprocessing import shared_memory
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, normalized_mutual_info_score

from ..training.ShallowDecoderDataset import _extract_values, _extract_variable_names

def compute_moments(f):
    nvars = f.shape[-1]
    means = []
    abs_means = []
    stds = []
    abs_stds = []
    cvs = []
    abs_cvs = []
    for k in range(nvars):
        x = f.reshape(-1,f.shape[-1])[...,k]
        mean = np.mean(x)
        std = np.std(x)
        cv = std/mean
        means.append(mean)
        stds.append(std)
        cvs.append(cv)

        x = np.abs(x)
        mean = np.mean(x)
        std = np.std(x)
        cv = std/mean
        abs_means.append(mean)
        abs_stds.append(std)
        abs_cvs.append(cv)

    return {'means': means, 'stds': stds, 'cvs': cvs, 'abs_means': abs_means, 'abs_stds':abs_stds, 'abs_cvs': abs_cvs}

def input_output_covariance(x,t):
    nvars = t.shape[-1]
    x = x.reshape(-1, x.shape[-1])
    t = t.reshape(-1, *t.shape[2:])

    vcov_norms = []
    vcorrcoef_norms = []
    for k in range(nvars):
        xt = np.concatenate([x,t[...,k]],-1)
        cov = np.cov(xt, rowvar=False)
        _cov_diag_factor = np.diag(cov)**-0.5
        corrcoef = np.expand_dims(_cov_diag_factor,0) * cov * np.expand_dims(_cov_diag_factor,1)
        vcov = cov[:x.shape[-1],x.shape[-1]:]
        vcorrcoef = corrcoef[:x.shape[-1],x.shape[-1]:]
        vcov_norms.append(np.sum(vcov**2)**0.5)
        vcorrcoef_norms.append(np.sum(vcorrcoef**2)**0.5)
    return {'covariance_norms': vcov_norms, 'correlation_coeff_norms': vcorrcoef_norms}

def input_output_spearmanr(x,t):
    nvars = t.shape[-1]
    x = x.reshape(-1, x.shape[-1])
    t = t.reshape(-1, *t.shape[2:])

    spearmanrs = []
    pvalues = []
    for k in range(nvars):
        sr = spearmanr(x,t[...,k])
        vsr = sr.correlation[:x.shape[-1],x.shape[-1]:]
        vp = sr.pvalue[:x.shape[-1],x.shape[-1]:]
        spearmanrs.append(np.sum(vsr**2)**0.5)
        pvalues.append(np.sum(vp**2)**0.5)
    return {'spearmanr': spearmanrs, 'pvalue': pvalues}

def _make_shared_array(X):
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    b = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
    b[:] = X[:]
    return shm, b

def _get_shared_array(buf_name, buf_dtype, buf_shape):
    existing_shm = shared_memory.SharedMemory(name=buf_name)
    X = np.ndarray(buf_shape, dtype=buf_dtype, buffer=existing_shm.buf)
    return existing_shm,X

def _calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = adjusted_mutual_info_score(np.sum(c_xy,axis=0), np.sum(c_xy,axis=1))
    return mi

def _calc_MI_worker(indices, xspec, tspec, mspec, bins):
    _shx, _x = _get_shared_array(*xspec)
    _sht, _t = _get_shared_array(*tspec)
    _shm, _mi = _get_shared_array(*mspec)

    for v,i,j in indices:
        mival = _calc_MI(_x[i], _t[v,j], bins)
        _mi[v,i,j] = mival

    return 0


def mutual_information(x,t,bins=50):
    ntvars = t.shape[-1]
    nxvars = x.shape[-1]
    
    shx, x = _make_shared_array(x.reshape(-1, x.shape[-1]).transpose((1,0)))
    xspec = (shx.name, x.dtype, x.shape)
    sht, t = _make_shared_array(t.reshape(-1, *t.shape[2:]).transpose((2,1,0)))
    tspec = (sht.name, t.dtype, t.shape)
    shm, mi_arr = _make_shared_array(np.zeros((t.shape[0],x.shape[0],t.shape[1])))
    mispec = (shm.name, mi_arr.dtype, mi_arr.shape)
    
    var_indices = list(itertools.product(range(t.shape[0]), range(x.shape[0]), range(t.shape[1])))
    var_indices = np.array_split(var_indices, mp.cpu_count())
    
    pool = mp.Pool(mp.cpu_count())
    
    pr = pool.starmap(_calc_MI_worker,
                      zip(
                          var_indices,
                          itertools.repeat(xspec, mp.cpu_count()),
                          itertools.repeat(tspec, mp.cpu_count()),
                          itertools.repeat(mispec, mp.cpu_count()),
                          itertools.repeat(bins, mp.cpu_count())
                      )
                  )

    return {'mi': list(map(lambda m: np.sum(m**2)**0.5, mi_arr))}
    

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('-b', type=int, default=50, help='mutual info bins')
    args = parser.parse_args()

    if args.cfg is not None:
        sensormask = json.load(open(args.cfg,'r'))['dataset']['sensor_masks']
    else:
        sensormask = None

    with h5py.File(args.file,'r') as f:
        varnames = _extract_variable_names(f)
        print(varnames)
    d = _extract_values(args.file, retain_time_dimension=True, retain_variable_dimension=True, sensor_masks=sensormask)
    moments = compute_moments(d[1])
    for stat in moments:
        print(f'{stat}: {moments[stat]}')
    covs = input_output_covariance(*d[:2])
    for stat in covs:
        print(f'{stat}: {covs[stat]}')
    srs = input_output_spearmanr(*d[:2])
    for stat in srs:
        print(f'{stat}: {srs[stat]}')
    mi = mutual_information(*d[:2], bins = args.b)
    for stat in mi:
        print(f'{stat}: {mi[stat]}')
    import pdb; pdb.set_trace()
        
    
