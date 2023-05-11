import os
import scipy.sparse as sp
import numpy as np
import lzma

def size2intype(size):
    if size < 256: return np.uint8
    if size < (1 << 16):
        return np.uint16
    if size < (1 << 32):
        return np.uint32
    if size < (1 << 32):
        return np.uint64

def als_st_features(x):
    if os.path.isfile(x):
        x = open(x) if not x.endswith(".xz") else lzma.open(x, 'rt')
    samples = next(x).strip().split('\t')
    return set(line.split('\t')[0] for line in x), samples

def parse_als_st_matrix(x):
    if os.path.isfile(x):
        x = open(x) if not x.endswith(".xz") else lzma.open(x, 'rt')
    samples = next(x).strip().split('\t')
    data = {}
    mcount = -1
    for line in x:
        toks = line.strip().split('\t')
        feature = toks[0]
        counts = list(map(int, toks[1:]))
        lmcount = max(counts)
        dtype = size2intype(lmcount)
        mcount = max(mcount, lmcount)
        data[feature] = np.array(counts, dtype=dtype)
    counts = np.array(counts)
    return {"data": data, "samples": samples, "dtype": size2intype(mcount)}


def parse_als_file_set(folder):
    import glob
    from functools import reduce
    paths = list(glob.iglob(folder + "/*tx*"))
    feature_and_samples = list(map(als_st_features, paths))
    individual_sample_lists = [x[1] for x in feature_and_samples]
    all_features = sorted(reduce(lambda x, y: x | y, (x[0] for x in feature_and_samples)))
    all_samples = reduce(lambda x, y: x + y, individual_sample_lists)
    shape = (len(all_samples), len(all_features))
    feature_id_map = {feature: fid for fid, feature in enumerate(all_features)}
    def parse_submat_from_file(path):
        data = parse_als_st_matrix(path)
        features = list(data['data'].keys())
        columns = list(data['data'].values())
        submat = np.zeros((shape[1], len(columns[0])), dtype=data['dtype'])
        for (feature, column) in data['data'].items():
            submat[feature_id_map[feature]] = column
        return sp.csr_matrix(submat.T.copy())
    submats = list(map(parse_submat_from_file, paths))
    assert all(x.shape[1] == shape[1] for x in submats)
    assert all(x.shape[0] == len(samples) for x, samples in  zip(submats, individual_sample_lists))
    total_map = sp.vstack(submats)
    return {"features": all_features, "samples": all_samples, "data": total_map}
