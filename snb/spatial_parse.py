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
    basename = os.path.split()[-1]
    basetoks = basename.split("_")
    sample, slide = basetoks[:2]
    x = open(x) if not x.endswith(".xz") else lzma.open(x, 'rt')
    samples = [f"{sample}_{slide}_{y}" for y in next(x).strip().split('\t')]
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
        data[feature] = np.array(counts, dtype=np.int32)
    counts = np.array(counts)
    return {"data": data, "samples": samples, "dtype": np.int32}


def parse_als_file_set(folder):
    '''
    From a folder containing many expression files, load them all as one large sparse matrix.
    '''
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
    maxv = max(map(np.max, submats))
    total_map = sp.vstack(submats, dtype=size2intype(maxv))
    coords = np.array([list(map(float, x.split("_")[-2:])) for x in all_samples])
    return {"features": all_features, "samples": all_samples, "data": total_map, "coords": coords}

def parse_aquila_folder(path):
    import gzip
    import json
    from scipy.io import mmread
    info = json.loads(open(path + "/info.json").read())
    markers = list(map(str.strip, gzip.open(path + "/markers_names.csv.gz", "rt")))[1:]
    xyzfp = gzip.open(path + "/cell_info.csv.gz", "rt")
    next(xyzfp) # Skip header
    xyz = np.array([list(map(float, x.strip().split(','))) for x in xyzfp])
    return {"info": info, "features": markers, "coords": xyz, "data": mmread(gzip.open("cell_exp.mtx.gz", "rt"))}

__all__ = ["parse_als_st_matrix", "als_st_features", "parse_als_file_set", "size2intype"]
