import copy
import glob
import h5py
import pickle
import numpy as np
import pandas as pd
import oddt
from oddt.fingerprints import ECFP


def ecfp(smiles):
    mol = oddt.toolkit.readstring("smi", smiles)
    fp = ECFP(mol, depth=2, size=4096, sparse=False)
    fpl = list(fp)
        
    rep = []
    for i in range(len(fpl)):
        rep.append(fpl[i])
    
    return rep


# load training data
trainset = open('/ihome/ghutchison/dlf57/ml-benchmark/train-ani.pkl', 'rb')
anitrain = pickle.load(trainset)

data = []
for ani in anitrain:
    smiles = ani['smiles']
    energy = ani['energy']
    
    # make bob representation
    rep = ecfp(smiles)

    d = {}
    d.update({'rep': rep})
    d.update({'energy': energy})
    data.append(d)

df = pd.DataFrame(data, columns=['rep', 'energy'])

# molecular descriptors for ML
ecfp = np.asarray(list(df['rep']), dtype=np.float16)
energy = np.asarray(list(df['energy']))
h5store = h5py.File('/zfs1/ghutchison/geoffh/ANI/ecfp-anitrain.h5', 'w')
h5store.create_dataset('ecfp', data=ecfp)
h5store.create_dataset('energy', data=energy)
h5store.close()
