'''
Script for training ANI in BOB represenation with Random Forest Regressor
and predicting the energy of the benchmarking set.
'''
import glob
import h5py
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pybel
import oddt
from oddt.fingerprints import ECFP
import re


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# ignore 'nan' values rather than throw a Runtime warning
np.seterr(divide='ignore', invalid='ignore')

ani = '/zfs1/ghutchison/geoffh/ANI/ecfp-anitrain.h5'
h5 = h5py.File(ani, 'r')
molecule = h5.get('ecfp')
energy = h5.get('energy')
molecule = np.array(molecule)
energy = np.array(energy)
h5.close()
        
n_estimators = 100
n_jobs = -1
regr = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
regr.fit(molecule, energy)

# store regressor
joblib_file = "/zfs1/ghutchison/geoffh/ANI/ECFP-RFR-joblib.pkl"
joblib.dump(regr, joblib_file)

# ecfp
# I know I can make ecfp w/ sdf but wanted to keep rep generation consistent
# ..between training set and test set
def ecfp(smiles):
    mol = oddt.toolkit.readstring("smi", smiles)
    fp = ECFP(mol, depth=2, size=4096, sparse=False)
    fpl = list(fp)
        
    rep = []
    for i in range(len(fpl)):
        rep.append(fpl[i])
    
    return rep

##### Bond Stretch #####
results = []
for sdf in sorted(glob.iglob('/ihome/ghutchison/dlf57/ml-benchmark/molecules/stretch/*/sdf/*.sdf'), key=numericalSort):
    name = sdf.split('ch/')[1].split('/sdf')[0]
    if name != 'HF':
        pt = sdf.split('sdf/')[1].split('.')[0]
        mol = next(pybel.readfile('sdf', sdf))
        smiles = mol.write('smi').split('\t')[0]
        rep = ecfp(smiles)
        pred = regr.predict([rep])[0]
        dict1 = {}
        dict1.update({'name': name})
        dict1.update({'point': pt})
        dict1.update({'ecfprfr_energy': pred})
        results.append(dict1)
        print('{} \t{} \t{}'.format(name, pt, pred))
    else:
        pt = sdf.split('sdf/')[1].split('.')[0]
        dict1 = {}
        dict1.update({'name': name})
        dict1.update({'point': pt})
        dict1.update({'ecfprfr_energy': np.nan})
        results.append(dict1)
        print('{} \t{} \t{}'.format(name, pt, pred))

df = pd.DataFrame(results, columns=['name', 'point', 'ecfprfr_energy'])
df.to_csv('/ihome/ghutchison/dlf57/ml-benchmark/data/ecfp-rfr.csv', index=False)


##### Dihedral #####
results = []
for sdf in sorted(glob.iglob('/ihome/ghutchison/dlf57/ml-benchmark/molecules/dihedral/*/sdf/*.sdf'), key=numericalSort):
    name = sdf.split('ral/')[1].split('/sdf')[0]
    if name == 'sucrose' or name == 'biphenyl-twist':
        phi = np.nan
        psi = np.nan
        theta = sdf.split('sdf/')[1].split('.')[0].split('_')[1]
    else:
        phi = sdf.split('sdf/')[1].split('.')[0].split('_')[1]
        psi = sdf.split('sdf/')[1].split('.')[0].split('_')[2]
        theta = np.nan
        
    mol = next(pybel.readfile('sdf', sdf))
    smiles = mol.write('smi').split('\t')[0]
    rep = ecfp(smiles)
    pred = regr.predict([rep])[0]

    d = {}
    d.update({'name': name})
    d.update({'phi': phi})
    d.update({'psi': psi})
    d.update({'theta': theta})
    d.update({'ecfprfr_energy': pred})
    results.append(d)
    print('{} \t{} \t{} \t{} \t{}'.format(name, phi, psi, theta, pred))
    
df = pd.DataFrame(results, columns=['name', 'phi', 'psi', 'theta', 'ecfprfr_energy'])
df.to_csv('/ihome/ghutchison/dlf57/ml-benchmark/data/ecfp-rfr-dihedral.csv', index=False)