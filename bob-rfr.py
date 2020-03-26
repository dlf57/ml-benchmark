'''
Script for training ANI in BOB represenation with Random Forest Regressor
and predicting the energy of the benchmarking set.
'''
import glob
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from chemreps.bag_of_bonds import bag_of_bonds
import re


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# ignore 'nan' values rather than throw a Runtime warning
np.seterr(divide='ignore', invalid='ignore')

ani = '/zfs1/ghutchison/geoffh/ANI/bob-ani.h5'
h5 = h5py.File(ani, 'r')
molecule = h5.get('bob')
energy = h5.get('energy')
molecule = np.array(molecule)
energy = np.array(energy)
h5.close()
        
n_estimators = 100
n_jobs = -1
regr = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
regr.fit(molecule, energy)

# store regressor
joblib_file = "/zfs1/ghutchison/geoffh/ANI/BOB-RFR-joblib.pkl"
joblib.dump(regr, joblib_file)


with open('/ihome/ghutchison/dlf57/Molecules/ANI/bob-ani_bags.pkl', 'rb') as f:
    set_bags = pickle.load(f)

bags = set_bags[0]
bag_sizes = set_bags[1]

results = []
for sdf in sorted(glob.iglob('/ihome/ghutchison/dlf57/ml-benchmark/molecules/*/sdf/*.sdf'), key=numericalSort):
    name = sdf.split('es/')[1].split('/sdf')[0]
    if name != 'HF':
        pt = sdf.split('sdf/')[1].split('.')[0]
        rep = bag_of_bonds(sdf, bags, bag_sizes)
        pred = regr.predict([rep])[0]
        dict1 = {}
        dict1.update({'name': name})
        dict1.update({'point': pt})
        dict1.update({'bobrfr_energy': pred})
        results.append(dict1)
        print('{} \t{} \t{}'.format(name, pt, pred))
    else:
        pass

df = pd.DataFrame(results, columns=['name', 'point', 'bobrfr_energy'])
df.to_csv('/ihome/ghutchison/dlf57/ml-benchmark/data/bob-rfr.csv', index=False)
