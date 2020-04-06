'''
Script for training ANI in BOB represenation with Random Forest Regressor
and predicting the energy of the benchmarking set.
'''
import glob
import h5py
import numpy as np
import pandas as pd
import pickle
from sklearn.kernel_ridge import KernelRidge
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
        
kernel = "rbf"
sigma = 4000
alpha = 1e-8
gamma = 1.0/(2*sigma**2)
regr = KernelRidge(gamma=gamma, kernel=kernel, alpha=alpha)
regr.fit(molecule[:100000], energy[:100000])

# store regressor
joblib_file = "/zfs1/ghutchison/geoffh/ANI/BOB-KRR-joblib.pkl"
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
        dict1.update({'bobkrr_energy': pred})
        results.append(dict1)
        print('{} \t{} \t{}'.format(name, pt, pred))
    else:
        pass

df = pd.DataFrame(results, columns=['name', 'point', 'bobkrr_energy'])
df.to_csv('/ihome/ghutchison/dlf57/ml-benchmark/data/bob-krr.csv', index=False)
