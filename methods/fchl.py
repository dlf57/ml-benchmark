import glob
import h5py
import cclib
import pickle
import pandas as pd
from qml.fchl import generate_representation
from qml.fchl import get_local_kernels
from qml.math import cho_solve
import numpy as np
import re
from chemreps.utils.molecule import Molecule

# ignore 'nan' values rather than throw a Runtime warning
np.seterr(divide='ignore', invalid='ignore')

__nuc = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
        'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14,
        'P': 15, 'S': 16, 'Cl': 17}


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# load training data
trainset = open('/ihome/ghutchison/dlf57/ml-benchmark/train-ani.pkl', 'rb')
anitrain = pickle.load(trainset)

reps = []
energies = []
for ani in anitrain:
    try:
        coords = ani['coordinates']
        elements = ani['species']
        energy = ani['energy']
        nuc = []
        for k in range(len(elements)):
            at_num = __nuc['{}'.format(elements[k])]
            nuc.append(at_num)

        rep = generate_representation(coords, nuc, max_size=45)
        reps.append(rep)
        energies.append(energy)
    except:
        print(ani['molecule'])
        print(ani['species'])
        print(ani['coordinates'])
        print(ani['energy'])
    

X = np.array(reps)[:5000]
y = np.array(energies)[:5000]

sigma = 2.5
K = get_local_kernels(X, X, [sigma], cut_distance=10.0)[0]
K[np.diag_indices_from(K)] += 1e-8

alpha = cho_solve(K, y)
np.save('/ihome/ghutchison/dlf57/ml-benchmark/alpha-sig25-5k.npz', alpha)

data = []
for out in sorted(glob.iglob('/ihome/ghutchison/dlf57/ml-benchmark/molecules/stretch/*/sdf/*.sdf'), key=numericalSort):
    name = out.split('stretch/')[1].split('/sdf')[0]
    pt = out.split('sdf/')[1].split('.')[0]
    if name != 'HF':
        mol = Molecule(out)
        coords = mol.xyz
        at_num = mol.at_num

        rep = generate_representation(coords, at_num, max_size=45)
        rep = np.array([rep])
        
        
        K_pred = get_local_kernels(rep, X, [sigma], cut_distance=10.0)[0]
        energy = list(np.dot(K_pred, alpha))[0]


        d = {}
        d.update({'name': name})
        d.update({'point': pt})
        d.update({'fchl_energy': energy})
        data.append(d)
    else:
        d = {}
        d.update({'name': name})
        d.update({'point': pt})
        d.update({'fchl_energy': np.nan})
        data.append(d)
        
    
df = pd.DataFrame(data, columns=['name', 'point', 'fchl_energy'])
df.to_csv('/ihome/ghutchison/dlf57/ml-benchmark/data/fchl-alpha25-5k.csv', index=False)




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
        
    mol = Molecule(out)
    coords = mol.xyz
    at_num = mol.at_num

    rep = generate_representation(coords, at_num, max_size=45)
    rep = np.array([rep])


    K_pred = get_local_kernels(rep, X, [sigma], cut_distance=10.0)[0]
    energy = list(np.dot(K_pred, alpha))[0]

    d = {}
    d.update({'name': name})
    d.update({'phi': phi})
    d.update({'psi': psi})
    d.update({'theta': theta})
    d.update({'fchl_energy': energy})
    results.append(d)
    print('{} \t{} \t{} \t{} \t{}'.format(name, phi, psi, theta, energy))
    
df = pd.DataFrame(results, columns=['name', 'phi', 'psi', 'theta', 'fchl_energy'])
df.to_csv('/ihome/ghutchison/dlf57/ml-benchmark/data/fchl-dihedral-alpha25-5k', index=False)
