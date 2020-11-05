import glob
import h5py
import pickle
import pybel
import numpy as np

def unpackani(h5file):
    h5 = h5py.File(h5file)
    results = []
    for key in h5.keys():
        item = h5[key]
        keys = [i for i in item.keys()]
        for k in keys:
            data = {'molecule': k}
            for j in item[k]:
                dataset = np.array(item[k][j].value)
                if type(dataset) is np.ndarray:
                    if dataset.size != 0:
                        if type(dataset[0]) is np.bytes_:
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({j: dataset})
            results.append(data)
    h5.close()
    return results


# smiles from test set to exclude from training
testsmi = ['C#N', 'N#C', 'C=[O]', '[C]=O', 'O=[C]', '[O]=C', 'c1ccccc1', 'N#N',
           'O', 'C', '[H][H]', 'C=C', 'N', 'C#C', 'CO', 'OC', 'F']

n_mols = 0
training = []
for gdb in sorted(glob.iglob('/zfs1/ghutchison/geoffh/ANI-1_release/*.h5')):
    ani = unpackani(gdb)
    for data in ani:
        # taking 167480 takes 5 geoms of all molecules in
        # ..1-7 while taking 5 geoms of half molecules in 8
        if n_mols < 167480:
            # parse smile from hdf5
            anismi = ''.join(data['smiles'])
            # write non-H smile w/ pybel
            mol = pybel.readstring('smi', anismi)
            smi = mol.write('smi').split('\t')[0]
            if smi in testsmi:
                # print to make sure the smile lines up with the species
                print('Excluded from training', smi, data['species'])
                pass
            else:
                # grab 5 geometries of that molecule
                for i in range(5):
                    molecule = data['molecule']
                    energy = data['energies'][i]
                    species = data['species']
                    coords = data['coordinates'][i]
                    smile = smi
                    d = {}
                    d.update({'molecule': molecule})
                    d.update({'energy': energy})
                    d.update({'species': species})
                    d.update({'coordinates': coords})
                    d.update({'smiles': smile})
                    training.append(d)
                    n_mols += 1

out = open('/ihome/ghutchison/dlf57/ml-benchmark/train-ani.pkl','wb')
pickle.dump(training, out)
out.close()

# test that it wrote the data
p_in = open("/ihome/ghutchison/dlf57/ml-benchmark/train-ani.pkl", "rb")
t_train = pickle.load(p_in)

print(t_train[0])
