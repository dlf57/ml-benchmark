import copy
import glob
import h5py
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from collections import OrderedDict
from math import sqrt

__nuc = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
        'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14,
        'P': 15, 'S': 16, 'Cl': 17}

def bag_updater(bag, bag_sizes):
    """
    Checks if new bag created is larger than previous bags and updates the
    bag size if it is larger

    Parameters
    -----------
    bag : dict
        dictionary of all the bags for the current molecule
    bag_sizes : dict
        dictionary of the largest bag sizes in the dataset
    """
    # grab the keys for the bag
    bag_keys = list(bag.keys())
    for i in range(len(bag_keys)):
        key = bag_keys[i]
        # check if the new bag has a larger size than what is in bag_sizes
        #   update the bag_sizes if it is larger, pass if it is not
        if key in bag_sizes:
            if bag[key] > bag_sizes[key]:
                bag_sizes[key] = bag[key]
        # if the bag is not in bag_sizes, add it to bag_sizes
        else:
            bag_sizes[key] = bag[key]

def bag_organizer(bag_set, bag_sizes):
    """
    Sorts bags by magnitude, pads, and concactenates into one feature list

    Parameters
    -----------
    bag_set : dict
        dictionary filled with all of the current molecules information
    bag_sizes : dict
        dictionary of the largest bag sizes in the dataset

    Returns
    --------
    feat_list : list
        sorted and padded feature list of the current molecule
    """
    feat_list = []
    bag_keys = list(bag_set.keys())
    for i in range(len(bag_keys)):
        # grab the size of the largest bag and length of current molecule bag
        size = bag_sizes[bag_keys[i]] + 1
        baglen = len(bag_set[bag_keys[i]])
        if baglen > (size - 1):
            raise Exception(
                '{}-bag size is too small. Increase size to {}.'.format(bag_keys[i], baglen))
        pad = size - baglen
        # sort the bag by magnitude and pad with zeros to make all same length
        bag_set[bag_keys[i]] = sorted(bag_set[bag_keys[i]], reverse=True)
        bag_set[bag_keys[i]].extend([0.] * pad)
        feat_list.append(bag_set[bag_keys[i]])

    return feat_list


def length(coordinates, atomi, atomj):
    """
    Returns the length between two atoms

    Parameters
    -----------
    molecule : object
        molecule object
    atomi, atomj : int
        atoms

    Returns
    --------
    rij : float
        length between the two
    """
    x = coordinates[atomi][0] - coordinates[atomj][0]
    y = coordinates[atomi][1] - coordinates[atomj][1]
    z = coordinates[atomi][2] - coordinates[atomj][2]
    rij = sqrt((x ** 2) + (y ** 2) + (z ** 2))
    return np.float16(rij)


def bag_of_bonds(coords, species, bags, bag_sizes):
    '''
    Parameters
    ---------
    coords: list
        atom coordinates
    species: list
        atom species
    bags: dict
        dict of all bags for the dataset
    bag_sizes: dict
        dict of size of the largest bags in the dataset

    Returns
    -------
    bob: vector
        vector of all bonds in the molecule
    '''
    # copy bags dict to ensure it does not get edited
    bag_set = copy.deepcopy(bags)
    # current_molecule = Molecule(mol_file)
    n_atom = len(species)
    sym = species
    at_num = []
    for elem in sym:
        atomic_number = __nuc['{}'.format(elem)]
        at_num.append(atomic_number)
    
    for i in range(n_atom):
        for j in range(i, n_atom):
            atomi = sym[i]
            atomj = sym[j]
            zi = at_num[i]
            zj = at_num[j]

            if i == j:
                mii = 0.5 * zi ** 2.4
                bag_set[atomi].append(mii)

            else:
                if zj > zi:
                        # swap ordering
                    atomi, atomj = atomj, atomi
                bond = "{}{}".format(atomi, atomj)

                # rij = sqrt((xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2)
                rij = length(coords, i, j)
                # print(rij)
                mij = (zi * zj) / rij

                bag_set[bond].append(mij)

    # sort bags by magnitude, pad, concactenate
    bob = bag_organizer(bag_set, bag_sizes)

    # flatten bob into one list and store as a np.array
    bob = np.array(list(chain.from_iterable(bob)), dtype=np.float16)

    return bob

# load bags
with open('/ihome/ghutchison/dlf57/ml-benchmark/dataset-bags.pkl', 'rb') as f:
    set_bags = pickle.load(f)

bags = set_bags[0]
bag_sizes = set_bags[1]

# load training data
trainset = open('/ihome/ghutchison/dlf57/ml-benchmark/train-ani.pkl', 'rb')
anitrain = pickle.load(trainset)

data = []
for ani in anitrain:
    coords = ani['coordinates']
    elements = ani['species']
    energy = ani['energy']
    
    # make bob representation
    rep = bag_of_bonds(coords, elements, bags, bag_sizes)

    d = {}
    d.update({'rep': rep})
    d.update({'energy': energy})
    data.append(d)

df = pd.DataFrame(data, columns=['rep', 'energy'])

# molecular descriptors for ML
bobrep = np.asarray(list(df['rep']), dtype=np.float16)
energy = np.asarray(list(df['energy']))
h5store = h5py.File('/zfs1/ghutchison/geoffh/ANI/bob-anitrain.h5', 'w')
h5store.create_dataset('bob', data=bobrep)
h5store.create_dataset('energy', data=energy)
h5store.close()
