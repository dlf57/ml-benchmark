#!/usr/bin/env python3 

import molgrid
import pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os, glob, argparse
import numpy as np
import sklearn
from sklearn.linear_model import *
from sklearn.metrics import *
from openbabel import openbabel as ob
from openbabel import pybel
import pandas as pd

class View(nn.Module):
    def __init__(self, shape):        
        super(View, self).__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(*self.shape)
        

class Net(nn.Module):  #celestial-sweep-390
    def __init__(self, dims):
        super(Net, self).__init__()
        self.modules = []
        self.residuals = []
        nchannels = dims[0] 
        dim = dims[1]
        ksize = 3
        pad = ksize//2
        fmult = 1
        func = F.elu
            
        inmultincr = 0
            
        for m in range(5):
            module = []          
            inmult = 1
            filters = int(64*fmult  )
            startchannels = nchannels
            for i in range(6):
                conv = nn.Conv3d(nchannels*inmult, filters, kernel_size=ksize, padding=pad)
                inmult += inmultincr
                self.add_module('conv_%d_%d'%(m,i), conv)
                module.append(conv)
                module.append(func)
                nchannels = filters
            
            if True:
                #create a 1x1x1 convolution to match input filters to output
                conv = nn.Conv3d(startchannels, nchannels, kernel_size=1, padding=0)
                self.add_module('resconv_%d'%m,conv)
                self.residuals.append(conv)
            #don't pool on last module
            if m < 5-1:
                pool = nn.MaxPool3d(2)
                self.add_module('pool_%d'%m,pool)
                module.append(pool)
                dim /= 2
            self.modules.append(module)
            fmult *= 2
            
        last_size = int(dim**3 * filters)
        lastmod = []
        lastmod.append(View((-1,last_size)))
        
        if 1024 > 0:
            fc = nn.Linear(last_size, 1024)
            self.add_module('hidden',fc)
            lastmod.append(fc)
            lastmod.append(func)
            last_size = 1024
            
        fc = nn.Linear(last_size, 1)
        self.add_module('fc',fc)
        lastmod.append(fc)
        lastmod.append(nn.Flatten())
        self.modules.append(lastmod)
            

    def forward(self, x):
        isdense = False
        isres = False
        isres = True
                        
        for (m,module) in enumerate(self.modules):
            prevconvs = []
            if isres and len(self.residuals) > m:
                passthrough = self.residuals[m](x)
            else:
                isres = False
            for (l,layer) in enumerate(module):
                if isinstance(layer, nn.Conv3d) and isdense:
                    if prevconvs:
                        #concate along channels
                        x = torch.cat((x,*prevconvs),1)
                if isres and l == len(module)-1:
                    #at last relu, do addition before
                    x = x + passthrough

                x = layer(x)
                
                if isinstance(layer, nn.Conv3d) and isdense:
                    prevconvs.append(x) #save for later

        return x


parser = argparse.ArgumentParser(description='Evaluate a structure with a cnn')
parser.add_argument('cnnmodel',metavar='cnn pytorch model file')
parser.add_argument('linearmodel',metavar='linear baseline model')
parser.add_argument('molfile',metavar='molecule file',nargs='+')
parser.add_argument('--device',default="cuda",help="device to use (cuda or cpu)")
parser.add_argument('-n',type=int,default=20,help="number of evaluations")
args = parser.parse_args()


gmaker = molgrid.GridMaker(resolution=0.5, dimension = 16-0.5)
batch_size = 1
dims = gmaker.grid_dimensions(4)

device = args.device
model = Net(dims).to(device)
model.load_state_dict(torch.load(args.cnnmodel,map_location=torch.device(args.device)))
model.eval();

linmodel = pickle.load(open(args.linearmodel,'rb'))

for molfile in args.molfile:
    try:
        ext = os.path.splitext(molfile)[1].lstrip('.')
        mol = next(pybel.readfile(ext,molfile))
        mol.OBMol.Center()

        elemmap = {1: 0, 6: 1, 7: 2, 8: 3} #type indices
        typeradii = [1.0, 1.6, 1.5, 1.4] 
        def mytyper(atom):
            i = elemmap[atom.GetAtomicNum()]
            r = typeradii[i]
            return (i,r)
        typer = molgrid.PythonCallbackIndexTyper(mytyper, 4, ['H','C','N','O'])

        tensor_shape = (1,)+dims
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device)

        predictions = []
        with torch.no_grad():
            labelvec = torch.zeros(1, dtype=torch.float32, device=device)
            
            c = molgrid.CoordinateSet(mol,typer)
            ex = molgrid.Example()
            ex.coord_sets.append(c)
            batch = molgrid.ExampleVec([ex])
            types = c.type_index.tonumpy()
            tcnts = np.array([np.count_nonzero(types == i) for i in range(4)])
            base = linmodel.predict([tcnts])

            start = time.time()
            for _ in range(args.n):
                gmaker.forward(batch, input_tensor, random_translation=2, random_rotation=True)  #create grid; randomly translate/rotate molecule
                output = model(input_tensor).cpu().numpy()
                pred = base[0]+output[0][0]
                #print(pred)
                predictions.append(pred)
                
            end = time.time()
            
            
        print("RESULT",molfile, np.mean(predictions), (end-start)/args.n, sep=',') 
    except KeyboardInterrupt:
        raise        
    except Exception as e:
        print("ERROR",molfile,e)
