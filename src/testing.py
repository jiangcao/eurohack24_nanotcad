# %%
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import numba as nb 
from bse import BSESolver


datasetname='/usr/scratch2/tortin16/jiacao/BSE_calc/agnr7/python/test/data_len10_ndiag14_nen600_'
indata=np.load(datasetname+'input.npz')   
outdata=np.load(datasetname+'output.npz')   
V=indata['coulomb']
GL=indata['G_lesser']
GG=indata['G_greater']
W=indata['W0_r']
nm_dev = int(indata['nm_dev'])
ndiag = int(indata['ndiag'])

bse=BSESolver(nm_dev,ndiag)
bse._preprocess()

num_E=GG.shape[-1]-1
bse._alloc_twobody_matrix(num_E=num_E)

bse._calc_noninteracting_twobody(GG,GL)

P = bse._solve_interacting_twobody(V,W)