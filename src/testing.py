# %%
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import cupy as cp
from bse import BSESolver
from qttools.utils.gpu_utils import xp


datasetname='/usr/scratch2/tortin16/jiacao/BSE_calc/agnr7/python/test/data_len10_ndiag14_nen600_'
indata=np.load(datasetname+'input.npz')   
outdata=np.load(datasetname+'output.npz')   

V=xp.array( indata['coulomb'] ,dtype=xp.complex128)
GL=xp.array( indata['G_lesser'] ,dtype=xp.complex128)
GG=xp.array( indata['G_greater'] ,dtype=xp.complex128)
W=xp.array( indata['W0_r'] ,dtype=xp.complex128)

nm_dev = int(indata['nm_dev'])
ndiag = int(indata['ndiag'])

ndiag=10
bse=BSESolver(nm_dev,ndiag)
bse._preprocess()

num_E=100
bse._alloc_twobody_matrix(num_E=num_E)

if comm.rank == 0:
   print('compute correlations ...', flush=True)

bse._calc_noninteracting_twobody(GG,GL)

if comm.rank == 0:
   print('solve ...', flush=True)

# P, Gamma = bse._solve_interacting_twobody(V,W)

P = bse._densesolve_interacting_twobody(V,W)
