# %%
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import cupy as cp
from bse import BSESolver


datasetname='/usr/scratch2/tortin16/jiacao/BSE_calc/agnr7/python/test/data_len10_ndiag14_nen600_'
indata=np.load(datasetname+'input.npz')   
outdata=np.load(datasetname+'output.npz')   

V=cp.array( indata['coulomb'] ,dtype=cp.complex128)
GL=cp.array( indata['G_lesser'] ,dtype=cp.complex128)
GG=cp.array( indata['G_greater'] ,dtype=cp.complex128)
W=cp.array( indata['W0_r'] ,dtype=cp.complex128)

nm_dev = int(indata['nm_dev'])
ndiag = int(indata['ndiag'])

#ndiag=1
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
