# %%
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import numba as nb 
from qttools.datastructures import DSBCOO
from bse import BSESolver
from mpi4py.MPI import COMM_WORLD as comm


# %%
num_sites = 50
cutoff = 4
bse=BSESolver(num_sites,cutoff)
bse._preprocess()


# %%
num_E=5
bse._alloc_twobody_matrix(num_E=num_E*2-1)

# %%
ARRAY_SHAPE = (num_sites, num_sites, num_E)

GG = np.random.rand(*ARRAY_SHAPE)
GL = np.random.rand(*ARRAY_SHAPE)
W = np.random.rand(num_sites,num_sites)
V = np.random.rand(num_sites,num_sites)

comm.barrier()
GG = comm.bcast(GG, root=0)
GL = comm.bcast(GL, root=0)

# %%
bse._calc_noninteracting_twobody(GG,GL)

# %%
bse._calc_kernel(V,W)


