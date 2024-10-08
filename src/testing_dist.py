import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBCOO
from qttools.utils.gpu_utils import xp
from scipy import sparse

from bse import BSESolverDist

datasetname = "/usr/scratch2/tortin16/jiacao/BSE_calc/agnr7/python/test/data_len10_ndiag14_nen600_"
indata = np.load(datasetname + "input.npz")
outdata = np.load(datasetname + "output.npz")

V = xp.array(indata["coulomb"], dtype=xp.complex128)
W = xp.array(indata["W0_r"], dtype=xp.complex128)
GL = xp.array(indata["G_lesser"], dtype=xp.complex128)
GG = xp.array(indata["G_greater"], dtype=xp.complex128)

# Prepare data.
ndiag = 4
for i, j in np.ndindex(GL.shape[:-1]):
    if np.abs(i - j) > ndiag:
        GL[i, j] = 0
        GG[i, j] = 0

block_sizes = [7] * 20
g_lesser = DSBCOO.from_sparray(
    sparse.coo_array(GL[..., 100].get()), block_sizes, (600,)
)
g_greater = DSBCOO.from_sparray(
    sparse.coo_array(GL[..., 100].get()), block_sizes, (600,)
)
gl_stack_section_offsets = xp.hstack(([0], xp.cumsum(g_lesser.stack_section_sizes)))
gg_stack_section_offsets = xp.hstack(([0], xp.cumsum(g_greater.stack_section_sizes)))
for i, j in zip(g_greater.rows, g_lesser.cols):
    g_lesser[i, j] = GL[
        i,
        j,
        gl_stack_section_offsets[comm.rank] : gl_stack_section_offsets[comm.rank + 1],
    ]
    g_greater[i, j] = GG[
        i,
        j,
        gg_stack_section_offsets[comm.rank] : gg_stack_section_offsets[comm.rank + 1],
    ]

nm_dev = int(indata["nm_dev"])
# ndiag = int(indata["ndiag"])

start_time = time.time()

bse = BSESolverDist(nm_dev, ndiag // 2)

bse._preprocess(rows=g_lesser.rows, cols=g_lesser.cols)

num_E = 10
bse._alloc_twobody_matrix(num_E=num_E)

# Visualize distribution of matrix elements.
# if comm.rank == 0:
#     fig, ax = plt.subplots()
#     rank_map = np.zeros_like(bse.L0mat.rows)
#     for i in range(comm.size):
#         rank_map[bse.L0mat.nnz_section_offsets[i] : bse.L0mat.nnz_section_offsets[i + 1]] = i

#     rows, cols = bse.L0mat.rows, bse.L0mat.cols

#     coo = sparse.coo_array((get_host(rank_map+1), (get_host(rows), get_host(cols))))
#     ax.matshow(np.abs(coo.toarray()))
#     plt.show()


if comm.rank == 0:
    print("compute correlations ...", flush=True)

bse._calc_noninteracting_twobody(g_greater, g_lesser, step_E=20)

if comm.rank == 0:
    finish_time = time.time()
    print(" compute time = ", finish_time - start_time)
    start_time = finish_time

#     print("solve ...", flush=True)

# P, Gamma = bse._solve_interacting_twobody(V, W)

# if comm.rank == 0:
#     finish_time = time.time()
#     print(" compute time = ", finish_time - start_time)
#     start_time = finish_time

#     print("dense solve ...", flush=True)

# P2, Gamma2 = bse._densesolve_interacting_twobody(V, W)

# if comm.rank == 0:
#     finish_time = time.time()
#     print(" compute time = ", finish_time - start_time)
#     start_time = finish_time

# print("rank=", comm.rank, "rel error=", np.sum(np.abs(P2 - P)) / np.sum(np.abs(P2)))

# filename = datasetname + "output"
# np.savez(filename + "_rank" + str(comm.rank), P=get_host(P))
