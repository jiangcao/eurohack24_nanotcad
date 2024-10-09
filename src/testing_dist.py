import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBCOO
from qttools.utils.gpu_utils import xp
from scipy import sparse

from bse import BSESolver, BSESolverDist

datasetname = "/usr/scratch2/tortin16/jiacao/BSE_calc/agnr7/python/test/data_len10_ndiag14_nen600_"
indata = np.load(datasetname + "input.npz")
outdata = np.load(datasetname + "output.npz")

V = xp.array(indata["coulomb"], dtype=xp.complex128)
W = xp.array(indata["W0_r"], dtype=xp.complex128)


def get_data(size) -> np.ndarray:
    """Returns some random complex boundary blocks."""
    # Generate a decaying random complex array.
    arr = np.triu(np.arange(size, 0, -1) + np.arange(size)[:, np.newaxis])
    arr = arr.astype(np.complex128)
    arr += arr.T
    arr **= 2
    # Add some noise.
    arr += np.random.rand(size, size) * arr + 1j * np.random.rand(size, size) * arr
    # Normalize.
    arr /= size**2
    # Make it diagonally dominant.
    np.fill_diagonal(arr, 2 * np.abs(arr.sum(-1).max() + arr.diagonal()))

    return arr


# Prepare data.
ndiag = 4
nm_dev = int(indata["nm_dev"])

if comm.rank == 0:
    GL = np.array(
        [get_data(nm_dev) for __ in range(600)], dtype=np.complex128
    ).swapaxes(0, -1)
    GG = np.array(
        [get_data(nm_dev) for __ in range(600)], dtype=np.complex128
    ).swapaxes(0, -1)
else:
    GL = None
    GG = None

GL = comm.bcast(GL, root=0)
GG = comm.bcast(GG, root=0)

if np.isnan(GL).any() or np.isnan(GG).any():
    raise ValueError("NaNs in data.")

GG = xp.array(GG)
GL = xp.array(GL)

if np.isnan(GL).any() or np.isnan(GG).any():
    raise ValueError("NaNs in data.")

bse = BSESolver(nm_dev, ndiag)

for i, j in np.ndindex(GL.shape[:-1]):
    if np.abs(i - j) > ndiag:
        GL[i, j] = 0
        GG[i, j] = 0

if np.isnan(GL).any() or np.isnan(GG).any():
    raise ValueError("NaNs in data.")

block_sizes = [7] * 20
g_lesser = DSBCOO.from_sparray(
    sparse.coo_array(GL[..., 100].get()), block_sizes, (600,)
)
g_greater = DSBCOO.from_sparray(
    sparse.coo_array(GG[..., 100].get()), block_sizes, (600,)
)
if np.isnan(g_lesser.data).any() or np.isnan(g_greater.data).any():
    raise ValueError("NaNs in data.")

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

# ndiag = int(indata["ndiag"])

start_time = time.time()

bse_dist = BSESolverDist(nm_dev, ndiag)

bse_dist._preprocess()
bse._preprocess()

num_E = 10
bse_dist._alloc_twobody_matrix(num_E=num_E)
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

bse_dist._calc_noninteracting_twobody(g_greater, g_lesser, step_E=20)
bse._calc_noninteracting_twobody(GG, GL, step_E=20)

if not np.allclose(bse_dist.L0mat.rows, bse.L0mat.rows):
    print("rows do not match!")
if not np.allclose(bse_dist.L0mat.cols, bse.L0mat.cols):
    print("rows do not match!")

print(
    "rank=",
    comm.rank,
    "rel data norm error=",
    (np.linalg.norm(bse_dist.L0mat.data) - np.linalg.norm(bse.L0mat.data))
    / np.linalg.norm(bse_dist.L0mat.data),
)
print(
    "rank=",
    comm.rank,
    "rel data elementwise error=",
    np.linalg.norm(bse_dist.L0mat.data - bse.L0mat.data)
    / np.linalg.norm(bse_dist.L0mat.data),
)

if comm.rank == comm.size - 1:
    print("save data ...", flush=True)
    np.save("dev/L0mat_dist_data.npy", bse_dist.L0mat.data[-1])
    np.save("dev/L0mat_dist_rows.npy", bse_dist.L0mat.rows)
    np.save("dev/L0mat_dist_cols.npy", bse_dist.L0mat.cols)
    np.save("dev/L0mat_data.npy", bse.L0mat.data[-1])
    np.save("dev/L0mat_rows.npy", bse.L0mat.rows)
    np.save("dev/L0mat_cols.npy", bse.L0mat.cols)


comm.barrier()

if comm.rank == 0:
    finish_time = time.time()
    print(" compute time = ", finish_time - start_time)
    start_time = finish_time

    print("solve ...", flush=True)

P, Gamma = bse_dist._solve_interacting_twobody(V, W)

comm.barrier()

if comm.rank == 0:
    finish_time = time.time()
    print(" compute time = ", finish_time - start_time)
    start_time = finish_time

    print("dense solve ...", flush=True)

P2, Gamma2 = bse_dist._densesolve_interacting_twobody(V, W)

comm.barrier()

if comm.rank == 0:
    finish_time = time.time()
    print(" compute time = ", finish_time - start_time)
    start_time = finish_time

print(
    "rank=", comm.rank, "rel error=", np.sum(np.abs(P2 - P)) / np.sum(np.abs(P2))
), "abs error=", np.sum(np.abs(P2 - P))

# filename = datasetname + "output"
# np.savez(filename + "_rank" + str(comm.rank), P=get_host(P))
