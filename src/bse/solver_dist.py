import time

import cupy as cp
import numba as nb
import numpy as np
from cupyx.profiler import time_range
from cupyx.scipy import sparse as cusparse
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.MPI import Request
from qttools.datastructures import DSBCOO
from qttools.utils.gpu_utils import get_device, get_host, synchronize_current_stream, xp
from qttools.utils.mpi_utils import check_gpu_aware_mpi
from scipy import sparse
from serinv.algs import ddbtasinv

GPU_AWARE = check_gpu_aware_mpi()


_compute_permutation_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void compute_permutation(
    const int* perm_rows,
    const int* perm_cols,
    const int* rows,
    const int* cols,
    int* permutation,
    int size
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        for (int j = 0; j < size; ++j) {
            int cond = (perm_rows[j] == rows[i]) && (perm_cols[j] == cols[i]);
            permutation[i] = permutation[i] * (1 - cond) + j * cond;
        }
    }
}
""",
    "compute_permutation",
)


@time_range()
def _compute_permutation_raw(
    perm_rows: cp.ndarray, perm_cols: cp.ndarray, rows: cp.ndarray, cols: cp.ndarray
):
    size = rows.size
    permutation = cp.zeros_like(rows, dtype=cp.int32)
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    _compute_permutation_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            perm_rows.astype(cp.int32),
            perm_cols.astype(cp.int32),
            rows.astype(cp.int32),
            cols.astype(cp.int32),
            permutation,
            size,
        ),
    )
    return permutation


_get_mapping_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void get_mapping(
    const int* row,
    const int* col,
    const int* L_rows,
    const int* L_cols,
    const int* L_nnz_section_offsets,
    int* L_idx,
    int row_size,
    int col_size,
    int my_rank,
    int L_rows_size,
    int num_offsets
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < row_size && j < col_size) {
        int ind = -1;
        for (int k = 0; k < L_rows_size; ++k) {
            int cond = (
                (L_rows[k] == row[i * col_size + j])
                && (L_cols[k] == col[i * col_size + j])
            );
            ind = ind * (1 - cond) + k * cond;
        }
        int data_rank = -1;
        for (int k = 0; k < num_offsets; ++k) {
            int cond = L_nnz_section_offsets[k] <= ind;
            data_rank = data_rank * (1 - cond) + k * cond;
        }
        int cond = (my_rank == data_rank);
        L_idx[i * col_size + j] = (ind - L_nnz_section_offsets[data_rank]) * cond + (cond - 1);
    }
}
""",
    "get_mapping",
)


@time_range()
def _get_mapping_raw(
    row: cp.ndarray,
    col: cp.ndarray,
    L_rows: cp.ndarray,
    L_cols: cp.ndarray,
    my_rank: int,
    L_nnz_section_offsets: cp.ndarray,
):
    row_size, col_size = row.shape
    L_idx = cp.full((row_size, col_size), -1, dtype=cp.int32)
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (row_size + threads_per_block[0] - 1) // threads_per_block[0],
        (col_size + threads_per_block[1] - 1) // threads_per_block[1],
    )
    _get_mapping_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            row.astype(cp.int32),
            col.astype(cp.int32),
            L_rows.astype(cp.int32),
            L_cols.astype(cp.int32),
            L_nnz_section_offsets.astype(cp.int32),
            L_idx,
            row_size,
            col_size,
            my_rank,
            L_rows.shape[0],
            L_nnz_section_offsets.shape[0],
        ),
    )
    return L_idx


class BSESolverDist:
    def __init__(self, num_sites: int, cutoff: int) -> None:
        self.num_sites = num_sites
        self.cutoff = cutoff

    @time_range()
    @nb.njit(parallel=True, fastmath=True)
    def _get_sparsity(size: np.int32, cutoff: np.int32, table: np.ndarray):
        nnz = 0
        coords = np.zeros((size, size), dtype=nb.boolean)
        for row in nb.prange(size):
            for col in nb.prange(size):
                i, j = table[:, row]
                k, l = table[:, col]
                if (
                    (abs(i - k) > cutoff)
                    or (abs(i - l) > cutoff)
                    or (abs(j - k) > cutoff)
                    or (abs(j - l) > cutoff)
                ):
                    continue
                coords[row, col] = True
                nnz += 1

        return nnz, coords

    @time_range()
    @nb.njit(parallel=True, fastmath=True)
    def _compute_permutation_numba(
        perm_rows: np.ndarray, perm_cols: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ):
        permutation = np.zeros_like(rows, dtype=np.int32)
        for i in nb.prange(rows.size):
            mask = (perm_rows == rows[i]) & (perm_cols == cols[i])
            permutation[i] = np.where(mask)[0][0]

        return permutation

    @time_range()
    def _compute_permutation(
        perm_rows: xp.ndarray, perm_cols: xp.ndarray, rows: xp.ndarray, cols: xp.ndarray
    ):
        mask = (perm_rows == rows[:, None]) & (perm_cols == cols[:, None])
        return mask.nonzero()[1]

    @time_range()
    @nb.njit(parallel=True, fastmath=True)
    def _get_mapping_numba(
        row: np.ndarray,
        col: np.ndarray,
        L_rows: np.ndarray,
        L_cols: np.ndarray,
        my_rank: int,
        L_nnz_section_offsets: np.ndarray,
    ):
        L_idx = np.zeros((row.shape[0], row.shape[1]), nb.int32) - 1
        for i in nb.prange(row.shape[0]):
            for j in nb.prange(row.shape[1]):
                ind = np.where((L_rows == row[i, j]) & (L_cols == col[i, j]))[0]
                if ind.size == 0:
                    continue
                data_rank = np.where(L_nnz_section_offsets <= ind[0])[0][-1]
                if my_rank != data_rank:
                    continue
                L_idx[i, j] = ind[0] - L_nnz_section_offsets[data_rank]

        return L_idx

    # Figure out the info to locate the needed nonzero elements (nnz) whichin an interaction range of `ndiag` of
    # the nnz on the i-th rank.
    #   - get_nnz_size: number of the nnz to gether
    #   - get_nnz_idx: indices ...
    #   - get_nnz_rank: on which ranks ...
    # For example, this gives all the nnz indices needed by i-th rank, which locates on the j-th rank
    #   > mask_i_needs_from_j = np.where(get_nnz_rank[i] == j)[0]
    #   > nnz_i_needs_from_j = get_nnz_idx[i][mask_i_needs_from_j]
    @time_range()
    @staticmethod
    def _determine_rank_map(offset, ndiag: int, rows: xp.ndarray, cols: xp.ndarray):
        """
        Figure out the info to locate the needed nonzero elements (nnz) whichin an interaction range of `ndiag` of
        the nnz on the i-th rank.
        """
        num_rank = len(offset) - 1
        get_nnz_idx = []
        get_nnz_rank = []
        get_nnz_size = []
        for rank in range(num_rank):

            min_row = rows[offset[rank] : offset[rank + 1]].min() - ndiag
            max_row = rows[offset[rank] : offset[rank + 1]].max() + ndiag
            min_col = cols[offset[rank] : offset[rank + 1]].min() - ndiag
            max_col = cols[offset[rank] : offset[rank + 1]].max() + ndiag

            mask = (
                (rows >= min_row)
                & (rows <= max_row)
                & (cols >= min_col)
                & (cols <= max_col)
            )

            idx = xp.where(mask)[0]
            idx_in_rank = xp.array([xp.where(offset <= ind)[0][-1] for ind in idx])

            get_nnz = xp.where(idx_in_rank != rank)[0]
            get_nnz_idx.append(idx[get_nnz])
            get_nnz_rank.append(idx_in_rank[get_nnz])
            get_nnz_size.append(get_nnz.size)

        return get_nnz_size, get_nnz_idx, get_nnz_rank

    @time_range()
    def _preprocess(self, rows, cols):
        """Computes some the sparsity pattern and the block-size."""
        # Sets self.size, self.table, self.inverse_table, self.nnz, self.rows, self.cols
        self._preprocess_bta()
        self.table_dist = xp.zeros((2, self.size), dtype=xp.int32)
        self.inverse_table_dist = (
            xp.zeros((self.num_sites, self.num_sites), dtype=xp.int32) * xp.nan
        )
        offset = 0
        for row, col in zip(rows, cols):
            self.table_dist[0, offset] = row
            self.table_dist[1, offset] = col
            self.inverse_table_dist[row, col] = offset
            offset += 1

        # for i in range(self.num_sites):
        #     l = max(0, i - self.cutoff)
        #     k = min(self.num_sites - 1, i + self.cutoff)
        #     for j in range(l, k + 1):
        #         self.table_dist[0, offset] = i
        #         self.table_dist[1, offset] = j
        #         self.inverse_table_dist[i, j] = offset
        #         offset += 1

        assert offset == self.size
        nnz, coords_dist = BSESolverDist._get_sparsity(
            self.size, self.cutoff, get_host(self.table_dist)
        )
        self.nnz = nnz
        self.rows_dist, self.cols_dist = coords_dist.nonzero()

    # preprocessing the sparsity pattern and decide the block_size and
    # num_blocks in the BTA matrix
    @time_range()
    def _preprocess_bta(self):
        """Computes some the sparsity pattern and the block-size."""
        self.size = self.num_sites**2 - (self.num_sites - self.cutoff - 1) * (
            self.num_sites - self.cutoff
        )  # compressed system size ~ 2*nm_dev*ndiag-ndiag*ndiag
        self.table = xp.zeros((2, self.size), dtype=xp.int32)
        self.inverse_table = (
            xp.zeros((self.num_sites, self.num_sites), dtype=xp.int32) * xp.nan
        )
        # construct a lookup table of reordered indices tip for the
        # "exchange" space， where we put the i=j.
        for i in range(self.num_sites):
            self.table[0, i] = i
            self.table[1, i] = i
            self.inverse_table[i, i] = i

        # then put the others, but within the ndiag
        offset = self.num_sites
        for i in range(self.num_sites):
            l = max(0, i - self.cutoff)
            k = min(self.num_sites - 1, i + self.cutoff)
            for j in range(l, k + 1):
                if i == j:
                    continue
                self.table[0, offset] = i
                self.table[1, offset] = j
                self.inverse_table[i, j] = offset
                offset += 1

        if (offset) != self.size:
            print(f"ERROR!, it={offset}, N={self.size}")

        # determine number of nnz and sparsity pattern
        table = get_host(self.table)
        self.nnz, coords = BSESolverDist._get_sparsity(self.size, self.cutoff, table)
        self.rows, self.cols = coords.nonzero()

        arrow_mask = (self.rows > self.num_sites) & (self.cols > self.num_sites)
        bandwidth = np.max(self.cols[arrow_mask] - self.rows[arrow_mask]) + 1

        self.blocksize = bandwidth  # <= 2*cutoff*(cutoff)
        self.num_blocks = int(np.ceil((self.size - self.num_sites) / self.blocksize))
        self.arrowsize = int(self.blocksize) * int(self.num_blocks)
        self.tipsize = self.num_sites
        self.totalsize = self.arrowsize + self.tipsize

        if comm.rank == 0:
            print("  N =", self.num_sites, flush=True)
            print("  N^2 =", self.num_sites**2, flush=True)
            print("  resized size =", self.totalsize, flush=True)
            print("  total arrow size=", self.arrowsize, flush=True)
            print("  arrow bandwidth=", bandwidth, flush=True)
            print("  arrow block size=", self.blocksize, flush=True)
            print("  arrow number of blocks=", self.num_blocks, flush=True)
            print("  nonzero elements=", self.nnz / 1e6, " Million", flush=True)
            print(
                "  nonzero ratio = ",
                self.nnz / (self.totalsize) ** 2 * 100,
                " %",
                flush=True,
            )
        return

    @time_range()
    def _alloc_twobody_matrix(self, num_E: int):
        ARRAY_SHAPE = (self.size, self.size)
        BLOCK_SIZES = [self.size] * 1
        GLOBAL_STACK_SHAPE = (num_E,)
        self.num_E = num_E
        data = np.zeros(len(self.rows_dist), dtype=xp.complex128)
        coords = (self.rows_dist, self.cols_dist)
        coo = sparse.coo_array((data, coords), shape=ARRAY_SHAPE)
        self.L0mat_dist = DSBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)
        del self.rows_dist
        del self.cols_dist
        if self.L0mat_dist.distribution_state == "stack":
            self.L0mat_dist.dtranspose()
        return

    @time_range()
    def _calc_noninteracting_twobody_old(self, GG: DSBCOO, GL: DSBCOO, step_E: int = 1):
        start_time = time.time()
        if self.L0mat_dist.distribution_state == "stack":
            self.L0mat_dist.dtranspose()
        if GG.distribution_state == "stack":
            GG.dtranspose()
        if GL.distribution_state == "stack":
            GL.dtranspose()
        finish_time = time.time()
        if comm.rank == 0:
            print(" dtranspose time=", finish_time - start_time, flush=True)
        start_time = finish_time

        G_nen = GG.data.shape[0]

        get_nnz_size, get_nnz_idx, get_nnz_rank = BSESolverDist._determine_rank_map(
            GG.nnz_section_offsets,
            self.cutoff,
            GG.rows,
            GG.cols,
        )

        gg_recbuf = [None] * comm.size
        gl_recbuf = [None] * comm.size
        gg_sendbuf = [None] * comm.size
        gl_sendbuf = [None] * comm.size

        synchronize_current_stream()
        for j in reversed(range(comm.size)):
            if j == comm.rank:
                continue
            inds_rank_to_j = get_nnz_idx[j][get_nnz_rank[j] == comm.rank]
            if not inds_rank_to_j.any():
                continue

            if not GPU_AWARE:
                gg_sendbuf[j] = np.zeros(
                    (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                )
                gl_sendbuf[j] = np.zeros(
                    (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                )
            else:
                gg_sendbuf[j] = xp.zeros(
                    (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                )
                gl_sendbuf[j] = xp.zeros(
                    (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                )

        reqs_gg = []
        for i in range(comm.size):
            if i == comm.rank:
                continue
            mask_buffer = get_nnz_rank[comm.rank] == i
            if not mask_buffer.any():
                continue

            if not GPU_AWARE:
                gg_recbuf[i] = np.zeros(
                    (G_nen, int(mask_buffer.sum())), dtype=np.complex128
                )
            else:
                gg_recbuf[i] = xp.zeros(
                    (G_nen, int(mask_buffer.sum())), dtype=np.complex128
                )

            print(f"Posting receive {i}-->{comm.rank}", flush=True)

            reqs_gg.append(comm.Irecv(gg_recbuf[i], source=i, tag=0))

        for j in reversed(range(comm.size)):
            if j == comm.rank:
                continue
            inds_rank_to_j = get_nnz_idx[j][get_nnz_rank[j] == comm.rank]
            if not inds_rank_to_j.any():
                continue

            print(f"Posting send {comm.rank}-->{j}", flush=True)

            gg_sendbuf[j] = GG.data[
                ..., inds_rank_to_j - GG.nnz_section_offsets[comm.rank]
            ]
            if not GPU_AWARE:
                gg_sendbuf[j] = get_host(gg_sendbuf[j])
            # if np.isnan(gg_sendbuf[j]).any():
            #     raise ValueError(f"rank {comm.rank}: gg send buffer contains NaNs")
            comm.Isend(gg_sendbuf[j], dest=j, tag=0)

        Request.Waitall(reqs_gg)

        reqs_gl = []
        for i in range(comm.size):
            if i == comm.rank:
                continue
            mask_buffer = get_nnz_rank[comm.rank] == i
            if not mask_buffer.any():
                continue

            if not GPU_AWARE:
                gl_recbuf[i] = np.zeros(
                    (G_nen, int(mask_buffer.sum())), dtype=np.complex128
                )
            else:
                gl_recbuf[i] = xp.zeros(
                    (G_nen, int(mask_buffer.sum())), dtype=np.complex128
                )

            print(f"Posting receive {i}-->{comm.rank}", flush=True)

            reqs_gl.append(comm.Irecv(gl_recbuf[i], source=i, tag=1))

        for j in reversed(range(comm.size)):
            if j == comm.rank:
                continue
            inds_rank_to_j = get_nnz_idx[j][get_nnz_rank[j] == comm.rank]
            if not inds_rank_to_j.any():
                continue

            print(f"Posting send {comm.rank}-->{j}", flush=True)

            gl_sendbuf[j] = GL.data[
                ..., inds_rank_to_j - GL.nnz_section_offsets[comm.rank]
            ]

            if not GPU_AWARE:
                gl_sendbuf[j] = get_host(gl_sendbuf[j])

            comm.Isend(gl_sendbuf[j], dest=j, tag=1)

        Request.Waitall(reqs_gl)

        if xp.isnan(GG.data).any():
            raise ValueError(f"rank {comm.rank}: GG contains NaNs")

        if comm.size > 1:
            gg_recbuf = xp.concatenate(
                [xp.array(gg) for gg in gg_recbuf if gg is not None], axis=-1
            )
            gl_recbuf = xp.concatenate(
                [xp.array(gl) for gl in gl_recbuf if gl is not None], axis=-1
            )

            if xp.isnan(gl_recbuf).any():
                raise ValueError(f"rank {comm.rank}: gl buffer contains NaNs")
            if xp.isnan(gg_recbuf).any():
                raise ValueError(f"rank {comm.rank}: gg buffer contains NaNs")

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "MPI send recv time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time

        local_elements = kron_correlate(GG.data, GL.data) - kron_correlate(
            GL.data, GG.data
        )

        start_inz_g = int(GG.nnz_section_offsets[comm.rank])
        end_inz_g = int(GG.nnz_section_offsets[comm.rank + 1])
        inz = xp.arange(start_inz_g, end_inz_g)

        row = self.inverse_table_dist[GG.rows[inz[:, None]], GG.cols[inz[:]]]
        col = self.inverse_table_dist[GG.cols[inz[:, None]], GG.rows[inz[:]]]

        inds = _get_mapping_raw(
            row,
            col,
            self.L0mat_dist.rows,
            self.L0mat_dist.cols,
            comm.rank,
            self.L0mat_dist.nnz_section_offsets,
        )
        valid = xp.where(inds != -1)

        self.L0mat_dist._data[
            xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
        ] = local_elements[G_nen : G_nen + self.num_E * step_E : step_E, *valid]

        if comm.size > 1:
            local_buf_elements = kron_correlate(GG.data, gl_recbuf) - kron_correlate(
                GL.data, gg_recbuf
            )

            row = self.inverse_table_dist[
                GG.rows[inz[:, None]], GG.cols[(get_nnz_idx[comm.rank])[:]]
            ]
            col = self.inverse_table_dist[
                GG.cols[inz[:, None]], GG.rows[(get_nnz_idx[comm.rank])[:]]
            ]

            inds = _get_mapping_raw(
                row,
                col,
                self.L0mat_dist.rows,
                self.L0mat_dist.cols,
                comm.rank,
                self.L0mat_dist.nnz_section_offsets,
            )
            valid = xp.where(inds != -1)

            self.L0mat_dist._data[
                xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
            ] = local_buf_elements[G_nen : G_nen + self.num_E * step_E : step_E, *valid]

            buf_local_elements = kron_correlate(gg_recbuf, GL.data) - kron_correlate(
                gl_recbuf, GG.data
            )

            row = self.inverse_table_dist[
                GG.rows[(get_nnz_idx[comm.rank])[:, None]], GG.cols[inz[:]]
            ]
            col = self.inverse_table_dist[
                GG.cols[(get_nnz_idx[comm.rank])[:, None]], GG.rows[inz[:]]
            ]

            inds = _get_mapping_raw(
                row,
                col,
                self.L0mat_dist.rows,
                self.L0mat_dist.cols,
                comm.rank,
                self.L0mat_dist.nnz_section_offsets,
            )
            valid = xp.where(inds != -1)

            self.L0mat_dist._data[
                xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
            ] = buf_local_elements[G_nen : G_nen + self.num_E * step_E : step_E, *valid]

            buf_elements = kron_correlate(gg_recbuf, gl_recbuf) - kron_correlate(
                gl_recbuf, gg_recbuf
            )
            row = self.inverse_table_dist[
                GG.rows[(get_nnz_idx[comm.rank])[:, None]],
                GG.cols[(get_nnz_idx[comm.rank])[:]],
            ]
            col = self.inverse_table_dist[
                GG.cols[(get_nnz_idx[comm.rank])[:, None]],
                GG.rows[(get_nnz_idx[comm.rank])[:]],
            ]

            inds = _get_mapping_raw(
                row,
                col,
                self.L0mat_dist.rows,
                self.L0mat_dist.cols,
                comm.rank,
                self.L0mat_dist.nnz_section_offsets,
            )
            valid = xp.where(inds != -1)

            self.L0mat_dist._data[
                xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
            ] = buf_elements[G_nen : G_nen + self.num_E * step_E : step_E, *valid]

        finish_time = time.time()
        print(
            " rank ", comm.rank, "compute time=", finish_time - start_time, flush=True
        )
        start_time = finish_time

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "barrier waiting time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time

        # transpose to stack distribution
        self.L0mat_dist.dtranspose()

        finish_time = time.time()
        if comm.rank == 0:
            print(" dtranspose time=", finish_time - start_time, flush=True)
        start_time = finish_time
        BLOCK_SIZES = [self.tipsize] + [self.blocksize] * self.num_blocks
        GLOBAL_STACK_SHAPE = (self.num_E,)

        perm_rows = self.inverse_table[*self.table_dist[:, self.L0mat_dist.rows]]
        perm_cols = self.inverse_table[*self.table_dist[:, self.L0mat_dist.cols]]

        permutation = _compute_permutation_raw(
            perm_rows, perm_cols, get_device(self.rows), get_device(self.cols)
        )

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "compute permutation time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time

        self.L0mat = DSBCOO(
            data=self.L0mat_dist.data[..., permutation],
            rows=get_device(self.rows),
            cols=get_device(self.cols),
            block_sizes=BLOCK_SIZES,
            global_stack_shape=GLOBAL_STACK_SHAPE,
        )
        del self.rows
        del self.cols
        del self.L0mat_dist

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "reorder L to BTA matrix time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time

        return

    @time_range()
    def _calc_noninteracting_twobody(self, GG: DSBCOO, GL: DSBCOO, step_E: int = 1):
        start_time = time.time()

        if self.L0mat_dist.distribution_state == "stack":
            self.L0mat_dist.dtranspose()
        if GG.distribution_state == "stack":
            GG.dtranspose()
        if GL.distribution_state == "stack":
            GL.dtranspose()
        finish_time = time.time()
        if comm.rank == 0:
            print(" dtranspose time=", finish_time - start_time, flush=True)
        start_time = finish_time

        G_nen = GG.data.shape[0]

        get_nnz_size, get_nnz_idx, get_nnz_rank = BSESolverDist._determine_rank_map(
            GG.nnz_section_offsets,
            self.cutoff,
            GG.rows,
            GG.cols,
        )

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "rank map compute time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time

        gg_recbuf = [None] * 2
        gl_recbuf = [None] * 2
        gg_sendbuf = [None] * 2
        gl_sendbuf = [None] * 2

        synchronize_current_stream()

        offsets = [
            i for i in range(-comm.size, comm.size) if i != 0
        ]  # [-2, -1, 1, 2]  # list of overlapping neighboring ranks

        for offset in offsets:
            # receive from comm.rank - offset
            i = comm.rank - offset
            if (i > -1) and (i < comm.size):
                mask_buffer = get_nnz_rank[comm.rank] == i
                if mask_buffer.any():
                    if not GPU_AWARE:
                        gg_recbuf[0] = np.zeros(
                            (G_nen, int(mask_buffer.sum())), dtype=np.complex128
                        )
                        gl_recbuf[0] = np.zeros(
                            (G_nen, int(mask_buffer.sum())), dtype=np.complex128
                        )
                    else:
                        gg_recbuf[0] = xp.zeros(
                            (G_nen, int(mask_buffer.sum())), dtype=xp.complex128
                        )
                        gl_recbuf[0] = xp.zeros(
                            (G_nen, int(mask_buffer.sum())), dtype=xp.complex128
                        )
                    print(f"Posting receive {i}-->{comm.rank}", flush=True)
                    reqs_gg = comm.Irecv(gg_recbuf[0], source=i, tag=0)
                    reqs_gl = comm.Irecv(gl_recbuf[0], source=i, tag=1)
            # send to comm.rank + offset
            j = comm.rank + offset
            if (j < comm.size) and (j > -1):
                inds_rank_to_j = get_nnz_idx[j][get_nnz_rank[j] == comm.rank]
                if inds_rank_to_j.any():
                    if not GPU_AWARE:
                        gg_sendbuf[0] = np.zeros(
                            (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                        )
                        gl_sendbuf[0] = np.zeros(
                            (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                        )
                    else:
                        gg_sendbuf[0] = xp.zeros(
                            (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                        )
                        gl_sendbuf[0] = xp.zeros(
                            (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
                        )
                    print(f"Posting send {comm.rank}-->{j}", flush=True)
                    gg_sendbuf[0] = GG.data[
                        ..., inds_rank_to_j - GG.nnz_section_offsets[comm.rank]
                    ]
                    gl_sendbuf[0] = GL.data[
                        ..., inds_rank_to_j - GG.nnz_section_offsets[comm.rank]
                    ]
                    if not GPU_AWARE:
                        gg_sendbuf[0] = get_host(gg_sendbuf[0])
                        gl_sendbuf[0] = get_host(gl_sendbuf[0])

                    comm.Isend(gg_sendbuf[0], dest=j, tag=0)
                    comm.Isend(gl_sendbuf[0], dest=j, tag=1)

            # non-local part
            if (i > -1) and (i < comm.size) and (mask_buffer.any()):
                # wait for some data comes in
                recbuf_idx = get_nnz_idx[comm.rank][get_nnz_rank[comm.rank] == i]
                Request.Waitall([reqs_gg, reqs_gl])
                dev_gl_recbuf = get_device(gl_recbuf[0])
                dev_gg_recbuf = get_device(gg_recbuf[0])
                synchronize_current_stream()
                ### Done transfer, now do some computations
                start_inz_g = int(GG.nnz_section_offsets[comm.rank])
                end_inz_g = int(GG.nnz_section_offsets[comm.rank + 1])
                inz = np.arange(start_inz_g, end_inz_g)
                # correlation of local G data with recv buffer
                local_buf_elements = kron_correlate(
                    GG.data, dev_gl_recbuf
                ) - kron_correlate(GL.data, dev_gg_recbuf)
                row = self.inverse_table_dist[
                    GG.rows[inz[:, None]], GG.cols[(recbuf_idx)[:]]
                ]
                col = self.inverse_table_dist[
                    GG.cols[inz[:, None]], GG.rows[(recbuf_idx)[:]]
                ]

                inds = _get_mapping_raw(
                    row,
                    col,
                    self.L0mat_dist.rows,
                    self.L0mat_dist.cols,
                    comm.rank,
                    self.L0mat_dist.nnz_section_offsets,
                )

                valid = xp.where(inds != -1)
                self.L0mat_dist._data[
                    xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
                ] = local_buf_elements[
                    G_nen : G_nen + self.num_E * step_E : step_E, *valid
                ]

                # recv buffer with local G data
                buf_local_elements = kron_correlate(
                    dev_gg_recbuf, GL.data
                ) - kron_correlate(dev_gl_recbuf, GG.data)
                row = self.inverse_table_dist[
                    GG.rows[recbuf_idx[:, None]], GG.cols[inz[:]]
                ]
                col = self.inverse_table_dist[
                    GG.cols[recbuf_idx[:, None]], GG.rows[inz[:]]
                ]
                inds = _get_mapping_raw(
                    row,
                    col,
                    self.L0mat_dist.rows,
                    self.L0mat_dist.cols,
                    comm.rank,
                    self.L0mat_dist.nnz_section_offsets,
                )
                valid = xp.where(inds != -1)
                self.L0mat_dist._data[
                    xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
                ] = buf_local_elements[
                    G_nen : G_nen + self.num_E * step_E : step_E, *valid
                ]
                # recv buffer with recv buffer
                buf_elements = kron_correlate(
                    dev_gg_recbuf, dev_gl_recbuf
                ) - kron_correlate(dev_gl_recbuf, dev_gg_recbuf)
                row = self.inverse_table_dist[
                    GG.rows[recbuf_idx[:, None]],
                    GG.cols[recbuf_idx[:]],
                ]
                col = self.inverse_table_dist[
                    GG.cols[recbuf_idx[:, None]],
                    GG.rows[recbuf_idx[:]],
                ]
                inds = _get_mapping_raw(
                    row,
                    col,
                    self.L0mat_dist.rows,
                    self.L0mat_dist.cols,
                    comm.rank,
                    self.L0mat_dist.nnz_section_offsets,
                )
                valid = xp.where(inds != -1)
                self.L0mat_dist._data[
                    xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
                ] = buf_elements[G_nen : G_nen + self.num_E * step_E : step_E, *valid]

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "non-local elements compute time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time

        # compute L with local data on the current rank
        start_inz_g = int(GG.nnz_section_offsets[comm.rank])
        end_inz_g = int(GG.nnz_section_offsets[comm.rank + 1])
        inz = np.arange(start_inz_g, end_inz_g)
        jnz = np.arange(start_inz_g, end_inz_g)
        # adjust this to match the GPU memory
        step_inz = 10
        for iinz in range(0, len(inz), step_inz):
            local_inz = inz[iinz] - int(GG.nnz_section_offsets[comm.rank])
            local_elements = kron_correlate(
                GG.data[:, local_inz : local_inz + step_inz], GL.data
            ) - kron_correlate(GL.data[:, local_inz : local_inz + step_inz], GG.data)
            row = self.inverse_table_dist[
                GG.rows[inz[local_inz : local_inz + step_inz, None]], GG.cols[jnz[:]]
            ]
            col = self.inverse_table_dist[
                GG.cols[inz[local_inz : local_inz + step_inz, None]], GG.rows[jnz[:]]
            ]
            inds = _get_mapping_raw(
                row,
                col,
                self.L0mat_dist.rows,
                self.L0mat_dist.cols,
                comm.rank,
                self.L0mat_dist.nnz_section_offsets,
            )
            valid = xp.where(inds != -1)
            self.L0mat_dist._data[
                xp.ix_(self.L0mat_dist._stack_padding_mask, inds[valid])
            ] = local_elements[G_nen : G_nen + self.num_E * step_E : step_E, *valid]

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "local elements compute time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time

        # transpose to stack distribution
        self.L0mat_dist.dtranspose()

        finish_time = time.time()
        if comm.rank == 0:
            print(" dtranspose time=", finish_time - start_time, flush=True)
        start_time = finish_time

        # reorder L0mat to BTA shape
        BLOCK_SIZES = [self.tipsize] + [self.blocksize] * self.num_blocks
        GLOBAL_STACK_SHAPE = (self.num_E,)
        # compute the permutation array to go from normal to BTA ordering
        perm_rows = self.inverse_table[*self.table_dist[:, self.L0mat_dist.rows]]
        perm_cols = self.inverse_table[*self.table_dist[:, self.L0mat_dist.cols]]
        permutation = _compute_permutation_raw(
            perm_rows, perm_cols, get_device(self.rows), get_device(self.cols)
        )
        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "compute permutation time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time
        self.L0mat = DSBCOO(
            data=self.L0mat_dist.data[..., permutation],
            rows=get_device(self.rows),
            cols=get_device(self.cols),
            block_sizes=BLOCK_SIZES,
            global_stack_shape=GLOBAL_STACK_SHAPE,
        )
        del self.rows
        del self.cols
        del self.L0mat_dist
        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "reorder to BTA matrix time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time
        return

    @time_range()
    def _calc_kernel(self, V: xp.array, W: xp.array):

        kernel_tip = xp.zeros((self.tipsize, self.tipsize), dtype=xp.complex128)
        kernel_diag = xp.zeros((self.totalsize - self.tipsize), dtype=xp.complex128)

        kernel_tip = -V[
            self.table[0, : self.num_sites], self.table[0, : self.num_sites]
        ] + xp.diag(
            xp.array(
                [W[self.table[0, i], self.table[0, i]] for i in range(self.num_sites)]
            )
        )

        for row in range(self.tipsize, self.size):
            i = self.table[0, row]
            j = self.table[1, row]
            kernel_diag[row - self.tipsize] += W[i, j]
        kernel_diag *= 1j
        kernel_tip *= 1j
        return kernel_tip, kernel_diag

    @time_range()
    def _densesolve_interacting_twobody(self, V: xp.array, W: xp.array):
        if self.L0mat.distribution_state != "stack":
            self.L0mat.dtranspose()
        kernel_tip, kernel_diag = self._calc_kernel(V, W)
        local_nen = self.L0mat.stack_shape[0]
        K = xp.zeros((self.size, self.size), dtype=xp.complex128)
        P = xp.zeros((self.tipsize, self.tipsize, local_nen), dtype=xp.complex128)
        Gamma = xp.zeros(
            (self.tipsize, self.tipsize, self.tipsize, local_nen),
            dtype=xp.complex128,
        )

        K[: self.tipsize, : self.tipsize] = kernel_tip
        K[self.tipsize :, self.tipsize :] = np.diag(
            kernel_diag[: self.size - self.tipsize]
        )

        table = self.table
        with time_range("dense solve", color_id=comm.rank):
            for ie in range(local_nen):
                print("rank=", comm.rank, "ie=", ie + 1, "/", local_nen, flush=True)
                data = self.L0mat.data[ie]
                coords = (self.L0mat.rows, self.L0mat.cols)

                L0 = cusparse.coo_matrix(
                    (data, coords), shape=(self.size, self.size)
                ).todense()

                A = -L0 @ K + xp.diag(xp.ones(self.size, dtype=xp.complex128))
                invA = xp.linalg.inv(A)
                # impose sparsity pattern of BTA, for a proper comparison with selected inversion
                invA_bta = _impose_bta_sparsity(
                    invA, self.blocksize, self.tipsize, self.num_blocks
                )
                A = invA_bta @ L0

                for row in range(self.tipsize):
                    for col in range(self.tipsize):
                        i = table[0, row]
                        j = table[0, col]
                        P[i, j, ie] = -1j * A[row, col]

                for row in range(self.tipsize):
                    i = table[0, row]
                    for col in range(self.tipsize, self.size):
                        j = table[0, col]
                        k = table[1, col]
                        Gamma[i, j, k, ie] = A[row, col]

        return P, Gamma

    @time_range()
    def _solve_interacting_twobody(self, V: xp.array, W: xp.array):
        if self.L0mat.distribution_state != "stack":
            self.L0mat.dtranspose()

        kernel_tip, kernel_diag = self._calc_kernel(V, W)

        K = xp.zeros((self.totalsize, self.totalsize), dtype=xp.complex128)
        K[: self.tipsize, : self.tipsize] = kernel_tip
        K[self.tipsize :, self.tipsize :] = np.diag(kernel_diag)

        A_arrow_right_blocks = xp.zeros(
            (self.num_blocks, self.blocksize, self.tipsize), dtype=xp.complex128
        )
        A_arrow_bottom_blocks = xp.zeros(
            (self.num_blocks, self.tipsize, self.blocksize), dtype=xp.complex128
        )
        A_diagonal_blocks = xp.zeros(
            (self.num_blocks, self.blocksize, self.blocksize), dtype=xp.complex128
        )
        A_upper_diagonal_blocks = xp.zeros(
            (self.num_blocks - 1, self.blocksize, self.blocksize), dtype=xp.complex128
        )
        A_lower_diagonal_blocks = xp.zeros(
            (self.num_blocks - 1, self.blocksize, self.blocksize), dtype=xp.complex128
        )

        local_nen = self.L0mat.stack_shape[0]
        P = xp.zeros((self.tipsize, self.tipsize, local_nen), dtype=xp.complex128)
        Gamma = xp.zeros(
            (self.tipsize, self.blocksize * self.num_blocks, local_nen),
            dtype=xp.complex128,
        )

        for ie in range(local_nen):
            with time_range("construct serinv inputs", color_id=comm.rank):
                print("rank=", comm.rank, "ie=", ie + 1, "/", local_nen, flush=True)
                # build system matrix: A = I - L0 @ K
                # Note: SerinV takes BTA pointing down, so the block ordering should be reversed and
                #       each block matrix should be transposed and flipped.
                A_arrow_tip_block = xp.transpose(
                    xp.flip(
                        -self.L0mat.stack[ie].blocks[0, 0] @ kernel_tip
                        + xp.eye(self.tipsize)
                    )
                )

                for k in range(self.num_blocks):
                    A_diagonal_blocks[-k - 1, :, :] = xp.transpose(
                        xp.flip(
                            -self.L0mat.stack[ie].blocks[k + 1, k + 1]
                            @ xp.diag(
                                kernel_diag[
                                    self.blocksize * k : self.blocksize * (k + 1)
                                ]
                            )
                        )
                        + xp.eye(self.blocksize)
                    )

                for k in range(self.num_blocks - 1):
                    A_upper_diagonal_blocks[-k - 1, :, :] = xp.transpose(
                        xp.flip(
                            -self.L0mat.stack[ie].blocks[k + 1, k + 2]
                            @ xp.diag(
                                kernel_diag[
                                    self.blocksize * (k + 1) : self.blocksize * (k + 2)
                                ]
                            )
                        )
                    )
                    A_lower_diagonal_blocks[-k - 1, :, :] = xp.transpose(
                        xp.flip(
                            -self.L0mat.stack[ie].blocks[k + 2, k + 1]
                            @ xp.diag(
                                kernel_diag[
                                    self.blocksize * (k) : self.blocksize * (k + 1)
                                ]
                            )
                        )
                    )

                for k in range(self.num_blocks):
                    A_arrow_bottom_blocks[-k - 1, :, :] = xp.transpose(
                        xp.flip(-self.L0mat.stack[ie].blocks[k + 1, 0] @ kernel_tip)
                    )
                    A_arrow_right_blocks[-k - 1, :, :] = xp.transpose(
                        xp.flip(
                            -self.L0mat.stack[ie].blocks[0, k + 1]
                            @ xp.diag(
                                kernel_diag[
                                    self.blocksize * k : self.blocksize * (k + 1)
                                ]
                            )
                        )
                    )

            # solve system matrix
            with time_range("serinv", color_id=comm.rank):
                (
                    X_diagonal_blocks_serinv,
                    X_lower_diagonal_blocks_serinv,
                    X_upper_diagonal_blocks_serinv,
                    X_arrow_bottom_blocks_serinv,
                    X_arrow_right_blocks_serinv,
                    X_arrow_tip_block_serinv,
                ) = ddbtasinv(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                    A_arrow_bottom_blocks,
                    A_arrow_right_blocks,
                    A_arrow_tip_block,
                )

            # first, we need to transpose and flip back the BTA matrix output from SerinV
            # extract P from tip of solution matrix L =  A^{-1} @ L0, and P := -i L_tip
            with time_range("extract P", color_id=comm.rank):
                tmp = (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                    @ self.L0mat.stack[ie].blocks[0, 0]
                )
                for k in range(self.num_blocks):
                    tmp += (
                        -1j
                        * xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :])
                        )
                        @ self.L0mat.stack[ie].blocks[k + 1, 0]
                    )
                # for row in range(self.tipsize):
                #     for col in range(self.tipsize):
                #         i = self.table[0, row]
                #         j = self.table[0, col]
                reorder = self.table[0, : self.tipsize]
                P[reorder[:, None], reorder, ie] = tmp[:, :]
                # extract Gamma from upper-arrow block of L = A^{-1} @ L0, and Gamma_ijk := L_iijk
                # L_01 = A_00 @ L0_01 + A_01 @ L0_11
                tmp2 = xp.zeros(
                    (self.tipsize, self.blocksize, self.num_blocks), dtype=xp.complex128
                )
                for k in range(self.num_blocks):
                    tmp2[:, :, k] += (
                        xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                        @ self.L0mat.stack[ie].blocks[0, k + 1]
                    )
                    tmp2[:, :, k] += (
                        xp.transpose(xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :]))
                        @ self.L0mat.stack[ie].blocks[k + 1, k + 1]
                    )
                    if k > 0:
                        tmp2[:, :, k] += (
                            xp.transpose(
                                xp.flip(X_arrow_right_blocks_serinv[-(k - 1) - 1, :, :])
                            )
                            @ self.L0mat.stack[ie].blocks[k, k + 1]
                        )
                    if k < self.num_blocks - 1:
                        tmp2[:, :, k] += (
                            xp.transpose(
                                xp.flip(X_arrow_right_blocks_serinv[-(k + 1) - 1, :, :])
                            )
                            @ self.L0mat.stack[ie].blocks[k + 2, k + 1]
                        )

                # i = self.table[0, :self.tipsize]
                # j = self.table[0, self.tipsize:self.size]
                # k = self.table[1, self.tipsize:self.size]
                Gamma[:, :, ie] = tmp2.reshape(
                    (tmp2.shape[0], tmp2.shape[1] * tmp2.shape[2])
                )

                # for row in range(self.tipsize):
                #     i = self.table[0, row]
                #     for ib in range(self.num_blocks):
                #         for ic in range(self.blocksize):
                #             col = ib * self.blocksize + ic + self.tipsize

                #             if col < self.size:
                #                 j = self.table[0, col]
                #                 k = self.table[1, col]

                #             Gamma[self.table[0, :], j, k, ie] = tmp2[:, ic, ib]
        return P, Gamma


def fftconvolve(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Convolves two 1D arrays using FFT.

    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.

    Returns
    -------
    np.ndarray
        Convolution of `a` and `b` including the "full" convolution.

    """
    n = len(a) + len(b) - 1
    a_fft = xp.fft.fft(a, n)
    b_fft = xp.fft.fft(b, n)
    return xp.fft.ifft(a_fft * b_fft)


@time_range()
def kron_correlate(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Convolves two 1D arrays using FFT and performs kronecker."""
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b[::-1], (n,), axes=(0,))

    with time_range("einsum", color_id=comm.rank):
        x_fft = xp.einsum("ei,ej->eij", a_fft, b_fft)

    return xp.fft.ifftn(x_fft, axes=(0,))


def correlate(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Computes the correlation of two 1D arrays.

    This is slightly different from the usual definition of correlation
    in signal processing, where the second array is conjugated.

    Here, we use the definition of correlation as the convolution of
    the first array with the reversed second array.


    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.

    Returns
    -------
    np.ndarray
        Correlation of `a` and `b` including the "full" correlation.

    """
    return fftconvolve(a, b[::-1])


def _impose_bta_sparsity(
    a: xp.ndarray,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Impose a block tridiagonal arrowhead sparsity to a dense array."""

    bta = a.copy()
    for i in range(n_diag_blocks):
        for j in range(n_diag_blocks):
            if abs(i - j) > 1:
                bta[
                    arrowhead_blocksize
                    + diagonal_blocksize * i : arrowhead_blocksize
                    + diagonal_blocksize * (i + 1),
                    arrowhead_blocksize
                    + diagonal_blocksize * j : arrowhead_blocksize
                    + diagonal_blocksize * (j + 1),
                ] = 0
    return bta
