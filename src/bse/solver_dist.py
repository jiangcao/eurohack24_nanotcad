import time

import numba as nb
import numpy as np
from cupyx.scipy import sparse as cusparse
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.MPI import Request
from qttools.datastructures import DSBCOO
from qttools.utils.gpu_utils import get_device, get_host, xp
from scipy import sparse
from serinv.algs import ddbtasinv


class BSESolverDist:
    def __init__(self, num_sites: int, cutoff: int) -> None:
        self.num_sites = num_sites
        self.cutoff = cutoff

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

    @nb.njit(parallel=True, fastmath=True)
    def _compute_permutation(
        perm_rows: np.ndarray, perm_cols: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ):
        permutation = np.zeros_like(rows, dtype=np.int32)
        for i in nb.prange(rows.size):
            mask = (perm_rows == rows[i]) & (perm_cols == cols[i])
            permutation[i] = np.where(mask)[0][0]

        return permutation

    # Figure out the info to locate the needed nonzero elements (nnz) whichin an interaction range of `ndiag` of
    # the nnz on the i-th rank.
    #   - get_nnz_size: number of the nnz to gether
    #   - get_nnz_idx: indices ...
    #   - get_nnz_rank: on which ranks ...
    # For example, this gives all the nnz indices needed by i-th rank, which locates on the j-th rank
    #   > mask_i_needs_from_j = np.where(get_nnz_rank[i] == j)[0]
    #   > nnz_i_needs_from_j = get_nnz_idx[i][mask_i_needs_from_j]
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

    def _preprocess(self):
        """Computes some the sparsity pattern and the block-size."""
        # Sets self.size, self.table, self.inverse_table, self.nnz, self.rows, self.cols
        self._preprocess_bta()
        self.table_dist = xp.zeros((2, self.size), dtype=xp.int32)
        self.inverse_table_dist = (
            xp.zeros((self.num_sites, self.num_sites), dtype=xp.int32) * xp.nan
        )
        offset = 0
        for i in range(self.num_sites):
            l = max(0, i - self.cutoff)
            k = min(self.num_sites - 1, i + self.cutoff)
            for j in range(l, k + 1):
                self.table_dist[0, offset] = i
                self.table_dist[1, offset] = j
                self.inverse_table_dist[i, j] = offset
                offset += 1
        assert offset == self.size
        nnz, coords_dist = BSESolverDist._get_sparsity(
            self.size, self.cutoff, get_host(self.table_dist)
        )
        self.nnz = nnz
        self.rows_dist, self.cols_dist = coords_dist.nonzero()

    # preprocessing the sparsity pattern and decide the block_size and
    # num_blocks in the BTA matrix
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

        gg_recbuf = [None] * comm.size
        gl_recbuf = [None] * comm.size
        gg_sendbuf = [None] * comm.size
        gl_sendbuf = [None] * comm.size

        for j in reversed(range(comm.size)):
            if j == comm.rank:
                continue
            inds_rank_to_j = get_nnz_idx[j][get_nnz_rank[j] == comm.rank]
            if not inds_rank_to_j.any():
                continue

            gg_sendbuf[j] = np.zeros(
                (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
            )
            gl_sendbuf[j] = np.zeros(
                (G_nen, int(inds_rank_to_j.size)), dtype=np.complex128
            )

        reqs_gg = []
        for i in range(comm.size):
            if i == comm.rank:
                continue
            mask_buffer = get_nnz_rank[comm.rank] == i
            if not mask_buffer.any():
                continue

            gg_recbuf[i] = np.zeros(
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

            gg_sendbuf[j] = get_host(
                GG.data[..., inds_rank_to_j - GG.nnz_section_offsets[comm.rank]]
            )
            if np.isnan(gg_sendbuf[j]).any():
                raise ValueError(f"rank {comm.rank}: gg send buffer contains NaNs")

            comm.Isend(gg_sendbuf[j], dest=j, tag=0)

        Request.Waitall(reqs_gg)

        reqs_gl = []
        for i in range(comm.size):
            if i == comm.rank:
                continue
            mask_buffer = get_nnz_rank[comm.rank] == i
            if not mask_buffer.any():
                continue

            gl_recbuf[i] = np.zeros(
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

            gl_sendbuf[j] = get_host(
                GL.data[..., inds_rank_to_j - GL.nnz_section_offsets[comm.rank]]
            )
            if np.isnan(gl_sendbuf[j]).any():
                raise ValueError(f"rank {comm.rank}: gl send buffer contains NaNs")

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

        start_inz = int(self.L0mat_dist.nnz_section_offsets[comm.rank])
        end_inz = int(self.L0mat_dist.nnz_section_offsets[comm.rank + 1])
        for inz in range(start_inz, end_inz):
            row = self.L0mat_dist.rows[inz]
            col = self.L0mat_dist.cols[inz]

            i = self.table_dist[0, row]
            j = self.table_dist[1, row]
            k = self.table_dist[0, col]
            l = self.table_dist[1, col]

            ind_ik = xp.where((GG.rows == i) & (GG.cols == k))[0]

            rank_ik = xp.where(GG.nnz_section_offsets <= ind_ik[0])[0][-1]
            if comm.rank == rank_ik:
                gg_ik = GG.data[..., ind_ik[0] - GG.nnz_section_offsets[rank_ik]]
                gl_ik = GL.data[..., ind_ik[0] - GL.nnz_section_offsets[rank_ik]]
            else:
                ind = xp.where(get_nnz_idx[comm.rank] == ind_ik[0])[0]
                # print(f"rank {comm.rank}:", ind, (i, j, k, l), flush=True)
                # print(f"rank {comm.rank}:", get_nnz_idx[comm.rank], flush=True)
                # print(f"rank {comm.rank}:", gg_recbuf[0], flush=True)
                # print(f"rank {comm.rank}:", ind_ik, flush=True)
                gg_ik = gg_recbuf[..., ind[0]]
                gl_ik = gl_recbuf[..., ind[0]]

            ind_lj = xp.where((GL.rows == l) & (GL.cols == j))[0]
            # print(ind_lj)
            # could happen that the G_{lj} is zero so not found in G
            rank_lj = xp.where(GL.nnz_section_offsets <= ind_lj[0])[0][-1]
            if comm.rank == rank_lj:
                gg_lj = GG.data[..., ind_lj[0] - GG.nnz_section_offsets[rank_lj]]
                gl_lj = GL.data[..., ind_lj[0] - GL.nnz_section_offsets[rank_lj]]
            else:
                ind = xp.where(get_nnz_idx[comm.rank] == ind_lj[0])[0]
                # print(f"rank {comm.rank}:", ind, (i, j, k, l), flush=True)
                # print(f"rank {comm.rank}:", get_nnz_idx[comm.rank], flush=True)
                # print(f"rank {comm.rank}:", gg_recbuf[0], flush=True)
                # print(f"rank {comm.rank}:", ind_lj, flush=True)
                gg_lj = gg_recbuf[..., ind[0]]
                gl_lj = gl_recbuf[..., ind[0]]

            L_ijkl = correlate(gg_ik, gl_lj) - correlate(gl_ik, gg_lj)
            self.L0mat_dist[row, col] = L_ijkl[
                G_nen : G_nen + self.num_E * step_E : step_E
            ]

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

        # if comm.rank == 0:
        #     np.save("L0_rows.npy", self.L0mat_dist.rows)
        #     np.save("L0_cols.npy", self.L0mat_dist.cols)
        #     np.save("L0_data.npy", self.L0mat_dist.data[0])

        #     np.save("bta_rows.npy", self.rows)
        #     np.save("bta_cols.npy", self.cols)

        #     np.save("table.npy", self.table)
        #     np.save("inverse_table.npy", self.inverse_table)
        #     np.save("table_dist.npy", self.table_dist)
        #     np.save("inverse_table_dist.npy", self.inverse_table_dist)

        # return

        # reorder L0mat to BTA shape
        # !!! an additional L0mat gets allocated temporarily !!!

        BLOCK_SIZES = [self.tipsize] + [self.blocksize] * self.num_blocks
        GLOBAL_STACK_SHAPE = (self.num_E,)
        # data = np.zeros(len(self.rows), dtype=xp.complex128)
        # coords = (self.rows, self.cols)
        # coo = sparse.coo_array((data, coords), shape=ARRAY_SHAPE)
        # self.L0mat = DSBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)

        perm_rows = self.inverse_table[*self.table_dist[:, self.L0mat_dist.rows]]
        perm_cols = self.inverse_table[*self.table_dist[:, self.L0mat_dist.cols]]

        permutation = BSESolverDist._compute_permutation(
            get_host(perm_rows), get_host(perm_cols), self.rows, self.cols
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
            (self.tipsize, self.tipsize, self.tipsize, local_nen),
            dtype=xp.complex128,
        )

        for ie in range(local_nen):
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
                            kernel_diag[self.blocksize * k : self.blocksize * (k + 1)]
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
                            kernel_diag[self.blocksize * (k) : self.blocksize * (k + 1)]
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
                            kernel_diag[self.blocksize * k : self.blocksize * (k + 1)]
                        )
                    )
                )

            # solve system matrix
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
            tmp = (
                -1j
                * xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                @ self.L0mat.stack[ie].blocks[0, 0]
            )
            for k in range(self.num_blocks):
                tmp += (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :]))
                    @ self.L0mat.stack[ie].blocks[k + 1, 0]
                )
            for row in range(self.tipsize):
                for col in range(self.tipsize):
                    i = self.table[0, row]
                    j = self.table[0, col]
                    P[i, j, ie] = tmp[row, col]
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

            for row in range(self.tipsize):
                i = self.table[0, row]
                for ib in range(self.num_blocks):
                    for ic in range(self.blocksize):
                        col = ib * self.blocksize + ic + self.tipsize

                        if col < self.size:
                            j = self.table[0, col]
                            k = self.table[1, col]

                            Gamma[i, j, k, ie] = tmp2[row, ic, ib]
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
