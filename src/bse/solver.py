import numba as nb
import numpy as np
from cupyx.scipy import sparse as cusparse
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBCOO
from qttools.utils.gpu_utils import get_device, get_host, xp
from serinv.algs import ddbtasinv


class BSESolver:
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

    # Figure out the info to locate the needed nonzero elements (nnz) whichin an interaction range of `ndiag` of
    # the nnz on the i-th rank.
    #   - get_nnz_size: number of the nnz to gether
    #   - get_nnz_idx: indices ...
    #   - get_nnz_rank: on which ranks ...
    # For example, this gives all the nnz indices needed by i-th rank, which locates on the j-th rank
    #   > mask_i_needs_from_j = np.where(get_nnz_rank[i] == j)[0]
    #   > nnz_i_needs_from_j = get_nnz_idx[i][mask_i_needs_from_j]
    def _determine_rank_map_DSBCOO(self, mat: DSBCOO, ndiag: int):
        """
        Figure out the info to locate the needed nonzero elements (nnz) whichin an interaction range of `ndiag` of
        the nnz on the i-th rank.
        """
        offset = mat.block_offsets
        num_rank = len(offset)
        get_nnz_idx = []
        get_nnz_rank = []
        get_nnz_size = []
        for rank in range(num_rank):

            min_row = mat.rows[offset[rank] : offset[rank + 1]].min()
            max_row = mat.rows[offset[rank] : offset[rank + 1]].max()
            min_col = mat.cols[offset[rank] : offset[rank + 1]].min()
            max_col = mat.cols[offset[rank] : offset[rank + 1]].max()

            mask = (
                (mat.rows > (min_row - ndiag))
                & (mat.rows < (max_row + ndiag))
                & (mat.cols > (min_col - ndiag))
                & (mat.cols < (max_col + ndiag))
            )

            idx = xp.where(mask)[0]
            idx_in_rank = xp.array([xp.where(offset <= ind)[0][-1] for ind in idx])

            get_nnz = xp.where(idx_in_rank != rank)[0]
            get_nnz_idx.append(idx[get_nnz])
            get_nnz_rank.append(idx_in_rank[get_nnz])
            get_nnz_size.append(get_nnz.size)

        return get_nnz_size, get_nnz_idx, get_nnz_rank

    # preprocessing the sparsity pattern and decide the block_size and
    # num_blocks in the BTA matrix
    def _preprocess(self):
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
        self.nnz, coords = BSESolver._get_sparsity(self.size, self.cutoff, table)
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
        BLOCK_SIZES = [self.tipsize] + [self.blocksize] * self.num_blocks
        GLOBAL_STACK_SHAPE = (num_E,)
        self.num_E = num_E
        data = xp.zeros(len(self.rows), dtype=xp.complex128)
        self.L0mat = DSBCOO(
            data=data,
            rows=get_device(self.rows),
            cols=get_device(self.cols),
            block_sizes=BLOCK_SIZES,
            global_stack_shape=GLOBAL_STACK_SHAPE,
        )
        del self.rows
        del self.cols
        if self.L0mat.distribution_state == "stack":
            self.L0mat.dtranspose()
        return

    def _calc_noninteracting_twobody(self, GG: xp.array, GL: xp.array, step_E: int):
        if self.L0mat.distribution_state == "stack":
            self.L0mat.dtranspose()
        nnz_section_offsets = np.hstack(([0], np.cumsum(self.L0mat.nnz_section_sizes)))
        start_inz = int(nnz_section_offsets[comm.rank])
        end_inz = int(nnz_section_offsets[comm.rank + 1])
        G_nen = GG.shape[-1]

        for inz in range(start_inz, end_inz):
            row = self.L0mat.rows[inz]
            col = self.L0mat.cols[inz]

            i = self.table[0, row]
            j = self.table[1, row]
            k = self.table[0, col]
            l = self.table[1, col]

            # print(row,col,i,j,k,l)

            L_ijkl = correlate(GG[i, k, :], GL[l, j, :]) - correlate(
                GL[i, k, :], GG[l, j, :]
            )
            self.L0mat[row, col] = L_ijkl[G_nen : G_nen + self.num_E * step_E : step_E]

        # transpose to stack distribution
        self.L0mat.dtranspose()
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
