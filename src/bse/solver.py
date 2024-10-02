from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBCOO,DSBSparse
import numba as nb 
from scipy import sparse
from qttools.utils.gpu_utils import xp

class BSESolver():
    def __init__(self,num_sites:int,cutoff:int) -> None:
        self.num_sites=num_sites
        self.cutoff=cutoff
        
    @nb.njit(parallel=True, fastmath=True)
    def _get_sparsity(size:int,cutoff:int,table:xp.ndarray):
        nnz = 0
        coords = xp.zeros((size, size), dtype=nb.boolean)
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
    
    # preprocessing the sparsity pattern and decide the block_size and
    # num_blocks in the BTA matrix
    def _preprocess(self):
        """Computes some the sparsity pattern and the block-size."""
        self.size = self.num_sites**2 - (self.num_sites - self.cutoff - 1) * (
            self.num_sites - self.cutoff
        )  # compressed system size ~ 2*nm_dev*ndiag-ndiag*ndiag
        self.table = xp.zeros((2, self.size), dtype=int)
        self.inverse_table = xp.zeros((self.num_sites, self.num_sites), dtype=int) * xp.nan
        # construct a lookup table of reordered indices tip for the
        # "exchange" space， where we put the i=j.
        for i in range(self.num_sites):
            self.table[:, i] = [i, i]
            self.inverse_table[i, i] = i

        # then put the others, but within the ndiag
        offset = self.num_sites
        for i in range(self.num_sites):
            l = max(0, i - self.cutoff)
            k = min(self.num_sites - 1, i + self.cutoff)
            for j in range(l, k + 1):
                if i == j:
                    continue
                self.table[:, offset] = [i, j]
                self.inverse_table[i, j] = offset
                offset += 1

        if (offset) != self.size:
            print(f"ERROR!, it={offset}, N={self.size}")

        # determine number of nnz and sparsity pattern
        self.nnz, coords = BSESolver._get_sparsity(self.size,self.cutoff,self.table)
        self.rows, self.cols = coords.nonzero()

        arrow_mask = (self.rows > self.num_sites) & (self.cols > self.num_sites)
        bandwidth = xp.max(self.cols[arrow_mask] - self.rows[arrow_mask]) + 1

        self.blocksize = bandwidth  # <= 2*cutoff*(cutoff)
        self.num_blocks = int(xp.ceil((self.size - self.num_sites) / self.blocksize))
        self.arrowsize = int(self.blocksize) * int(self.num_blocks)
        self.tipsize = self.num_sites
        self.totalsize = self.arrowsize + self.tipsize

        if (comm.rank == 0):
            print("  total arrow size=", self.arrowsize, flush=True)
            print("  arrow bandwidth=", bandwidth, flush=True)
            print("  arrow block size=", self.blocksize, flush=True)
            print("  arrow number of blocks=", self.num_blocks, flush=True)
            print("  nonzero elements=", self.nnz / 1e6, " Million", flush=True)
            print("  nonzero ratio = ", self.nnz / (self.totalsize) ** 2 * 100, " %", flush=True)
        return


    def _alloc_twobody_matrix(self,num_E:int):
        ARRAY_SHAPE = (self.totalsize, self.totalsize)
        BLOCK_SIZES = xp.concatenate([[self.tipsize],self.blocksize*xp.ones(self.num_blocks,dtype=int)])
        GLOBAL_STACK_SHAPE = (num_E,)        
        data = xp.zeros(len(self.rows),dtype=xp.complex128)        
        coords = (self.rows,self.cols)                
        coo = sparse.coo_array((data,coords),shape=ARRAY_SHAPE)
        self.L0mat = DSBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)        
        del self.rows
        del self.cols
        if (self.L0mat.distribution_state == 'stack'):
            self.L0mat.dtranspose()
        return
    
    
    def _calc_noninteracting_twobody(self,GG:xp.array,GL:xp.array):
        if (self.L0mat.distribution_state == 'stack'):
            self.L0mat.dtranspose()
        nnz_section_offsets = xp.hstack(([0], xp.cumsum(self.L0mat.nnz_section_sizes)))
        start_inz = nnz_section_offsets[comm.rank]
        end_inz = nnz_section_offsets[comm.rank+1]
        
        for inz in range(start_inz,end_inz):
            row=self.L0mat.rows[inz]
            col=self.L0mat.cols[inz]

            i=self.table[0,row]
            j=self.table[1,row]
            k=self.table[0,col]
            l=self.table[1,col]
        
            L_ijkl  = correlate(GG[i,k,:],GL[l,j,:]) - correlate(GL[i,k,:],GG[l,j,:])            
            self.L0mat[row,col] = L_ijkl            
        # transpose to stack distribution
        self.L0mat.dtranspose()
        return
    
    def _calc_kernel(self,V:xp.array,W:xp.array):
        
        kernel_tip=xp.zeros((self.tipsize,self.tipsize),dtype=xp.complex128)
        kernel_diag=xp.zeros((self.totalsize-self.tipsize),dtype=xp.complex128)

        for row in range(self.num_sites):
            for col in range(self.num_sites):        
                i=self.table[0,row]
                k=self.table[0,col]
                kernel_tip[row,col] += -1j * V[i,k]
                if (row == col):
                    kernel_tip[row,col] += 1j * W[i,i]
        for row in range(self.num_sites,self.totalsize):
            if (row < self.size):
                i=self.table[0,row]
                j=self.table[1,row]
                kernel_diag[row-self.tipsize] += 1j * W[i,j]
            
        return kernel_tip, kernel_diag
    

    def _solve_interacting_twobody(self,V:xp.array,W:xp.array):

        kernel_tip,kernel_diag = self._calc_kernel(V,W)
        L0_tip = self.L0mat.blocks[0,0]
        # build system matrix

        return
    

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