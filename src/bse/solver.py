from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBCOO,DSBSparse
import numba as nb 
from scipy import sparse
from qttools.utils.gpu_utils import xp, get_host
import numpy as np
from cupyx import scipy as cux

class BSESolver():
    def __init__(self,num_sites:int,cutoff:int) -> None:
        self.num_sites=num_sites
        self.cutoff=cutoff
        
    @nb.njit(parallel=True, fastmath=True)
    def _get_sparsity(size:np.int32,cutoff:np.int32,table:np.ndarray):
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
    
    # preprocessing the sparsity pattern and decide the block_size and
    # num_blocks in the BTA matrix
    def _preprocess(self):
        """Computes some the sparsity pattern and the block-size."""
        self.size = self.num_sites**2 - (self.num_sites - self.cutoff - 1) * (
            self.num_sites - self.cutoff
        )  # compressed system size ~ 2*nm_dev*ndiag-ndiag*ndiag
        self.table = xp.zeros((2, self.size), dtype=xp.int32)
        self.inverse_table = xp.zeros((self.num_sites, self.num_sites), dtype=xp.int32) * xp.nan
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
        table= get_host(self.table)
        self.nnz, coords = BSESolver._get_sparsity(self.size,self.cutoff,table)
        self.rows, self.cols = coords.nonzero()

        arrow_mask = (self.rows > self.num_sites) & (self.cols > self.num_sites)
        bandwidth = np.max(self.cols[arrow_mask] - self.rows[arrow_mask]) + 1

        self.blocksize = bandwidth  # <= 2*cutoff*(cutoff)
        self.num_blocks = int(np.ceil((self.size - self.num_sites) / self.blocksize))
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
        BLOCK_SIZES = np.concatenate([[self.tipsize],self.blocksize*np.ones(self.num_blocks,dtype=int)])
        GLOBAL_STACK_SHAPE = (num_E,)        
        self.num_E=num_E
        data = np.zeros(len(self.rows),dtype=xp.complex128)        
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
        nnz_section_offsets = np.hstack(([0], np.cumsum(self.L0mat.nnz_section_sizes)))
        start_inz = int(nnz_section_offsets[comm.rank])
        end_inz = int(nnz_section_offsets[comm.rank+1])
        G_nen = GG.shape[-1]
        
        for inz in range(start_inz,end_inz):
            row=self.L0mat.rows[inz]
            col=self.L0mat.cols[inz]

            i=self.table[0,row]
            j=self.table[1,row]
            k=self.table[0,col]
            l=self.table[1,col]

            # print(row,col,i,j,k,l)
        
            L_ijkl  = correlate(GG[i,k,:],GL[l,j,:]) - correlate(GL[i,k,:],GG[l,j,:])            
            self.L0mat[row,col] = L_ijkl[G_nen:G_nen+self.num_E] 

        # transpose to stack distribution
        self.L0mat.dtranspose()
        return
    
    def _calc_kernel(self,V:xp.array,W:xp.array):
        
        kernel_tip=xp.zeros((self.tipsize,self.tipsize), dtype=xp.complex128)
        kernel_diag=xp.zeros((self.totalsize-self.tipsize), dtype=xp.complex128)

        kernel_tip = - V[self.table[0,:self.num_sites], self.table[0,:self.num_sites]] + xp.diag(xp.array([W[self.table[0,i],self.table[0,i]] for i in range(self.num_sites)]))
        
        # for row in range(self.num_sites):
        #     for col in range(self.num_sites):        
        #         i=self.table[0,row]
        #         k=self.table[0,col]                         
        #         print(V[i,k])       
        #         kernel_tip[row,col] += - V[i,k]
        #         if (row == col):
        #             kernel_tip[row,col] += W[i,i]
        for row in range(self.num_sites,self.totalsize):
            if (row < self.size):
                i=self.table[0,row]
                j=self.table[1,row]
                kernel_diag[row-self.tipsize] += W[i,j]
        kernel_diag *=  1j
        kernel_tip *= 1j
        return kernel_tip, kernel_diag

    def _densesolve_interacting_twobody(self,V:xp.array,W:xp.array):
        if (self.L0mat.distribution_state != 'stack'):
            self.L0mat.dtranspose()    
        kernel_tip,kernel_diag = self._calc_kernel(V,W)

        K = np.zeros((self.size,self.size),dtype=xp.complex128)    
        P= np.zeros((self.tipsize,self.tipsize,self.L0mat.stack_shape[0]),dtype=xp.complex128)
        Gamma= np.zeros((self.tipsize,self.tipsize,self.tipsize,self.L0mat.stack_shape[0]),dtype=xp.complex128)

        K[:self.tipsize,:self.tipsize] = get_host(kernel_tip)
        K[self.tipsize:,self.tipsize:] = np.diag(get_host(kernel_diag)[:self.size-self.tipsize])

        table= get_host(self.table)

        for ie in range(self.L0mat.stack_shape[0]):
            print('ie=',ie,flush=True)
            data = self.L0mat.data[ie]
            coords = (get_host(self.L0mat.rows), get_host(self.L0mat.cols))            
            L0 = sparse.coo_matrix((get_host(data),coords),
                                   shape= (self.size, self.size)).todense()
            k=get_host(K)
            A = - L0 @ k + np.diag(np.ones(self.size, dtype=np.complex128))
            A = np.linalg.inv(A) @ L0
            
            for row in range(self.tipsize):
                for col in range(self.tipsize):
                    i=table[0,row]
                    j=table[0,col]
                    P[i,j,ie] = A[row,col]

            for row in range(self.tipsize):
                i = table[0, row]
                for col in range(self.tipsize, self.size):                        
                    j = table[0,col]
                    k = table[1,col]
                    Gamma[i,j,k,ie] = A[row, col]        
        return P, Gamma

    def _solve_interacting_twobody(self,V:xp.array,W:xp.array):
        if (self.L0mat.distribution_state != 'stack'):
            self.L0mat.dtranspose()

        kernel_tip,kernel_diag = self._calc_kernel(V,W)

        A_upperarrow = xp.zeros((self.tipsize,self.blocksize,self.num_blocks),dtype=xp.complex128)
        A_lowerarrow = xp.zeros((self.blocksize,self.tipsize,self.num_blocks),dtype=xp.complex128)
        A_diag= xp.zeros((self.blocksize,self.blocksize,self.num_blocks),dtype=xp.complex128)
        A_upper= xp.zeros((self.blocksize,self.blocksize,self.num_blocks-1),dtype=xp.complex128)
        A_lower= xp.zeros((self.blocksize,self.blocksize,self.num_blocks-1),dtype=xp.complex128)

        P= xp.zeros((self.tipsize,self.tipsize,self.L0mat.stack_shape[0]),dtype=xp.complex128)
        Gamma= xp.zeros((self.tipsize,self.tipsize,self.tipsize,self.L0mat.stack_shape[0]),dtype=xp.complex128)
        
        for ie in range(self.L0mat.stack_shape[0]):
            # build system matrix: A = I - L0 @ K

            A_tip = - self.L0mat.stack[ie].blocks[0,0] @ kernel_tip + xp.eye(self.tipsize)
            
            for k in range(self.num_blocks):
                A_diag[:,:,k] = - self.L0mat.stack[ie].blocks[k+1,k+1] @ xp.diag( kernel_diag[self.blocksize*k:self.blocksize*(k+1)] ) + xp.eye(self.blocksize)

            for k in range(self.num_blocks-1):
                A_upper[:,:,k] = - self.L0mat.stack[ie].blocks[k+1,k+2] @ xp.diag( kernel_diag[self.blocksize*(k+1):self.blocksize*(k+2)] )
                A_lower[:,:,k] = - self.L0mat.stack[ie].blocks[k+2,k+1] @ xp.diag( kernel_diag[self.blocksize*(k):self.blocksize*(k+1)] )                

            for k in range(self.num_blocks):
                A_lowerarrow[:,:,k] = - self.L0mat.stack[ie].blocks[k+1,0] @ kernel_tip
                A_upperarrow[:,:,k] = - self.L0mat.stack[ie].blocks[0,k+1] @ xp.diag( kernel_diag[self.blocksize*k:self.blocksize*(k+1)] )

            # need to flip the BTA matrix before calling solver ?

            # solve system matrix in-place, call SerinV

            # need to flip back the BTA matrix ?

            # extract P from tip of solution matrix L =  A^{-1} @ L0, and P := -i L_tip      
            tmp = xp.zeros((self.tipsize,self.tipsize), dtype=xp.complex128)            
            tmp = - 1j * A_tip @ self.L0mat.stack[ie].blocks[0,0] 
            for k in range(self.num_blocks):
                tmp += - 1j * A_upperarrow[:,:,k] @ self.L0mat.stack[ie].blocks[k+1,0]
            for row in range(self.tipsize):
                for col in range(self.tipsize):
                    i=self.table[0,row]
                    j=self.table[0,col]
                    P[i,j,ie] = tmp[row,col]
            # extract Gamma from upper-arrow block of L = A^{-1} @ L0, and Gamma_ijk := L_iijk
            # L_01 = A_00 @ L0_01 + A_01 @ L0_11
            tmp2 = xp.zeros((self.tipsize,self.blocksize,self.num_blocks), dtype=xp.complex128)            
            for k in range(self.num_blocks):
                tmp2[:,:,k] += A_tip @ self.L0mat.stack[ie].blocks[0,k+1]
                tmp2[:,:,k] += A_upperarrow[:,:,k] @ self.L0mat.stack[ie].blocks[k+1,k+1]    
                if k > 0:
                    tmp2[:,:,k] += A_upperarrow[:,:,k-1] @ self.L0mat.stack[ie].blocks[k,k+1]
                if k < self.num_blocks-1:
                    tmp2[:,:,k] += A_upperarrow[:,:,k+1] @ self.L0mat.stack[ie].blocks[k+2,k+1]
            
            for row in range(self.tipsize):
                i = self.table[0, row]
                for ib in range(self.num_blocks):
                    for ic in range(self.blocksize):
                        col = ib*self.blocksize + ic + self.tipsize

                        if col < self.size:
                            j = self.table[0,col]
                            k = self.table[1,col]

                            Gamma[i,j,k,ie] = tmp2[row, ic, ib]
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