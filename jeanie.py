import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, max_n, max_m, max_k, max_l, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    K = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for j in range(1, max_j + 1):
        for i in range(1, max_i + 1):
            for m in range(1, max_m + 1):
                for n in range(1, max_n + 1):
                    for p in range(n_passes):

                        # The index is actually 'p - tid' but need to force it in-bounds
                        L = max(0, min(p - tid, max_l - 1))

                        # For simplicity, we define i, j which start from 1 (offset from I, J)
                        k = K + 1
                        l = L + 1

                        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
                        if K + L == p and (K < max_k and L < max_l):
                            # Don't compute if outside bandwidth
                            if not (abs(m - l) > bandwidth > 0):
                            # if not ((abs(i - j) > bandwidth > 0) or (abs(m - l) > bandwidth > 0)):
                            # if not ((abs(i - j) > bandwidth > 0)  or (abs(m - n) > bandwidth > 0) or (abs(k - l) > bandwidth > 0)):
                            # if not ((abs(i - j) > bandwidth > 0) or (abs(j - n) > bandwidth > 0) or (abs(m - n) > bandwidth > 0) or (abs(m - k) > bandwidth > 0) or (abs(k - l) > bandwidth > 0)):
                                r00 = -R[b, i, j, n, m, k - 1, l - 1] * inv_gamma
                                r01 = -R[b, i, j, n, m, k - 1, l] * inv_gamma
                                r02 = -R[b, i, j, n, m, k, l - 1] * inv_gamma

                                r03 = -R[b, i - 1, j, n, m, k - 1, l - 1] * inv_gamma
                                r04 = -R[b, i - 1, j, n, m, k - 1, l] * inv_gamma
                                r05 = -R[b, i - 1, j, n, m, k, l - 1] * inv_gamma

                                r06 = -R[b, i, j - 1, n, m, k - 1, l - 1] * inv_gamma
                                r07 = -R[b, i, j - 1, n, m, k - 1, l] * inv_gamma
                                r08 = -R[b, i, j - 1, n, m, k, l - 1] * inv_gamma

                                r09 = -R[b, i - 1, j - 1, n, m, k - 1, l - 1] * inv_gamma
                                r10 = -R[b, i - 1, j - 1, n, m, k - 1, l] * inv_gamma
                                r11 = -R[b, i - 1, j - 1, n, m, k, l - 1] * inv_gamma

                                r12 = -R[b, i, j - 1, n, m, k, l] * inv_gamma
                                r13 = -R[b, i - 1, j, n, m, k, l] * inv_gamma
                                r14 = -R[b, i - 1, j - 1, n, m, k, l] * inv_gamma

                                r15 = -R[b, i, j, n-1, m, k - 1, l - 1] * inv_gamma
                                r16 = -R[b, i, j, n-1, m, k - 1, l] * inv_gamma
                                r17 = -R[b, i, j, n-1, m, k, l - 1] * inv_gamma

                                r18 = -R[b, i - 1, j, n-1, m, k - 1, l - 1] * inv_gamma
                                r19 = -R[b, i - 1, j, n-1, m, k - 1, l] * inv_gamma
                                r20 = -R[b, i - 1, j, n-1, m, k, l - 1] * inv_gamma

                                r21 = -R[b, i, j - 1, n-1, m, k - 1, l - 1] * inv_gamma
                                r22 = -R[b, i, j - 1, n-1, m, k - 1, l] * inv_gamma
                                r23 = -R[b, i, j - 1, n-1, m, k, l - 1] * inv_gamma

                                r24 = -R[b, i - 1, j - 1, n-1, m, k - 1, l - 1] * inv_gamma
                                r25 = -R[b, i - 1, j - 1, n-1, m, k - 1, l] * inv_gamma
                                r26 = -R[b, i - 1, j - 1, n-1, m, k, l - 1] * inv_gamma

                                r27 = -R[b, i, j - 1, n-1, m, k, l] * inv_gamma
                                r28 = -R[b, i - 1, j, n-1, m, k, l] * inv_gamma
                                r29 = -R[b, i - 1, j - 1, n-1, m, k, l] * inv_gamma

                                r30 = -R[b, i, j, n, m-1, k - 1, l - 1] * inv_gamma
                                r31 = -R[b, i, j, n, m-1, k - 1, l] * inv_gamma
                                r32 = -R[b, i, j, n, m-1, k, l - 1] * inv_gamma

                                r33 = -R[b, i - 1, j, n, m-1, k - 1, l - 1] * inv_gamma
                                r34 = -R[b, i - 1, j, n, m-1, k - 1, l] * inv_gamma
                                r35 = -R[b, i - 1, j, n, m-1, k, l - 1] * inv_gamma

                                r36 = -R[b, i, j - 1, n, m-1, k - 1, l - 1] * inv_gamma
                                r37 = -R[b, i, j - 1, n, m-1, k - 1, l] * inv_gamma
                                r38 = -R[b, i, j - 1, n, m-1, k, l - 1] * inv_gamma

                                r39 = -R[b, i - 1, j - 1, n, m-1, k - 1, l - 1] * inv_gamma
                                r40 = -R[b, i - 1, j - 1, n, m-1, k - 1, l] * inv_gamma
                                r41 = -R[b, i - 1, j - 1, n, m-1, k, l - 1] * inv_gamma

                                r42 = -R[b, i, j - 1, n, m-1, k, l] * inv_gamma
                                r43 = -R[b, i - 1, j, n, m-1, k, l] * inv_gamma
                                r44 = -R[b, i - 1, j - 1, n, m-1, k, l] * inv_gamma

                                r45 = -R[b, i, j, n - 1, m-1, k - 1, l - 1] * inv_gamma
                                r46 = -R[b, i, j, n - 1, m-1, k - 1, l] * inv_gamma
                                r47 = -R[b, i, j, n - 1, m-1, k, l - 1] * inv_gamma

                                r48 = -R[b, i - 1, j, n - 1, m-1, k - 1, l - 1] * inv_gamma
                                r49 = -R[b, i - 1, j, n - 1, m-1, k - 1, l] * inv_gamma
                                r50 = -R[b, i - 1, j, n - 1, m-1, k, l - 1] * inv_gamma

                                r51 = -R[b, i, j - 1, n - 1, m-1, k - 1, l - 1] * inv_gamma
                                r52 = -R[b, i, j - 1, n - 1, m-1, k - 1, l] * inv_gamma
                                r53 = -R[b, i, j - 1, n - 1, m-1, k, l - 1] * inv_gamma

                                r54 = -R[b, i - 1, j - 1, n - 1, m-1, k - 1, l - 1] * inv_gamma
                                r55 = -R[b, i - 1, j - 1, n - 1, m-1, k - 1, l] * inv_gamma
                                r56 = -R[b, i - 1, j - 1, n - 1, m-1, k, l - 1] * inv_gamma

                                r57 = -R[b, i, j - 1, n - 1, m-1, k, l] * inv_gamma
                                r58 = -R[b, i - 1, j, n - 1, m-1, k, l] * inv_gamma
                                r59 = -R[b, i - 1, j - 1, n - 1, m-1, k, l] * inv_gamma

                                r60 = -R[b, i, j, n, m - 1, k, l] * inv_gamma
                                r61 = -R[b, i, j, n - 1, m, k, l] * inv_gamma
                                r62 = -R[b, i, j, n - 1, m - 1, k, l] * inv_gamma



                                rmax = max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(
                                    max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(
                                        max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(
                                            max(max(max(max(r00, r01), r02), r03), r04), r05), r06), r07), r08), r09),
                                            r10), r11), r12), r13), r14), r15), r16), r17), r18), r19),
                                            r20), r21), r22), r23), r24), r25), r26), r27), r28), r29),
                                        r30), r31), r32), r33), r34), r35), r36), r37), r38), r39),
                                        r40), r41), r42), r43), r44), r45), r46), r47), r48), r49),
                                    r50), r51), r52), r53), r54), r55), r56), r57), r58), r59),
                                    r60), r61), r62)
                                rsum = math.exp(r00 - rmax) + math.exp(r01 - rmax) + math.exp(r02 - rmax) + math.exp(
                                    r03 - rmax) + math.exp(r04 - rmax) + math.exp(r05 - rmax) + math.exp(r06 - rmax) + math.exp(
                                    r07 - rmax) + math.exp(r08 - rmax) + math.exp(r09 - rmax) + math.exp(r10 - rmax) + math.exp(
                                    r11 - rmax) + math.exp(r12 - rmax) + math.exp(r13 - rmax) + math.exp(r14 - rmax) + math.exp(
                                    r15 - rmax) + math.exp(r16 - rmax) + math.exp(r17 - rmax) + math.exp(r18 - rmax) + math.exp(
                                    r19 - rmax) + math.exp(r20 - rmax) + math.exp(r21 - rmax) + math.exp(r22 - rmax) + math.exp(
                                    r23 - rmax) + math.exp(r24 - rmax) + math.exp(r25 - rmax) + math.exp(r26 - rmax) + math.exp(
                                    r27 - rmax) + math.exp(r28 - rmax) + math.exp(r29 - rmax) + math.exp(r30 - rmax) + math.exp(
                                    r31 - rmax) + math.exp(r32 - rmax) + math.exp(r33 - rmax) + math.exp(r34 - rmax) + math.exp(
                                    r35 - rmax) + math.exp(r36 - rmax) + math.exp(r37 - rmax) + math.exp(r38 - rmax) + math.exp(
                                    r39 - rmax) + math.exp(r40 - rmax) + math.exp(r41 - rmax) + math.exp(r42 - rmax) + math.exp(
                                    r43 - rmax) + math.exp(r44 - rmax) + math.exp(r45 - rmax) + math.exp(r46 - rmax) + math.exp(
                                    r47 - rmax) + math.exp(r48 - rmax) + math.exp(r49 - rmax) + math.exp(r50 - rmax) + math.exp(
                                    r51 - rmax) + math.exp(r52 - rmax) + math.exp(r53 - rmax) + math.exp(r54 - rmax) + math.exp(
                                    r55 - rmax) + math.exp(r56 - rmax) + math.exp(r57 - rmax) + math.exp(r58 - rmax) + math.exp(
                                    r59 - rmax) + math.exp(r60 - rmax) + math.exp(r61 - rmax) + math.exp(r62 - rmax)
                                softmin = -gamma * (math.log(rsum) + rmax)
                                R[b, i, j, n, m, k, l] = D[b, i - 1, j - 1, n - 1, m - 1, k - 1, l - 1] + softmin

                        # Wait for other threads in this block
                        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, max_n, max_m, max_u, max_v, n_passes, E):
    k = cuda.blockIdx.x

    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    U = tid

    for j in range(max_j, 0, -1):
        for i in range(max_i, 0, -1):
            for m in range(max_m, 0, -1):
                for n in range(max_n, 0, -1):
                    for p in range(n_passes):

                        # Reverse the order to make the loop go backward
                        rev_p = n_passes - p - 1

                        # convert tid to I, J, then i, j
                        V = max(0, min(rev_p - tid, max_v - 1))

                        u = U + 1
                        v = V + 1

                        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
                        if U + V == rev_p and (U < max_u and V < max_v):

                            if math.isinf(R[k, i, j, n, m, u, v]):
                                R[k, i, j, n, m, u, v] = -math.inf

                            # Don't compute if outside bandwidth
                            if not (abs(m - v) > bandwidth > 0):
                            # if not ((abs(i - j) > bandwidth > 0) or (abs(m - v) > bandwidth > 0)):
                            # if not ((abs(i - j) > bandwidth > 0)  or (abs(m - n) > bandwidth > 0) or (abs(u - v) > bandwidth > 0)):
                            # if not ((abs(i - j) > bandwidth > 0) or (abs(j - n) > bandwidth > 0) or (abs(m - n) > bandwidth > 0) or (abs(m - u) > bandwidth > 0) or (abs(u - v) > bandwidth > 0)):
                                e00 = math.exp((R[k, i, j, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[k, i, j, n, m, u + 1, v]) * inv_gamma)
                                e01 = math.exp((R[k, i, j, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[k, i, j, n, m, u, v + 1]) * inv_gamma)
                                e02 = math.exp(
                                    (R[k, i, j, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[k, i, j, n, m, u + 1, v + 1]) * inv_gamma)

                                e03 = math.exp(
                                    (R[k, i + 1, j, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[k, i + 1, j, n, m, u + 1, v]) * inv_gamma)
                                e04 = math.exp(
                                    (R[k, i + 1, j, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[k, i + 1, j, n, m, u, v + 1]) * inv_gamma)
                                e05 = math.exp((R[k, i + 1, j, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n, m, u + 1, v + 1]) * inv_gamma)

                                e06 = math.exp(
                                    (R[k, i, j + 1, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[k, i, j + 1, n, m, u + 1, v]) * inv_gamma)
                                e07 = math.exp(
                                    (R[k, i, j + 1, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[k, i, j + 1, n, m, u, v + 1]) * inv_gamma)
                                e08 = math.exp((R[k, i, j + 1, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n, m, u + 1, v + 1]) * inv_gamma)

                                e09 = math.exp((R[k, i + 1, j + 1, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m, u + 1, v]) * inv_gamma)
                                e10 = math.exp((R[k, i + 1, j + 1, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m, u, v + 1]) * inv_gamma)
                                e11 = math.exp((R[k, i + 1, j + 1, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m, u + 1, v + 1]) * inv_gamma)

                                e12 = math.exp((R[k, i + 1, j, n, m, u, v] - R[k, i, j, n, m, u, v] - D[k, i + 1, j, n, m, u, v]) * inv_gamma)
                                e13 = math.exp((R[k, i, j + 1, n, m, u, v] - R[k, i, j, n, m, u, v] - D[k, i, j + 1, n, m, u, v]) * inv_gamma)
                                e14 = math.exp(
                                    (R[k, i + 1, j + 1, n, m, u, v] - R[k, i, j, n, m, u, v] - D[k, i + 1, j + 1, n, m, u, v]) * inv_gamma)

                                e15 = math.exp((R[k, i, j, n+1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n+1, m, u + 1, v]) * inv_gamma)
                                e16 = math.exp((R[k, i, j, n+1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n+1, m, u, v + 1]) * inv_gamma)
                                e17 = math.exp(
                                    (R[k, i, j, n+1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n+1, m, u + 1, v + 1]) * inv_gamma)

                                e18 = math.exp(
                                    (R[k, i + 1, j, n+1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n+1, m, u + 1, v]) * inv_gamma)
                                e19 = math.exp(
                                    (R[k, i + 1, j, n+1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n+1, m, u, v + 1]) * inv_gamma)
                                e20 = math.exp((R[k, i + 1, j, n+1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n+1, m, u + 1, v + 1]) * inv_gamma)

                                e21 = math.exp(
                                    (R[k, i, j + 1, n+1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n+1, m, u + 1, v]) * inv_gamma)
                                e22 = math.exp(
                                    (R[k, i, j + 1, n+1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n+1, m, u, v + 1]) * inv_gamma)
                                e23 = math.exp((R[k, i, j + 1, n+1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n+1, m, u + 1, v + 1]) * inv_gamma)

                                e24 = math.exp((R[k, i + 1, j + 1, n+1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n+1, m, u + 1, v]) * inv_gamma)
                                e25 = math.exp((R[k, i + 1, j + 1, n+1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n+1, m, u, v + 1]) * inv_gamma)
                                e26 = math.exp((R[k, i + 1, j + 1, n+1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n+1, m, u + 1, v + 1]) * inv_gamma)

                                e27 = math.exp((R[k, i + 1, j, n+1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n+1, m, u, v]) * inv_gamma)
                                e28 = math.exp((R[k, i, j + 1, n+1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n+1, m, u, v]) * inv_gamma)
                                e29 = math.exp(
                                    (R[k, i + 1, j + 1, n+1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n+1, m, u, v]) * inv_gamma)

                                e30 = math.exp((R[k, i, j, n, m+1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m+1, u + 1, v]) * inv_gamma)
                                e31 = math.exp((R[k, i, j, n, m+1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m+1, u, v + 1]) * inv_gamma)
                                e32 = math.exp(
                                    (R[k, i, j, n, m+1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n, m+1, u + 1, v + 1]) * inv_gamma)

                                e33 = math.exp(
                                    (R[k, i + 1, j, n, m+1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n, m+1, u + 1, v]) * inv_gamma)
                                e34 = math.exp(
                                    (R[k, i + 1, j, n, m+1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n, m+1, u, v + 1]) * inv_gamma)
                                e35 = math.exp((R[k, i + 1, j, n, m+1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n, m+1, u + 1, v + 1]) * inv_gamma)

                                e36 = math.exp(
                                    (R[k, i, j + 1, n, m+1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n, m+1, u + 1, v]) * inv_gamma)
                                e37 = math.exp(
                                    (R[k, i, j + 1, n, m+1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n, m+1, u, v + 1]) * inv_gamma)
                                e38 = math.exp((R[k, i, j + 1, n, m+1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n, m+1, u + 1, v + 1]) * inv_gamma)

                                e39 = math.exp((R[k, i + 1, j + 1, n, m+1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m+1, u + 1, v]) * inv_gamma)
                                e40 = math.exp((R[k, i + 1, j + 1, n, m+1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m+1, u, v + 1]) * inv_gamma)
                                e41 = math.exp((R[k, i + 1, j + 1, n, m+1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m+1, u + 1, v + 1]) * inv_gamma)

                                e42 = math.exp((R[k, i + 1, j, n, m+1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n, m+1, u, v]) * inv_gamma)
                                e43 = math.exp((R[k, i, j + 1, n, m+1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n, m+1, u, v]) * inv_gamma)
                                e44 = math.exp(
                                    (R[k, i + 1, j + 1, n, m+1, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n, m+1, u, v]) * inv_gamma)

                                e45 = math.exp((R[k, i, j, n+1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n+1, m + 1, u + 1, v]) * inv_gamma)
                                e46 = math.exp((R[k, i, j, n+1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n+1, m + 1, u, v + 1]) * inv_gamma)
                                e47 = math.exp(
                                    (R[k, i, j, n+1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n+1, m + 1, u + 1, v + 1]) * inv_gamma)

                                e48 = math.exp(
                                    (R[k, i + 1, j, n+1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n+1, m + 1, u + 1, v]) * inv_gamma)
                                e49 = math.exp(
                                    (R[k, i + 1, j, n+1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n+1, m + 1, u, v + 1]) * inv_gamma)
                                e50 = math.exp((R[k, i + 1, j, n+1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n+1, m + 1, u + 1, v + 1]) * inv_gamma)

                                e51 = math.exp(
                                    (R[k, i, j + 1, n+1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n+1, m + 1, u + 1, v]) * inv_gamma)
                                e52 = math.exp(
                                    (R[k, i, j + 1, n+1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n+1, m + 1, u, v + 1]) * inv_gamma)
                                e53 = math.exp((R[k, i, j + 1, n+1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n+1, m + 1, u + 1, v + 1]) * inv_gamma)

                                e54 = math.exp((R[k, i + 1, j + 1, n+1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n+1, m + 1, u + 1, v]) * inv_gamma)
                                e55 = math.exp((R[k, i + 1, j + 1, n+1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n+1, m + 1, u, v + 1]) * inv_gamma)
                                e56 = math.exp(
                                    (R[k, i + 1, j + 1, n+1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n+1, m + 1, u + 1, v + 1]) * inv_gamma)

                                e57 = math.exp((R[k, i + 1, j, n+1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n+1, m + 1, u, v]) * inv_gamma)
                                e58 = math.exp((R[k, i, j + 1, n+1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n+1, m + 1, u, v]) * inv_gamma)
                                e59 = math.exp(
                                    (R[k, i + 1, j + 1, n+1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n+1, m + 1, u, v]) * inv_gamma)

                                e60 = math.exp((R[k, i, j, n + 1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n + 1, m, u, v]) * inv_gamma)
                                e61 = math.exp((R[k, i, j, n, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m + 1, u, v]) * inv_gamma)
                                e62 = math.exp(
                                    (R[k, i, j, n + 1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n + 1, m + 1, u, v]) * inv_gamma)



                                E[k, i, j, n, m, u, v] = E[k, i, j, n, m, u + 1, v] * e00 + E[k, i, j, n, m, u, v + 1] * e01 + E[
                                    k, i, j, n, m, u + 1, v + 1] * e02 + E[k, i + 1, j, n, m, u + 1, v] * e03 + E[k, i + 1, j, n, m, u, v + 1] * e04 + \
                                                   E[k, i + 1, j, n, m, u + 1, v + 1] * e05 + E[k, i, j + 1, n, m, u + 1, v] * e06 + E[
                                                       k, i, j + 1, n, m, u, v + 1] * e07 + E[k, i, j + 1, n, m, u + 1, v + 1] * e08 + E[
                                                       k, i + 1, j + 1, n, m, u + 1, v] * e09 + E[k, i + 1, j + 1, n, m, u, v + 1] * e10 + \
                                                   E[k, i + 1, j + 1, n, m, u + 1, v + 1] * e11 + E[k, i + 1, j, n, m, u, v] * e12 + E[
                                                       k, i, j + 1, n, m, u, v] * e13 + E[k, i + 1, j + 1, n, m, u, v] * e14 + E[k, i, j, n+1, m, u + 1, v] * e15 + E[k, i, j, n+1, m, u, v + 1] * e16 + E[
                                    k, i, j, n+1, m, u + 1, v + 1] * e17 + E[k, i + 1, j, n+1, m, u + 1, v] * e18 + E[k, i + 1, j, n+1, m, u, v + 1] * e19 + \
                                                   E[k, i + 1, j, n+1, m, u + 1, v + 1] * e20 + E[k, i, j + 1, n+1, m, u + 1, v] * e21 + E[
                                                       k, i, j + 1, n+1, m, u, v + 1] * e22 + E[k, i, j + 1, n+1, m, u + 1, v + 1] * e23 + E[
                                                       k, i + 1, j + 1, n+1, m, u + 1, v] * e24 + E[k, i + 1, j + 1, n+1, m, u, v + 1] * e25 + \
                                                   E[k, i + 1, j + 1, n+1, m, u + 1, v + 1] * e26 + E[k, i + 1, j, n+1, m, u, v] * e27 + E[
                                                       k, i, j + 1, n+1, m, u, v] * e28 + E[k, i + 1, j + 1, n+1, m, u, v] * e29 + E[k, i, j, n, m+1, u + 1, v] * e30 + E[k, i, j, n, m+1, u, v + 1] * e31 + E[
                                    k, i, j, n, m+1, u + 1, v + 1] * e32 + E[k, i + 1, j, n, m+1, u + 1, v] * e33 + E[k, i + 1, j, n, m+1, u, v + 1] * e34 + \
                                                   E[k, i + 1, j, n, m+1, u + 1, v + 1] * e35 + E[k, i, j + 1, n, m+1, u + 1, v] * e36 + E[
                                                       k, i, j + 1, n, m+1, u, v + 1] * e37 + E[k, i, j + 1, n, m+1, u + 1, v + 1] * e38 + E[
                                                       k, i + 1, j + 1, n, m+1, u + 1, v] * e39 + E[k, i + 1, j + 1, n, m+1, u, v + 1] * e40 + \
                                                   E[k, i + 1, j + 1, n, m+1, u + 1, v + 1] * e41 + E[k, i + 1, j, n, m+1, u, v] * e42 + E[
                                                       k, i, j + 1, n, m+1, u, v] * e43 + E[k, i + 1, j + 1, n, m+1, u, v] * e44 + E[k, i, j, n+1, m+1, u + 1, v] * e45 + E[k, i, j, n+1, m+1, u, v + 1] * e46 + E[
                                    k, i, j, n+1, m+1, u + 1, v + 1] * e47 + E[k, i + 1, j, n+1, m+1, u + 1, v] * e48 + E[k, i + 1, j, n+1, m+1, u, v + 1] * e49 + \
                                                   E[k, i + 1, j, n+1, m+1, u + 1, v + 1] * e50 + E[k, i, j + 1, n+1, m+1, u + 1, v] * e51 + E[
                                                       k, i, j + 1, n+1, m+1, u, v + 1] * e52 + E[k, i, j + 1, n+1, m+1, u + 1, v + 1] * e53 + E[
                                                       k, i + 1, j + 1, n+1, m+1, u + 1, v] * e54 + E[k, i + 1, j + 1, n+1, m+1, u, v + 1] * e55 + \
                                                   E[k, i + 1, j + 1, n+1, m+1, u + 1, v + 1] * e56 + E[k, i + 1, j, n+1, m+1, u, v] * e57 + E[
                                                       k, i, j + 1, n+1, m+1, u, v] * e58 + E[k, i + 1, j + 1, n+1, m+1, u, v] * e59 + E[k, i, j, n+1, m, u, v] * e60 + E[
                                                       k, i, j, n, m+1, u, v] * e61 + E[k, i, j, n+1, m+1, u, v] * e62

                        # Wait for other threads in this block
                        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]

        V_P = D.shape[3]
        V_Q = D.shape[4]

        V_N = D.shape[5]
        V_M = D.shape[6]

        threads_per_block = max(V_N, V_M)

        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2, V_P + 2, V_Q + 2, V_N + 2, V_M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0, 0, 0, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, V_P, V_Q, V_N, V_M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, R, gamma, bandwidth)

        l1 = D.shape[4]
        l2 = D.shape[6]
        bw = int(bandwidth.item())

        if l1 < l2:
            if bw >= abs(l1 - l2) or bw == 0:
                V = R[:, -2, -2, -2, -2, -2, -2]
            else:
                V = R[:, -2, -2, -2, -2, -2, l1 - l2 + int(bandwidth.item())-2]
        elif l1 > l2:
            if bw >= abs(l1 - l2) or bw == 0:
                V = R[:, -2, -2, -2, -2, -2, -2]
            else:
                V = R[:, -2, -2, -2, l2 - l1 + int(bandwidth.item())-2, -2, -2]

        else:
            V = R[:, -2, -2, -2, -2, -2, -2]



        return V # R[:, -2, -2, -2, -2, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]

        V_P = D.shape[3]
        V_Q = D.shape[4]

        V_N = D.shape[5]
        V_M = D.shape[6]

        threads_per_block = max(V_N, V_M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2, V_P + 2, V_Q + 2, V_N + 2, V_M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1, 1:V_P + 1, 1:V_Q + 1, 1:V_N + 1, 1:V_M + 1] = D

        E = torch.zeros((B, N + 2, M + 2, V_P + 2, V_Q + 2, V_N + 2, V_M + 2), dtype=dtype, device=dev)
        # E[:, -1, -1, -1, -1, -1, -1] = 1

        l1 = D.shape[4]
        l2 = D.shape[6]
        bw = int(bandwidth.item())

        # print(l1, l2, '---')

        if l1 < l2:
            if bw >= abs(l1 - l2) or bw == 0:
                E[:, -1, -1, -1, -1, -1, -1] = 1

                R[:, :, :, :, :, :, -1] = -math.inf
                R[:, :, :, :, :, -1, :] = -math.inf
                R[:, :, :, :, -1, :, :] = -math.inf
                R[:, :, :, -1, :, :, :] = -math.inf
                R[:, :, -1, :, :, :, :] = -math.inf
                R[:, -1, :, :, :, :, :] = -math.inf

                R[:, -1, -1, :, :, :, :] = -math.inf
                R[:, -1, :, -1, :, :, :] = -math.inf
                R[:, -1, :, :, -1, :, :] = -math.inf
                R[:, -1, :, :, :, -1, :] = -math.inf
                R[:, -1, :, :, :, :, -1] = -math.inf
                R[:, :, -1, -1, :, :, :] = -math.inf
                R[:, :, -1, :, -1, :, :] = -math.inf
                R[:, :, -1, :, :, -1, :] = -math.inf
                R[:, :, -1, :, :, :, -1] = -math.inf
                R[:, :, :, -1, -1, :, :] = -math.inf
                R[:, :, :, -1, :, -1, :] = -math.inf
                R[:, :, :, -1, :, :, -1] = -math.inf
                R[:, :, :, :, -1, -1, :] = -math.inf
                R[:, :, :, :, -1, :, -1] = -math.inf
                R[:, :, :, :, :, -1, -1] = -math.inf

                R[:, -1, -1, -1, :, :, :] = -math.inf
                R[:, -1, -1, :, -1, :, :] = -math.inf
                R[:, -1, -1, :, :, -1, :] = -math.inf
                R[:, -1, -1, :, :, :, -1] = -math.inf
                R[:, -1, :, -1, -1, :, :] = -math.inf
                R[:, -1, :, -1, :, -1, :] = -math.inf
                R[:, -1, :, -1, :, :, -1] = -math.inf
                R[:, -1, :, :, -1, -1, :] = -math.inf
                R[:, -1, :, :, -1, :, -1] = -math.inf
                R[:, -1, :, :, :, -1, -1] = -math.inf
                R[:, :, -1, -1, -1, :, :] = -math.inf
                R[:, :, -1, -1, :, -1, :] = -math.inf
                R[:, :, -1, -1, :, :, -1] = -math.inf
                R[:, :, -1, :, -1, -1, :] = -math.inf
                R[:, :, -1, :, -1, :, -1] = -math.inf
                R[:, :, -1, :, :, -1, -1] = -math.inf
                R[:, :, :, -1, -1, -1, :] = -math.inf
                R[:, :, :, -1, -1, :, -1] = -math.inf
                R[:, :, :, -1, :, -1, -1] = -math.inf
                R[:, :, :, :, -1, -1, -1] = -math.inf

                R[:, :, :, -1, -1, -1, -1] = -math.inf
                R[:, :, -1, :, -1, -1, -1] = -math.inf
                R[:, :, -1, -1, :, -1, -1] = -math.inf
                R[:, :, -1, -1, -1, :, -1] = -math.inf
                R[:, :, -1, -1, -1, -1, :] = -math.inf
                R[:, -1, :, :, -1, -1, -1] = -math.inf
                R[:, -1, :, -1, :, -1, -1] = -math.inf
                R[:, -1, :, -1, -1, :, -1] = -math.inf
                R[:, -1, :, -1, -1, -1, :] = -math.inf
                R[:, -1, -1, :, :, -1, -1] = -math.inf
                R[:, -1, -1, :, -1, :, -1] = -math.inf
                R[:, -1, -1, :, -1, -1, :] = -math.inf
                R[:, -1, -1, -1, :, :, -1] = -math.inf
                R[:, -1, -1, -1, :, -1, :] = -math.inf
                R[:, -1, -1, -1, -1, :, :] = -math.inf

                R[:, -1, -1, -1, -1, -1, :] = -math.inf
                R[:, -1, -1, -1, -1, :, -1] = -math.inf
                R[:, -1, -1, -1, :, -1, -1] = -math.inf
                R[:, -1, -1, :, -1, -1, -1] = -math.inf
                R[:, -1, :, -1, -1, -1, -1] = -math.inf
                R[:, :, -1, -1, -1, -1, -1] = -math.inf

                R[:, -1, -1, -1, -1, -1, -1] = R[:, -2, -2, -2, -2, -2, -2]
            else:
                E[:, -1, -1, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = 1

                R[:, :, :, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, :, :, :, -1, :] = -math.inf
                R[:, :, :, :, -1, :, :] = -math.inf
                R[:, :, :, -1, :, :, :] = -math.inf
                R[:, :, -1, :, :, :, :] = -math.inf
                R[:, -1, :, :, :, :, :] = -math.inf

                R[:, -1, -1, :, :, :, :] = -math.inf
                R[:, -1, :, -1, :, :, :] = -math.inf
                R[:, -1, :, :, -1, :, :] = -math.inf
                R[:, -1, :, :, :, -1, :] = -math.inf
                R[:, -1, :, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, -1, :, :, :] = -math.inf
                R[:, :, -1, :, -1, :, :] = -math.inf
                R[:, :, -1, :, :, -1, :] = -math.inf
                R[:, :, -1, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, :, -1, -1, :, :] = -math.inf
                R[:, :, :, -1, :, -1, :] = -math.inf
                R[:, :, :, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, :, :, -1, -1, :] = -math.inf
                R[:, :, :, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, :, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf

                R[:, -1, -1, -1, :, :, :] = -math.inf
                R[:, -1, -1, :, -1, :, :] = -math.inf
                R[:, -1, -1, :, :, -1, :] = -math.inf
                R[:, -1, -1, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, :, -1, -1, :, :] = -math.inf
                R[:, -1, :, -1, :, -1, :] = -math.inf
                R[:, -1, :, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, :, :, -1, -1, :] = -math.inf
                R[:, -1, :, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, :, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, -1, -1, :, :] = -math.inf
                R[:, :, -1, -1, :, -1, :] = -math.inf
                R[:, :, -1, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, :, -1, -1, :] = -math.inf
                R[:, :, -1, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, :, -1, -1, -1, :] = -math.inf
                R[:, :, :, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, :, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, :, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf

                R[:, :, :, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, -1, -1, -1, :] = -math.inf
                R[:, -1, :, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, :, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, :, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, :, -1, -1, -1, :] = -math.inf
                R[:, -1, -1, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, -1, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, -1, :, -1, -1, :] = -math.inf
                R[:, -1, -1, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, -1, -1, :, -1, :] = -math.inf
                R[:, -1, -1, -1, -1, :, :] = -math.inf

                R[:, -1, -1, -1, -1, -1, :] = -math.inf
                R[:, -1, -1, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, -1, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, -1, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, -1, :, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf
                R[:, :, -1, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -math.inf

                R[:, -1, -1, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = R[:, -2, -2, -2, -2, -2, l1 - l2 + int(bandwidth.item())-2]


        elif l1 > l2:
            if bw >= abs(l1 - l2) or bw == 0:
                E[:, -1, -1, -1, -1, -1, -1] = 1

                R[:, :, :, :, :, :, -1] = -math.inf
                R[:, :, :, :, :, -1, :] = -math.inf
                R[:, :, :, :, -1, :, :] = -math.inf
                R[:, :, :, -1, :, :, :] = -math.inf
                R[:, :, -1, :, :, :, :] = -math.inf
                R[:, -1, :, :, :, :, :] = -math.inf

                R[:, -1, -1, :, :, :, :] = -math.inf
                R[:, -1, :, -1, :, :, :] = -math.inf
                R[:, -1, :, :, -1, :, :] = -math.inf
                R[:, -1, :, :, :, -1, :] = -math.inf
                R[:, -1, :, :, :, :, -1] = -math.inf
                R[:, :, -1, -1, :, :, :] = -math.inf
                R[:, :, -1, :, -1, :, :] = -math.inf
                R[:, :, -1, :, :, -1, :] = -math.inf
                R[:, :, -1, :, :, :, -1] = -math.inf
                R[:, :, :, -1, -1, :, :] = -math.inf
                R[:, :, :, -1, :, -1, :] = -math.inf
                R[:, :, :, -1, :, :, -1] = -math.inf
                R[:, :, :, :, -1, -1, :] = -math.inf
                R[:, :, :, :, -1, :, -1] = -math.inf
                R[:, :, :, :, :, -1, -1] = -math.inf

                R[:, -1, -1, -1, :, :, :] = -math.inf
                R[:, -1, -1, :, -1, :, :] = -math.inf
                R[:, -1, -1, :, :, -1, :] = -math.inf
                R[:, -1, -1, :, :, :, -1] = -math.inf
                R[:, -1, :, -1, -1, :, :] = -math.inf
                R[:, -1, :, -1, :, -1, :] = -math.inf
                R[:, -1, :, -1, :, :, -1] = -math.inf
                R[:, -1, :, :, -1, -1, :] = -math.inf
                R[:, -1, :, :, -1, :, -1] = -math.inf
                R[:, -1, :, :, :, -1, -1] = -math.inf
                R[:, :, -1, -1, -1, :, :] = -math.inf
                R[:, :, -1, -1, :, -1, :] = -math.inf
                R[:, :, -1, -1, :, :, -1] = -math.inf
                R[:, :, -1, :, -1, -1, :] = -math.inf
                R[:, :, -1, :, -1, :, -1] = -math.inf
                R[:, :, -1, :, :, -1, -1] = -math.inf
                R[:, :, :, -1, -1, -1, :] = -math.inf
                R[:, :, :, -1, -1, :, -1] = -math.inf
                R[:, :, :, -1, :, -1, -1] = -math.inf
                R[:, :, :, :, -1, -1, -1] = -math.inf

                R[:, :, :, -1, -1, -1, -1] = -math.inf
                R[:, :, -1, :, -1, -1, -1] = -math.inf
                R[:, :, -1, -1, :, -1, -1] = -math.inf
                R[:, :, -1, -1, -1, :, -1] = -math.inf
                R[:, :, -1, -1, -1, -1, :] = -math.inf
                R[:, -1, :, :, -1, -1, -1] = -math.inf
                R[:, -1, :, -1, :, -1, -1] = -math.inf
                R[:, -1, :, -1, -1, :, -1] = -math.inf
                R[:, -1, :, -1, -1, -1, :] = -math.inf
                R[:, -1, -1, :, :, -1, -1] = -math.inf
                R[:, -1, -1, :, -1, :, -1] = -math.inf
                R[:, -1, -1, :, -1, -1, :] = -math.inf
                R[:, -1, -1, -1, :, :, -1] = -math.inf
                R[:, -1, -1, -1, :, -1, :] = -math.inf
                R[:, -1, -1, -1, -1, :, :] = -math.inf

                R[:, -1, -1, -1, -1, -1, :] = -math.inf
                R[:, -1, -1, -1, -1, :, -1] = -math.inf
                R[:, -1, -1, -1, :, -1, -1] = -math.inf
                R[:, -1, -1, :, -1, -1, -1] = -math.inf
                R[:, -1, :, -1, -1, -1, -1] = -math.inf
                R[:, :, -1, -1, -1, -1, -1] = -math.inf

                R[:, -1, -1, -1, -1, -1, -1] = R[:, -2, -2, -2, -2, -2, -2]
            else:
                E[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = 1

                R[:, :, :, :, :, :, -1] = -math.inf
                R[:, :, :, :, :, -1, :] = -math.inf
                R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf
                R[:, :, :, -1, :, :, :] = -math.inf
                R[:, :, -1, :, :, :, :] = -math.inf
                R[:, -1, :, :, :, :, :] = -math.inf

                R[:, -1, -1, :, :, :, :] = -math.inf
                R[:, -1, :, -1, :, :, :] = -math.inf
                R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf
                R[:, -1, :, :, :, -1, :] = -math.inf
                R[:, -1, :, :, :, :, -1] = -math.inf
                R[:, :, -1, -1, :, :, :] = -math.inf
                R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf
                R[:, :, -1, :, :, -1, :] = -math.inf
                R[:, :, -1, :, :, :, -1] = -math.inf
                R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf
                R[:, :, :, -1, :, -1, :] = -math.inf
                R[:, :, :, -1, :, :, -1] = -math.inf
                R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, :, :, :, :, -1, -1] = -math.inf

                R[:, -1, -1, -1, :, :, :] = -math.inf
                R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf
                R[:, -1, -1, :, :, -1, :] = -math.inf
                R[:, -1, -1, :, :, :, -1] = -math.inf
                R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf
                R[:, -1, :, -1, :, -1, :] = -math.inf
                R[:, -1, :, -1, :, :, -1] = -math.inf
                R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, -1, :, :, :, -1, -1] = -math.inf
                R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf
                R[:, :, -1, -1, :, -1, :] = -math.inf
                R[:, :, -1, -1, :, :, -1] = -math.inf
                R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, :, -1, :, :, -1, -1] = -math.inf
                R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, :, :, -1, :, -1, -1] = -math.inf
                R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -math.inf

                R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -math.inf
                R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -math.inf
                R[:, :, -1, -1, :, -1, -1] = -math.inf
                R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -math.inf
                R[:, -1, :, -1, :, -1, -1] = -math.inf
                R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, -1, -1, :, :, -1, -1] = -math.inf
                R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, -1, -1, -1, :, :, -1] = -math.inf
                R[:, -1, -1, -1, :, -1, :] = -math.inf
                R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -math.inf

                R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -math.inf
                R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -math.inf
                R[:, -1, -1, -1, :, -1, -1] = -math.inf
                R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -math.inf
                R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -math.inf
                R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -math.inf

                R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = R[:, -2, -2, -2, l2 - l1 + int(bandwidth.item())-2, -2, -2]


        else:
            E[:, -1, -1, -1, -1, -1, -1] = 1

            R[:, :, :, :, :, :, -1] = -math.inf
            R[:, :, :, :, :, -1, :] = -math.inf
            R[:, :, :, :, -1, :, :] = -math.inf
            R[:, :, :, -1, :, :, :] = -math.inf
            R[:, :, -1, :, :, :, :] = -math.inf
            R[:, -1, :, :, :, :, :] = -math.inf

            R[:, -1, -1, :, :, :, :] = -math.inf
            R[:, -1, :, -1, :, :, :] = -math.inf
            R[:, -1, :, :, -1, :, :] = -math.inf
            R[:, -1, :, :, :, -1, :] = -math.inf
            R[:, -1, :, :, :, :, -1] = -math.inf
            R[:, :, -1, -1, :, :, :] = -math.inf
            R[:, :, -1, :, -1, :, :] = -math.inf
            R[:, :, -1, :, :, -1, :] = -math.inf
            R[:, :, -1, :, :, :, -1] = -math.inf
            R[:, :, :, -1, -1, :, :] = -math.inf
            R[:, :, :, -1, :, -1, :] = -math.inf
            R[:, :, :, -1, :, :, -1] = -math.inf
            R[:, :, :, :, -1, -1, :] = -math.inf
            R[:, :, :, :, -1, :, -1] = -math.inf
            R[:, :, :, :, :, -1, -1] = -math.inf

            R[:, -1, -1, -1, :, :, :] = -math.inf
            R[:, -1, -1, :, -1, :, :] = -math.inf
            R[:, -1, -1, :, :, -1, :] = -math.inf
            R[:, -1, -1, :, :, :, -1] = -math.inf
            R[:, -1, :, -1, -1, :, :] = -math.inf
            R[:, -1, :, -1, :, -1, :] = -math.inf
            R[:, -1, :, -1, :, :, -1] = -math.inf
            R[:, -1, :, :, -1, -1, :] = -math.inf
            R[:, -1, :, :, -1, :, -1] = -math.inf
            R[:, -1, :, :, :, -1, -1] = -math.inf
            R[:, :, -1, -1, -1, :, :] = -math.inf
            R[:, :, -1, -1, :, -1, :] = -math.inf
            R[:, :, -1, -1, :, :, -1] = -math.inf
            R[:, :, -1, :, -1, -1, :] = -math.inf
            R[:, :, -1, :, -1, :, -1] = -math.inf
            R[:, :, -1, :, :, -1, -1] = -math.inf
            R[:, :, :, -1, -1, -1, :] = -math.inf
            R[:, :, :, -1, -1, :, -1] = -math.inf
            R[:, :, :, -1, :, -1, -1] = -math.inf
            R[:, :, :, :, -1, -1, -1] = -math.inf

            R[:, :, :, -1, -1, -1, -1] = -math.inf
            R[:, :, -1, :, -1, -1, -1] = -math.inf
            R[:, :, -1, -1, :, -1, -1] = -math.inf
            R[:, :, -1, -1, -1, :, -1] = -math.inf
            R[:, :, -1, -1, -1, -1, :] = -math.inf
            R[:, -1, :, :, -1, -1, -1] = -math.inf
            R[:, -1, :, -1, :, -1, -1] = -math.inf
            R[:, -1, :, -1, -1, :, -1] = -math.inf
            R[:, -1, :, -1, -1, -1, :] = -math.inf
            R[:, -1, -1, :, :, -1, -1] = -math.inf
            R[:, -1, -1, :, -1, :, -1] = -math.inf
            R[:, -1, -1, :, -1, -1, :] = -math.inf
            R[:, -1, -1, -1, :, :, -1] = -math.inf
            R[:, -1, -1, -1, :, -1, :] = -math.inf
            R[:, -1, -1, -1, -1, :, :] = -math.inf

            R[:, -1, -1, -1, -1, -1, :] = -math.inf
            R[:, -1, -1, -1, -1, :, -1] = -math.inf
            R[:, -1, -1, -1, :, -1, -1] = -math.inf
            R[:, -1, -1, :, -1, -1, -1] = -math.inf
            R[:, -1, :, -1, -1, -1, -1] = -math.inf
            R[:, :, -1, -1, -1, -1, -1] = -math.inf

            R[:, -1, -1, -1, -1, -1, -1] = R[:, -2, -2, -2, -2, -2, -2]

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, V_P, V_Q, V_N, V_M,
                                                            n_passes, cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1, 1:V_P + 1, 1:V_Q + 1, 1:V_N + 1, 1:V_M + 1]
        return grad_output.view(-1, 1, 1, 1, 1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]

    V_P = D.shape[3]
    V_Q = D.shape[4]

    V_N = D.shape[5]
    V_M = D.shape[6]

    R = np.ones((B, N + 2, M + 2, V_P + 2, V_Q + 2, V_N + 2, V_M + 2)) * np.inf
    R[:, 0, 0, 0, 0, 0, 0] = 0

    for b in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                for m in range(1, V_Q + 1):
                    for n in range(1, V_P + 1):
                        for l in range(1, V_M + 1):
                            for k in range(1, V_N + 1):

                                # Check the pruning condition
                                if 0 < bandwidth < np.abs(m - l):
                                # if 0 < bandwidth < np.abs(i - j) or 0 < bandwidth < np.abs(m - l):
                                # if 0 < bandwidth < np.abs(i - j) or 0 < bandwidth < np.abs(m - n) or 0 < bandwidth < np.abs(k-l):
                                # if 0 < bandwidth < np.abs(i - j) or 0 < bandwidth < np.abs(j - n) or 0 < bandwidth < np.abs(m - n) or 0 < bandwidth < np.abs(m - k) or 0 < bandwidth < np.abs(k-l):
                                    continue

                                r00 = -R[b, i, j, n, m, k - 1, l - 1] / gamma
                                r01 = -R[b, i, j, n, m, k - 1, l] / gamma
                                r02 = -R[b, i, j, n, m, k, l - 1] / gamma

                                r03 = -R[b, i - 1, j, n, m, k - 1, l - 1] / gamma
                                r04 = -R[b, i - 1, j, n, m, k - 1, l] / gamma
                                r05 = -R[b, i - 1, j, n, m, k, l - 1] / gamma

                                r06 = -R[b, i, j - 1, n, m, k - 1, l - 1] / gamma
                                r07 = -R[b, i, j - 1, n, m, k - 1, l] / gamma
                                r08 = -R[b, i, j - 1, n, m, k, l - 1] / gamma

                                r09 = -R[b, i - 1, j - 1, n, m, k - 1, l - 1] / gamma
                                r10 = -R[b, i - 1, j - 1, n, m, k - 1, l] / gamma
                                r11 = -R[b, i - 1, j - 1, n, m, k, l - 1] / gamma

                                r12 = -R[b, i, j - 1, n, m, k, l] / gamma
                                r13 = -R[b, i - 1, j, n, m, k, l] / gamma
                                r14 = -R[b, i - 1, j - 1, n, m, k, l] / gamma

                                r15 = -R[b, i, j, n - 1, m, k - 1, l - 1] / gamma
                                r16 = -R[b, i, j, n - 1, m, k - 1, l] / gamma
                                r17 = -R[b, i, j, n - 1, m, k, l - 1] / gamma

                                r18 = -R[b, i - 1, j, n - 1, m, k - 1, l - 1] / gamma
                                r19 = -R[b, i - 1, j, n - 1, m, k - 1, l] / gamma
                                r20 = -R[b, i - 1, j, n - 1, m, k, l - 1] / gamma

                                r21 = -R[b, i, j - 1, n - 1, m, k - 1, l - 1] / gamma
                                r22 = -R[b, i, j - 1, n - 1, m, k - 1, l] / gamma
                                r23 = -R[b, i, j - 1, n - 1, m, k, l - 1] / gamma

                                r24 = -R[b, i - 1, j - 1, n - 1, m, k - 1, l - 1] / gamma
                                r25 = -R[b, i - 1, j - 1, n - 1, m, k - 1, l] / gamma
                                r26 = -R[b, i - 1, j - 1, n - 1, m, k, l - 1] / gamma

                                r27 = -R[b, i, j - 1, n - 1, m, k, l] / gamma
                                r28 = -R[b, i - 1, j, n - 1, m, k, l] / gamma
                                r29 = -R[b, i - 1, j - 1, n - 1, m, k, l] / gamma

                                r30 = -R[b, i, j, n, m - 1, k - 1, l - 1] / gamma
                                r31 = -R[b, i, j, n, m - 1, k - 1, l] / gamma
                                r32 = -R[b, i, j, n, m - 1, k, l - 1] / gamma

                                r33 = -R[b, i - 1, j, n, m - 1, k - 1, l - 1] / gamma
                                r34 = -R[b, i - 1, j, n, m - 1, k - 1, l] / gamma
                                r35 = -R[b, i - 1, j, n, m - 1, k, l - 1] / gamma

                                r36 = -R[b, i, j - 1, n, m - 1, k - 1, l - 1] / gamma
                                r37 = -R[b, i, j - 1, n, m - 1, k - 1, l] / gamma
                                r38 = -R[b, i, j - 1, n, m - 1, k, l - 1] / gamma

                                r39 = -R[b, i - 1, j - 1, n, m - 1, k - 1, l - 1] / gamma
                                r40 = -R[b, i - 1, j - 1, n, m - 1, k - 1, l] / gamma
                                r41 = -R[b, i - 1, j - 1, n, m - 1, k, l - 1] / gamma

                                r42 = -R[b, i, j - 1, n, m - 1, k, l] / gamma
                                r43 = -R[b, i - 1, j, n, m - 1, k, l] / gamma
                                r44 = -R[b, i - 1, j - 1, n, m - 1, k, l] / gamma

                                r45 = -R[b, i, j, n - 1, m - 1, k - 1, l - 1] / gamma
                                r46 = -R[b, i, j, n - 1, m - 1, k - 1, l] / gamma
                                r47 = -R[b, i, j, n - 1, m - 1, k, l - 1] / gamma

                                r48 = -R[b, i - 1, j, n - 1, m - 1, k - 1, l - 1] / gamma
                                r49 = -R[b, i - 1, j, n - 1, m - 1, k - 1, l] / gamma
                                r50 = -R[b, i - 1, j, n - 1, m - 1, k, l - 1] / gamma

                                r51 = -R[b, i, j - 1, n - 1, m - 1, k - 1, l - 1] / gamma
                                r52 = -R[b, i, j - 1, n - 1, m - 1, k - 1, l] / gamma
                                r53 = -R[b, i, j - 1, n - 1, m - 1, k, l - 1] / gamma

                                r54 = -R[b, i - 1, j - 1, n - 1, m - 1, k - 1, l - 1] / gamma
                                r55 = -R[b, i - 1, j - 1, n - 1, m - 1, k - 1, l] / gamma
                                r56 = -R[b, i - 1, j - 1, n - 1, m - 1, k, l - 1] / gamma

                                r57 = -R[b, i, j - 1, n - 1, m - 1, k, l] / gamma
                                r58 = -R[b, i - 1, j, n - 1, m - 1, k, l] / gamma
                                r59 = -R[b, i - 1, j - 1, n - 1, m - 1, k, l] / gamma

                                r60 = -R[b, i, j, n, m - 1, k, l] / gamma
                                r61 = -R[b, i, j, n - 1, m, k, l] / gamma
                                r62 = -R[b, i, j, n - 1, m - 1, k, l] / gamma

                                rmax = max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(
                                    max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(
                                        max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(
                                            max(max(max(max(r00, r01), r02), r03), r04), r05), r06), r07), r08), r09),
                                            r10), r11), r12), r13), r14), r15), r16), r17), r18), r19),
                                            r20), r21), r22), r23), r24), r25), r26), r27), r28), r29),
                                        r30), r31), r32), r33), r34), r35), r36), r37), r38), r39),
                                        r40), r41), r42), r43), r44), r45), r46), r47), r48), r49),
                                    r50), r51), r52), r53), r54), r55), r56), r57), r58), r59),
                                    r60), r61), r62)
                                rsum = np.exp(r00 - rmax) + np.exp(r01 - rmax) + np.exp(r02 - rmax) + np.exp(
                                    r03 - rmax) + np.exp(r04 - rmax) + np.exp(r05 - rmax) + np.exp(
                                    r06 - rmax) + np.exp(
                                    r07 - rmax) + np.exp(r08 - rmax) + np.exp(r09 - rmax) + np.exp(
                                    r10 - rmax) + np.exp(
                                    r11 - rmax) + np.exp(r12 - rmax) + np.exp(r13 - rmax) + np.exp(
                                    r14 - rmax) + np.exp(
                                    r15 - rmax) + np.exp(r16 - rmax) + np.exp(r17 - rmax) + np.exp(
                                    r18 - rmax) + np.exp(
                                    r19 - rmax) + np.exp(r20 - rmax) + np.exp(r21 - rmax) + np.exp(
                                    r22 - rmax) + np.exp(
                                    r23 - rmax) + np.exp(r24 - rmax) + np.exp(r25 - rmax) + np.exp(
                                    r26 - rmax) + np.exp(
                                    r27 - rmax) + np.exp(r28 - rmax) + np.exp(r29 - rmax) + np.exp(
                                    r30 - rmax) + np.exp(
                                    r31 - rmax) + np.exp(r32 - rmax) + np.exp(r33 - rmax) + np.exp(
                                    r34 - rmax) + np.exp(
                                    r35 - rmax) + np.exp(r36 - rmax) + np.exp(r37 - rmax) + np.exp(
                                    r38 - rmax) + np.exp(
                                    r39 - rmax) + np.exp(r40 - rmax) + np.exp(r41 - rmax) + np.exp(
                                    r42 - rmax) + np.exp(
                                    r43 - rmax) + np.exp(r44 - rmax) + np.exp(r45 - rmax) + np.exp(
                                    r46 - rmax) + np.exp(
                                    r47 - rmax) + np.exp(r48 - rmax) + np.exp(r49 - rmax) + np.exp(
                                    r50 - rmax) + np.exp(
                                    r51 - rmax) + np.exp(r52 - rmax) + np.exp(r53 - rmax) + np.exp(
                                    r54 - rmax) + np.exp(
                                    r55 - rmax) + np.exp(r56 - rmax) + np.exp(r57 - rmax) + np.exp(
                                    r58 - rmax) + np.exp(
                                    r59 - rmax) + np.exp(r60 - rmax) + np.exp(r61 - rmax) + np.exp(r62 - rmax)

                                # the log-sum-exp stabilization trick
                                softmin = - gamma * (np.log(rsum) + rmax)
                                R[b, i, j, n, m, k, l] = D[b, i - 1, j - 1, n-1, m-1, k - 1, l - 1] + softmin
    return R


# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]

    V_P = D_.shape[3]
    V_Q = D_.shape[4]

    V_N = D_.shape[5]
    V_M = D_.shape[6]

    D = np.zeros((B, N + 2, M + 2, V_P + 2, V_Q + 2, V_N + 2, V_M + 2))
    E = np.zeros((B, N + 2, M + 2, V_P + 2, V_Q + 2, V_N + 2, V_M + 2))

    D[:, 1:N + 1, 1:M + 1, 1:V_P + 1, 1:V_Q + 1, 1:V_N + 1, 1:V_M + 1] = D_

    l1 = D_.shape[4]
    l2 = D_.shape[6]
    bw = int(bandwidth.item())

    # print(l1, l2, '---')

    if l1 < l2:
        if bw >= abs(l1 - l2) or bw == 0:
            E[:, -1, -1, -1, -1, -1, -1] = 1

            R[:, :, :, :, :, :, -1] = -np.inf
            R[:, :, :, :, :, -1, :] = -np.inf
            R[:, :, :, :, -1, :, :] = -np.inf
            R[:, :, :, -1, :, :, :] = -np.inf
            R[:, :, -1, :, :, :, :] = -np.inf
            R[:, -1, :, :, :, :, :] = -np.inf

            R[:, -1, -1, :, :, :, :] = -np.inf
            R[:, -1, :, -1, :, :, :] = -np.inf
            R[:, -1, :, :, -1, :, :] = -np.inf
            R[:, -1, :, :, :, -1, :] = -np.inf
            R[:, -1, :, :, :, :, -1] = -np.inf
            R[:, :, -1, -1, :, :, :] = -np.inf
            R[:, :, -1, :, -1, :, :] = -np.inf
            R[:, :, -1, :, :, -1, :] = -np.inf
            R[:, :, -1, :, :, :, -1] = -np.inf
            R[:, :, :, -1, -1, :, :] = -np.inf
            R[:, :, :, -1, :, -1, :] = -np.inf
            R[:, :, :, -1, :, :, -1] = -np.inf
            R[:, :, :, :, -1, -1, :] = -np.inf
            R[:, :, :, :, -1, :, -1] = -np.inf
            R[:, :, :, :, :, -1, -1] = -np.inf

            R[:, -1, -1, -1, :, :, :] = -np.inf
            R[:, -1, -1, :, -1, :, :] = -np.inf
            R[:, -1, -1, :, :, -1, :] = -np.inf
            R[:, -1, -1, :, :, :, -1] = -np.inf
            R[:, -1, :, -1, -1, :, :] = -np.inf
            R[:, -1, :, -1, :, -1, :] = -np.inf
            R[:, -1, :, -1, :, :, -1] = -np.inf
            R[:, -1, :, :, -1, -1, :] = -np.inf
            R[:, -1, :, :, -1, :, -1] = -np.inf
            R[:, -1, :, :, :, -1, -1] = -np.inf
            R[:, :, -1, -1, -1, :, :] = -np.inf
            R[:, :, -1, -1, :, -1, :] = -np.inf
            R[:, :, -1, -1, :, :, -1] = -np.inf
            R[:, :, -1, :, -1, -1, :] = -np.inf
            R[:, :, -1, :, -1, :, -1] = -np.inf
            R[:, :, -1, :, :, -1, -1] = -np.inf
            R[:, :, :, -1, -1, -1, :] = -np.inf
            R[:, :, :, -1, -1, :, -1] = -np.inf
            R[:, :, :, -1, :, -1, -1] = -np.inf
            R[:, :, :, :, -1, -1, -1] = -np.inf

            R[:, :, :, -1, -1, -1, -1] = -np.inf
            R[:, :, -1, :, -1, -1, -1] = -np.inf
            R[:, :, -1, -1, :, -1, -1] = -np.inf
            R[:, :, -1, -1, -1, :, -1] = -np.inf
            R[:, :, -1, -1, -1, -1, :] = -np.inf
            R[:, -1, :, :, -1, -1, -1] = -np.inf
            R[:, -1, :, -1, :, -1, -1] = -np.inf
            R[:, -1, :, -1, -1, :, -1] = -np.inf
            R[:, -1, :, -1, -1, -1, :] = -np.inf
            R[:, -1, -1, :, :, -1, -1] = -np.inf
            R[:, -1, -1, :, -1, :, -1] = -np.inf
            R[:, -1, -1, :, -1, -1, :] = -np.inf
            R[:, -1, -1, -1, :, :, -1] = -np.inf
            R[:, -1, -1, -1, :, -1, :] = -np.inf
            R[:, -1, -1, -1, -1, :, :] = -np.inf

            R[:, -1, -1, -1, -1, -1, :] = -np.inf
            R[:, -1, -1, -1, -1, :, -1] = -np.inf
            R[:, -1, -1, -1, :, -1, -1] = -np.inf
            R[:, -1, -1, :, -1, -1, -1] = -np.inf
            R[:, -1, :, -1, -1, -1, -1] = -np.inf
            R[:, :, -1, -1, -1, -1, -1] = -np.inf

            R[:, -1, -1, -1, -1, -1, -1] = R[:, -2, -2, -2, -2, -2, -2]
        else:
            E[:, -1, -1, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = 1

            R[:, :, :, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, :, :, :, -1, :] = -np.inf
            R[:, :, :, :, -1, :, :] = -np.inf
            R[:, :, :, -1, :, :, :] = -np.inf
            R[:, :, -1, :, :, :, :] = -np.inf
            R[:, -1, :, :, :, :, :] = -np.inf

            R[:, -1, -1, :, :, :, :] = -np.inf
            R[:, -1, :, -1, :, :, :] = -np.inf
            R[:, -1, :, :, -1, :, :] = -np.inf
            R[:, -1, :, :, :, -1, :] = -np.inf
            R[:, -1, :, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, -1, :, :, :] = -np.inf
            R[:, :, -1, :, -1, :, :] = -np.inf
            R[:, :, -1, :, :, -1, :] = -np.inf
            R[:, :, -1, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, :, -1, -1, :, :] = -np.inf
            R[:, :, :, -1, :, -1, :] = -np.inf
            R[:, :, :, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, :, :, -1, -1, :] = -np.inf
            R[:, :, :, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, :, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf

            R[:, -1, -1, -1, :, :, :] = -np.inf
            R[:, -1, -1, :, -1, :, :] = -np.inf
            R[:, -1, -1, :, :, -1, :] = -np.inf
            R[:, -1, -1, :, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, :, -1, -1, :, :] = -np.inf
            R[:, -1, :, -1, :, -1, :] = -np.inf
            R[:, -1, :, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, :, :, -1, -1, :] = -np.inf
            R[:, -1, :, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, :, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, -1, -1, :, :] = -np.inf
            R[:, :, -1, -1, :, -1, :] = -np.inf
            R[:, :, -1, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, :, -1, -1, :] = -np.inf
            R[:, :, -1, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, :, -1, -1, -1, :] = -np.inf
            R[:, :, :, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, :, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, :, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf

            R[:, :, :, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, -1, -1, -1, :] = -np.inf
            R[:, -1, :, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, :, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, :, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, :, -1, -1, -1, :] = -np.inf
            R[:, -1, -1, :, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, -1, :, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, -1, :, -1, -1, :] = -np.inf
            R[:, -1, -1, -1, :, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, -1, -1, :, -1, :] = -np.inf
            R[:, -1, -1, -1, -1, :, :] = -np.inf

            R[:, -1, -1, -1, -1, -1, :] = -np.inf
            R[:, -1, -1, -1, -1, :, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, -1, -1, :, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, -1, :, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, -1, :, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf
            R[:, :, -1, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = -np.inf

            R[:, -1, -1, -1, -1, -1, l1 - l2 + int(bandwidth.item())-1] = R[:, -2, -2, -2, -2, -2, l1 - l2 + int(bandwidth.item())-2]


    elif l1 > l2:
        if bw >= abs(l1 - l2) or bw == 0:
            E[:, -1, -1, -1, -1, -1, -1] = 1

            R[:, :, :, :, :, :, -1] = -np.inf
            R[:, :, :, :, :, -1, :] = -np.inf
            R[:, :, :, :, -1, :, :] = -np.inf
            R[:, :, :, -1, :, :, :] = -np.inf
            R[:, :, -1, :, :, :, :] = -np.inf
            R[:, -1, :, :, :, :, :] = -np.inf

            R[:, -1, -1, :, :, :, :] = -np.inf
            R[:, -1, :, -1, :, :, :] = -np.inf
            R[:, -1, :, :, -1, :, :] = -np.inf
            R[:, -1, :, :, :, -1, :] = -np.inf
            R[:, -1, :, :, :, :, -1] = -np.inf
            R[:, :, -1, -1, :, :, :] = -np.inf
            R[:, :, -1, :, -1, :, :] = -np.inf
            R[:, :, -1, :, :, -1, :] = -np.inf
            R[:, :, -1, :, :, :, -1] = -np.inf
            R[:, :, :, -1, -1, :, :] = -np.inf
            R[:, :, :, -1, :, -1, :] = -np.inf
            R[:, :, :, -1, :, :, -1] = -np.inf
            R[:, :, :, :, -1, -1, :] = -np.inf
            R[:, :, :, :, -1, :, -1] = -np.inf
            R[:, :, :, :, :, -1, -1] = -np.inf

            R[:, -1, -1, -1, :, :, :] = -np.inf
            R[:, -1, -1, :, -1, :, :] = -np.inf
            R[:, -1, -1, :, :, -1, :] = -np.inf
            R[:, -1, -1, :, :, :, -1] = -np.inf
            R[:, -1, :, -1, -1, :, :] = -np.inf
            R[:, -1, :, -1, :, -1, :] = -np.inf
            R[:, -1, :, -1, :, :, -1] = -np.inf
            R[:, -1, :, :, -1, -1, :] = -np.inf
            R[:, -1, :, :, -1, :, -1] = -np.inf
            R[:, -1, :, :, :, -1, -1] = -np.inf
            R[:, :, -1, -1, -1, :, :] = -np.inf
            R[:, :, -1, -1, :, -1, :] = -np.inf
            R[:, :, -1, -1, :, :, -1] = -np.inf
            R[:, :, -1, :, -1, -1, :] = -np.inf
            R[:, :, -1, :, -1, :, -1] = -np.inf
            R[:, :, -1, :, :, -1, -1] = -np.inf
            R[:, :, :, -1, -1, -1, :] = -np.inf
            R[:, :, :, -1, -1, :, -1] = -np.inf
            R[:, :, :, -1, :, -1, -1] = -np.inf
            R[:, :, :, :, -1, -1, -1] = -np.inf

            R[:, :, :, -1, -1, -1, -1] = -np.inf
            R[:, :, -1, :, -1, -1, -1] = -np.inf
            R[:, :, -1, -1, :, -1, -1] = -np.inf
            R[:, :, -1, -1, -1, :, -1] = -np.inf
            R[:, :, -1, -1, -1, -1, :] = -np.inf
            R[:, -1, :, :, -1, -1, -1] = -np.inf
            R[:, -1, :, -1, :, -1, -1] = -np.inf
            R[:, -1, :, -1, -1, :, -1] = -np.inf
            R[:, -1, :, -1, -1, -1, :] = -np.inf
            R[:, -1, -1, :, :, -1, -1] = -np.inf
            R[:, -1, -1, :, -1, :, -1] = -np.inf
            R[:, -1, -1, :, -1, -1, :] = -np.inf
            R[:, -1, -1, -1, :, :, -1] = -np.inf
            R[:, -1, -1, -1, :, -1, :] = -np.inf
            R[:, -1, -1, -1, -1, :, :] = -np.inf

            R[:, -1, -1, -1, -1, -1, :] = -np.inf
            R[:, -1, -1, -1, -1, :, -1] = -np.inf
            R[:, -1, -1, -1, :, -1, -1] = -np.inf
            R[:, -1, -1, :, -1, -1, -1] = -np.inf
            R[:, -1, :, -1, -1, -1, -1] = -np.inf
            R[:, :, -1, -1, -1, -1, -1] = -np.inf

            R[:, -1, -1, -1, -1, -1, -1] = R[:, -2, -2, -2, -2, -2, -2]
        else:

            E[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = 1

            R[:, :, :, :, :, :, -1] = -np.inf
            R[:, :, :, :, :, -1, :] = -np.inf
            R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf
            R[:, :, :, -1, :, :, :] = -np.inf
            R[:, :, -1, :, :, :, :] = -np.inf
            R[:, -1, :, :, :, :, :] = -np.inf

            R[:, -1, -1, :, :, :, :] = -np.inf
            R[:, -1, :, -1, :, :, :] = -np.inf
            R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf
            R[:, -1, :, :, :, -1, :] = -np.inf
            R[:, -1, :, :, :, :, -1] = -np.inf
            R[:, :, -1, -1, :, :, :] = -np.inf
            R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf
            R[:, :, -1, :, :, -1, :] = -np.inf
            R[:, :, -1, :, :, :, -1] = -np.inf
            R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf
            R[:, :, :, -1, :, -1, :] = -np.inf
            R[:, :, :, -1, :, :, -1] = -np.inf
            R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, :, :, :, :, -1, -1] = -np.inf

            R[:, -1, -1, -1, :, :, :] = -np.inf
            R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf
            R[:, -1, -1, :, :, -1, :] = -np.inf
            R[:, -1, -1, :, :, :, -1] = -np.inf
            R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf
            R[:, -1, :, -1, :, -1, :] = -np.inf
            R[:, -1, :, -1, :, :, -1] = -np.inf
            R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, -1, :, :, :, -1, -1] = -np.inf
            R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf
            R[:, :, -1, -1, :, -1, :] = -np.inf
            R[:, :, -1, -1, :, :, -1] = -np.inf
            R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, :, -1, :, :, -1, -1] = -np.inf
            R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, :, :, -1, :, -1, -1] = -np.inf
            R[:, :, :, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -np.inf

            R[:, :, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -np.inf
            R[:, :, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -np.inf
            R[:, :, -1, -1, :, -1, -1] = -np.inf
            R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, -1, :, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -np.inf
            R[:, -1, :, -1, :, -1, -1] = -np.inf
            R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, -1, -1, :, :, -1, -1] = -np.inf
            R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, -1, -1, -1, :, :, -1] = -np.inf
            R[:, -1, -1, -1, :, -1, :] = -np.inf
            R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, :] = -np.inf

            R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, :] = -np.inf
            R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, :, -1] = -np.inf
            R[:, -1, -1, -1, :, -1, -1] = -np.inf
            R[:, -1, -1, :, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -np.inf
            R[:, -1, :, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -np.inf
            R[:, :, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = -np.inf

            R[:, -1, -1, -1, l2 - l1 + int(bandwidth.item())-1, -1, -1] = R[:, -2, -2, -2, l2 - l1 + int(bandwidth.item())-2, -2, -2]


    else:
        E[:, -1, -1, -1, -1, -1, -1] = 1

        R[:, :, :, :, :, :, -1] = -np.inf
        R[:, :, :, :, :, -1, :] = -np.inf
        R[:, :, :, :, -1, :, :] = -np.inf
        R[:, :, :, -1, :, :, :] = -np.inf
        R[:, :, -1, :, :, :, :] = -np.inf
        R[:, -1, :, :, :, :, :] = -np.inf

        R[:, -1, -1, :, :, :, :] = -np.inf
        R[:, -1, :, -1, :, :, :] = -np.inf
        R[:, -1, :, :, -1, :, :] = -np.inf
        R[:, -1, :, :, :, -1, :] = -np.inf
        R[:, -1, :, :, :, :, -1] = -np.inf
        R[:, :, -1, -1, :, :, :] = -np.inf
        R[:, :, -1, :, -1, :, :] = -np.inf
        R[:, :, -1, :, :, -1, :] = -np.inf
        R[:, :, -1, :, :, :, -1] = -np.inf
        R[:, :, :, -1, -1, :, :] = -np.inf
        R[:, :, :, -1, :, -1, :] = -np.inf
        R[:, :, :, -1, :, :, -1] = -np.inf
        R[:, :, :, :, -1, -1, :] = -np.inf
        R[:, :, :, :, -1, :, -1] = -np.inf
        R[:, :, :, :, :, -1, -1] = -np.inf

        R[:, -1, -1, -1, :, :, :] = -np.inf
        R[:, -1, -1, :, -1, :, :] = -np.inf
        R[:, -1, -1, :, :, -1, :] = -np.inf
        R[:, -1, -1, :, :, :, -1] = -np.inf
        R[:, -1, :, -1, -1, :, :] = -np.inf
        R[:, -1, :, -1, :, -1, :] = -np.inf
        R[:, -1, :, -1, :, :, -1] = -np.inf
        R[:, -1, :, :, -1, -1, :] = -np.inf
        R[:, -1, :, :, -1, :, -1] = -np.inf
        R[:, -1, :, :, :, -1, -1] = -np.inf
        R[:, :, -1, -1, -1, :, :] = -np.inf
        R[:, :, -1, -1, :, -1, :] = -np.inf
        R[:, :, -1, -1, :, :, -1] = -np.inf
        R[:, :, -1, :, -1, -1, :] = -np.inf
        R[:, :, -1, :, -1, :, -1] = -np.inf
        R[:, :, -1, :, :, -1, -1] = -np.inf
        R[:, :, :, -1, -1, -1, :] = -np.inf
        R[:, :, :, -1, -1, :, -1] = -np.inf
        R[:, :, :, -1, :, -1, -1] = -np.inf
        R[:, :, :, :, -1, -1, -1] = -np.inf

        R[:, :, :, -1, -1, -1, -1] = -np.inf
        R[:, :, -1, :, -1, -1, -1] = -np.inf
        R[:, :, -1, -1, :, -1, -1] = -np.inf
        R[:, :, -1, -1, -1, :, -1] = -np.inf
        R[:, :, -1, -1, -1, -1, :] = -np.inf
        R[:, -1, :, :, -1, -1, -1] = -np.inf
        R[:, -1, :, -1, :, -1, -1] = -np.inf
        R[:, -1, :, -1, -1, :, -1] = -np.inf
        R[:, -1, :, -1, -1, -1, :] = -np.inf
        R[:, -1, -1, :, :, -1, -1] = -np.inf
        R[:, -1, -1, :, -1, :, -1] = -np.inf
        R[:, -1, -1, :, -1, -1, :] = -np.inf
        R[:, -1, -1, -1, :, :, -1] = -np.inf
        R[:, -1, -1, -1, :, -1, :] = -np.inf
        R[:, -1, -1, -1, -1, :, :] = -np.inf

        R[:, -1, -1, -1, -1, -1, :] = -np.inf
        R[:, -1, -1, -1, -1, :, -1] = -np.inf
        R[:, -1, -1, -1, :, -1, -1] = -np.inf
        R[:, -1, -1, :, -1, -1, -1] = -np.inf
        R[:, -1, :, -1, -1, -1, -1] = -np.inf
        R[:, :, -1, -1, -1, -1, -1] = -np.inf

        R[:, -1, -1, -1, -1, -1, -1] = R[:, -2, -2, -2, -2, -2, -2]


    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                for m in range(V_Q, 0, -1):
                    for n in range(V_P, 0, -1):
                        for v in range(V_M, 0, -1):
                            for u in range(V_N, 0, -1):

                                if np.isinf(R[k, i, j, n, m, u, v]):
                                    R[k, i, j, n, m, u, v] = -np.inf

                                # Check the pruning condition
                                # if 0 < bandwidth < np.abs(i - j) or 0 < bandwidth < np.abs(v - m):
                                if 0 < bandwidth < np.abs(v - m):
                                # if 0 < bandwidth < np.abs(i - j) or 0 < bandwidth < np.abs(n - m) or 0 < bandwidth < np.abs(u-v):
                                # if 0 < bandwidth < np.abs(i - j) or 0 < bandwidth < np.abs(j - n) or 0 < bandwidth < np.abs(n - m) or 0 < bandwidth < np.abs(m - u) or 0 < bandwidth < np.abs(u - v):
                                    continue

                                e00 = np.exp((R[k, i, j, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m, u + 1, v]) / gamma)
                                e01 = np.exp((R[k, i, j, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m, u, v + 1]) / gamma)
                                e02 = np.exp(
                                    (R[k, i, j, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n, m, u + 1, v + 1]) / gamma)

                                e03 = np.exp(
                                    (R[k, i + 1, j, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n, m, u + 1, v]) / gamma)
                                e04 = np.exp(
                                    (R[k, i + 1, j, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n, m, u, v + 1]) / gamma)
                                e05 = np.exp((R[k, i + 1, j, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n, m, u + 1, v + 1]) / gamma)

                                e06 = np.exp(
                                    (R[k, i, j + 1, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n, m, u + 1, v]) / gamma)
                                e07 = np.exp(
                                    (R[k, i, j + 1, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n, m, u, v + 1]) / gamma)
                                e08 = np.exp((R[k, i, j + 1, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n, m, u + 1, v + 1]) / gamma)

                                e09 = np.exp((R[k, i + 1, j + 1, n, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m, u + 1, v]) / gamma)
                                e10 = np.exp((R[k, i + 1, j + 1, n, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m, u, v + 1]) / gamma)
                                e11 = np.exp((R[k, i + 1, j + 1, n, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m, u + 1, v + 1]) / gamma)

                                e12 = np.exp((R[k, i + 1, j, n, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n, m, u, v]) / gamma)
                                e13 = np.exp((R[k, i, j + 1, n, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n, m, u, v]) / gamma)
                                e14 = np.exp(
                                    (R[k, i + 1, j + 1, n, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n, m, u, v]) / gamma)

                                e15 = np.exp((R[k, i, j, n + 1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n + 1, m, u + 1, v]) / gamma)
                                e16 = np.exp((R[k, i, j, n + 1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n + 1, m, u, v + 1]) / gamma)
                                e17 = np.exp(
                                    (R[k, i, j, n + 1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n + 1, m, u + 1, v + 1]) / gamma)

                                e18 = np.exp(
                                    (R[k, i + 1, j, n + 1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n + 1, m, u + 1, v]) / gamma)
                                e19 = np.exp(
                                    (R[k, i + 1, j, n + 1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n + 1, m, u, v + 1]) / gamma)
                                e20 = np.exp((R[k, i + 1, j, n + 1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n + 1, m, u + 1, v + 1]) / gamma)

                                e21 = np.exp(
                                    (R[k, i, j + 1, n + 1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n + 1, m, u + 1, v]) / gamma)
                                e22 = np.exp(
                                    (R[k, i, j + 1, n + 1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n + 1, m, u, v + 1]) / gamma)
                                e23 = np.exp((R[k, i, j + 1, n + 1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n + 1, m, u + 1, v + 1]) / gamma)

                                e24 = np.exp((R[k, i + 1, j + 1, n + 1, m, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n + 1, m, u + 1, v]) / gamma)
                                e25 = np.exp((R[k, i + 1, j + 1, n + 1, m, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n + 1, m, u, v + 1]) / gamma)
                                e26 = np.exp((R[k, i + 1, j + 1, n + 1, m, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n + 1, m, u + 1, v + 1]) / gamma)

                                e27 = np.exp((R[k, i + 1, j, n + 1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n + 1, m, u, v]) / gamma)
                                e28 = np.exp((R[k, i, j + 1, n + 1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n + 1, m, u, v]) / gamma)
                                e29 = np.exp(
                                    (R[k, i + 1, j + 1, n + 1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n + 1, m, u, v]) / gamma)

                                e30 = np.exp((R[k, i, j, n, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m + 1, u + 1, v]) / gamma)
                                e31 = np.exp((R[k, i, j, n, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m + 1, u, v + 1]) / gamma)
                                e32 = np.exp(
                                    (R[k, i, j, n, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n, m + 1, u + 1, v + 1]) / gamma)

                                e33 = np.exp(
                                    (R[k, i + 1, j, n, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n, m + 1, u + 1, v]) / gamma)
                                e34 = np.exp(
                                    (R[k, i + 1, j, n, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n, m + 1, u, v + 1]) / gamma)
                                e35 = np.exp((R[k, i + 1, j, n, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n, m + 1, u + 1, v + 1]) / gamma)

                                e36 = np.exp(
                                    (R[k, i, j + 1, n, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n, m + 1, u + 1, v]) / gamma)
                                e37 = np.exp(
                                    (R[k, i, j + 1, n, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n, m + 1, u, v + 1]) / gamma)
                                e38 = np.exp((R[k, i, j + 1, n, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n, m + 1, u + 1, v + 1]) / gamma)

                                e39 = np.exp((R[k, i + 1, j + 1, n, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m + 1, u + 1, v]) / gamma)
                                e40 = np.exp((R[k, i + 1, j + 1, n, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m + 1, u, v + 1]) / gamma)
                                e41 = np.exp((R[k, i + 1, j + 1, n, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n, m + 1, u + 1, v + 1]) / gamma)

                                e42 = np.exp((R[k, i + 1, j, n, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n, m + 1, u, v]) / gamma)
                                e43 = np.exp((R[k, i, j + 1, n, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n, m + 1, u, v]) / gamma)
                                e44 = np.exp(
                                    (R[k, i + 1, j + 1, n, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n, m + 1, u, v]) / gamma)

                                e45 = np.exp((R[k, i, j, n + 1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n + 1, m + 1, u + 1, v]) / gamma)
                                e46 = np.exp((R[k, i, j, n + 1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n + 1, m + 1, u, v + 1]) / gamma)
                                e47 = np.exp(
                                    (R[k, i, j, n + 1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n + 1, m + 1, u + 1, v + 1]) / gamma)

                                e48 = np.exp(
                                    (R[k, i + 1, j, n + 1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n + 1, m + 1, u + 1, v]) / gamma)
                                e49 = np.exp(
                                    (R[k, i + 1, j, n + 1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j, n + 1, m + 1, u, v + 1]) / gamma)
                                e50 = np.exp((R[k, i + 1, j, n + 1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n + 1, m + 1, u + 1, v + 1]) / gamma)

                                e51 = np.exp(
                                    (R[k, i, j + 1, n + 1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n + 1, m + 1, u + 1, v]) / gamma)
                                e52 = np.exp(
                                    (R[k, i, j + 1, n + 1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j + 1, n + 1, m + 1, u, v + 1]) / gamma)
                                e53 = np.exp((R[k, i, j + 1, n + 1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n + 1, m + 1, u + 1, v + 1]) / gamma)

                                e54 = np.exp((R[k, i + 1, j + 1, n + 1, m + 1, u + 1, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n + 1, m + 1, u + 1, v]) / gamma)
                                e55 = np.exp((R[k, i + 1, j + 1, n + 1, m + 1, u, v + 1] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j + 1, n + 1, m + 1, u, v + 1]) / gamma)
                                e56 = np.exp(
                                    (R[k, i + 1, j + 1, n + 1, m + 1, u + 1, v + 1] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n + 1, m + 1, u + 1, v + 1]) / gamma)

                                e57 = np.exp((R[k, i + 1, j, n + 1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i + 1, j, n + 1, m + 1, u, v]) / gamma)
                                e58 = np.exp((R[k, i, j + 1, n + 1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j + 1, n + 1, m + 1, u, v]) / gamma)
                                e59 = np.exp(
                                    (R[k, i + 1, j + 1, n + 1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i + 1, j + 1, n + 1, m + 1, u, v]) / gamma)

                                e60 = np.exp((R[k, i, j, n + 1, m, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n + 1, m, u, v]) / gamma)
                                e61 = np.exp((R[k, i, j, n, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                    k, i, j, n, m + 1, u, v]) / gamma)
                                e62 = np.exp(
                                    (R[k, i, j, n + 1, m + 1, u, v] - R[k, i, j, n, m, u, v] - D[
                                        k, i, j, n + 1, m + 1, u, v]) / gamma)

                                E[k, i, j, n, m, u, v] = E[k, i, j, n, m, u + 1, v] * e00 + E[
                                    k, i, j, n, m, u, v + 1] * e01 + E[
                                                             k, i, j, n, m, u + 1, v + 1] * e02 + E[
                                                             k, i + 1, j, n, m, u + 1, v] * e03 + E[
                                                             k, i + 1, j, n, m, u, v + 1] * e04 + \
                                                         E[k, i + 1, j, n, m, u + 1, v + 1] * e05 + E[
                                                             k, i, j + 1, n, m, u + 1, v] * e06 + E[
                                                             k, i, j + 1, n, m, u, v + 1] * e07 + E[
                                                             k, i, j + 1, n, m, u + 1, v + 1] * e08 + E[
                                                             k, i + 1, j + 1, n, m, u + 1, v] * e09 + E[
                                                             k, i + 1, j + 1, n, m, u, v + 1] * e10 + \
                                                         E[k, i + 1, j + 1, n, m, u + 1, v + 1] * e11 + E[
                                                             k, i + 1, j, n, m, u, v] * e12 + E[
                                                             k, i, j + 1, n, m, u, v] * e13 + E[
                                                             k, i + 1, j + 1, n, m, u, v] * e14 + E[
                                                             k, i, j, n + 1, m, u + 1, v] * e15 + E[
                                                             k, i, j, n + 1, m, u, v + 1] * e16 + E[
                                                             k, i, j, n + 1, m, u + 1, v + 1] * e17 + E[
                                                             k, i + 1, j, n + 1, m, u + 1, v] * e18 + E[
                                                             k, i + 1, j, n + 1, m, u, v + 1] * e19 + \
                                                         E[k, i + 1, j, n + 1, m, u + 1, v + 1] * e20 + E[
                                                             k, i, j + 1, n + 1, m, u + 1, v] * e21 + E[
                                                             k, i, j + 1, n + 1, m, u, v + 1] * e22 + E[
                                                             k, i, j + 1, n + 1, m, u + 1, v + 1] * e23 + E[
                                                             k, i + 1, j + 1, n + 1, m, u + 1, v] * e24 + E[
                                                             k, i + 1, j + 1, n + 1, m, u, v + 1] * e25 + \
                                                         E[k, i + 1, j + 1, n + 1, m, u + 1, v + 1] * e26 + E[
                                                             k, i + 1, j, n + 1, m, u, v] * e27 + E[
                                                             k, i, j + 1, n + 1, m, u, v] * e28 + E[
                                                             k, i + 1, j + 1, n + 1, m, u, v] * e29 + E[
                                                             k, i, j, n, m + 1, u + 1, v] * e30 + E[
                                                             k, i, j, n, m + 1, u, v + 1] * e31 + E[
                                                             k, i, j, n, m + 1, u + 1, v + 1] * e32 + E[
                                                             k, i + 1, j, n, m + 1, u + 1, v] * e33 + E[
                                                             k, i + 1, j, n, m + 1, u, v + 1] * e34 + \
                                                         E[k, i + 1, j, n, m + 1, u + 1, v + 1] * e35 + E[
                                                             k, i, j + 1, n, m + 1, u + 1, v] * e36 + E[
                                                             k, i, j + 1, n, m + 1, u, v + 1] * e37 + E[
                                                             k, i, j + 1, n, m + 1, u + 1, v + 1] * e38 + E[
                                                             k, i + 1, j + 1, n, m + 1, u + 1, v] * e39 + E[
                                                             k, i + 1, j + 1, n, m + 1, u, v + 1] * e40 + \
                                                         E[k, i + 1, j + 1, n, m + 1, u + 1, v + 1] * e41 + E[
                                                             k, i + 1, j, n, m + 1, u, v] * e42 + E[
                                                             k, i, j + 1, n, m + 1, u, v] * e43 + E[
                                                             k, i + 1, j + 1, n, m + 1, u, v] * e44 + E[
                                                             k, i, j, n + 1, m + 1, u + 1, v] * e45 + E[
                                                             k, i, j, n + 1, m + 1, u, v + 1] * e46 + E[
                                                             k, i, j, n + 1, m + 1, u + 1, v + 1] * e47 + E[
                                                             k, i + 1, j, n + 1, m + 1, u + 1, v] * e48 + E[
                                                             k, i + 1, j, n + 1, m + 1, u, v + 1] * e49 + \
                                                         E[k, i + 1, j, n + 1, m + 1, u + 1, v + 1] * e50 + E[
                                                             k, i, j + 1, n + 1, m + 1, u + 1, v] * e51 + E[
                                                             k, i, j + 1, n + 1, m + 1, u, v + 1] * e52 + E[
                                                             k, i, j + 1, n + 1, m + 1, u + 1, v + 1] * e53 + E[
                                                             k, i + 1, j + 1, n + 1, m + 1, u + 1, v] * e54 + E[
                                                             k, i + 1, j + 1, n + 1, m + 1, u, v + 1] * e55 + \
                                                         E[k, i + 1, j + 1, n + 1, m + 1, u + 1, v + 1] * e56 + E[
                                                             k, i + 1, j, n + 1, m + 1, u, v] * e57 + E[
                                                             k, i, j + 1, n + 1, m + 1, u, v] * e58 + E[
                                                             k, i + 1, j + 1, n + 1, m + 1, u, v] * e59 + E[
                                                             k, i, j, n + 1, m, u, v] * e60 + E[
                                                             k, i, j, n, m + 1, u, v] * e61 + E[
                                                             k, i, j, n + 1, m + 1, u, v] * e62

    return E[:, 1:N + 1, 1:M + 1, 1:V_P + 1, 1:V_Q + 1, 1:V_N + 1, 1:V_M + 1]


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)

        l1 = D.shape[4]
        l2 = D.shape[6]
        bw = int(bandwidth.item())

        if l1 < l2:
            if bw >= abs(l1 - l2) or bw == 0:
                V = R[:, -2, -2, -2, -2, -2, -2]
            else:
                V = R[:, -2, -2, -2, -2, -2, l1 - l2 + int(bandwidth.item()) - 2]
        elif l1 > l2:
            if bw >= abs(l1 - l2) or bw == 0:
                V = R[:, -2, -2, -2, -2, -2, -2]
            else:
                V = R[:, -2, -2, -2, l2 - l1 + int(bandwidth.item()) - 2, -2, -2]

        else:
            V = R[:, -2, -2, -2, -2, -2, -2]

        return V # R[:, -2, -2, -2, -2, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)

        return grad_output.view(-1, 1, 1, 1, 1, 1, 1).expand_as(E) * E, None, None


class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        # vx, vy is added by Lei for the viewpoints
        # print(x.shape, y.shape)
        bx, lx, v1x, v2x, dx = x.shape
        by, ly, v1y, v2y, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions
        # assert vx == vy  # Equal viewpoints --- added by Lei

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
            print(
                "SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
            use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    def _calc_distance_matrix(self, x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        sigma = 0.5

        n = x.size(1)
        m = y.size(1)

        v1x = x.size(2)
        v2x = x.size(3)

        v1y = y.size(2)
        v2y = y.size(3)

        d = x.size(4)

        x = x.unsqueeze(2).expand(-1, n, m, v1x, v2x, d)
        x = x.unsqueeze(4).expand(-1, n, m, v1x, v1y, v2x, d)
        x = x.unsqueeze(6).expand(-1, n, m, v1x, v1y, v2x, v2y, d)

        # print(x.shape, ' == x')

        y = y.unsqueeze(1).expand(-1, n, m, v1y, v2y, d)
        y = y.unsqueeze(3).expand(-1, n, m, v1x, v1y, v2y, d)
        y = y.unsqueeze(5).expand(-1, n, m, v1x, v1y, v2x, v2y, d)

        # print(y.shape, ' == y')

        # euclidean distance
        # print(torch.pow(x - y, 2).sum(7).shape, '--------')
        # rbf distance

        return (2 - 2 * torch.exp(-1 * sigma * torch.pow(x - y, 2))).sum(7)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x view x dims
        :param Y: The other batch of examples, batch_size x seq_len x view x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            D_xy = self._calc_distance_matrix(X, Y)
            D_xx = self._calc_distance_matrix(X, X)
            D_yy = self._calc_distance_matrix(Y, Y)

            out_xy = func_dtw(D_xy, self.gamma, self.bandwidth)
            out_xx = func_dtw(D_xx, self.gamma, self.bandwidth)
            out_yy = func_dtw(D_yy, self.gamma, self.bandwidth)

            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self._calc_distance_matrix(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)