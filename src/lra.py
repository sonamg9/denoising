
import numpy as np
#----------------------------------------------------------------
#----------------------------------------------------------------
def low_rank_approximation(matrix, rank):
    # Perform singular value decomposition (SVD)
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Truncate the singular values and vectors to the specified rank
    U_rank = U[:, :rank]
    S_rank = np.diag(S[:rank])
    Vt_rank = Vt[:rank, :]

    # Reconstruct the low-rank approximation
    approx_matrix = U_rank @ S_rank @ Vt_rank


    # compute retention
    # Calculate total energy (sum of squares of singular values)
    total_energy = np.sum(S**2)

    # Calculate retained energy for the given number of components
    retained_energy = np.sum(S[: rank ]**2)

    # Calculate percentage of energy retained
    percentage_retained = (retained_energy / total_energy)

    return approx_matrix , percentage_retained

#----------------------------------------------------------------
#----------------------------------------------------------------

def compute_stats(original_matrix, approx_matrix):

    # Calculate the absolute column-wise differences
    abs_diff = np.abs(original_matrix - approx_matrix)

    # Find the maximum absolute column sum
    infinity_norm = np.max(np.sum(abs_diff, axis=1))

    # size normalized Frobenius norm
    fnorm = np.linalg.norm(original_matrix - approx_matrix, 'fro') / abs_diff.size

    abs_max_diff = np.max( abs_diff )

    # standard deviation in the sum of all cols of difference matrix
    stdev = np.std(np.sum(abs_diff, axis=1))

    return fnorm, infinity_norm,  abs_max_diff , stdev

#----------------------------------------------------------------
#----------------------------------------------------------------

def evaluate_stats(inputx, input2D):

    rank_start = 2
    rank_end   = 20

    size = rank_end - rank_start
    print  ( '\n\t Total tests : ', size)
    out = np.zeros((size, 6 ))

    for i in range(rank_start , rank_end , 1):

        aprx, rtn = low_rank_approximation( input2D , i)

        res = compute_stats( input2D , aprx )

        out [i-2,0] = i
        out [i-2,1] = res[0]
        out [i-2,2] = res[1]
        out [i-2,3] = res[2]
        out [i-2,4] = res[3]
        out [i-2,5] = rtn


    return out

#----------------------------------------------------------------
#----------------------------------------------------------------
