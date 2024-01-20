
import numpy as np
from scipy.signal import savgol_filter
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
    out = np.zeros((size, 10 ))

    for i in range(rank_start , rank_end , 1):

        aprx, rtn = low_rank_approximation( input2D , i)

        res = compute_stats( input2D , aprx )

        out [i-2,0] = i
        out [i-2,1] = res[0]  # fnorm
        out [i-2,2] = res[1]  #infinity_norm
        out [i-2,3] = res[2]  # abs_max_diff
        out [i-2,4] = res[3]  # stdev
        out [i-2,5] = rtn     # retention

    #compute diff
    for i in range(rank_start , rank_end-1 , 1):

        out [i-2,6] = out [i-1,1] - out [i-2,1]  # fnorm
        out [i-2,7] = out [i-1,2] - out [i-2,2]  #infinity_norm
        out [i-2,8] = out [i-1,3] - out [i-2,3]  # abs_max_diff
        out [i-2,9] = out [i-1,4] - out [i-2,4]  # stdev

    # -------------------------------------
    return out

#----------------------------------------------------------------
def compute_snr_eVec(matrix, last_rank):
    # Perform singular value decomposition (SVD)
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    out = np.zeros((last_rank,5))

    for i in range(out.shape[0]):
        out[i,0] = i+1
        out[i,1] = compute_SNR( U[:,i] , 5 , 2 )
        out[i,2] = compute_SNR( U[:,i] , 7 , 3 )
        out[i,3] = compute_SNR( U[:,i] , 9 , 4 )
        out[i,4] = compute_SNR( U[:,i] , 11 , 5 )

    return out

#----------------------------------------------------------------

def compute_SNR(vector1D, window, pol):

    # Calculate smoothed values with Savitzky-Golay method
    smoothed = savgol_filter( vector1D, window_length=window, polyorder=pol)

    noise = vector1D - smoothed

    sdev_signal = np.std(smoothed)
    sdev_noise = np.std(noise)

    return (sdev_signal / sdev_noise )

#----------------------------------------------------------------
#----------------------------------------------------------------
