import torch
import time
import torch.nn as nn
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

from utils.tensorops import contains_non_float_values, get_gmm_lfbgf
from external_util.softdtw import SoftDTW
from external_util.ot_pytorch import sink
from utils.plotter import plot_gmm_informative

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def TCC_loss(sequences, alignment_variance=0.1, softmax_temp=1):
    normalizer = nn.Softmax(dim=0).to(device)
    # loss term 2 is TCC loss
    tcc_count = 0
    all_tcc_losses = None
    ###################################################
    ### iterate through primary sequences
    for i, primary in enumerate(sequences):
        primary = primary.to(device)
        idx_range = torch.arange(0, primary.shape[0], dtype=torch.float).to(device).detach()
        ###################################################
        ### iterate through secondary sequences
        for j, secondary in enumerate(sequences):
            secondary = secondary.to(device)
            #################################
            # get intermittent variables
            ALPHA = normalizer(-torch.cdist(primary, secondary, p=2).T / softmax_temp).T
            SNN = ALPHA @ secondary
            BETA = normalizer(-torch.cdist(SNN, primary, p=2).T / softmax_temp).T
            mus = BETA @ idx_range
            variances = BETA @ torch.square(idx_range - mus)
            
            #################################
            # get TCC loss
            loss_terms = torch.divide(torch.square(idx_range - mus), variances) + alignment_variance * torch.log(torch.sqrt(variances) + 1e-20)
            
            if all_tcc_losses is None:
                all_tcc_losses = torch.sum(loss_terms) 
            else: 
                all_tcc_losses += torch.sum(loss_terms)
            tcc_count += 1
    return all_tcc_losses / tcc_count



def IntraContrast_loss(sequence, idx_range, window=7, margin=2):
    N = sequence.shape[0]
    DX = torch.cdist(sequence, sequence, p=2)
    margin_identity = create_margin_identity_matrix(N, window).to(device)
    margin_nidentity = 1 - margin_identity
    relu = nn.ReLU()

    ones = torch.ones((N, N)).to(device)
    idx_dup = ones * idx_range
    W_matrix = (torch.square(idx_dup - idx_dup.T) + 1)
    outside_window = margin_nidentity * W_matrix * relu(margin - DX)
    inside_window = margin_identity * W_matrix * margin
    return (outside_window + inside_window).sum()



def LAV_loss(sequences, min_temp=.1, cr_coefficient=0.01):
    loss_term = None
    softdtw = SoftDTW(gamma=min_temp)
    for i, v1 in enumerate(sequences):
        N = v1.shape[0]
        idx_range_N = torch.arange(0, N, dtype=torch.float).to(device)
        ###################################################
        ### iterate through secondary sequences
        for j, v2 in enumerate(sequences):
            if i == j: # skip if same sequence
                continue
            M = v2.shape[0]
            idx_range_M = torch.arange(0, M, dtype=torch.float).to(device)
            soft_dtw_term = softdtw(v1, v2)
            x_cr_term = IntraContrast_loss(v1, idx_range_N, window=15)
            y_cr_term = IntraContrast_loss(v2, idx_range_M, window=15)
            if loss_term is None:
                loss_term = soft_dtw_term + cr_coefficient * (x_cr_term + y_cr_term)
            else:
                loss_term += soft_dtw_term + cr_coefficient * (x_cr_term + y_cr_term)
    return loss_term




def VAVA_loss(sequences, maxIter=20, lambda1=1.0,lambda2=0.1, gamma=.5, zeta=0.5, delta=0.6,global_step=None):
    loss_term = None
    phi = max(min(0.999**(global_step + 1),0.999),0.001)
    for i, v1 in enumerate(sequences):
        N = v1.shape[0]

        max_comparisons = min(40, N)
        indices_to_check = torch.randperm(N)[:max_comparisons].to(device).sort().values
        ind_bool = torch.zeros(N, dtype=torch.bool).to(device)
        ind_bool[indices_to_check] = True
        idx_range_N = torch.arange(0, N+1, dtype=torch.float).to(device)
        ###################################################
        ### iterate through secondary sequences
        for j, v2 in enumerate(sequences):
            if i == j: # skip if same sequence
                continue
            M = v2.shape[0]
            idx_range_M = torch.arange(0, M+1, dtype=torch.float).to(device)
            lambda1 = lambda1*(N+M)
            lambda2 = lambda2*(N*M)/4.0
            
            # OT
            D = torch.cdist(v1, v2, p=2)
            D = torch.cat((torch.ones(M).to(device).detach()[None, :] * zeta, D), dim=0)
            D = torch.cat((torch.ones(N+1).to(device).detach()[:, None] * zeta, D), dim=1)
            dist, T = sink(D, reg=N, cuda=True, numItermax=maxIter)
            
            lc = get_diag_consistency_matrix(N+1, M+1, idx_range_N, idx_range_M)
            lo = get_diag_optimality_matrix(T, N+1, M+1, idx_range_N, idx_range_M)     
            Pc = torch.exp(-(torch.square(lc)) / (2 * delta**2)) / delta * np.sqrt(2 * np.pi)
            Po = torch.exp(-(torch.square(lo)) / (2 * delta**2)) / delta * np.sqrt(2 * np.pi)
            P = phi * Pc + (1-phi) * Po

            # I(T)
            consistency_coefficients = get_consistency_matrix(N+1, M+1, idx_range_N, idx_range_M)
            optimality_coefficients = get_denom_diag_opt_matrix(T, N+1, M+1, idx_range_N, idx_range_M)
            Ic_T = torch.sum(T * consistency_coefficients)
            Io_T = torch.sum(T * optimality_coefficients)
            I_T = phi * Ic_T + (1-phi) *  Io_T

            # KL(T || P)
            KLT = T + 1e-5
            KLP= P + 1e-5
            KL_T_P = torch.sum(KLT * torch.log(torch.divide(KLT, KLP)))

            # C(X)
            C_X = IntraContrast_loss(v1, idx_range_N[:-1], window=15)

            # C(Y)
            C_Y = IntraContrast_loss(v2, idx_range_M[:-1], window=15)

            # C(X, Y)
            A = get_A(T, N+1, M+1)
            Abar = get_Abar(T, N+1, M+1)
            C_XY = torch.sum(A * D - Abar * D)
            vava_loss = dist - lambda1 * I_T + lambda2 * KL_T_P
            cr_loss = C_X + C_Y + C_XY

            if None in [loss_term]:
                loss_term = vava_loss + gamma * cr_loss
            else: 
                loss_term += vava_loss + gamma * cr_loss
    return loss_term



def GTCC_loss(
        sequences,
        n_components=1,
        gamma=1,
        delta=1,
        dropouts=None,
        softmax_temp=1,
        alignment_variance=0,
        max_gmm_iters=8,
        epoch=0
    ):
    assert .05 <= delta <= 1
    tiny_number = 0
    all_tcc_losses = None
    drop_min = gamma**(epoch + 1)
    ###################################################
    ### iterate through primary sequences
    for i, primary in enumerate(sequences):
        primary = primary.to(device)
        N = primary.shape[0]
        margin = round(N * delta)

        max_comparisons = 20
        indices_to_check = torch.randperm(N)[:max_comparisons].to(device).sort().values
        ind_bool = torch.zeros(N, dtype=torch.bool).to(device)
        ind_bool[indices_to_check] = True
        del indices_to_check
        idx_range = torch.arange(0, N, dtype=torch.float).to(device).detach()
        indbool_idx_range = idx_range[ind_bool]
        margin_identity = create_stochastic_margin_identity_matrix(N, margin).to(device)[ind_bool]

        ###################################################
        ### iterate through secondary sequences
        for j, secondary in enumerate(sequences):
            if i == j: # skip if same sequence
                continue
            M = secondary.shape[0]
            # get the drop vectors!
            if gamma < 1:
                BX = primary @ dropouts[j][:-1].squeeze() + dropouts[j][-1]
                BX = (BX - BX.mean()) / BX.std()
                BX = drop_min + (1-drop_min) * nn.Sigmoid()(BX)[ind_bool]
            
            secondary = secondary.to(device)
            cdist = torch.cdist(primary, secondary, p=2)[ind_bool]
            ALPHA_exp = torch.exp(-cdist / softmax_temp) + tiny_number
            ALPHA = (ALPHA_exp.T / (ALPHA_exp.sum(dim=1))).T
            
            gmm = torch.zeros((ALPHA.shape[0], n_components, M)).to(device)
            spread = torch.zeros((ALPHA.shape[0], n_components)).to(device)
            for u in range(ALPHA.shape[0]):
                start = time.time()
                g, s, _ = get_gmm_lfbgf(
                    ALPHA[u], n_components, max_iters=max_gmm_iters
                )
                g = g.to(device)
                s = s.to(device)

                gmm[u] = g
                spread[u] = s
                
            SNNs = (gmm @ secondary)
            prim_expanded = primary.unsqueeze(0)
            snn_cdist = torch.cdist(SNNs, prim_expanded, p=2)

            BETAs = (margin_identity * torch.exp(-snn_cdist / softmax_temp).permute(1,0,2)).permute(1, 0, 2) + tiny_number
            BETAs = (BETAs.permute(2, 0, 1) / (BETAs.sum(dim=2) + 1e-6)).permute(1, 2, 0)

            mus = (BETAs @ idx_range).unsqueeze(-1)
            
            variances = torch.sum(BETAs * torch.square(idx_range - mus), dim=2)

            index_margin_identity = idx_range * margin_identity
            
            spread = spread.to(device)
            for t in range(max_comparisons):
                idx_mask = index_margin_identity[t]
                set_of_mus = mus[t]
                set_of_vars = variances[t]
                this_spread = spread[t]
                idx_mask = idx_mask[margin_identity[t].bool()].unsqueeze(0)

                each_mu_error = torch.square(indbool_idx_range[t] - set_of_mus).squeeze()
                if alignment_variance > 0:
                    tcc = each_mu_error + alignment_variance * torch.log(torch.sqrt(set_of_vars) + 1e-20)
                else:
                    tcc = each_mu_error

                if contains_non_float_values(tcc):
                    print("contains_non_float_values(tcc)")
                    print(set_of_vars)
                    print(set_of_mus)
                    exit(1)
                align_loss = torch.inner(tcc, this_spread)
                if contains_non_float_values(1/align_loss):
                    print("contains_non_float_values(1/align_loss)")
                    print(align_loss)
                    exit(1)

                if gamma < 1:
                    tcc_loss_term = BX[t] * align_loss + (1-BX[t]) * (1 / align_loss)
                else:
                    tcc_loss_term = align_loss

                if None in [all_tcc_losses]:
                    all_tcc_losses = tcc_loss_term
                else: 
                    all_tcc_losses += tcc_loss_term

    return all_tcc_losses


#########################################
# below function is only for GTCC
#########################################
def create_stochastic_margin_identity_matrix(size, width):
    m = int(np.random.choice([0, 1]) * width)
    mm = 1 - torch.triu(torch.ones(size, size), diagonal=m+1).T
    wm = torch.triu(torch.ones(size, size), diagonal=width - m)
    insurance = torch.zeros((size, size))
    insurance[:width, :width] = 1
    insurance[-width:, -width:] = 1
    insurance = insurance.bool()
    return torch.logical_or(torch.logical_xor(wm, mm), insurance).float()


#########################################
# below function is only for IntraContrastive
#########################################
def create_margin_identity_matrix(size, margin):
    # Create two identity matrices of the specified size
    identity = torch.eye(size)
    mask = torch.triu(torch.ones(size, size), diagonal=0)
    mask2 = 1-torch.triu(torch.ones(size, size), diagonal=margin)
    # Combine the identity matrix and the shifted identity to create the margin identity
    margin_identity = margin_identity = mask2 * mask
    return (margin_identity + (margin_identity.T - identity)).float()


#########################################
# below functions are only for VAVA
#########################################
def get_diag_consistency_matrix(N, M, idx_range_row, idx_range_col):
    left_denom = idx_range_col / M
    right_denom = idx_range_row / N

    ones = torch.ones((N, M)).to(device)
    coeffs = torch.abs(ones * left_denom - (ones.T * right_denom).T) / np.sqrt(1/N**2 + 1/M**2)
    return coeffs


def get_diag_optimality_matrix(T, N, M, idx_range_row, idx_range_col):
    i_bests = T.argmax(dim=1)
    j_bests = T.argmax(dim=0)
    right_denom = torch.abs(idx_range_row - i_bests) / N
    left_denom = torch.abs(idx_range_col - j_bests) / N

    ones = torch.ones((N, M)).to(device)
    coeffs = (torch.abs(ones * left_denom) + torch.abs((ones.T * right_denom).T)) / (2 * np.sqrt(1/N**2 + 1/M**2))
    return coeffs


def get_denom_diag_opt_matrix(T, N, M, idx_range_row, idx_range_col):
    i_bests = T.argmax(dim=1)
    j_bests = T.argmax(dim=0)
    right_denom = torch.square((idx_range_row - i_bests) / N)
    left_denom = torch.square((idx_range_col - j_bests) / N)
    ones = torch.ones((N, M)).to(device)
    denoms = (1/2) * ((ones * left_denom) + (ones.T * right_denom).T) + 1
    coeffs = 1 / denoms
    return coeffs


def get_Abar(T, N, M):
    # Find the row maximums and column maximums
    row_min = T.min(dim=1, keepdim=True).values
    col_min = T.min(dim=0, keepdim=True).values
    # Create a boolean mask for the elements that meet the condition
    mask = (T == row_min) & (T == col_min)

    # Convert the boolean mask to integers (1s and 0s)
    result_matrix = mask.to(torch.int)
    return result_matrix


def get_A(T, N, M):
    # Find the row maximums and column maximums
    row_max = T.max(dim=1, keepdim=True).values
    col_max = T.max(dim=0, keepdim=True).values
    # Create a boolean mask for the elements that meet the condition
    mask = (T == row_max) & (T == col_max)

    # Convert the boolean mask to integers (1s and 0s)
    result_matrix = mask.to(torch.int)
    return result_matrix


def get_consistency_matrix(N, M, idx_range_row, idx_range_col):
    left_denom = idx_range_col / M
    right_denom = idx_range_row / N
    ones = torch.ones((N, M)).to(device)
    coeffs = 1 / (torch.square(ones * left_denom - (ones.T * right_denom).T) + 1)
    return coeffs
    