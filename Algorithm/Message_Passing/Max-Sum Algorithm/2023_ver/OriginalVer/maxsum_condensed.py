import numpy as np
import matplotlib.pyplot as plt
import time

INF = 10**60  # infinity
DAMP = 0.0  # between 0 and 1. 0 for fastest change.
N_NODE = 5  # number of nodes per group
N_ITER = N_NODE*10
bLogSumExp = False
np.set_printoptions(precision=5)


def main():
    # rng = np.random.default_rng(0)
    w = np.random.uniform(0, 1, (N_NODE, N_NODE))
    print(f"weights:\n{w}")
    alpha = np.zeros((N_NODE, N_NODE))
    rho = np.zeros((N_NODE, N_NODE))
    tic = time.time()
    for i in range(N_ITER):
        alpha = update_alpha(alpha, rho, w, bLogSumExp)
        rho = update_rho(alpha, rho, w, bLogSumExp)

    D = conclude_update(alpha, rho)
    is_valid = check_validity(D)
    toc = time.time()
    print(f"matching time: {(toc - tic)/1000}ms")
    if is_valid:
        print("Sucessful bipartite matching")
        show_match(w, D)
    else:
        print("Pairing unsucessful.")


def log_sum_exp(input_array):
    return np.log(np.sum(np.exp(input_array)))


def update_alpha(alpha, rho, w, bLogSumExp):
    old = alpha
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = rho + w/2
            if bLogSumExp:
                tmp_ith_row_except_ij = np.delete(tmp[i, :], j)
                new[i, j] = w[i, j]/2 - log_sum_exp(tmp_ith_row_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = w[i, j]/2 -max(tmp[i, :])
    return new*(1-DAMP) + old*(DAMP)


def update_rho(alpha, rho, w, bLogSumExp):
    old = rho
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = alpha + w/2
            if bLogSumExp:
                tmp_jth_col_except_ij = np.delete(tmp[:, j], i)
                new[i, j] = w[i, j]/2 - log_sum_exp(tmp_jth_col_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = w[i, j]/2 -max(tmp[:, j])
    return new*(1-DAMP) + old*(DAMP)


def conclude_update(alpha, rho):
    D_row = alpha + rho
    D_col = alpha + rho
    for row in range(N_NODE):
        idx_row_max = np.argmax(D_row[row, :])
        D_row[row, :] = 0
        D_row[row, idx_row_max] = 1
    for col in range(N_NODE):
        idx_col_max = np.argmax(D_col[:, col])
        D_col[:, col] = 0
        D_col[idx_col_max, col] = 1
    D = np.multiply(D_row, D_col)
    return D

def conclude_update_nn(alpha, rho):
    D_row = np.copy(alpha + rho)
    D_col = np.copy(alpha + rho)
    temp = np.copy(alpha + rho)
    for row in range(N_NODE):
        idx_row_max = np.argmax(D_row[row, :])
        D_row[row,:] = 0
        D_row[row, idx_row_max] = 1
    for col in range(N_NODE):
        idx_col_max = np.argmax(D_col[:, col])
        D_col[:, col] = 0
        D_col[idx_col_max, col] = 1
    D = np.multiply(D_row, D_col)
    a = np.arange(N_NODE)
    while np.sum(D) != N_NODE:
        try:
            D_nonzero_idx = np.nonzero(D)
            b1 = np.setdiff1d(a, D_nonzero_idx[0])
            b2 = np.setdiff1d(a, D_nonzero_idx[1])
            D_remain = temp[b1, b2]
            indices = np.where(temp == np.max(D_remain))
            D[indices] = 1
        except:
            D = D
    return D


def check_validity(D):
    rowsum = np.sum(D, axis=1)
    colsum = np.sum(D, axis=0)
    if np.all(rowsum==1) and np.all(colsum==1):
        return 1
    else:
        return 0


def show_match(w, D):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Preferences')
    plt.imshow(w, origin='lower', cmap='gray')
    plt.colorbar(orientation='vertical')
    x = np.linspace(0, N_NODE-1, N_NODE)
    y = np.argmax(D, axis=1)
    plt.scatter(y, x,
                marker='d',
                color='red')
    plt.show()


if __name__=="__main__":
    main()
    dataset_alpha_rho_star = np.concatenate((alpha_star,
                                             rho_star),
                                            axis=1)
