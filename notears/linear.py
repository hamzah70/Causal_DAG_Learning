import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from sklearn.linear_model import LinearRegression
import pandas as pd


def notears_linear(X_full, X, y, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, g_tol=1e-8, target_node=0):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        # print(X.shape, W.shape)
        M = X_full @ W
        if loss_type == 'l2':
            R = X_full - M
            loss = 0.5 / X_full.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X_full.shape[0] * X_full.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X_full.shape[0] * (np.logaddexp(0, M) - X_full * M).sum()
            G_loss = 1.0 / X_full.shape[0] * X.T @ (sigmoid(M) - X_full)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X_full.shape[0] * (S - X_full * M).sum()
            G_loss = 1.0 / X_full.shape[0] * X.T @ (S - X_full)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _g(X_, W_, y, beta):
        W_ = np.delete(W_, target_node, axis=1)
        W_ = np.delete(W_, target_node, axis=0)

        d = W_.shape[0]
        z = X_ - X_ @ (W_.T)

        # T_ = np.linalg.inv((np.identity(d) - (W_)))

        # g_x = y - beta @ W_.T @ X_.T - beta @ z.T
        g_x = y - X_@W_@beta - z@beta
        g_x = g_x.sum()  # L1 norm
        # g_x = max(g_x) #INFINITY NORM
        # temp = [i**2 for i in g_x]
        # result_g_x = sum(temp)

        G_g = -(X_@beta).T @ np.sign(y - (X_@W_ + z) @ beta)  # L1 norm
        # G_g = -2*(X_@beta).T @ g_x                #L2 norm
        # print(G_g)

        return abs(g_x)*(10**-6), G_g*(10**-10)

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""

        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        g, G_g = _g(X_, W, y, beta)
        obj = loss + 0.5 * rho * h * h + 0.5  * g * \
            g + alpha * h + gamma * g + lambda1 * w.sum()
        # print(obj)
        g_term = (gamma + rho * g) * G_g
        # print(g_term, rho)
        G_smooth = G_loss + (rho * h + alpha) * G_h + g_term

        # G_smooth_2 = g_term

        g_obj = np.concatenate(
            (G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        # print(g_obj.shape)
        return obj, g_obj

    n, d = X_full.shape
    w_est, rho, alpha, gamma, h, g = np.zeros(
        2 * d * d), 1.0, 0.0, 0.0, np.inf, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]
    reg = LinearRegression().fit(X_, y)
    beta = reg.coef_
    # g_new = 0
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new, g_new = None, None, None
        while rho < rho_max:

            sol = sopt.minimize(
                _func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            g_new, _ = _g(X_, _adj(w_new), y, beta)
            if h_new > 0.25 * h:
                rho *= 10
            else:
                print("2")
                break
        w_est, h, g = w_new, h_new, g_new
        alpha += rho * h
        gamma += rho * g

        if h <= h_tol or rho >= rho_max or g <= g_tol:
            print(h, h_tol)
            print(rho, rho_max)
            print(g, g_tol)
            print("Breaking here")
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == '__main__':
    import utils
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    target_node = 10
    X_ = np.delete(X, target_node, axis=1)
    # W_ = np.delete(W_true, target_node, axis=1)
    # W_ = np.delete(W_, target_node, axis=0)
    y = X[:, target_node]
    # B_test = np.delete(B_true, target_node, axis=1)
    # B_test = np.delete(B_test, target_node, axis=0)
    B_test = B_true

    # print(X_.shape)
    # print(type(X_))

    ### METABRIC
    df = pd.read_csv('data.csv')
    X_ = df.loc[:, df.columns != 'Survival Time']
    X_ = X_.to_numpy()
    print(X_.shape)

    y = df['Survival Time']
    y = y.to_numpy()
    X = df.to_numpy()

    ### BOSTON DATASET
    # df = pd.read_csv('boston.csv')
    # X_ = df.loc[:, df.columns != 'MEDV']
    # X_ = X_.to_numpy()
    # print(X_.shape)

    # y = df['MEDV']
    # y = y.to_numpy()
    # X = df.to_numpy()

    W_est = notears_linear(X, X_, y, lambda1=0.1, loss_type='l2', target_node=10)
    assert utils.is_dag(W_est)
    print(W_est.shape)
    np.savetxt('W_est_metabric.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_test, W_est != 0)
    print(acc)
