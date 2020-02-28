import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import cvxopt as opt
from cvxopt import blas, solvers
solvers.options['show_progress'] = 0

"""
Several ways of realizing MVP methods and Efficient Frontiers
"""


# Simulation Method
# Main Contributor: Yukang Zhou
def run_simulation(ret, num):
    '''
    This method takes in a DataFrame of stock return and a int of simulation times.
    This method returns a tuple as a result containing volatilities, returns, sharpe ratios and weights of simulated portfolios.
    '''
    starttime = datetime.datetime.now()
    log_ret = ret
    cov = log_ret.cov()

    all_weights = []
    ret_arr = []
    vol_arr = []
    sharpe_arr = []

    for x in range(num):
        # Weights
        #weights = -1 + 2 * np.array( np.random.random(len(log_ret.columns)) )
        weights = np.array(np.random.random(len(log_ret.columns)))
        weights = weights / np.sum(weights)

        # Expected volatility
        P1 = np.dot(cov * 252, weights)
        sigma = np.dot(weights.T, P1)
        sigma = np.sqrt(sigma)
#        if sigma >= 0.19:
#            continue
        vol_arr.append(sigma)

        # Save weights
        all_weights.append(weights)

        # Expected return
        ret = np.sum(log_ret.mean() * weights * 252)
        ret_arr.append(ret)

        # Sharpe Ratio
        sharpe_arr.append(ret / sigma)

    endtime = datetime.datetime.now()
    simu_time = (endtime - starttime).seconds
    print("Simulation Time:", simu_time)
    return((vol_arr, ret_arr, sharpe_arr, all_weights))


def get_ScatterPlot(result):
    '''
    This method takes in a tuple result of run_simulation.
    This method plots the scatter plot of volatilities and returns colored by sharpe ratios.
    '''
    vol_arr = result[0]
    ret_arr = result[1]
    sharpe_arr = result[2]
    plt.figure(figsize=(12, 8))
    # The Scatter points are colored by its sharpe_ratio
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='summer', alpha=0.6)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')


def get_Frontier(result):
    '''
    This method takes in a tuple result of run_simulation.
    This method plots the effective frontier of simulated portfolios and returns the list of coordinates.
    '''

    vol_arr = result[0]
    ret_arr = result[1]

    # The effective frontier is the portofilos which has the minimum of
    # variance given its return
    x_front = []
    y_front = []
    df = pd.DataFrame({'Ret': ret_arr, 'Vol': vol_arr})
    df = df.sort_values(by="Ret")
    df.reset_index(drop=True, inplace=True)
    mvp = min(df['Vol'])
    mvp_pos = df[df['Vol'] == mvp].index.tolist()[0]
    for i in range(mvp_pos + 1):
        if i == 0:
            x_front.append(df['Vol'][i])
            y_front.append(df['Ret'][i])
        elif i < mvp_pos and df['Vol'][i] < x_front[-1]:
            x_front.append(df['Vol'][i])
            y_front.append(df['Ret'][i])
        elif i == mvp_pos:
            x_front.append(df['Vol'][i])
            y_front.append(df['Ret'][i])

    df = df.sort_values(by="Ret", ascending=False)
    df.reset_index(drop=True, inplace=True)
    mvp_pos = sims - 1 - mvp_pos
    x_gear = []
    y_gear = []
    for i in range(mvp_pos):
        if i == 0:
            x_gear.append(df['Vol'][i])
            y_gear.append(df['Ret'][i])
        elif i < mvp_pos and df['Vol'][i] < x_gear[-1]:
            x_gear.append(df['Vol'][i])
            y_gear.append(df['Ret'][i])

    x_front = x_front + list(reversed(x_gear))
    y_front = y_front + list(reversed(y_gear))

    plt.plot(x_front, y_front, 'r--', linewidth=3)
    # plt.savefig('Markowitz_Simulation.png')
    # plt.show()
    return((x_front, y_front))


def get_MVP(result):
    '''
    This method takes in a tuple result of run_simulation.
    This method returns the minimum variance portfolio given the simulation result.
    '''
    vol_arr = list(result[0])
    ret_arr = result[1]
    sharpe_arr = result[2]
    all_weights = result[3]

    vol = min(vol_arr)
    pos = vol_arr.index(vol)
    ret = ret_arr[pos]
    sharpe = sharpe_arr[pos]
    weights = all_weights[pos]

    print("The Minimum Variance Portofio")
    print("Return:", ret)
    print("Volatility:", vol)
    print("Sharpe Ratio:", sharpe)
    print("Weights:", weights)
    print()

    return((ret, vol, sharpe, weights))


def get_HSR(result):
    '''
    This method takes in a tuple result of run_simulation.
    This method returns the highest sharpe ratio portfolio given the simulation result.
    '''
    vol_arr = result[0]
    ret_arr = result[1]
    sharpe_arr = list(result[2])
    all_weights = result[3]

    sharpe = max(sharpe_arr)
    pos = sharpe_arr.index(sharpe)
    vol = vol_arr[pos]
    ret = ret_arr[pos]
    weights = all_weights[pos]

    print("The Highest Sharpe Ratio Portofio")
    print("Return:", ret)
    print("Volatility:", vol)
    print("Sharpe Ratio:", sharpe)
    print("Weights:", weights)
    print()

    return((ret, vol, sharpe, weights))


# Unconstraint and Constraint MVP Method and Effective Frontier
# Main Contributor: Zehao Dong

def get_return(df):
    ret = np.log(df.pct_change() + 1)
    return ret


def get_cov_matrix(df):
    ret = get_return(df)
    cov = ret.cov().values * 252
    return cov


def MVP(df):
    ret = get_return(df)
    r = np.mean(ret.T, axis=1) * 252
    cov = get_cov_matrix(df)
    n = len(cov)
    mus = np.linspace(0, 0.14, 1000)

    # Convert to cvxopt matrices
    S = 2 * opt.matrix(cov)
    #pbar = opt.matrix(np.mean(returns, axis=1))
    pbar = opt.matrix(0.0, (n, 1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix([opt.matrix(1.0, (1, n)), opt.matrix(r, (1, n))], (2, n))
    portfolios = [solvers.qp(S, pbar, G, h, A, opt.matrix([1.0, mu], (2, 1)))['x']
                  for mu in mus]
    returns = [blas.dot(opt.matrix(r), x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    w = portfolios[risks.index(min(risks))]
    min_ret = returns[risks.index(min(risks))]
    min_risk = min(risks)
    return pd.Series(w), returns, risks, (min_ret, min_risk)


def UMVP(df):
    ret = np.log(df.pct_change() + 1)
    r = np.mean(ret.T, axis=1)
    r = np.matrix(r)
    cov = np.matrix(ret.cov().to_numpy())
    n = len(cov)
    l = np.array(n * [1])
    # global min variance
    w = np.matrix(l) * cov.I / np.sum(np.matrix(l) * cov.I)
    return pd.Series(np.array(w)[0])


def plot_frontier(df):
    # fig, axs = plt.subplots()
    _, returns, risks, min_loc = MVP(df)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(risks, returns, linewidth=4, color='black')
    plt.plot(min_loc[1], min_loc[0], 'ro', label="point")
    plt.text(min_loc[1] + 0.002, min_loc[0] - 0.003, 'MVP', fontsize=18)

    # return axs
    # plt.show()


if __name__ == '__main__':
    df = pd.read_csv('nasdaq_100.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
    df_in = df[df.index < '2015-01-01']
    df_out = df[df.index >= '2015-01-01']
