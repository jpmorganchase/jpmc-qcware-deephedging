import QuantLib as ql

from BlackScholes import BlackScholesProcess
from utils import HyperParams
import numpy as np
from tqdm import tqdm

def gen_paths_quantlib(hps, seed=0):
    calculation_date = ql.Date.todaysDate()
    maturity_date = ql.Date.todaysDate() + hps.n_steps
    day_count = ql.Actual365Fixed()  # Actual/Actual (ISDA)
    # Length of one time-step (as fraction of a year).
    dt = day_count.yearFraction(calculation_date, calculation_date + 1)
    maturity = hps.n_steps*dt  # Maturities (in the unit of a year)

    stochastic_process = BlackScholesProcess(s0=hps.S0, sigma=hps.sigma, risk_free=hps.risk_free,
                                             dividend=hps.dividend, day_count=day_count, seed=seed)

    S = stochastic_process.gen_path(maturity, hps.n_steps, hps.n_paths)

    return S


def gen_paths(hps):
    ''' Generate paths for geometric Brownian motion.
    Parameters
    ==========
    S0 : float
        initial stock/index value
    r : float
        constant short rate
    sigma : float
        constant volatility
    T : float
        final time horizon
    M : int
        number of time steps/intervals
    I : int
        number of paths to be simulated
        
    Returns
    =======
    paths : ndarray, shape (M + 1, I)
        simulated paths given the parameters
    '''
    S0 = hps.S0
    r = hps.risk_free
    sigma = hps.sigma
    T = hps.n_steps/365
    M = hps.n_steps
    I = hps.n_paths
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in tqdm(range(1, M + 1)):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths.T