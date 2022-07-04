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
    '''
    dt = 1/365
    paths = np.zeros((hps.n_steps + 1, hps.n_paths), np.float64)
    paths[0] = hps.S0
    for t in tqdm(range(1, hps.n_steps + 1)):
        rand = np.random.standard_normal(hps.n_paths)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((hps.risk_free - 0.5 * hps.sigma ** 2) * dt +
                                         hps.sigma * np.sqrt(dt) * rand)
    return paths.T