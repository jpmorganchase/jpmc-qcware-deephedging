import QuantLib as ql

from BlackScholes import BlackScholesProcess
from utils import HyperParams


hps = HyperParams()

def gen_paths(n_paths, n_steps, seed=0):
    calculation_date = ql.Date.todaysDate()
    maturity_date = ql.Date.todaysDate() + n_steps
    day_count = ql.Actual365Fixed() # Actual/Actual (ISDA)        
    # Length of one time-step (as fraction of a year).
    dt = day_count.yearFraction(calculation_date,calculation_date + 1) 
    maturity = n_steps*dt # Maturities (in the unit of a year)

    stochastic_process = BlackScholesProcess(s0 = hps.S0, sigma = hps.sigma, risk_free = hps.risk_free, \
                            dividend = hps.dividend, day_count = day_count, seed=seed)

    S = stochastic_process.gen_path(maturity, n_steps, n_paths)

    return S