import numpy as np
from tqdm import tqdm
from utils import HyperParams

def gen_paths(hps):
    ''' Generate paths for geometric Brownian motion.
    Args:
        hps: HyperParams
    Returns:
        paths: (n_paths, n_steps + 1) array of paths
    '''
    dt = 1/252
    paths = np.zeros((hps.n_steps + 1, hps.n_paths), np.float64)
    paths[0] = hps.S0
    for t in tqdm(range(1, hps.n_steps + 1)):
        rand = np.random.standard_normal(hps.n_paths)
        rand = (rand - rand.mean()) / rand.std()
        if hps.discrete_path:
            rand = np.asarray([2*int(i>0)-1 for i in rand])
            paths[t] = paths[t - 1] + rand
        else:
            paths[t] = paths[t - 1] * np.exp((hps.risk_free - 0.5 * hps.sigma ** 2) * dt +
                                            hps.sigma * np.sqrt(dt) * rand)
    return paths.T
