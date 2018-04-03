import numpy as np 
import pandas as pd 
import pymc3 as pm, theano.tensor as tt
import matplotlib.pyplot as plt
import time, datetime
from scipy import stats

from pymc3.distributions import Interpolated


def convertTime(t):
    x = time.strptime(t,'%M:%S')
    return float(int(datetime.timedelta(minutes=x.tm_min,seconds=x.tm_sec).total_seconds()))






def player_model(df, mask):

    shots=df['shot_made'].values[mask]

    with pm.Model() as bernoulli_model: 
        # Distributions are PyMC3 objects.
        # Specify prior using Uniform object.
        p_prior = pm.Uniform('p', 0, 1)  
        
        # Specify likelihood using Bernoulli object.
        like = pm.Bernoulli('likelihood', p=p_prior, 
                            observed=shots)  

    with bernoulli_model:

        step = pm.Metropolis()
        
        shot_trace=pm.sample(10000, step=step)

    return shot_trace

def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)

if __name__=='__main__':

    df = pd.read_csv('free_throws.csv')

    df['t_seconds']=df['time'].apply(convertTime)
    df['t_seconds']/=df['t_seconds'].max()



    # T=df['t_seconds'].values[mask]

    player_grouping=df.groupby(['player'])

    shots_made=player_grouping['shot_made'].sum().values
    attempts=player_grouping['shot_made'].count().values
    names=np.array([name for name, _ in player_grouping['player']])

    N_players=len(attempts)

    with pm.Model() as model:

        global_skill = pm.Uniform('global_skill', lower=0.0, upper=1.0)

        kappa_log = pm.Exponential('kappa_log', lam=1.5)
        kappa = pm.Deterministic('kappa', tt.exp(kappa_log))

        individual_player_skills = pm.Beta('individual_player_skills', alpha=global_skill*kappa, beta=(1.0-global_skill)*kappa, shape=N_players)
        likelihood = pm.Binomial('likelihood', n=attempts, p=individual_player_skills, observed=shots_made)

        trace = pm.sample(2000, tune=1000, chains=2)

pm.traceplot(trace, varnames=['global_skill'])

skill_levels=np.mean(trace['individual_player_skills'], axis=0)

#Testing