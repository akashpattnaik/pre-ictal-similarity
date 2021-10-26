'''
This script conducts a power analysis to determine the necessary sample size
'''
# %% Imports
# estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower
import numpy as np
from matplotlib import pyplot as plt

# %%
# parameters for power analysis
effect = 0.5
alpha = 0.05
power = 0.8
# perform power analysis
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)

# %%
fig, ax = plt.subplots()
# parameters for power analysis
sample_sizes = np.array(range(40, 100))
# calculate power curves from multiple power analyses
analysis = TTestIndPower()
analysis.plot_power(
    dep_var='nobs', 
    nobs=sample_sizes, 
    effect_size=np.array([effect]), 
    ax=ax)
ax.set_ylabel("Statistical Power (a.u.)")
ax.set_xlabel("Number of Patients")
ax.axhline(0.75, ls='--', color='grey')
ax.axhline(0.85, ls='--', color='grey')
fig.show()

# %%
