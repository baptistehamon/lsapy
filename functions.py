"Suitability function definitions"

import numpy as np
import scipy.stats as stats

class SuitabilityFunction:
    def __init__(self, name, params):
        pass

def sigmoid():
    pass

def asymsigmoid(x, c, d, m):
    return 1 / (np.power(1 + np.power(x/c, d), m))

import matplotlib.pyplot as plt
from scipy.stats import genlogistic
from scipy.optimize import curve_fit


indicator_vals = np.array([700,750,800,850,1000])
suit_vals = np.array([0,0.25,0.5,0.75,1])

x = np.linspace(0, 1200, 100)
y_asymsigmoid = asymsigmoid(x, 501, -13.1, 296)

popt, _ = curve_fit(genlogistic.cdf, indicator_vals, suit_vals, p0=[1, np.median(x),1], maxfev=15000)
y_genlogistic = genlogistic.cdf(x, *popt)

 = genlogistic.fit(indicator_vals)

plt.plot(x, y_asymsigmoid, label='Asymmetric Sigmoid')
plt.plot(x, y_genlogistic, label='Generalized Logistic')
plt.scatter(indicator_vals, suit_vals, c='r')
plt.legend()
plt.show()
