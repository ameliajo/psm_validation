import numpy as np
from numpy.random import random
from scipy import special
from scipy.special import erf, gamma, gammaincc, gammaincinv
from misc import getAlphaMinMax
from scipy.stats import norm

def sample_simple_rejection(n,gaussian_info,betaMin,betaMax,Ein,kbT,A,lambda_s,TnVec3,cVec,numRuns):
    found = False
    while not found:
        numRuns += 1
        mean, sigma, scaling = gaussian_info[n]
        guessBeta = np.random.normal(mean,sigma,1)[0]
        if guessBeta < betaMin or guessBeta > betaMax:
            continue

        alphaMin,alphaMax = getAlphaMinMax(Ein,guessBeta,kbT,A)
        nContribs_val = -((gammaincc(n+2,lambda_s*alphaMax)-gammaincc(n+2,lambda_s*alphaMin)))/lambda_s 
        Tn_piece = np.interp(guessBeta*kbT,TnVec3[n][0],TnVec3[n][1],left=0,right=0)
        guessVal = nContribs_val * Tn_piece /cVec[n]
        sample_point = random()*scaling*norm.pdf([guessBeta],mean,sigma)
        
        if sample_point < guessVal:
            xi = random()*(gammaincc(n+2,lambda_s*alphaMin)-gammaincc(n+2,lambda_s*alphaMax))
            alpha = gammaincinv(n+2, xi)/lambda_s
            return guessBeta,alpha,numRuns

        
def sample_gaussian_approx(n,gaussian_info,betaMin,betaMax,Ein,kbT,A,aboveThisUseGaussian,lambda_s,numRuns):
    found = False
    while not found:
        numRuns += 1
        mean, sigma, scaling = gaussian_info[n-aboveThisUseGaussian]
        guessBeta = np.random.normal(mean,sigma,1)[0]
        if guessBeta < betaMin or guessBeta > betaMax:
            continue
        alphaMin,alphaMax = getAlphaMinMax(Ein,guessBeta,kbT,A)
        xi = random()*(gammaincc(n+2,lambda_s*alphaMin)-gammaincc(n+2,lambda_s*alphaMax))
        alpha = gammaincinv(n+2, xi)/lambda_s
        return guessBeta,alpha,numRuns

    
def sample_TnCDF_method(n,gaussian_info,betaMin,betaMax,TnCDFs,Ein,kbT,A,lambda_s,max_vals,numRuns):
    found = False
    while not found:
        numRuns += 1
        xi_minimum = np.interp(-Ein/kbT,TnCDFs[n][0],TnCDFs[n][1])
        xi = random()*(1-xi_minimum) + xi_minimum
        guessBeta = np.interp(xi,TnCDFs[n][1],TnCDFs[n][0])
        max_val_wn = max_vals[0]
        alphaMin,alphaMax = getAlphaMinMax(Ein,guessBeta,kbT,A)
        nContribs_val = -((gammaincc(n+2,lambda_s*alphaMax)-gammaincc(n+2,lambda_s*alphaMin)))
        if random()*max_val_wn*1.05 < nContribs_val:
            xi = random()*(gammaincc(n+2,lambda_s*alphaMin)-gammaincc(n+2,lambda_s*alphaMax))
            alpha = gammaincinv(n+2, xi)/lambda_s
            return guessBeta,alpha,numRuns


from scipy.special import erf, gamma, gammaincc, gammaincinv
def get_max_vals_TnCDF_sampling(Ein_vec,kbT,A,lambda_s):
    n = 0
    max_vals = []
    for this_ein in Ein_vec:
        betaMin,betaMax = -this_ein/kbT,20
        temp_x,temp_y = np.linspace(betaMin*(1-1e-6),betaMax*10,10000),[]
        for this_x in temp_x:
            alphaMin,alphaMax = getAlphaMinMax(this_ein,this_x,kbT,A)
            temp_y.append(-((gammaincc(n+2,lambda_s*alphaMax)-gammaincc(n+2,lambda_s*alphaMin))))
        max_vals.append(max(temp_y))
    return max_vals
