import numpy as np
from scipy import special
from scipy.special import erf, gamma, gammaincc, gammaincinv
from misc import getAlphaMinMax

def getWeightOfNthTerm(Ein,kbT,A,fullBetas,n,lambda_s,betaMin,betaMax):
    valid_betas_i = np.where((fullBetas > betaMin+1e-6)&(fullBetas <= betaMax))
    trapz_over_alpha = []
    for b in valid_betas_i[0]:
        beta = float(fullBetas[b])
        alphaMin,alphaMax = getAlphaMinMax(Ein,beta,kbT,A)
        value = -((gammaincc(n+1,lambda_s*alphaMax)-gammaincc(n+1,lambda_s*alphaMin)))
        trapz_over_alpha.append(value)
    return trapz_over_alpha

def getNContribution(Ein,kbT,A,nphon,fullBetas,lambda_s,betaMin,betaMax):
    nContribsBetaDep = []
    for n in range(nphon):
        if n < nphon:
            nContributionVec = getWeightOfNthTerm(Ein,kbT,A,fullBetas,n+1,lambda_s,betaMin,betaMax)
            nContribsBetaDep.append(nContributionVec)
        else:
            print("help")

    return nContribsBetaDep


#def getCn(nContribsBetaDep,TnVec2,valid_betas,nphon,kbT):
def getCn(Ein,betaMin,betaMax,kbT,A,nphon,lambda_s,TnVec2):
    fullBetas = np.linspace(betaMin,betaMax,500)
    valid_betas = fullBetas[np.where((fullBetas > betaMin+1e-6) & (fullBetas <= betaMax))]
    nContribsBetaDep = np.array(getNContribution(Ein,kbT,A,nphon,fullBetas,lambda_s,betaMin,betaMax))

    cVec = [0.0]*nphon
    for n in range(nphon):
        added = []
        for b in range(len(valid_betas)):
            added.append(nContribsBetaDep[n][b]*np.interp(valid_betas[b]*kbT,TnVec2[n][0],TnVec2[n][1]) )
        cVec[n] = np.trapz(added,valid_betas)
    cVec = cVec[:np.where(cVec>1e-5*max(cVec))[-1][-1]+1]
    cVec /= sum(cVec)
    return cVec


#cVec = getCn(nContribsBetaDep,TnVec2,valid_betas,nphon,kbT)
#cVec = cVec[:np.where(cVec>1e-10)[-1][-1]+1]


