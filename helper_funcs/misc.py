import openmc
from readLeaprOutput import readLeaprOutput, thinOutTn, getRealisticTn, readTn, getRho
import os
import numpy as np

kb      =  8.617333262145e-5

from scipy.integrate import cumtrapz
def get_TnCDFs(TnVec3,kbT):
    TnCDFs = []
    for n in range(len(TnVec3)):
        cdf = np.concatenate(([0],cumtrapz(TnVec3[n][1],TnVec3[n][0])))
        TnCDFs.append([np.array(TnVec3[n][0])/kbT,cdf/max(cdf)])
    return TnCDFs


def getAlphaMinMax(Ein,beta,kbT,A):
    alphaMin = ((Ein)**0.5 - (Ein+beta*kbT)**0.5)**2 / (A*kbT)
    alphaMax = ((Ein)**0.5 + (Ein+beta*kbT)**0.5)**2 / (A*kbT)
    return alphaMin,alphaMax


#def get_setup(case,njoy_exec,base_directory='./NJOY_results'):
def get_setup(case,njoy_exec,base_directory='./NJOY_results',T_name=None):
    directory = base_directory+'/'+case+'/'
    if T_name:
        directory += T_name+'/'

    inputFile = directory + 'tsl-'+case+'.leapr'

    if not os.path.isfile(directory+'./tape24'):
        #!{njoy_exec+' < '+inputFile+' && mv output Tn.dat tape24 '+directory}
        print(os.popen(njoy_exec+' < '+inputFile+' && mv output Tn.dat tape24 '+directory).read())

    tsl = openmc.data.ThermalScattering.from_endf(directory+'/tape24')
    T   = round(tsl.kTs[0]/kb,1)
    A   = tsl.atomic_weight_ratio
    kbT = tsl.kTs[0]
    delta,rho_E_L,rhoVals_L = getRho(inputFile)
    lambda_s = float(os.popen('grep lambda '+directory+'/output ').read().split()[-1])

    alphas_leapr, betas_leapr, maxSab_leapr, sab_leapr = readLeaprOutput(directory+'./tape24',None,T)
    sab_leapr,alphas,fullBetas = sab_leapr.T,np.array(alphas_leapr[:]),np.array(betas_leapr[:])

    _,TnVec2 = readTn(directory,delta/kbT)
    for i in range(len(TnVec2)):
        TnVec2[i][0] *= kbT

    TnVec3 = []
    for i in range(len(TnVec2)):
        final_x,final_y = getRealisticTn(TnVec2[i][0],TnVec2[i][1])
        TnVec3.append([final_x,final_y])
    return T,kbT,A,rho_E_L,rhoVals_L,lambda_s,sab_leapr,alphas,fullBetas,TnVec2,TnVec3

