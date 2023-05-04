import numpy as np
import openmc
from scipy.interpolate import interp1d


def readLeaprOutput(path,mat,desiredTemp=None):
    maxSab = 0.0
    data = openmc.data.ThermalScattering.from_endf(path)
    kb      =  8.617333262145e-5
    alphas = data.inelastic.data['sab']['alpha']
    betas  = data.inelastic.data['sab']['beta' ]
    if desiredTemp:
        sab   = data.inelastic.data['sab'][desiredTemp]
    else:
        assert(len(list(data.inelastic.data['sab'].keys()))==3)
        desiredTemp = [key for key in list(data.inelastic.data['sab'].keys()) if key not in ['alpha','beta']][0]
        sab   = data.inelastic.data['sab'][desiredTemp]

    print('Getting information for temperature: ',desiredTemp,'K')
    alphas = [(0.0253/(kb*desiredTemp))*a for a in alphas]

    nbeta = len(betas)
    sab_array = np.zeros((2*len(betas)-1,len(alphas)))
    fullBetas = []
    for ibeta in range(len(betas)):
        beta = betas[ibeta]
        beta *= 0.0253/(kb*desiredTemp)
        fullBetas.append(beta)

        if ibeta != 0 and beta > 1e-10:
            fullBetas.append(-beta)
        this_sab = sab.T[ibeta]
        if max(this_sab) > maxSab: maxSab = max(this_sab)
        sab_array[nbeta-ibeta-1] = [val*np.exp( beta*0.5) for val in this_sab]
        sab_array[nbeta+ibeta-1] = [val*np.exp(-beta*0.5) for val in this_sab]
    fullBetas.sort()
    return alphas, fullBetas, maxSab, sab_array


def readTn(directory,deltab):
    TnVec = []

    with open(directory+'Tn.dat','r') as f:
        while True:
            line = f.readline()
            if len(line.split()) == 0:
                break
            n = int(line.split('=')[1])
            Tn = []
            line = f.readline()
            while '---' not in line:
                Tn.append(float(line))
                line = f.readline()
            betas = [deltab*i for i in range(len(Tn))]
            TnVec.append([n,betas,Tn])


    TnVec2 = []
    for i in range(len(TnVec)):
        Tn = TnVec[i][2]
        bgrid = np.array(range(len(Tn)))*deltab
        Tn_pos = np.array(Tn)*np.exp(-bgrid)
        Tn_neg = np.array(Tn[::-1])
        bgrid_full = np.concatenate((-1*bgrid[::-1],bgrid[1:]))
        Tn_full = np.concatenate((Tn_neg,Tn_pos[1:]))
        TnVec2.append([bgrid_full,Tn_full])
    return TnVec,TnVec2



def getRho(fileName):
    with open(fileName,'r') as f:
        lines = f.readlines()
        i_rho = None
        for i,line in enumerate(lines):
            try:
                if len(line.split()) == 3:
                    if line.split()[-1] == 'temperature':
                         i_rho = i+1
            except:
                continue
        delta = float(lines[i_rho].split()[0])
        nVals = int(float(lines[i_rho].split()[1]))
        i, j = 0,0
        rhoVals = []
        while j < nVals:
            this_line = [val for val in lines[i_rho+i+1].split() if val != '/']
            j += len(this_line)
            i += 1
            rhoVals += [float(x) for x in this_line]
        rho_E = [delta*i for i in range(nVals)]
        return delta,rho_E,rhoVals




"""
def thinOutTn(Tn_x,Tn_y):
    final_x = [val for val in Tn_x]
    final_y = [val for val in Tn_y]

    while True:
        x_new = final_x[::4]
        y_new = final_y[::4]
        f_new = interp1d(x_new,y_new,bounds_error=False)
        works = True
        for i,val in enumerate(Tn_x):
            if Tn_y[i] > 0.01*max(Tn_y) and abs(f_new(val)-Tn_y[i]) > 0.1*Tn_y[i]:
                works = False
                break
        if works:
            final_x = x_new[:]
            final_y = y_new[:]
        else:
            break
    return final_x,final_y
"""

def thinOutTn(Tn_x,Tn_y):
    trial_x = np.linspace(Tn_x[0],Tn_x[-1],500)
    trial_y = np.interp(trial_x,Tn_x,Tn_y)
    final_x = np.copy(trial_x)
    final_y = np.copy(trial_y)
    while True:
        x_new = final_x[::2]
        y_new = final_y[::2]
        f_new = interp1d(x_new,y_new,bounds_error=False)
        works = True
        
        approximated_y = np.interp(trial_x,x_new,y_new)
        for i in range(len(trial_x)):
            if trial_y[i] > 0.01*max(trial_y) and abs(approximated_y[i]-trial_y[i]) > 0.1*trial_y[i]:
                works = False
                break

        if works:
            final_x = x_new[:]
            final_y = y_new[:]
        else:
            break
    return final_x,final_y

# def getRealisticTn(Tn_x,Tn_y):
#     f1 = interp1d(Tn_x[:],Tn_y[:])
#     if len(np.where(Tn_y < 1e-5)[0]) <= 2:
#         return thinOutTn(Tn_x,Tn_y)
#     first_big_val = np.where(Tn_y>1e-5*max(Tn_y))[0][0]
#     last_big_val  = np.where(Tn_y>1e-5*max(Tn_y))[0][-1]
#     shorter_x = Tn_x[first_big_val:last_big_val]
#     shorter_y = Tn_y[first_big_val:last_big_val]
#     return thinOutTn(shorter_x,shorter_y)

def getRealisticTn(Tn_x,Tn_y):
    f1 = interp1d(Tn_x[:],Tn_y[:])
    if len(np.where(Tn_y < 1e-7)[0]) <= 2:
        return thinOutTn(Tn_x,Tn_y)
    first_big_val = np.where(Tn_y>1e-5*max(Tn_y))[0][0]-1
    if first_big_val < 0:
        first_big_val = 0
    last_big_val  = np.where(Tn_y>1e-5*max(Tn_y))[0][-1]
    shorter_x = Tn_x[first_big_val:last_big_val]
    shorter_y = Tn_y[first_big_val:last_big_val]
    return thinOutTn(shorter_x,shorter_y)



"""
import sys
sys.path.append('/Users/ameliajo/getPyENDFtk/ENDFtk/bin')
sys.path.append('/Users/ameliajo/getPyENDFtk/ENDFtk/build')
sys.path.append('/Users/ameliatrainer/getPyENDFtk/ENDFtk/build')
import ENDFtk


def readLeaprOutput(path,mat,desiredTemp=None):
    maxSab = 0.0

    endfTape = ENDFtk.tree.Tape.from_file(path)
    h2o = endfTape.MAT(mat)
    MT4 = h2o.MF(7).MT(4).parse()
    functions = MT4.scattering_law.scattering_functions
    i = 0
    if desiredTemp:
        i = [i for i in range(len(functions[0].temperatures)) \
            if abs(functions[0].temperatures[i] - desiredTemp) < 1]
        assert(len(i)==1)
        i = i[0]
    else:
        assert( len(functions[0].thermal_scattering_values) == 1 )

    print('            Getting information for temperature: ',functions[0].temperatures[i],'K')
    
    nbeta = len(functions)
    alphas = functions[0].alphas.to_list()
    sab_array = np.zeros((2*nbeta-1,len(alphas)))
    betas = []
    for ibeta in range(len(functions)):
        func = functions[ibeta]
        beta = func.beta
        if MT4.LAT == 1:
            beta *= 0.0253/(kb*functions[0].temperatures[i])
        betas.append(beta)
        if ibeta != 0 and beta > 1e-10:
            betas.append(-beta)
        sab  = func.thermal_scattering_values[i].to_list()
        if max(sab) > maxSab: maxSab = max(sab)
        sab_array[nbeta-ibeta-1] = [val*np.exp( beta*0.5) for val in sab]
        sab_array[nbeta+ibeta-1] = [val*np.exp(-beta*0.5) for val in sab]
    if MT4.LAT == 1:
        alphas = [(0.0253/(kb*functions[0].temperatures[i]))*a for a in alphas]

    betas.sort()
    sab_array = sab_array.T

    alphas = np.array(alphas[:])
    betas = np.array(betas[:])
    return alphas, betas, maxSab, sab_array

"""
