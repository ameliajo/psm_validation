import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from misc import getAlphaMinMax
from scipy import special
from scipy.special import erf, gamma, gammaincc, gammaincinv
#from tqdm import tqdm


def conduct_search(x_,y_,m_bounds,w_bounds,lowest_area,best_yet):
    for m in np.linspace(m_bounds[0],m_bounds[1],12):
        for w in np.linspace(w_bounds[0],w_bounds[1],18):
            g  = norm.pdf(x_,m,w)
            min_scaling = max(y_)/max(g)
            for s in np.linspace(min_scaling,1.3*min_scaling,8):
                gaussian = g*s
                if (gaussian >= y_).all():
                    area = np.trapz(gaussian,x_)
                    if area < lowest_area:
                        lowest_area=area
                        best_yet = [m,w,s,gaussian]
    return lowest_area,best_yet


def get_x_y_for_rejection_envelopes(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax):
    x_ = np.linspace(betaMin+1e-6,betaMax,2000)
    main = []
    for n in range(len(cVec)):
        y_ = []
        nContribs = []
        for guessBeta in x_:
            alphaMin,alphaMax = getAlphaMinMax(Ein,guessBeta,kbT,A)
            nContribs_val = -((gammaincc(n+2,lambda_s*alphaMax)-gammaincc(n+2,lambda_s*alphaMin)))/lambda_s
            y_.append(nContribs_val * np.interp(guessBeta*kbT,TnVec3[n][0],TnVec3[n][1],left=0,right=0)/cVec[n])
            nContribs.append(nContribs_val)
        main.append([x_,y_,nContribs])
    return main


def get_bounding_func_for_rejection(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax):
    gaussian_info = []
    xy_vals = get_x_y_for_rejection_envelopes(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax)

    #for i in tqdm(range(len(cVec))):
    for i in range(len(cVec)):
        x_,y_,_ = xy_vals[i]

        mean      = x_[np.where(y_==max(y_))][0]
        width     = x_[np.where(y_>1e-4*max(y_))[0][-1]]-x_[np.where(y_>1e-4*max(y_))[0][0]]
        gaussian  = norm.pdf(x_,mean,width)
        scaling   = 1.1*max(y_)/max(gaussian)
        gaussian *= scaling

        lowest_area = np.trapz(gaussian,x_)
        best_yet = [mean,width,scaling]

        lowest_area,best_yet = conduct_search(x_,y_,[mean-0.25*width,mean+0.15*width],[0.1*width,1.5*width],lowest_area,best_yet)
        mean,width,scaling,gaussian = best_yet

        lowest_area,best_yet = conduct_search(x_,y_,[mean-0.1*width,mean+0.08*width],[0.6*width,1.2*width],lowest_area,best_yet)
        mean,width,scaling,gaussian = best_yet

        #print(starting_area>lowest_area,'\t\t',starting_area,lowest_area)
        gaussian_info.append([mean,width,scaling])
    return gaussian_info


def get_approx_fit_for_rejection(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax,aboveThisUseGaussian):
    gaussian_info = []
    xy_vals = get_x_y_for_rejection_envelopes(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax)

    for i in range(aboveThisUseGaussian,len(cVec)):
        x_,y_,_ = xy_vals[i]
        mean      = x_[np.where(y_==max(y_))][0]
        width     = x_[np.where(y_>1e-4*max(y_))[0][-1]]-x_[np.where(y_>1e-4*max(y_))[0][0]]

        def gaussian_func(x, scaling, mean, sigma):
            y = scaling*np.exp(-(x-mean)**2/(2.0*sigma**2))
            return y
        parameters, covariance = curve_fit(gaussian_func, x_, y_, p0=[1.0,mean,width])
        scaling,mean,sigma = parameters
        # fit_y = gaussian_func(x_, scaling, mean, sigma)
        # plt.plot(x_,y_); plt.plot(x_, fit_y)
        gaussian_info.append([mean,abs(sigma),scaling])
    return gaussian_info

