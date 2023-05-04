import numpy as np
import h5py
from readLeaprOutput import getRealisticTn
import os
from misc import kb
from egrid import egrid
from tqdm import tqdm
from cn import getCn
from gaussian_fits_and_bounding_funcs import get_bounding_func_for_rejection,\
                                             get_approx_fit_for_rejection
from sampling import get_max_vals_TnCDF_sampling


def get_Tn_vals(rho_x,rho_y,kbT,A,nphon=100):
    #nphon = 100 # Maximum number of phonon excitations you'd like to consider. 
    #            # This number is likely going to be decreased, because at a 
    #            # certain point, the many phonon contributions will be negligible
    #            # Should likely not need more than 100-200

    # Normalize phonon dos
    beta  = rho_x/kbT
    rho_y /= np.trapz(rho_y,beta)

    # Start generating Tn Scattering Values
    Pb = rho_y[1:] / (2*beta[1:]*np.sinh(beta[1:]/2))
    P0 = rho_y[1] / (beta[1])**2
    Pb = np.concatenate((np.array([P0]),Pb))

    # Generate Debye-Waller Coefficient
    dwf_pos = np.trapz(Pb*np.exp(-beta/2),beta)
    dwf_neg = np.trapz(Pb*np.exp( beta/2),beta)
    lambda_s = dwf_pos+dwf_neg 

    # Begin preparing scattering convolution arguments
    T1_neg = Pb*np.exp(beta/2)/lambda_s
    T1_pos = T1_neg*np.exp(-beta)
    T1     = np.concatenate((T1_neg[::-1][:-1],T1_pos))
    betas  = np.concatenate((-beta[::-1][:-1],beta))

    TnVec2 = [[betas,T1]]
    for _ in range(nphon):
        Tnew = np.convolve(T1,TnVec2[-1][1])
        beta_new = np.array([0.001/kbT*i for i in range(len(Tnew))])
        beta_new -= max(beta_new)*0.5
        TnVec2.append([beta_new,Tnew/np.trapz(Tnew,beta_new)])

    for i in range(len(TnVec2)):
        TnVec2[i][0] *= kbT
        
    TnVec3 = []
    for i in range(len(TnVec2)):
        final_x,final_y = getRealisticTn(TnVec2[i][0],TnVec2[i][1])
        TnVec3.append([final_x,final_y])
    return TnVec2,TnVec3,lambda_s




def fill_psm_into_hdf5(psm_group,aboveThisUseGaussian,kbT,A,lambda_s,TnVec2,max_vals,Ein_vec,cutoff_energy,info):
    psm_group.attrs['aboveThisUseGaussian'] = aboveThisUseGaussian
    psm_group.attrs['kbT'] = kbT
    psm_group.attrs['A'] = A
    psm_group.attrs['debyeWaller'] = lambda_s

    # T1_start_ = psm_group.create_dataset('T1_start',()    ,dtype='f8',data=T1_x[0])
    # T1_delta_ = psm_group.create_dataset('T1_delta',()    ,dtype='f8',data=(T1_x[1]-T1_x[0]))
    T1_x = TnVec2[0][0]
    T1_y = TnVec2[0][1]

    psm_group.attrs['T1_start'] = T1_x[0]
    psm_group.attrs['T1_delta'] = T1_x[1]-T1_x[0]

    T1_       = psm_group.create_dataset('T1',(len(T1_y),),dtype='f8',data=T1_y)
    max_vals_ = psm_group.create_dataset('maxVals',(len(max_vals),),dtype='f8',data=max_vals)
    energy_   = psm_group.create_dataset('energy',(len(Ein_vec),),dtype='f8',data=Ein_vec)
    psm_group.attrs['cutoff_energy'] = cutoff_energy


    cVec_long = []
    cVec_offsets = []
    for i in range(len(info)):
        cVec_offsets.append(len(cVec_long))
        cVec_long = np.concatenate((cVec_long,info[i][1]))

    cvecs_1 = psm_group.create_dataset('cvec',(len(cVec_long),),dtype='f8',data=cVec_long)
    cvecs_1.attrs['offsets'] = cVec_offsets


    means    = []
    widths   = []
    scalings = []
    for i in range(len(info)):
        energy = Ein_vec[i]
        # print(i,energy)
        for j in range(len(info[i][0])):
            # print("  ",j,len(info[i][1]),"need all 3?",energy<cutoff_energy,"     ",info[i][0][j],'\t',info[i][1][0])
            if len(info[i][0][j]) > 0:
                means.append(info[i][0][j][0])
                widths.append(info[i][0][j][1])
                if energy < cutoff_energy:
                    scalings.append(info[i][0][j][2])

    gauss_means_    = psm_group.create_dataset('gaussian_means'   ,(len(means)   ,),dtype='f8',data=means)
    gauss_widths_   = psm_group.create_dataset('gaussian_widths'  ,(len(widths)  ,),dtype='f8',data=widths)
    gauss_scalings_ = psm_group.create_dataset('gaussian_scalings',(len(scalings),),dtype='f8',data=scalings)

    
    
    
from scipy.stats import norm
def conduct_search2(x_,y_,m_bounds,w_bounds,lowest_area,best_yet):
#     print('----',best_yet)
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
                        best_yet = [m,w,s]
    return lowest_area,best_yet

from gaussian_fits_and_bounding_funcs import get_x_y_for_rejection_envelopes
def get_bounding_func_for_rejection2(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax):
    gaussian_info = []
    xy_vals = get_x_y_for_rejection_envelopes(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax)

    #for i in tqdm(range(len(cVec))):
    for i in range(len(cVec)):
        x_,y_,_ = xy_vals[i]
        #plt.plot(x_,y_)
        mean      = x_[np.where(y_==max(y_))][0]
        width     = x_[np.where(y_>1e-4*max(y_))[0][-1]]-x_[np.where(y_>1e-4*max(y_))[0][0]]
        gaussian  = norm.pdf(x_,mean,width)
        scaling   = 1.1*max(y_)/max(gaussian)
        gaussian *= scaling

        lowest_area = np.trapz(gaussian,x_)
        best_yet = [mean,width,scaling]

        lowest_area,best_yet = conduct_search2(x_,y_,[mean-0.25*width,mean+0.15*width],[0.1*width,1.5*width],lowest_area,best_yet)
        mean,width,scaling = best_yet
        gaussian  = norm.pdf(x_,mean,width)/max(gaussian) * scaling

        lowest_area,best_yet = conduct_search2(x_,y_,[mean-0.1*width,mean+0.08*width],[0.6*width,1.2*width],lowest_area,best_yet)
        mean,width,scaling = best_yet
        gaussian  = norm.pdf(x_,mean,width)/max(gaussian) * scaling
        gaussian_info.append([mean,width,scaling])
        #plt.plot(x_,gaussian)
    return gaussian_info
    
    
    
def write_hdf5_info(case,rho_x,rho_y,aboveThisUseGaussian,cutoff_energy,
                    endf_openmc_directory,nphon=100,desired_temperatures=None):
    if os.path.isfile(case+'.h5'):
#         !{'rm '+case+'.h5'}
        os.remove(case+'.h5')
    if os.path.isfile(case+'_old.h5'):
#         !{'rm '+case+'.h5'}
        os.remove(case+'_old.h5')

#     !{'cp '+endf_openmc_directory+case+'.h5 ./'}
    os.system('cp '+endf_openmc_directory+case+'.h5 ./'+case+'_old.h5')

    f1 = h5py.File(case+'_old.h5', 'r+')
    f2 = h5py.File('./'+case+'.h5', 'w')

    f2.attrs['filetype'] = f1.attrs['filetype']
    f2.attrs['version'] = f1.attrs['version']

    g1 = f1[case]
    g2 = f2.create_group(case)

    for attrs_key in g1.attrs.keys():
        g2.attrs[attrs_key] = g1.attrs[attrs_key]

    f1.copy(case+'/kTs',f2[case],'kTs')

    max_energy = f1[case].attrs['energy_max']
    A = f1[case].attrs['atomic_weight_ratio']
    desired_kTs = np.array([f1[case]['kTs'][val][()] for val in f1[case]['kTs']])
    desired_Ts  = desired_kTs/kb
    kT_dict = {}
    for key in f1[case]['kTs'].keys():
        kT_dict[key] = f1[case]['kTs'][key][()]

    accepted_Ts = {}
    for T_name in kT_dict:
        if desired_temperatures:
            found_temp = False
            for desired_T in desired_temperatures:
                if abs(int(T_name.replace('K',''))-desired_T) < 3:
                    found_temp = True
            if not found_temp:
                continue
        print(T_name)
        kbT = kT_dict[T_name]
        accepted_Ts[T_name] = kbT/kb
        TnVec2,TnVec3,lambda_s = get_Tn_vals(rho_x,rho_y,kbT,A,nphon)
        Ein_vec = egrid[np.where(egrid <= max_energy*(1+1e-5))]
        T1_x, T1_y = getRealisticTn(TnVec2[0][0],TnVec2[0][1])

        info = []
        cvecs = []
        max_vals = []
        for Ein in tqdm(Ein_vec):
            betaMin = -Ein/kbT
            betaMax = 20.0
            # print('\t\t Getting cVec')
            cVec = getCn(Ein,betaMin,betaMax,kbT,A,nphon,lambda_s,TnVec2)
            if Ein < cutoff_energy:
                # print('\t\t Calculating Gaussian Envelope')
#                 gaussian_info = get_bounding_func_for_rejection(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax)
                gaussian_info = get_bounding_func_for_rejection2(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax)
            else:
                # print('\t\t Calculating Gaussian Fit')
                gaussian_info = get_approx_fit_for_rejection(TnVec3,Ein,kbT,A,lambda_s,cVec,betaMin,betaMax,
                                                             aboveThisUseGaussian)
            # TnCDFs = get_TnCDFs(TnVec3,kbT)
            # print('\t\t Computing Max Vals (new sampling)')
            max_vals.append(get_max_vals_TnCDF_sampling([Ein,],kbT,A,lambda_s)[0])

            # print('\t\t Packing it all up...')
            info_this_energy = [gaussian_info,cVec]
            info.append(info_this_energy)

        # psm_data[T_name] = [info,max_vals,cvecs,TnVec2,lambda_s]

        g2.create_group(T_name)
        g1.copy(key+'/elastic',g2[T_name],'elastic')

        t1 = g1[T_name]
        t2 = g2[T_name]

        i1 = t1['inelastic']
        i2 = t2.create_group('inelastic')
        i1.copy('xs',i2,'xs')

        i2.create_group('distribution')

        i2['distribution'].attrs['type'] = np.bytes_('incoherent_inelastic_phonon_sampling_method')
        fill_psm_into_hdf5(i2['distribution'],aboveThisUseGaussian,kbT,A,lambda_s,TnVec2,max_vals,
                           Ein_vec,cutoff_energy,info)

    f1.close()
    f2.close()
    return accepted_Ts,max_energy