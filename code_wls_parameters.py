import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit
import h5py

from BAP_functions import *


input_file = "blank_tls_only.h5"
output_file = "tls_wls_params_fknee.npy" 
output_file_alg = "tls_wls_params_alg.npy" 
nperseg = 2**16


with h5py.File(input_file, "r") as f:
    data_tls_blank = f["SPAXEL0"]["data"][...]
    times_tls_blank = f["OBSATTRS"]["times"][...]
f.close()
    
# Reading data with atmospheric and photon noise
with h5py.File("source_atm_plus_photon.h5", "r") as f:  
    data_atm_plus_photon_source = f["SPAXEL0"]["data"][...]
    frequencies_atm_plus_photon_source = f["OBSATTRS"]["frequencies"][...]
    times_atm_plus_phton_source = f["OBSATTRS"]["times"][...] 
f.close()    
    
# Reading data with atmospheric
with h5py.File("source_atm_only.h5", "r") as f:  
    data_atm_source = f["SPAXEL0"]["data"][...]
    frequencies_atm_source = f["OBSATTRS"]["frequencies"][...]
    times_atm_source = f["OBSATTRS"]["times"][...] 
    
f.close()

dt = np.mean(np.diff(times_tls_blank))
fs = 1.0 / dt
num_channels = data_tls_blank.shape[0]

print("sampling frequency = ", fs, "Hz")

wls_params_alg = np.zeros((num_channels, 3)) #Row = channel, col0 = alpha, col1 = beta, col2 = C
data_photon_blank = data_atm_plus_photon_source - data_atm_source      #Create photon noise only data set

for channel in range(num_channels):
    

    
    
    f_welch, Pxx_ch = welch(data_tls_blank[channel, :], fs=fs, nperseg=nperseg, detrend='constant')
    _, Pxx_photon = welch(data_photon_blank[channel, :], fs= fs, nperseg=nperseg, detrend='constant')
    
        
    #f_floor = f_welch > f_knee_photon #C is based on the part where the photon noise is dominant
    C_estimate = np.mean(Pxx_photon) #Take the mean of the PSD
    
    if np.max(Pxx_ch) <= 0:
        wls_params_alg[channel, :] = [0, 0,0] #For the empty channels
        continue
    
    intersection_tls_photon = np.where(Pxx_ch < (1.5 * C_estimate))[0]      #finds the knee freq or intersection with photon noise and tls noise, fluctuation -> 1.5
    f_upper_bound = f_welch[intersection_tls_photon[0]] * 0.8 if len(intersection_tls_photon) > 0 else 0.18 #fluctuations -> times .8
    #f_upper_bound = np.clip(f_upper_bound, 0.05, 0.25)  
    
    
    seg = (f_welch > f_knee_atmos) & (f_welch < f_knee_photon)
    #seg = (f_welch > 0) & (f_welch < f_knee_atmos)
    #seg = ((f_welch > 0) & (f_welch < f_knee_atmos)) | (f_welch > f_knee_photon)#Ensures that WLS is fitted on the segment between the knee freq
    
    f_fit = f_welch[seg]
    Pxx_fit = Pxx_ch[seg] 
    

    ln_f = np.log(f_fit)        #Take ln to make the model linear
    x = ln_S = np.log(Pxx_fit)
    

    w = 1.0 / (Pxx_fit**2 + 1e-30) 
    #w = np.ones_like(Pxx_fit)
    #w = weight_matrix
    
    
    HTWH = np.array([               # Compute H.T @ W @ H 
        [np.sum(w), np.sum(w * ln_f)],
        [np.sum(w * ln_f), np.sum(w * ln_f**2)]
    ])
    
    
    HTWx = np.array([           # Compute H.T @ W @ x which (2x1 vector)
        np.sum(w * x),
        np.sum(w * ln_f * x)
    ])
    

    try:
        theta_hat = np.linalg.inv(HTWH) @ HTWx
        
        wls_params_alg[channel, 0] = np.exp(theta_hat[0])  # alpha_wls
        wls_params_alg[channel, 1] = -theta_hat[1]         # beta_wls
        wls_params_alg[channel, 2] = C_estimate             #C
    except np.linalg.LinAlgError:
        wls_params_alg[channel, :] = [0, 0, 0]
        continue




np.save(output_file_alg, wls_params_alg)

