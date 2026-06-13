import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit
import h5py

from BAP_functions import *


input_file = "blank_tls_only.h5"
output_file_alg = "tls_wls_params_alg.npy" 
nperseg = 2**16

# Reading data with tls noise
with h5py.File(input_file, "r") as f:
    data_tls_blank = f["SPAXEL0"]["data"][...]
    times_tls_blank = f["OBSATTRS"]["times"][...]
f.close()
    
# Reading data with atmospheric noise, photon noise and source
with h5py.File("source_atm_plus_photon.h5", "r") as f:  
    data_atm_plus_photon_source = f["SPAXEL0"]["data"][...]
    frequencies_atm_plus_photon_source = f["OBSATTRS"]["frequencies"][...]
    times_atm_plus_phton_source = f["OBSATTRS"]["times"][...] 
f.close()    
    
# Reading data with atmospheric noise and source
with h5py.File("source_atm_only.h5", "r") as f:  
    data_atm_source = f["SPAXEL0"]["data"][...]
    frequencies_atm_source = f["OBSATTRS"]["frequencies"][...]
    times_atm_source = f["OBSATTRS"]["times"][...] 
    
f.close()

dt = np.mean(np.diff(times_tls_blank))      #Calculates the average time between samples
fs = 1.0 / dt                               #Calculates the sample frequency of DESHIMA 2.0 simulations
num_channels = data_tls_blank.shape[0]

print("sampling frequency = ", fs, "Hz")

wls_params_alg = np.zeros((num_channels, 3))                             #Row = channel, col0 = alpha, col1 = beta, col2 = C
data_photon_blank = data_atm_plus_photon_source - data_atm_source        #Creates photon noise only data set

for channel in range(num_channels):
    
    f_welch, Pxx_ch = welch(data_tls_blank[channel, :], fs=fs, nperseg=nperseg, detrend='constant')         #PSD of TLS noise
    _, Pxx_photon = welch(data_photon_blank[channel, :], fs= fs, nperseg=nperseg, detrend='constant')       #PSD of photon noise
    
    C_estimate = np.mean(Pxx_photon)             #Calculates the mean of the photon noise and stores in C
    
    if np.max(Pxx_ch) <= 0:
        wls_params_alg[channel, :] = [0, 0,0]           #For the empty channels
        continue
    
    intersection_tls_photon = np.where(Pxx_ch < (1.5 * C_estimate))[0]      #Calculates where the photon noise and TLS intersect
    f_upper_bound = f_welch[intersection_tls_photon[0]] * 0.8 if len(intersection_tls_photon) > 0 else 0.18 #Multiplies intersection frequency with 0.8 due to fluctuations
    
    
    seg = (f_welch > f_knee_atmos) & (f_welch < f_knee_photon)          #Range where TLS noise is dominant
    
    f_fit = f_welch[seg]
    Pxx_fit = Pxx_ch[seg] 
    

    ln_f = np.log(f_fit)        #Makes noise model linear
    x = ln_S = np.log(Pxx_fit)
    

    w = 1.0 / (Pxx_fit**2 + 1e-30)  #initializes the weight matrix

    
    
    HTWH = np.array([                               #Computes H.T @ W @ H 
        [np.sum(w), np.sum(w * ln_f)],
        [np.sum(w * ln_f), np.sum(w * ln_f**2)]
    ])
    
    
    HTWx = np.array([                               #Compute H.T @ W @ x  (2x1 vector)
        np.sum(w * x),
        np.sum(w * ln_f * x)
    ])
    

    try:
        theta_hat = np.linalg.inv(HTWH) @ HTWx
        
        wls_params_alg[channel, 0] = np.exp(theta_hat[0])  # alpha_wls
        wls_params_alg[channel, 1] = -theta_hat[1]         # beta_wls
        wls_params_alg[channel, 2] = C_estimate            #C
    except np.linalg.LinAlgError:
        wls_params_alg[channel, :] = [0, 0, 0]
        continue




np.save(output_file_alg, wls_params_alg)                   #Saves parameters in output file

