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
    data = f["SPAXEL0"]["data"][...]
    times = f["OBSATTRS"]["times"][...]

dt = np.mean(np.diff(times))
fs = 1.0 / dt
num_channels = data.shape[0]



wls_params_alg = np.zeros((num_channels, 2)) #Row = channel, col0 = alpha, col1 = beta


for channel in range(num_channels):
    f_welch, Pxx_ch = welch(data[channel, :], fs=fs, nperseg=nperseg, detrend='constant')
    
    if np.max(Pxx_ch) <= 0:
        wls_params_alg[channel, :] = [0, 0] #For the empty channels
        continue
    
    seg = (f_welch > f_knee_atmos) & (f_welch < f_knee_photon)
    #seg = (f_welch > 0) & (f_welch < f_knee_atmos)
    #seg = ((f_welch > 0) & (f_welch < f_knee_atmos)) | (f_welch > f_knee_photon)#Ensures that WLS is fitted on the segment between the knee freq

    f_fit = f_welch[1:]
    Pxx_fit = Pxx_ch[1:]
    

    ln_f = np.log(f_fit)        #Take ln to make the model linear
    x = ln_S = np.log(Pxx_fit)
    

    w = 1.0 / (Pxx_fit**2 + 1e-30) 
    
    
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
    except np.linalg.LinAlgError:
        wls_params_alg[channel, :] = [0, 0]
        continue




np.save(output_file_alg, wls_params_alg)

