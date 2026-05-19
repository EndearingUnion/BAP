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


wls_params = np.zeros((num_channels, 2)) #Row = channel, col0 = alpha, col1 = beta

f_knee_atmos = 0.2 #Hz
f_knee_photon = 8 #Hz

for channel in range(num_channels):
    f_welch, Pxx_ch = welch(data[channel, :], fs=fs, nperseg=nperseg, detrend='constant')
    
    seg = (f_welch > f_knee_atmos) & (f_welch < f_knee_photon) #Ensures that WLS is fitted on the segment between the knee freq

    f_fit = f_welch[seg]
    Pxx_fit = Pxx_ch[seg]
    

    weights = np.maximum(Pxx_fit, 1e-20) 
    

    popt, _ = curve_fit(power_law, f_fit, Pxx_fit, sigma=weights, p0=[1e-5, 0.5])
        
    wls_params[channel, 0] = popt[0]  # alpha_wls
    wls_params[channel, 1] = popt[1]  # beta_wls




np.save(output_file, wls_params)



wls_params_alg = np.zeros((num_channels, 2)) #Row = channel, col0 = alpha, col1 = beta


for channel in range(num_channels):
    f_welch, Pxx_ch = welch(data[channel, :], fs=fs, nperseg=nperseg, detrend='constant')
    
    if np.max(Pxx_ch) <= 0:
        wls_params_alg[channel, :] = [0, 0] #For the empty channels
        continue
    
    seg = ((f_welch > 0) & (f_welch < f_knee_atmos)) | (f_welch > f_knee_photon)#Ensures that WLS is fitted on the segment between the knee freq

    f_fit = f_welch[seg]
    Pxx_fit = Pxx_ch[seg]
    
    # #f_fit = f_welch[1:]
    # #Pxx_fit = Pxx_ch[1:]
    
    ln_f = np.log(f_fit)        #Take ln to make the model linear
    x = ln_S = np.log(Pxx_fit)
    
    # H = np.column_stack((np.ones_like(ln_f), ln_f)) #Observation matrix H

    # #W = np.diag(np.maximum(Pxx_fit, 1e-20)) #Weights matrix W
    # #W = np.diag(np.ones_like(ln_f))
    # #W = np.diag(np.log(Pxx_fit))
    # W = np.diag(1.0 / Pxx_fit**2)
    

    
    # HTW = H.T @ W #Calculate the the matrix multiplication of tranpose H times W
    
    # theta_hat = np.linalg.inv(HTW @ H) @ HTW @ x
    

    
        
    # wls_params_alg[channel, 0] = np.exp(theta_hat[0])  # alpha_wls
    # wls_params_alg[channel, 1] = -theta_hat[1]  # beta_wls
    

    w = 1.0 / (Pxx_fit**2 + 1e-30) 
    
    # Compute H.T @ W @ H without creating a 2D W matrix
    HTWH = np.array([
        [np.sum(w), np.sum(w * ln_f)],
        [np.sum(w * ln_f), np.sum(w * ln_f**2)]
    ])
    
    # Compute H.T @ W @ x which results in a 2x1 vector
    HTWx = np.array([
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

