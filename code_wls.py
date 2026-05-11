import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate
from scipy.optimize import curve_fit

import h5py

from BAP_functions import *




#reading the data with tls noise only
f = h5py.File("blank_tls_only.h5", "r")

with h5py.File("blank_tls_only.h5", "r") as f:  #Reading data
    data_tls = f["SPAXEL0"]["data"][...]
    frequencies_tls = f["OBSATTRS"]["frequencies"][...]
    times_tls = f["OBSATTRS"]["times"][...] 
    
f.close()


#reading data with atmospheric and photon noise
f = h5py.File("blank_atm_plus_photon.h5", "r")

with h5py.File("blank_atm_plus_photon.h5", "r") as f:  #Reading data
    data_atm_plus_photon = f["SPAXEL0"]["data"][...]
    frequencies_atm_plus_photon = f["OBSATTRS"]["frequencies"][...]
    times_atm_plus_phton = f["OBSATTRS"]["times"][...] 
    
f.close()




#Parameters
dt = np.mean(np.diff(times_atm_plus_phton))   #Average sampling interval MOET OP times_tls_photon_atm
fs = 1.0 / dt                   #Determining the sample freq
nperseg = 2**16
channel = 14
# beta = 0.5




#Weighted Least Squares 
f_tls_actual, Pxx_tls_actual = welch(data_tls[channel, :], fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz], WLS will fit to this curve

f_fit = f_tls_actual[1:]        #Remove the 0 Hz bin
Pxx_fit = Pxx_tls_actual[1:]    #Remove the 0 Hz bin

weights = Pxx_fit #the actual 


param_opt, param_cov = curve_fit(power_law, f_fit, Pxx_fit, sigma=weights, p0=[scaling_factors[channel], 0.5]) #p0 are the initial guess for alpha and beta
alpha_fit, beta_fit = param_opt

print(f"Fitted Scaling Factor (alpha): {alpha_fit}")
print(f"Fitted Beta (Slope): {beta_fit}")