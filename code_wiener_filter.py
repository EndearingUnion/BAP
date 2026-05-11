import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate

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
beta = 0.5012009845887407




x_t = data_tls[channel, :] + data_atm_plus_photon[channel, :]                   #DATA moet data_tls_photon_atm zijn   
f_welch, Pxx = welch(x_t, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]    
    
S_nn = tls_estimation(f_welch = f_welch, channel = channel, beta = beta)#*(2.365877666756682e-05)/scaling_factors[channel] #unit [K^2/Hz]

_, S_ss = welch(data_atm_plus_photon[channel, :], fs=fs, nperseg=nperseg)    #unit [K^2/Hz]


freqs = np.fft.rfftfreq(n = len(x_t), d=1/fs) 

S_nn_interpolated = np.interp(freqs, f_welch, S_nn)     #Interpolation to make Snn and Sss the same size
S_ss_interpolated = np.interp(freqs, f_welch, S_ss)


y_t = wiener_filter(x_t, S_ss_interpolated, S_nn_interpolated)
f_final, Pxx_final = welch(y_t, fs=fs, nperseg=nperseg)


print("scaling factor", scaling_factors[channel])





plt.figure(figsize=(10, 6))
#plt.loglog(f_welch, Pxx, label="Input (TLS + Atm + Photon)", alpha=0.5)
plt.loglog(f_final, Pxx_final, label="Output (Filtered)", alpha=0.7)
plt.loglog(f_welch, S_ss, label="Target (Atm + Photon Only)", alpha=0.7)

plt.title(f"Wiener Filter Results - Channel {channel}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [K²/Hz]")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()


