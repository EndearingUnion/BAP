import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate

import h5py

from BAP_functions import *

# Reading data with tls noise only
with h5py.File("blank_tls_only.h5", "r") as f:  
    data_tls = f["SPAXEL0"]["data"][...]
    frequencies_tls = f["OBSATTRS"]["frequencies"][...]
    times_tls = f["OBSATTRS"]["times"][...] 

# Reading data with atmospheric and photon noise
with h5py.File("blank_atm_plus_photon.h5", "r") as f:  
    data_atm_plus_photon = f["SPAXEL0"]["data"][...]
    frequencies_atm_plus_photon = f["OBSATTRS"]["frequencies"][...]
    times_atm_plus_phton = f["OBSATTRS"]["times"][...] 

# Parameters
dt = np.mean(np.diff(times_atm_plus_phton))   
fs = 1.0 / dt                   
nperseg = 2**16
channel = 14

alpha = wls_param[channel, 0] 
beta = wls_param[channel, 1] 

print("alpha = ", alpha)
print("beta = ", beta)

x_t = data_tls[channel, :] + data_atm_plus_photon[channel, :]                  
f_welch, Pxx = welch(x_t, fs=fs, nperseg=nperseg, detrend='constant')    

S_nn = tls_estimation(f_welch=f_welch, channel=channel, alpha=alpha, beta=beta)
_, S_ss = welch(data_atm_plus_photon[channel, :], fs=fs, nperseg=nperseg)    

freqs = np.fft.rfftfreq(n=len(x_t), d=1/fs) 


# Voor alg wls comp
alpha_alg = wls_param_alg[channel, 0] 
beta_alg = wls_param_alg[channel, 1]


print("alpha alg= ", alpha_alg)
print("beta alg= ", beta_alg)

#Plot tls only
f_tls, Pxx_tls = welch(data_tls[channel, :], fs=fs, nperseg=nperseg)


S_nn_interpolated_alg = np.zeros_like(freqs)
S_nn_interpolated_alg[1:] = alpha_alg * (freqs[1:] ** -beta_alg) 

S_ss_interpolated = np.interp(freqs, f_welch, S_ss)

y_t_alg = wiener_filter(x_t, S_ss_interpolated, S_nn_interpolated_alg)
f_final_alg, Pxx_final_alg = welch(y_t_alg, fs=fs, nperseg=nperseg)







plt.figure(figsize=(10, 6))
plt.axvline(x=f_knee_atmos, label="Knee Frequency Atmospheric Noise", color='red', linestyle='--')
plt.axvline(x=f_knee_photon, label="Knee Frequency Photon Noise", color='red', linestyle='--')

#plt.loglog(f_final, Pxx_final, label="Output (Filtered)", alpha=0.7)
plt.loglog(f_welch, S_ss, label="Target (Atm + Photon Only)", alpha=0.7, color="black")
#plt.loglog(f_final_knee, Pxx_final_knee, label="Output (Filtered using WLS with fknee)", alpha=0.7)
plt.loglog(f_final_alg, Pxx_final_alg, label="Output (Filtered using WLS algebraic)", alpha=0.7, color="green")
plt.loglog(f_tls, Pxx_tls, label="TLS noise", alpha=0.7)
plt.loglog(f_tls[1:], alpha_alg * f_tls[1:]**-beta_alg, label="WLS model", color='red', linestyle='--')

plt.title(f"Wiener Filter Results on Blank Dataset using WLS - Channel {channel}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [K²/Hz]")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()




