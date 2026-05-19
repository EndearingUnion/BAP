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
channel = 70

alpha = wls_param[channel, 0] 
beta = wls_param[channel, 1] 



print("alpha = ", alpha)
print("beta = ", beta)


x_t = data_tls[channel, :] + data_atm_plus_photon[channel, :]                   #DATA moet data_tls_photon_atm zijn   
f_welch, Pxx = welch(x_t, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]    



S_nn = tls_estimation(f_welch = f_welch, channel = channel, alpha = alpha, beta = beta)#unit [K^2/Hz]




_, S_ss = welch(data_atm_plus_photon[channel, :], fs=fs, nperseg=nperseg)    #unit [K^2/Hz]


freqs = np.fft.rfftfreq(n = len(x_t), d=1/fs) 

S_nn_interpolated = np.interp(freqs, f_welch, S_nn)     #Interpolation to make Snn and Sss the same size
S_ss_interpolated = np.interp(freqs, f_welch, S_ss)

y_t = wiener_filter(x_t, S_ss_interpolated, S_nn_interpolated)
f_final, Pxx_final = welch(y_t, fs=fs, nperseg=nperseg)

#voor knee comp
alpha_knee = wls_param_knee[channel, 0] 
beta_knee = wls_param_knee[channel, 1] 
print("alpha knee= ", alpha_knee)
print("beta knee= ", beta_knee)
S_nn_knee = tls_estimation(f_welch = f_welch, channel = channel, alpha = alpha_knee, beta = beta_knee)#unit [K^2/Hz]
S_nn_interpolated_knee = np.interp(freqs, f_welch, S_nn_knee)
y_t_knee = wiener_filter(x_t, S_ss_interpolated, S_nn_interpolated_knee)
f_final_knee, Pxx_final_knee = welch(y_t_knee, fs=fs, nperseg=nperseg)

f_knee_atmos = 0.2 #Hz
f_knee_photon = 8 #Hz


#voor alg wls comp
#voor knee comp
alpha_alg = wls_param_alg[channel, 0] 
beta_alg = wls_param_alg[channel, 1] 
print("alpha alg= ", alpha_alg)
print("beta alg= ", beta_alg)
S_nn_alg = tls_estimation(f_welch = f_welch, channel = channel, alpha = alpha_alg, beta = beta_alg)#unit [K^2/Hz]
S_nn_interpolated_alg = np.interp(freqs, f_welch, S_nn_alg)
y_t_alg = wiener_filter(x_t, S_ss_interpolated, S_nn_interpolated_alg)
f_final_alg, Pxx_final_alg = welch(y_t_alg, fs=fs, nperseg=nperseg)

#Plot tls only
f_tls, Pxx_tls = welch(data_tls[channel, :], fs=fs, nperseg=nperseg)



plt.figure(figsize=(10, 6))
#plt.loglog(f_welch, Pxx, label="Input (TLS + Atm + Photon)", alpha=0.5)
plt.axvline(x = f_knee_atmos, label = "Knee Frequency Atmospheric Noise", color = 'red', linestyle = '--')
plt.axvline(x = f_knee_photon, label = "Knee Frequency Photon Noise", color = 'red', linestyle = '--')

plt.loglog(f_final, Pxx_final, label="Output (Filtered)", alpha=0.7)
plt.loglog(f_welch, S_ss, label="Target (Atm + Photon Only)", alpha=0.7, color = "black")

plt.loglog(f_final_knee, Pxx_final_knee, label="Output (Filtered using WLS with fknee)", alpha=0.7)
plt.loglog(f_final_alg, Pxx_final_alg, label="Output (Filtered using WLS algebraic)", alpha=0.7)

plt.loglog(f_welch, S_nn_alg, label="TLS estimation (using WLS algebraic)", alpha=0.7)


plt.loglog(f_tls, Pxx_tls, label="TLS noise", alpha=0.7)

plt.title(f"Wiener Filter Results on Blank Dataset using WLS - Channel {channel}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [K²/Hz]")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()


