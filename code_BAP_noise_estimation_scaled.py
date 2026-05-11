import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch
from scipy.signal import spectrogram, impulse, correlate

import h5py

from BAP_functions import *


#reading the data
f = h5py.File("blank_atm_plus_photon.h5", "r")
for group in f:                                 #Print the groups in blank_tls_only
    print(group)
# metadata = f["OBSATTRS"][...] #second element selects the array in the group

with h5py.File("blank_tls_only.h5", "r") as f:  #Print the arrays in OBSATTRS
    print(list(f["OBSATTRS"].keys()))
        
with h5py.File("blank_tls_only.h5", "r") as f:  #Print arrays in SPAXEL0
    print(list(f["SPAXEL0"].keys()))
    
with h5py.File("blank_tls_only.h5", "r") as f:  #Reading data
    data = f["SPAXEL0"]["data"][...]
    az_spax = f["SPAXEL0"]["az_spax"][...]
    el_spax = f["SPAXEL0"]["el_spax"][...]
    
    frequencies = f["OBSATTRS"]["frequencies"][...]
    times = f["OBSATTRS"]["times"][...]
    
    
f.close()




#parameters
beta = 0.5 #beta parameter
n_samples = len(times)
print("length data =", n_samples)
dt = np.mean(np.diff(times))   #Average sampling interval
fs = 1.0 / dt
channel = 14
k_B = 1.380649e-23  # Boltzmann constan [J/K]
nperseg = 2**16



#TLS noise estimation
tls_estimate = tls_noise_estimation(n_samples, fs) #unitless



#actual noise
#Welch PSD estimate (from EE3S1 Lab)
tls_actual = data[channel, :] #Unit Kelvin (actual TLs noise)

f_welch, Pxx_actual = welch(tls_actual, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]

#ASD_K = np.sqrt(Pxx_actual) #ASD in [K/√Hz]
#ASD_W = k_B*ASD_K*channel_bandwidth(channel) #ASD in [W/√Hz]            available noise power telecommunication and sensing lecture 3 slide 10



#Welch PSD estimate (from EE3S1 Lab)
tls_estimate = tls_estimate #Unit Kelvin

_, Pxx_estimation = welch(tls_estimate, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]

scaling_factor = np.mean(Pxx_actual[1:] / Pxx_estimation[1:])
Pxx_estimation_scaled = Pxx_estimation*scaling_factor


#Pxx_estimation_scaled = Pxx_estimation*scalar(Pxx_actual, Pxx_estimation)

#ASD_estimation_scaled = np.sqrt(Pxx_estimation_scaled)*channel_bandwidth(channel)*k_B #ASD in [w/√Hz]
#S(f) = A*f^-beta (source Modeling scaled processes and 1/fβ noise by the nonlinear stochastic differential equations)

ASD_actual = np.sqrt(Pxx_actual) * k_B #[w/√Hz]
ASD_estimation = np.sqrt(Pxx_estimation_scaled) * k_B #[w/√Hz]



#plotting a tls model to test the found parameters
tls_model_psd = np.zeros_like(f_welch)
tls_model_psd[1:] = f_welch[1:]**-(beta)
scaling_factor = np.median(Pxx_actual[1:] / tls_model_psd[1:])

tls_model_scaled = scaling_factor*tls_model_psd
ASD_model = np.sqrt(tls_model_scaled) * k_B

print("scaling factor =", scaling_factor)



plt.figure()
plt.loglog(f_welch, ASD_actual, label="Actual TLS noise", alpha=0.7)
plt.loglog(f_welch, ASD_estimation, label="Estimated TLS noise", color = 'orange')
plt.loglog(f_welch, ASD_model, label="Modelled TLS noise", color = 'green')
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (W/√Hz)")
plt.title(f"Comparison of Estimated vs. Actual TLS nosie")
plt.grid(True, which="both")
plt.legend()
plt.show()