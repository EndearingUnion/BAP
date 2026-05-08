import numpy as np
from scipy.signal import welch
import h5py


input_file = "blank_tls_only.h5"
output_file = "tls_scaling_factors.npy"
beta = 0.5
nperseg = 2**16


with h5py.File(input_file, "r") as f:
    data = f["SPAXEL0"]["data"][...]
    times = f["OBSATTRS"]["times"][...]


dt = np.mean(np.diff(times))
fs = 1.0 / dt
num_channels = data.shape[0]

#Welch PSD estimate (from EE3S1 Lab)
_, Pxx_temp = welch(data[0, :], fs=fs, nperseg=nperseg, detrend='constant')
freqs = np.fft.rfftfreq(nperseg) * fs 


model_raw = np.zeros_like(Pxx_temp) #S = f^-0.5 used to find the scaling factor
model_raw[1:] = freqs[1:]**-beta


scaling_factors = np.zeros(num_channels)



for channel in range(num_channels):
    _, Pxx_ch = welch(data[channel, :], fs=fs, nperseg=nperseg, detrend='constant')#calculates the actual tls noise PSD from the simulations
    
    factor = np.mean(Pxx_ch[1:] / model_raw[1:]) #Calculates the scaling factor of the current channel
    scaling_factors[channel] = factor
    


np.save(output_file, scaling_factors)

print('scaling factors', scaling_factors)