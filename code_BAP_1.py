import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate

import h5py

#Constants
k_B = 1.381e-23  # Boltzmann constan [J/K]



#Select channel
channel = 270



#reading the data
f = h5py.File("blank_atm_only.h5", "r")
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



# #Print shape of data
# print("data shape:", data.shape)
# print("frequencies shape:", frequencies.shape)

# #Plot TLS noise over time
# plt.plot(data[channel, :])
# plt.xlabel("Time index")
# plt.ylabel("Signal")
# plt.title(f"Channel {channel}: TLS noise over time")



#Determining the sample freq
dt = np.mean(np.diff(times))   #Average sampling interval
fs = 1.0 / dt

print("dt =", dt)
print("fs =", fs)



#Welch PSD estimate (from EE3S1 Lab)
x = data[channel, :] #Unit Kelvin
fs = fs 
nperseg = 2**16

f_welch, Pxx = welch(x, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]

ASD_K = np.sqrt(Pxx) #ASD in [K/√Hz]
ASD_W = k_B*ASD_K #ASD in [W/√Hz]            available noise power telecommunication and sensing lecture 3 slide 10

plt.figure()
plt.loglog(f_welch, ASD_W, label="TLS noise")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (W/√Hz)")
plt.title(f"Channel {channel} - Welch PSD")
plt.grid(True, which="both")
plt.legend()
plt.show()



