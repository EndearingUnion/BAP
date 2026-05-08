import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate

import h5py


#import the scaling factors
scaling_factors = np.load("tls_scaling_factors.npy")


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






def wiener_filter(x_t, S_ss, S_nn):
    
    """
    Applies a Wiener filter to time-series data on input.
    
    X(f)H(f) = Y(f)
    
    H(f) = Wiener Filter
    Y(f) = desired signal
    
    x: time-domain signal (Atm + Photon + TLS)
    fs: sampling frequency
    S_ss: The psd of desired signal (photon + atmospheric noise)
    S_nn: The psd of noise (simulated TLS noise)
    """
    
    n = len(x_t)
    
    X_f = np.fft.rfft(x_t)                    #Transform input to frequency domain
        
    H_f = S_ss / (S_ss + S_nn + 1e-30)      #Wiener filter H(f) = Sss / (Sss + Snn)
    
    Y_f = X_f * H_f                         #Apply the Wiener filter with convolution
    
    y_t = np.fft.irfft(Y_f, n=n)
    
    return y_t



def tls_estimation(f_welch, channel, beta):
    tls_model_psd = np.zeros_like(f_welch)
    tls_model_psd[1:] = f_welch[1:]**-(beta)
    tls_model_scaled = scaling_factors[channel]*tls_model_psd
    
    return tls_model_scaled



#Parameters
dt = np.mean(np.diff(times_atm_plus_phton))   #Average sampling interval MOET OP times_tls_photon_atm
fs = 1.0 / dt                   #Determining the sample freq
nperseg = 2**16
channel = 14
beta = 0.5




x_t = data_tls[channel, :] + data_atm_plus_photon[channel, :]                   #DATA moet data_tls_photon_atm zijn   
f_welch, Pxx = welch(x_t, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]    
    
S_nn = tls_estimation(f_welch = f_welch, channel = channel, beta = beta) #unit [K^2/Hz]

_, S_ss = welch(data_atm_plus_photon[channel, :], fs=fs, nperseg=nperseg)    #unit [K^2/Hz]


freqs = np.fft.rfftfreq(n = len(x_t), d=1/fs) 

S_nn_interpolated = np.interp(freqs, f_welch, S_nn)     #Interpolation to make Snn and Sss the same size
S_ss_interpolated = np.interp(freqs, f_welch, S_ss)




y_t = wiener_filter(x_t, S_ss_interpolated, S_nn_interpolated)



f_final, Pxx_final = welch(y_t, fs=fs, nperseg=nperseg)

plt.figure(figsize=(10, 6))
plt.loglog(f_welch, Pxx, label="Input (TLS + Atm + Photon)", alpha=0.5)
plt.loglog(f_final, Pxx_final, label="Output (Filtered)")
plt.loglog(f_welch, S_ss, label="Target (Atm + Photon Only)")

plt.title(f"Wiener Filter Results - Channel {channel}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [K²/Hz]")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()


