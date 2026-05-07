import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch
from scipy.signal import spectrogram, impulse, correlate

import h5py




#reading the data
f = h5py.File("blank_tls_only.h5", "r")
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
beta = 1.0  #beta parameter
n_samples = len(times)
print("length data =", n_samples)
dt = np.mean(np.diff(times))   #Average sampling interval
fs = 1.0 / dt
channel = 14
k_B = 1.381e-23  # Boltzmann constan [J/K]
nperseg = 2**16






#This chunk of code is taken from  www.socsci.ru.nl/wilberth/python/noise.html by drs. W.C.P. van Ham and altered to fit our project
def pink_spectrum(f, beta=beta, f_min = 0, f_max = np.inf):
    """
    Define a pink (1/f) spectrum
        f     = array of frequencies
        f_min = minimum frequency for band pass
        f_max = maximum frequency for band pass
    """
    #Power = Amplitude^2, so Power ~ (f^-(beta/2))^2 = f^-beta 
    s = f**-(beta/2.0)  #To achieve the amplitude spectrum [W/√Hz]
    s[np.logical_or(f < f_min, f > f_max)] = 0    # apply band pass
    return s






#This chunk of code is based on the code from www.socsci.ru.nl/wilberth/python/noise.html by drs. W.C.P. van Ham and altered to fit our project


def tls_noise_estimation(n_samples, fs):
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)  #Real FFT for real-valued time signals
                                                #d is the interval between each sample
                                                #n_samples = len(f["SPAXEL0"]["data"][...])
    spectrum = np.zeros_like(freqs, dtype='complex')      # make complex numbers for spectrum
    spectrum[1:] = pink_spectrum(freqs[1:])               # get spectrum amplitude for all frequencies except f=0, to aviod div by 0
    phases = np.random.uniform(0, 2*np.pi, len(freqs)-1)  # random phases for all frequencies except f=0
    spectrum[1:] *= np.exp(1j*phases)                     # apply random phases to the amplitude
    noise = np.fft.irfft(spectrum)                        # return the reverse fourier transform to get time series
    noise = np.pad(noise, (0, n_samples - len(noise)), 'constant') # add zero for odd number of input samples
    
    return noise




def scalar(Pxx, Pxx_estimation):
    scalar = np.mean(Pxx/Pxx_estimation)
    return scalar



#Channel bandwidth (NEeded to convert K to Watt using Johnson-Nyquist from literature source)
#P = kTb telecommunications and sensing lecture 3 slide 10
def channel_bandwidth(channel):
    bw = np.diff(frequencies)
    channel_bw = bw[channel]
    #print("channel bandwidth = ", channel_bw)
    return channel_bw


#TLS noise estimation
tls_estimate = tls_noise_estimation(n_samples, fs)



#actual noise
#Welch PSD estimate (from EE3S1 Lab)
x = data[channel, :] #Unit Kelvin

f_welch, Pxx_actual = welch(x, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]

ASD_K = np.sqrt(Pxx_actual) #ASD in [K/√Hz]
ASD_W = k_B*ASD_K*channel_bandwidth(channel) #ASD in [W/√Hz]            available noise power telecommunication and sensing lecture 3 slide 10



#Welch PSD estimate (from EE3S1 Lab)
x = tls_estimate #Unit Kelvin

f_welch, Pxx_estimation = welch(x, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]

Pxx_estimation_scaled = Pxx_estimation*scalar(Pxx_actual, Pxx_estimation)

ASD_estimation_scaled = np.sqrt(Pxx_estimation_scaled)*channel_bandwidth(channel)*k_B #ASD in [w/√Hz]
#S(f) = A*f^-beta (source Modeling scaled processes and 1/fβ noise by the nonlinear stochastic differential equations)



plt.figure()
plt.loglog(f_welch, ASD_estimation_scaled, label="TLS noise estimation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (W/√Hz)")
plt.title(f"Estimation of TLS nosie - Welch PSD")
plt.grid(True, which="both")
plt.legend()
plt.show()