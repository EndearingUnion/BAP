import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate
from hrtem_filter import filters
from scipy.signal.windows import gaussian
import h5py

#Constants
k_B = 1.381e-23  # Boltzmann constan [J/K]



#Select channel
channel = 14



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


#Determining the sample freq
dt = np.mean(np.diff(times))   #Average sampling interval
fs = 1.0 / dt

print("dt =", dt)
print("fs =", fs)


#Channel bandwidth (NEeded to convert K to Watt using Johnson-Nyquist from literature source)
#P = kTb
bw = np.diff(frequencies)
channel_bw = bw[channel]
print("channel bandwidth = ", channel_bw)


#Welch PSD estimate (from EE3S1 Lab)
x = data[channel, :] #Unit Kelvin
fs = fs 
nperseg = 2**16

f_welch, Pxx = welch(x, fs=fs, nperseg = nperseg, detrend='constant') #Unit [K/Hz]

ASD_K = np.sqrt(Pxx) #ASD in [K/√Hz]
ASD_W = k_B*ASD_K*channel_bw #ASD in [W/√Hz]            available noise power telecommunication and sensing lecture 3 slide 10



#Wiener filter
raw_data = data[channel, :]

# filtered_data = wiener(raw_data, mysize = 11)

# plt.figure(figsize=(10, 5))
# plt.plot(raw_data, label="Raw Signal", alpha=0.5)
# plt.plot(filtered_data, label="Wiener Filtered", color='red')
# plt.title(f"Wiener Filter applied to Channel {channel}")
# plt.legend()
# plt.show()


#Frequency domain wiener filter


def gaussian_kernel(kernel_size=3):
    kernel_size = int(kernel_size)
    h = gaussian(kernel_size, kernel_size / 3)
    
    # 2D Gaussian via outer product
    h = np.asarray(h).flatten()
    
    # Normalize
    h /= np.sum(h)
    return h

def wiener_filter_time(data, kernel, K):
    # Ensure kernel is normalized
    data = np.asarray(data)
    n_samples = data.size

    kernel = np.asarray(kernel)
    kernel /= np.sum(kernel)
    
    # Copy image and move to frequency domain
    dummy = np.copy(data)
    dummy = fft(dummy)
    
    # Move kernel to frequency domain, padded to image size
    kernel = fft(kernel, n=n_samples)
    
    # The Wiener Gain formula: H* / (|H|^2 + K)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    
    # Apply filter in frequency domain
    dummy = dummy * kernel
    
    # Transform back to spatial domain
    dummy = np.abs(ifft(dummy))
    return dummy


def wiener_filter_frequency(data, kernel, K):
    # Ensure kernel is normalized
    data = np.asarray(data)
    n_samples = data.size

    kernel = np.asarray(kernel)
    kernel /= np.sum(kernel)
    
    # Copy image and move to frequency domain
    dummy = np.copy(data)
    dummy = fft(dummy)
    
    # Move kernel to frequency domain, padded to image size
    kernel = fft(kernel, n=n_samples)
    
    # The Wiener Gain formula: H* / (|H|^2 + K)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    
    # Apply filter in frequency domain
    dummy = dummy * kernel

    return dummy

# Create a 1D kernel (try size 5 or 7)
my_kernel = gaussian_kernel(kernel_size=7)

# Apply the filter to your specific channel
# Try K between 0.01 and 0.1 depending on noise levels
filtered_result_time = wiener_filter_time(raw_data, my_kernel, K=0.01)

filtered_result_freq = wiener_filter_frequency(raw_data, my_kernel, K=0.01)

# # Plotting the comparison
# plt.plot(raw_data, label="Raw", alpha=0.5)
# plt.plot(filtered_result_time, label="1D Wiener Filtered", color='red')
# plt.legend()
# plt.show()

# 2. Frequency-Domain Performance Plot
# To see the "Frequency" result, we look at the Magnitude Spectrum
plt.figure(figsize=(10, 4))
# Convert complex result to magnitude
freq_magnitude = np.abs(filtered_result_freq) 
# Create frequency axis for plotting
freq_axis = np.fft.fftfreq(len(raw_data), d=dt)

plt.semilogy(freq_axis[:len(freq_axis)//2], freq_magnitude[:len(freq_axis)//2], color='blue')
plt.title("Frequency Spectrum (Magnitude) of Filtered Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True, which="both", alpha=0.3)
plt.show()


