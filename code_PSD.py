import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import h5py
from BAP_functions import * 
#Constants
k_B = 1.381e-23  # Boltzmann constan [J/K]
nperseg = 2**16
channel = 14


def load_data(filename):
    #load directory
    data = {}
    
    with h5py.File(filename, "r") as f:
        # Load OBSATTRS
        if "OBSATTRS" in f:
            obs_group = f["OBSATTRS"]
            data['freqs'] = obs_group["frequencies"][...]
            data['times'] = obs_group["times"][...]
        
        # Load SPAXEL0
        if "SPAXEL0" in f:
            spax_group = f["SPAXEL0"]
            data['data'] = spax_group["data"][...]
            data['az']   = spax_group["az_spax"][...]
            data['el']   = spax_group["el_spax"][...]

        print(f"Successfully loaded: {filename}")
        
    return data


#load all of the data
tls_only   = load_data("blank_tls_only.h5")
atm_photon = load_data("blank_atm_plus_photon.h5")
atm_only = load_data("blank_atm_only.h5")

#calculate the photon data
photon_only = atm_photon['data'] - atm_only['data']

# create a new dictionary for photon noise
photon_only = {
    'data': photon_only,
    'times': atm_photon['times'],
    'freqs': atm_photon['freqs'],
    'az': atm_photon['az'],
    'el': atm_photon['el']
}

print(atm_photon['data'].shape)
print(atm_only['data'].shape)

sum = tls_only['data'] + atm_photon['data']

# create a new dictionary for sum noise
sum = {
    'data': sum,
    'times': atm_photon['times'],
    'freqs': atm_photon['freqs'],
    'az': atm_photon['az'],
    'el': atm_photon['el']
}


#Determining the sample freq
dt = np.mean(np.diff(sum['times']))
fs = 1.0 / dt

#Channel bandwidth (Needed to convert K to Watt using Johnson-Nyquist from literature source)
#P = kTb
bw = np.diff(sum["freqs"])
channel_bw = bw[channel]

print(f"Sampling Frequency (fs): {fs:.2f} Hz")
print(f"Channel {channel} Bandwidth: {channel_bw:.2f} Hz")


datasets = {
    "TLS Only": tls_only['data'],
    "Atmosphere Only": atm_only['data'],
    "Photon Only": photon_only['data'],
    'Sum': sum['data']
}

print(np.mean(tls_only['data']))
print(np.mean(atm_photon['data']))
print(np.mean(sum['data'])) # This should be the sum of the two above

plt.figure(figsize=(10, 6))

for label, data_array in datasets.items():
   #Welch PSD estimate (from EE3S1 Lab)
    x = data_array[channel, :]                              #Unit is Kelvin
    
    f_welch, Pxx = welch(x, fs=fs, nperseg=nperseg, detrend='constant')
    
    # Convert Kelvin PSD to Watt ASD
    # Pxx is in [K^2/Hz], sqrt(Pxx) is [K/√Hz]
    #ASD_K = np.sqrt(Pxx)                                    #ASD in [K/√Hz]
    #SD_W = k_B*Pxx                          #ASD in [W/√Hz]            available noise power telecommunication and sensing lecture 3 slide 10
    
    # Plot the ADS of all the data in the same plot
    plt.loglog(f_welch, Pxx, label=label, alpha = 0.7)
    

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"PSD ($K^2/Hz$)")
plt.axvline(x = f_knee_atmos, label = "Knee Frequency Atmospheric Noise", color = 'red', linestyle = '--')
plt.axvline(x = f_knee_photon, label = "Knee Frequency Photon Noise", color = 'red', linestyle = '--')
plt.title(f"Comparison of Noise Components - Channel {channel}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()