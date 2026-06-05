import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate
from scipy.fft import fft, ifft, fftshift, fftfreq

import h5py, pickle

with open('source.pkl', 'rb') as f:
    source_dict = pickle.load(f)

f.close()


print(source_dict.keys())

frequencies_source = source_dict['frequencies (Hz)']
temp_source = source_dict['source temperature (K)']



f = h5py.File("blank_atm_only.h5", "r")
for group in f:                                 #Print the groups in blank_tls_only
    print(group)
# metadata = f["OBSATTRS"][...] #second element selects the array in the group

with h5py.File("blank_tls_only.h5", "r") as f:  #Print the arrays in OBSATTRS
    print(list(f["OBSATTRS"].keys()))
        
with h5py.File("blank_tls_only.h5", "r") as f:  #Print arrays in SPAXEL0
    print(list(f["SPAXEL0"].keys()))
    
f.close

with h5py.File("blank_atm_only.h5", "r") as f:  
    data_atm_blank = f["SPAXEL0"]["data"][...]
    frequencies_atm_blank = f["OBSATTRS"]["frequencies"][...]
    times_atm_blank = f["OBSATTRS"]["times"][...]

f.close()


with h5py.File("blank_tls_only.h5", "r") as f:  
    data_tls_blank = f["SPAXEL0"]["data"][...]
    frequencies_tls_blank = f["OBSATTRS"]["frequencies"][...]
    times_tls_blank = f["OBSATTRS"]["times"][...] 

    
f.close()

# Reading data with atmospheric and photon noise
with h5py.File("source_atm_plus_photon.h5", "r") as f:  
    data_atm_plus_photon_source = f["SPAXEL0"]["data"][...]
    frequencies_atm_plus_photon_source = f["OBSATTRS"]["frequencies"][...]
    times_atm_plus_phton_source = f["OBSATTRS"]["times"][...]
    az_switched = f["OBSATTRS"]["az"][...]          #az = 0: looking at the source, az != 0 looking off source 
f.close()


eta_atm = np.load("eta_atmosphere.npy") #Load eta_atm
tls_noise_parameters = np.load("tls_wls_params_alg.npy")

print("Unique Azimuth values:", np.unique(az_switched))

T_P_ATM = 273 #T_p,atm Kelvin


num_channels = data_atm_blank.shape[0]
num_times = len(times_atm_blank)
dt = np.mean(np.diff(times_atm_plus_phton_source)) 


x_opt_channels = np.zeros(num_channels)

print("Shape eta_atm",eta_atm.shape)

y = data_atm_plus_photon_source + data_tls_blank - data_atm_blank #y contains tls noise, photon noise + source signal


d_switch = np.mean(az_switched == 0)
print(f"Duty Cycle: {d_switch}")              #Astronomisch correct om het duty cycle te noemen?


x_opt_channels = np.zeros(num_channels)
variance_channels = np.zeros(num_channels)
noise_level_channels = np.zeros(num_channels)





for channel in range(300):
    
    #TLS noise parameters
    alpha = tls_noise_parameters[channel, 0]
    beta = tls_noise_parameters[channel, 1]
    C = tls_noise_parameters[channel, 2]        #Dit hernoemen, terminology klopt niet, is voor photon noise
    
    
    

    a_vec = eta_atm[channel, :].copy()             #Atmoshpere transmission coefficient
    off_source_mask = (az_switched != 0)            #Frequencies where looking off source 
    on_source_mask = (az_switched == 0)             
    a_vec[off_source_mask] = 0                         #Sets a_vec(=eta) to zero
    
    
    y_vec = y[channel, :]   #y vec containing the source signal, photon and TLS noise
    y_vec_off = y_vec[off_source_mask]       #y vec containing off source measurement
    y_vec_on = y_vec[on_source_mask]       #y vec containing on source measurement
    
    min_length = min(len(y_vec_on), len(y_vec_off))     #Arrays have different sizes
    z_vec = y_vec_on[:min_length] - y_vec_off[:min_length]  #z vec, chopper method
    a_vec = a_vec[:min_length]
    
    
    x_opt = np.dot(a_vec, z_vec) / np.dot(a_vec, a_vec) #x optimal
    
    x_opt_channels[channel] = x_opt     #/d_switch
    
    
    #Performence metrics
    # variance_channels[channel] = np.var(z_vec) / np.dot(a_vec, a_vec)
    # noise_level_channels[channel] = np.sqrt(variance_channels[channel])
    
    
    
np.save("x_optimal_chopper.npy", x_opt_channels )
        


# freqs, Pxx = welch(z_vec, fs=1/dt, nperseg=1024)

# plt.figure(figsize=(10, 4))
# plt.loglog(freqs, Pxx, linewidth=1.5)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Z PSD ($K^2/Hz$)")
# plt.title("Power Spectrum of $z$ (Should look flat)")
# plt.grid(True, which="both", ls="--", alpha=0.5)
# plt.show()


np.save("x_optimal_chopper.npy", x_opt_channels )
np.save("chopper_variance_channels.npy", variance_channels)
np.save("chopper_noise_level_channels.npy", noise_level_channels)
snr_channels = temp_source / np.where(noise_level_channels == 0, 1e-6, noise_level_channels)
np.save("chopper_snr_channels.npy", snr_channels)
np.save("chopper_optimal_switching.npy", x_opt_channels )



