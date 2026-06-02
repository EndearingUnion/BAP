import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate
from scipy.fft import fft, ifft, fftshift, fftfreq

import h5py

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


#data_complete = data_atm_plus_photon_source + data_tls_blank    #T_meas, contains signal from the source plus TLS, photon and atmospheric noise
#data_source_tls_photon = data_atm_plus_photon_source + data_tls_blank - data_atm_blank

x_opt_channels = np.zeros(num_channels)

print("Shape eta_atm",eta_atm.shape)

y = data_atm_plus_photon_source + data_tls_blank - data_atm_blank #y contains tls noise, photon noise + source signal


d_switch = np.mean(az_switched == 0)
print(f"Duty Cycle: {d_switch}")              #Astronomisch correct om het duty cycle te noemen?



plt.figure(figsize=(10, 6))
plt.plot(frequencies_atm_plus_photon_source / 1e9, y.mean(axis=1), label="Ground Truth ($x$)", color="black", linewidth=1)
#plt.plot(frequencies_source / 1e9, x_opt, label="Reconstructed Source", color="red", alpha = 0.7, linewidth=1)




plt.xlabel("Frequency (GHz)")
plt.ylabel("Source Temperature $T_A^*$ (Kelvin)")
plt.title("Ground Truth Source")
plt.legend()
plt.grid(True)
plt.show()


for channel in range(300):
    
    #TLS noise parameters
    alpha = tls_noise_parameters[channel, 0]
    beta = tls_noise_parameters[channel, 1]
    C = tls_noise_parameters[channel, 2]        #Dit hernoemen, terminology klopt niet, is voor photon noise
    
    
    
    #Data model \vec{y} = \vec{a}x + \mathbf{n}
    #T_meas = data_complete[channel, :]      #Measured temperature
    a_vec = eta_atm[channel, :].copy()             #Atmoshpere transmission coefficient
    off_source_mask = (az_switched != 0)            #Frequencies where looking off source              
    a_vec[off_source_mask] = 0                         #Sets a_vec(=eta) to zero
    
    
    y_vec = y[channel, :]#T_meas - (1 - a_vec)*T_P_ATM    #y vec containing the source signal, photon and TLS noise
    #y_vec = a_vec * temp_source[channel]  #AANPASSEN NAAR vec_a times x (gt)
        
    freqs = fftfreq(num_times, d=dt)
    S_nn = np.zeros_like(freqs)
    
    nonzero_mask = freqs != 0
    S_nn[nonzero_mask] = alpha * np.abs(freqs[nonzero_mask])**-beta + C #Is n in the model, photon and TLS noise
    S_nn[~nonzero_mask] = C
    
    
    
    W = 1.0 / np.sqrt(S_nn) #Rn ^-1/2, Snn = F{Rn}, Rn = F^-1{Snn}
    y_vec_tilde = np.real(ifft(W * fft(y_vec))) #Whitening(?)
    a_vec_tilde = np.real(ifft(W * fft(a_vec))) #Whitening(?)
    
    
    x_opt = np.dot(a_vec_tilde, y_vec_tilde) / np.dot(a_vec_tilde, a_vec_tilde) #x optimal
    
    x_opt_channels[channel] = x_opt     #/d_switch
    
    np.save("x_optimal_switching.npy", x_opt_channels )
