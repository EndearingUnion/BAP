import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate

import h5py
import pickle


with open('source.pkl', 'rb') as f:
    source_dict = pickle.load(f)

f.close()


print(source_dict.keys())

frequencies_source = source_dict['frequencies (Hz)']
temp_source = source_dict['source temperature (K)']


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
f.close()

with h5py.File("blank_atm_plus_photon.h5", "r") as f:  
    data_atm_plus_photon_blank = f["SPAXEL0"]["data"][...]
    frequencies_atm_plus_photon_blank = f["OBSATTRS"]["frequencies"][...]
    times_atm_plus_phton_blank = f["OBSATTRS"]["times"][...]
f.close()



eta_atm = np.load("eta_atmosphere.npy") #Load eta_atm
tls_noise_parameters = np.load("tls_wls_params_alg.npy")



T_P_ATM = 273 #T_p,atm Kelvin


num_channels = data_atm_blank.shape[0]
num_times = len(times_atm_blank)
dt = np.mean(np.diff(times_atm_plus_phton_source)) 

photon_noise = data_atm_plus_photon_blank  - data_atm_blank


#data_complete = data_atm_plus_photon_source + data_tls_blank    #T_meas, contains signal from the source plus TLS, photon and atmospheric noise
#data_source_tls_photon = data_atm_plus_photon_source + data_tls_blank - data_atm_blank

x_opt_channels = np.zeros(num_channels)
variance_channels = np.zeros(num_channels)
noise_level_channels = np.zeros(num_channels)

print("Shape eta_atm",eta_atm.shape)
print("Shape freq source",frequencies_source.shape)
print("Shape temp source",temp_source.shape)

for channel in range(300):
    
    #TLS noise parameters
    alpha = tls_noise_parameters[channel, 0]
    beta = tls_noise_parameters[channel, 1]
    C = tls_noise_parameters[channel, 2]        #Dit hernoemen, terminology klopt niet, is voor photon noise
    
    
    
    #Data model \vec{y} = \vec{a}x + \mathbf{n}     
    #T_meas = data_complete[channel, :]      #Measured temperature
    a_vec = eta_atm[channel, :]          #Atmoshpere transmission coefficient
    
    #y_vec = data_complete[channel, :] - data_atm_blank[channel, :]#T_meas - (1 - a_vec)*T_P_ATM    #y vec containing the source signal, photon and TLS noise
    y_vec = a_vec * temp_source[channel]  + data_tls_blank[channel, :] + photon_noise[channel, :]     #AANPASSEN NAAR vec_a times x (gt)
        
    freqs = fftfreq(num_times, d=dt)
    S_nn = np.zeros_like(freqs)
    
    nonzero_mask = freqs != 0
    S_nn[nonzero_mask] = alpha * np.abs(freqs[nonzero_mask])**-beta + C #Is n in the model, photon and TLS noise
    S_nn[~nonzero_mask] = C
    
    
    
    W = 1.0 / np.sqrt(S_nn) #Rn ^-1/2, Snn = F{Rn}, Rn = F^-1{Snn}
    y_vec_tilde = np.real(ifft(W * fft(y_vec))) #Whitening(?)
    a_vec_tilde = np.real(ifft(W * fft(a_vec))) #Whitening(?)
    
    
    x_opt = np.dot(a_vec_tilde, y_vec_tilde) / np.dot(a_vec_tilde, a_vec_tilde) #x optimal
    
    x_opt_channels[channel] = x_opt


    #varience = 1/sum(a_tile)
    sum_a_vec_tilde = np.sum(a_vec_tilde**2)
    variance_channels[channel] = (1/ sum_a_vec_tilde)
    noise_level_channels[channel] = np.sqrt(variance_channels[channel])
    
    np.save("x_optimal.npy", x_opt_channels )
    np.save("varience_channels.npy", variance_channels)
    np.save("noise_level_channels", noise_level_channels)

