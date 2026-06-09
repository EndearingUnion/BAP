import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq
import h5py, pickle

with open('source.pkl', 'rb') as f:
    source_dict = pickle.load(f)

frequencies_source = source_dict['frequencies (Hz)']
temp_source = source_dict['source temperature (K)']

with h5py.File("blank_atm_only.h5", "r") as f:
    data_atm_blank = f["SPAXEL0"]["data"][...]
    times_atm_blank = f["OBSATTRS"]["times"][...]

with h5py.File("blank_tls_only.h5", "r") as f:
    data_tls_blank = f["SPAXEL0"]["data"][...]

with h5py.File("blank_atm_plus_photon.h5", "r") as f:
    data_atm_plus_photon_blank = f["SPAXEL0"]["data"][...]
    times_atm_plus_phton_blank = f["OBSATTRS"]["times"][...]
    az_switched = f["OBSATTRS"]["az"][...]

eta_atm = np.load("eta_atmosphere.npy")
tls_noise_parameters = np.load("tls_wls_params_alg.npy")

num_channels = data_atm_blank.shape[0]
dt = np.mean(np.diff(times_atm_plus_phton_blank))

data_photon_tls_noise = data_tls_blank + data_atm_plus_photon_blank - data_atm_blank

samples_per_cycle = 12
num_cycles = data_photon_tls_noise.shape[1] // samples_per_cycle

x_opt_channels = np.zeros(num_channels)
variance_channels = np.zeros(num_channels)
noise_level_channels = np.zeros(num_channels)
snr_channels = np.zeros(num_channels)

t0 = 6 * dt 
scaling_factor = 0

# Separate the tracking vectors into continuous blocks of length num_cycles
z_store = np.zeros((300, num_cycles))
a_store = np.zeros((300, num_cycles))

for channel in range(300):

    alpha = tls_noise_parameters[channel, 0]
    beta = tls_noise_parameters[channel, 1]
    C = tls_noise_parameters[channel, 2]

    a_vec = eta_atm[channel, :].copy()
    n_vec = data_photon_tls_noise[channel]
    atm_vec = data_atm_blank[channel]

    y_on_series = np.zeros(num_cycles)
    y_off_series = np.zeros(num_cycles)
    a_on_series = np.zeros(num_cycles)

    for i in range(num_cycles):
        start_on_source = i * samples_per_cycle
        end_on_source = start_on_source + 6
        start_off_source = end_on_source
        end_off_source = start_off_source + 6
        
        atm_on_mean  = np.mean(atm_vec[start_on_source:end_on_source])    
        atm_off_mean = np.mean(atm_vec[start_off_source:end_off_source])
        
        y_on_series[i] = np.mean((a_vec[start_on_source:end_on_source] * temp_source[channel]) + n_vec[start_on_source:end_on_source]) + scaling_factor * atm_on_mean
        y_off_series[i] = np.mean(n_vec[start_off_source:end_off_source]) + scaling_factor * atm_off_mean
        
        a_on_series[i] = np.mean(a_vec[start_on_source:end_on_source])
        
    # Construct a physically continuous chopped stream
    y_stream = y_on_series - y_off_series
    a_stream = a_on_series

    fs = 1 / dt
    fs_downsampled = 1 / (dt * samples_per_cycle)
    
    N_z = len(y_stream)
    f_bins = fftfreq(N_z, d = 1 / fs_downsampled)
    
    def s_nn_mod(f_bin):
        f_new = np.where(f_bin == 0, 1e-6, f_bin)   
        estimated_noise = (alpha / (np.abs(f_new)**beta)) + C   
        transfer_function = 4 * (np.sin(np.pi * f_new * t0)**2)     
        return transfer_function * estimated_noise
    
    s_nn_downsampled = 0.5 * (s_nn_mod(f_bins / 2.0) + s_nn_mod((f_bins / 2.0) - (fs / 2.0))) 
    s_nn_downsampled = s_nn_downsampled + C 
    
    
    delta_f = fs_downsampled / N_z
    W = (1 / np.sqrt(s_nn_downsampled)) * np.sqrt(delta_f)
    
    z_vec_tilde = np.real(ifft(W * fft(y_stream))) 
    a_vec_tilde = np.real(ifft(W * fft(a_stream))) 
    
    x_opt = np.mean(z_vec_tilde / a_vec_tilde) 
    x_opt_channels[channel] = x_opt

    z_store[channel] = z_vec_tilde
    a_store[channel] = a_vec_tilde
    
    sum_a_vec_tilde = np.sum(a_vec_tilde**2)
    variance_channels[channel] = 1.0 / sum_a_vec_tilde
    noise_level_channels[channel] = np.sqrt(variance_channels[channel])
    snr_channels[:300] = temp_source[:300] / np.where(noise_level_channels[:300] == 0, 1e-6, noise_level_channels[:300])

np.save("z_whitened_chopper.npy", z_store)
np.save("x_optimal_chopper_whitened.npy",   x_opt_channels)
np.save("a_vec_tilde_chopper_whitened.npy", a_store)
np.save("chopper_variance_channels_whitened.npy",    variance_channels)
np.save("chopper_noise_level_channels_whitened.npy", noise_level_channels)
np.save("chopper_snr_channels_whitened.npy",         snr_channels)