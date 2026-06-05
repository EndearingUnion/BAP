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
for group in f:
    print(group)

with h5py.File("blank_tls_only.h5", "r") as f:
    print(list(f["OBSATTRS"].keys()))

with h5py.File("blank_tls_only.h5", "r") as f:
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

with h5py.File("blank_atm_plus_photon.h5", "r") as f:
    data_atm_plus_photon_blank = f["SPAXEL0"]["data"][...]
    frequencies_atm_plus_photon_blank = f["OBSATTRS"]["frequencies"][...]
    times_atm_plus_phton_blank = f["OBSATTRS"]["times"][...]
    az_switched = f["OBSATTRS"]["az"][...]
f.close()

eta_atm = np.load("eta_atmosphere.npy")
tls_noise_parameters = np.load("tls_wls_params_alg.npy")

print("Unique Azimuth values:", np.unique(az_switched))

T_P_ATM = 273

num_channels = data_atm_blank.shape[0]
num_times = len(times_atm_blank)
dt = np.mean(np.diff(times_atm_plus_phton_blank))

print("Shape eta_atm", eta_atm.shape)

data_photon_tls_noise = data_tls_blank + data_atm_plus_photon_blank - data_atm_blank

d_switch = np.mean(az_switched == 0)
print(f"Duty Cycle: {d_switch}")

samples_per_cycle = 12
num_cycles = data_photon_tls_noise.shape[1] // samples_per_cycle

x_opt_channels = np.zeros(num_channels)
variance_channels = np.zeros(num_channels)
noise_level_channels = np.zeros(num_channels)
snr_channels = np.zeros(num_channels)

z_store = np.zeros((300, num_cycles))
a_store = np.zeros((300, num_cycles))

for channel in range(300):

    alpha = tls_noise_parameters[channel, 0]
    beta = tls_noise_parameters[channel, 1]
    C = tls_noise_parameters[channel, 2]

    a_vec = eta_atm[channel, :].copy()
    n_vec = data_photon_tls_noise[channel]

    total_samples = num_cycles * samples_per_cycle

    y_on_source = np.zeros(num_cycles)
    y_off_source = np.zeros(num_cycles)
    a_on_source = np.zeros(num_cycles)

    for i in range(num_cycles):
        start_on_source = i * samples_per_cycle
        end_on_source = start_on_source + 6
        start_off_source = end_on_source
        end_off_source = start_off_source + 6

        y_on_source[i] = np.mean((a_vec[start_on_source:end_on_source] * temp_source[channel]) + n_vec[start_on_source:end_on_source])
        y_off_source[i] = np.mean(n_vec[start_off_source:end_off_source])
        a_on_source[i] = np.mean(a_vec[start_on_source:end_on_source])

    z_vec = y_on_source - y_off_source

    x_opt = np.dot(a_on_source, z_vec) / np.dot(a_on_source, a_on_source)
    x_opt_channels[channel] = x_opt

    z_store[channel] = z_vec
    a_store[channel] = a_on_source

residuals = z_store - a_store * temp_source[:300, np.newaxis]
sigma_n_sq = np.var(residuals, axis=1, ddof=1)
a_dot_a    = np.sum(a_store**2, axis=1)
variance_channels[:300] = sigma_n_sq / a_dot_a
noise_level_channels[:300] = np.sqrt(variance_channels[:300])
snr_channels[:300] = temp_source[:300] / np.where(noise_level_channels[:300] == 0, 1e-6, noise_level_channels[:300])

np.save("x_optimal_chopper.npy",   x_opt_channels)
np.save("chopper_variance_channels.npy",    variance_channels)
np.save("chopper_noise_level_channels.npy", noise_level_channels)
np.save("chopper_snr_channels.npy",         snr_channels)