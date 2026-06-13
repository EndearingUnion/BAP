import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import butter, buttord, convolve, firwin, freqz, lfilter, kaiserord, iirdesign, welch, wiener
from scipy.signal import spectrogram, impulse, correlate

import h5py

#Reads atmospheric noise only dataset
with h5py.File("blank_atm_only.h5", "r") as f:  
    data_atm_blank = f["SPAXEL0"]["data"][...]
    frequencies_atm_blank = f["OBSATTRS"]["frequencies"][...]
    times_atm_blank = f["OBSATTRS"]["times"][...]

f.close()

T_P_ATM = 273                           #T_p,atm Kelvin


num_channels = data_atm_blank.shape[0]
num_times = len(times_atm_blank)

print("channels number =", num_channels)

eta_atm = np.zeros((num_channels, num_times)) #Row = channel, collumn = time




for channel in range(num_channels):
    eta_atm[channel, :] = 1 - (data_atm_blank[channel, :] / T_P_ATM)      #eta_atm = 1-T_sky/T_p,atm


np.save("eta_atmosphere.npy", eta_atm )                                     #Saves eta_atm for each point in time




