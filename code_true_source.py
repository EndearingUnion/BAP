import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py


with open('source.pkl', 'rb') as f:
    source_dict = pickle.load(f)

f.close()

with h5py.File("blank_atm_only.h5", "r") as f:  
    data_atm_blank = f["SPAXEL0"]["data"][...]
    frequencies_atm_blank = f["OBSATTRS"]["frequencies"][...]
    times_atm_blank = f["OBSATTRS"]["times"][...]

f.close()


with h5py.File("source_atm_only.h5", "r") as f:  
    data_atm_source = f["SPAXEL0"]["data"][...]
    frequencies_atm_source = f["OBSATTRS"]["frequencies"][...]
    times_atm_source = f["OBSATTRS"]["times"][...] 
f.close()


with h5py.File("blank_tls_only.h5", "r") as f:  
    data_tls_blank = f["SPAXEL0"]["data"][...]
    frequencies_tls_blank = f["OBSATTRS"]["frequencies"][...]
    times_tls_blank = f["OBSATTRS"]["times"][...]

f.close()


with h5py.File("source_tls_only.h5", "r") as f:  
    data_tls_source = f["SPAXEL0"]["data"][...]
    frequencies_tls_source = f["OBSATTRS"]["frequencies"][...]
    times_tls_source = f["OBSATTRS"]["times"][...] 
f.close()








print(source_dict.keys())

frequencies_source = source_dict['frequencies (Hz)']
t_source = source_dict['source temperature (K)']

x_opt = np.load("x_optimal.npy")
x_opt_switch = np.load("x_optimal_switching.npy")

#gt_source_switch = (data_tls_source - data_tls_blank).mean(axis=1)
#gt_source_switch = (data_atm_source - data_atm_blank).mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(frequencies_source / 1e9, t_source, label="Ground Truth ($x$)", color="black", linewidth=1)
#plt.plot(frequencies_source / 1e9, x_opt, label="Reconstructed Source", color="red", alpha = 0.7, linewidth=1)

#position switching
plt.plot(frequencies_source / 1e9, x_opt_switch, label="Reconstructed Source Position Switching", color="green", linewidth=1, linestyle = '--')
#plt.plot(frequencies_source / 1e9, data_tls_blank, label="TLS Noise", color="red", alpha = 0.7, linewidth=1 , linestyle = '--')


plt.xlabel("Frequency (GHz)")
plt.ylabel("Source Temperature $T_A^*$ (Kelvin)")
plt.title("Ground Truth Source")
plt.legend()
plt.grid(True)
plt.show()


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# ax1.plot(frequencies_source / 1e9, t_source, label="Ground Truth ($x$)", color="black", linewidth=1)
# ax1.plot(frequencies_source / 1e9, x_opt, label="Reconstructed Source", color="red", alpha=0.7, linewidth=1)

# ax1.set_xlabel("Frequency (GHz)")
# ax1.set_ylabel("Source Temperature $T_A^*$ (Kelvin)")
# ax1.set_title("Complete Source Observation Reconstruction")
# ax1.legend()
# ax1.grid(True)

# #ax2.plot(frequencies_atm_source / 1e9, gt_source_switch, label="Ground Truth ($x$)", color="black", linewidth=1)
# ax2.plot(frequencies_atm_source / 1e9, x_opt_switch, label="Reconstructed Source", color="red", alpha=0.7, linewidth=1)


# ax2.set_xlabel("Frequency (GHz)")
# ax2.set_title("Position Switching Reconstruction")
# ax2.legend()
# ax2.grid(True)


# plt.tight_layout()
# plt.show()