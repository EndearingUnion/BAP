import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('source.pkl', 'rb') as f:
    source_dict = pickle.load(f)

frequencies_source = source_dict['frequencies (Hz)']
temp_source = source_dict['source temperature (K)']


num_channels = 300
freq_axis_ghz = frequencies_source[:num_channels] / 1e9


noise_cs = np.load("cs_noise_level_channels.npy")[:num_channels]
noise_psw = np.load("PSW_noise_level_channels.npy")[:num_channels]
noise_chopper = np.load("chopper_noise_level_channels.npy")[:num_channels]


snr_cs = np.load("cs_snr_channels.npy")[:num_channels]
snr_psw = np.load("PSW_snr_channels.npy")[:num_channels]
snr_chopper = np.load("chopper_snr_channels.npy")[:num_channels]


x_opt = np.load("x_optimal.npy")
x_opt_switch = np.load("x_optimal_switching.npy")
x_opt_chopper = np.load("x_optimal_chopper.npy")



#Performence metrics:
error_staring   = x_opt - temp_source
error_switching = x_opt_switch - temp_source
error_chopper   = x_opt_chopper - temp_source

print("--- MEAN BIAS (Kelvin) ---")
print(f"Constant Staring:   {np.mean(error_staring):.6f} K")
print(f"Position Switching: {np.mean(error_switching):.6f} K")
print(f"Chopper Method:     {np.mean(error_chopper):.6f} K")
print()


print("--- ROOT MEAN SQUARE ERROR (Kelvin) ---")
print(f"Constant Staring:   {np.sqrt(np.mean(error_staring**2)):.6f} K")
print(f"Position Switching: {np.sqrt(np.mean(error_switching**2)):.6f} K")
print(f"Chopper Method:     {np.sqrt(np.mean(error_chopper**2)):.6f} K")


# ratio_chopper = np.mean(noise_chopper)/np.mean(noise_cs)
# print("Ratio chopper =", ratio_chopper)


ratio_psw = np.mean(noise_psw)/np.mean(noise_cs)
print("Ratio psw =", ratio_psw)


plt.figure(figsize=(11, 5), dpi=100)

plt.plot(frequencies_source[:300] / 1e9, noise_cs[:300], color='Red', linestyle='--', linewidth=1, label='No Position Switching')
plt.plot(frequencies_source[:300] / 1e9, noise_psw[:300], color='Purple', linestyle='--',linewidth=1, label='Position Switching')
plt.plot(frequencies_source[:300] / 1e9, noise_chopper[:300], color='green', linestyle='--',linewidth=1, label='Chopper Method')

plt.title("Noise Level ($\sigma_x$) Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Frequencies (GHz)", fontsize=12)
plt.ylabel("Noise Level $\sigma_x$ (Kelvin)", fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()


plt.figure(figsize=(11, 5), dpi=100)

plt.plot(range(300) , noise_cs[:300], color='Red', linestyle='--', linewidth=1, label='No Position Switching')
plt.plot(range(300) , noise_psw[:300], color='Purple', linestyle='--',linewidth=1, label='Position Switching')
plt.plot(range(300) , noise_chopper[:300], color='Purple', linestyle='--',linewidth=1, label='Chopper Method')

plt.title("Noise Level ($\sigma_x$) Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Channels", fontsize=12)
plt.ylabel("Noise Level $\sigma_x$ (Kelvin)", fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()

