import numpy as np
scaling_factors = np.load("tls_scaling_factors.npy") #import the scaling factors
wls_param = np.load("tls_wls_params.npy")
wls_param_knee = np.load("tls_wls_params_fknee.npy")
wls_param_alg = np.load("tls_wls_params_alg.npy")

#Noise estimation functions

#This chunk of code is taken from  www.socsci.ru.nl/wilberth/python/noise.html by drs. W.C.P. van Ham and altered to fit our project
def pink_spectrum(f, beta, f_min = 0, f_max = np.inf):
    """
    Define a pink (1/f) spectrum
        f     = array of frequencies
        f_min = minimum frequency for band pass
        f_max = maximum frequency for band pass
    """
    #Power = Amplitude^2, so Power ~ (f^-(beta/2))^2 = f^-beta 
    s = f**-(beta/2.0)  #To achieve the amplitude spectrum divide by 2 [W/√Hz]
    return s



#This chunk of code is based on the code from www.socsci.ru.nl/wilberth/python/noise.html by drs. W.C.P. van Ham and altered to fit our project


def tls_noise_estimation(n_samples, fs):
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)  #Real FFT for real-valued time signals
                                                #d is the interval between each sample
                                                #n_samples = len(f["SPAXEL0"]["data"][...])
    spectrum = np.zeros_like(freqs, dtype='complex')      # make complex numbers for spectrum
    spectrum[1:] = pink_spectrum(freqs[1:])               # get spectrum amplitude for all frequencies except f=0, to aviod div by 0
    phases = np.random.uniform(0, 2*np.pi, len(freqs)-1)  # random phases for all frequencies except f=0
    spectrum[1:] *= np.exp(1j*phases)                     # apply random phases to the amplitude
    noise = np.fft.irfft(spectrum, n=n_samples) # return the reverse fourier transform to get time series
    
    return noise




 #Channel bandwidth (NEeded to convert K to Watt using Johnson-Nyquist from literature source)
# #P = kTb telecommunications and sensing lecture 3 slide 10
# def channel_bandwidth(channel):
#     bw = np.diff(frequencies)
#     channel_bw = bw[channel]
#     #print("channel bandwidth = ", channel_bw)
#     return channel_bw




def wiener_filter(x_t, S_ss, S_nn):
    
    """
    Applies a Wiener filter to time-series data on input.
    
    X(f)H(f) = Y(f)
    
    H(f) = Wiener Filter
    Y(f) = desired signal
    
    x: time-domain signal (Atm + Photon + TLS)
    fs: sampling frequency
    S_ss: The psd of desired signal (photon + atmospheric noise)
    S_nn: The psd of noise (simulated TLS noise)
    """
    
    n = len(x_t)
    
    X_f = np.fft.rfft(x_t)                    #Transform input to frequency domain
    H_f = S_ss / (S_ss + S_nn + 1e-30)      #Wiener filter H(f) = Sss / (Sss + Snn)
    Y_f = X_f * H_f                         #Apply the Wiener filter in frequency domain
    y_t = np.fft.irfft(Y_f, n=n)            #Transform back to time domain
    
    return y_t



def tls_estimation(f_welch, channel, alpha, beta):         #Models the TLS based on alpha/f**-beta
    tls_model_psd = np.zeros_like(f_welch)
    tls_model_psd[1:] = f_welch[1:]**-(beta)        #Avoid first bin which is zero
    tls_model_scaled = alpha*tls_model_psd       #Scales the noise model
    
    return tls_model_scaled



#Weighted Least Squares functions
def power_law(f, alpha, beta):
    """Mathematical model for TLS noise: S(f) = alpha * f^-beta"""
    return (alpha * f**-beta) 

def tls_estimation_wls(f_grid, A_fit, beta_fit): #uses the param found via WLS to model the TLS noise
    tls_model_psd = np.zeros_like(f_grid)
    # Avoid 0 Hz
    tls_model_psd[1:] = power_law(f_grid[1:], A_fit, beta_fit)
    return tls_model_psd