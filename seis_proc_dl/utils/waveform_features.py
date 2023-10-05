import numpy as np

class WaveformFeatures():

    @staticmethod
    def compute_snr(X, Y, T, num_channels, phase_type = "P", azims = None):
        """
        A simple SNR computation routine.  For z this computes the 10*log_10 of 
        the ratio of the expectations of the squared amplitudes.
        """
        n_rows = len(T)
        snrs = np.zeros(n_rows) - 999.0 
        
        z_ind = 0
        if num_channels > 1:
            z_ind = 2

        if (phase_type != "P" and azims is None):
            print("Warning azims is none - don't use these SNRs")
        for i in range(n_rows):
            # Noise window
            if (T[i] < 0):  
                continue
            y_idx = np.where(Y[i,:] == 1)[0]
            i0 = max(0, y_idx[0] - len(y_idx))
            i1 = y_idx[0]
            if (phase_type == "P"):
                x_signal = X[i, y_idx, z_ind] # Index 2 should be Z
                x_noise = X[i, i0:i1, z_ind]
                x_signal = x_signal - np.mean(x_signal)
                x_noise = x_noise - np.mean(x_noise)
            else:
                if (azims is None):
                    alpha = 0
                else:
                    alpha = azims[i]*np.pi/180

                n_signal = X[i, y_idx, 0]
                n_noise  = X[i, i0:i1, 0]
                e_signal = X[i, y_idx, 1]
                e_noise  = X[i, i0:i1, 1]

                # transverse signal
                t_signal = -n_signal*np.sin(alpha) + e_signal*np.cos(alpha)
                t_noise  = -n_noise*np.sin(alpha)  + e_noise*np.cos(alpha)
                x_signal = t_signal - np.mean(t_signal)
                x_noise = t_noise - np.mean(t_noise)
                
            s2 = np.multiply(x_signal, x_signal) # element-wise
            n2 = np.multiply(x_noise, x_noise) # element-wise
            snr = np.mean(s2)/np.mean(n2) # Ratio of exepctations
            snr = 10*np.log10(snr)
            snrs[i] = snr
        return snrs
