import numpy as np





def generate_data(num_samples, L_d, L_q, psi_f):

    psi_d = np.random.uniform(-0.5, 0.5, num_samples)
    psi_q = np.random.uniform(-0.5, 0.5, num_samples)
    theta_m = np.random.uniform(0, 2 * np.pi, num_samples)

    i_d = (psi_d - psi_f) / L_d
    i_q = psi_q / L_q
    tau_m = 1.5 * (psi_d * i_q - psi_q * i_d)

    return np.stack([psi_d, psi_q, theta_m], axis=1), np.stack([i_d, i_q, tau_m], axis=1)