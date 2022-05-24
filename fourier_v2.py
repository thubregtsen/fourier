# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: myenv
#     language: python
#     name: myenv
# ---

# +
# imports

import pennylane as qml
from pennylane import numpy as np
from scipy.stats import unitary_group
from scipy.fftpack import fft
import matplotlib # for setting the rcparam
import matplotlib.pyplot as plt # for regular use



# +
# define our circuit

dev1 = qml.device("default.qubit", wires=1)

@qml.qnode(dev1, diff_method="backprop")
def circuit(params):
    qml.QubitUnitary(U1, wires=0)
    qml.RX(params[0], wires=0)
    qml.QubitUnitary(U2, wires=0)
#    qml.RX(params[0], wires=0)
#    qml.QubitUnitary(U3, wires=0)
#    qml.RX(params[0], wires=0)
#    qml.QubitUnitary(U4, wires=0)
    return qml.expval(qml.PauliZ(0))



# +
# run the simulation

repetitions = 100 # number of experiments to average over
plot_intermediate = False # whether or not we want plotting

# to save a historic log
log_real = []
log_imag = []
log_norm = []

for rep in range(repetitions):
    # sample new unitaries for every experiment
    U1 = unitary_group.rvs(2)
    U2 = unitary_group.rvs(2)
    U3 = unitary_group.rvs(2)
    U4 = unitary_group.rvs(2)

    resolution = 100 # number of sample points in every experiment to swipe the parameters
    results = []
    for i in range(resolution):
        params = [i * 2 * np.pi / resolution] # update the parameter value in the swipe
        results.append(circuit(params)) # run the circuit

    # plot the swipe if desired
    if True == plot_intermediate:
        plt.plot(results)
        plt.show()

    # calculate the fft
#    N = resolution
#    T = 1.0 / N
    xf = np.arange(-resolution//2, resolution//2)
    yf = np.fft.fftshift(fft(results))/ (resolution//2)

    # append to the historic log
    log_real.append(np.real(yf))
    log_imag.append(np.imag(yf))
    log_norm.append([np.linalg.norm(x) for x in yf])

    # plot if desired
    if True == plot_intermediate:
        plt.plot(xf, log_real[-1], label="real")
        plt.plot(xf, log_imag[-1], label="imag")
        plt.plot(xf, log_norm[-1], label="norm")

        x_res = 10
        plt.xticks(np.arange(-x_res, x_res+1, 1))
        plt.xlim(-x_res, x_res)
        plt.legend()
        plt.grid()
        plt.show()


# +
# Calculate and plot averages

matplotlib.rcParams["figure.dpi"] = 200 # plot resolution in jupyter notebooks

# plot the average data using error bars depicting variance
plt.errorbar(xf, np.average(log_real, axis=0), yerr=np.var(log_real), fmt=".", label="real", color="blue")
plt.errorbar(xf, np.average(log_imag, axis=0), yerr=np.var(log_imag), fmt=".", label="imag", color="orange")
plt.errorbar(xf, np.average(log_norm, axis=0), yerr=np.var(log_norm), fmt=".", label="norm", color="green")

x_res = 10
plt.xticks(np.arange(-x_res, x_res+1, 1))
plt.xlim(-x_res, x_res)
plt.legend()
plt.grid()
plt.show()


