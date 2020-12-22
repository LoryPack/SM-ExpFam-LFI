from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator

from src.functions import jacobian_second_order
from src.networks import createDefaultNNWithDerivatives

n_outputs = range(1, 21)
times_jac = []
times_forward = []

# if you have saved the files before and only want to create plots:
# times_forward = np.load("results/times_forward.npy")
# times_jac = np.load("results/times_jac.npy")

data = torch.randn(5000, 100)
data.requires_grad = True

# study time complexity wrt number outputs in the NN:

for out_size in n_outputs:
    print(out_size)
    net = createDefaultNNWithDerivatives(100, out_size)()
    start = time()
    y1 = net(data)
    f1, s1 = jacobian_second_order(data, y1)
    times_jac.append(time() - start)
    start = time()
    y2, f2, s2 = net.forward_and_derivatives(data)
    times_forward.append(time() - start)

print(times_jac)
print(times_forward)

if len(times_jac) > 0:
    np.save("results/times_jac", np.array(times_jac))
if len(times_forward) > 0:
    np.save("results/times_forward", np.array(times_forward))

fig, ax = plt.subplots(1, 1)
ax.plot(n_outputs, times_jac)
ax.set_title(r"Naive autodif")
ax.set_xlabel(r"Output size ($d_s$)")
ax.set_ylabel(r"Time ($s$)")
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.savefig("results/NN_derivative_time_autodiff.pdf", bbox_inches="tight")
plt.show()
plt.close()

fig, ax = plt.subplots(1, 1)
ax.plot(n_outputs, times_forward)
ax.set_title(r"Forward computation")
ax.set_xlabel(r"Output size ($d_s$)")
ax.set_ylabel(r"Time ($s$)")
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.savefig("results/NN_derivative_time_forward.pdf", bbox_inches="tight")
plt.show()
