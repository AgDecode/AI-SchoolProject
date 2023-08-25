import numpy as np

np_encode_inputs = np.array([x for x in [1.0, -0.69491525, -0.58181818]])
np_encode_outputs = np.array([x for x in [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


print(sigmoid(-0.69491525))
