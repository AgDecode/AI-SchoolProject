import numpy as np
import random
import pandas as pd

par = np.array([x for x in [1.0, -0.69491525, -0.58181818]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def signaltonoise(a, axis=0, ddof=0):
    """
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.
    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.
    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def ml_function(raw_array):
    # ========================== SNR array preparing ==========================
    snr_array = []  # np.zeros(par.max_snr_points)
    for i in range(0, len(raw_array) - par.disc_freq * par.snr_timewindow_sec, 1 * par.disc_freq):
        # snr_array[i - par.ioffset] = signaltonoise(raw_array[i - par.ioffset:i])
        snr_array.append(signaltonoise(raw_array[i:i + par.disc_freq * par.snr_timewindow_sec]))
        # print(signaltonoise(raw_array[i - 30:i]))
    # print("SNR:", snr_array)
    snr_array = np.array(snr_array)
    # ==========================

    # ========================== SNR normalization ==========================
    if (par.norm_type == 0):
        par.snr_max_value = max(np.max(snr_array), par.snr_max_value)
        par.snr_min_value = min(np.min(snr_array), par.snr_min_value)
        snr_array = (snr_array - par.snr_min_value) / (par.snr_max_value - par.snr_min_value)
        # print("renormed SNR:", snr_array)
    if (par.norm_type == 2):
        par.snr_max_value = max(np.max(snr_array), par.snr_max_value)
        par.snr_min_value = min(np.min(snr_array), par.snr_min_value)
        s = 0
        for i in range(len(snr_array)):
            s = s + snr_array[i]
        par.snr_mean_total_sum = par.snr_mean_total_sum + s
        par.snr_mean_total_num = par.snr_mean_total_num + len(snr_array)
        par.snr_mean_value = par.snr_mean_total_sum / par.snr_mean_total_num
        coef = min(par.snr_max_value - par.snr_mean_value, par.snr_mean_value - par.snr_min_value)
        # snr_array = (snr_array - par.snr_mean_value)/coef
        snr_array = (snr_array - 1.3669) / 0.43403017029525803
        # print(par.snr_mean_value, coef)
    # ==========================

    # ========================== INN layers reading from file ==========================
    dump = np.genfromtxt("matrix/W12.dat", delimiter='\t', dtype=np.float)
    ml_function.W12 = dump[:, :dump.shape[1] - 1]
    # print("W12", W12)
    ml_function.bias1 = 0

    dump = np.genfromtxt("matrix/W23.dat", delimiter='\t', dtype=np.float)
    ml_function.W23 = dump[:, :dump.shape[1] - 1]
    # print("W23", W23)
    ml_function.bias2 = 0

    dump = np.genfromtxt("matrix/W34.dat", delimiter='\t', dtype=np.float)
    ml_function.W34 = dump[:, :dump.shape[1] - 1]
    # print("W34", W34)
    ml_function.bias3 = 0

    dump = np.genfromtxt("matrix/W45.dat", delimiter='\t', dtype=np.float)
    ml_function.W45 = dump[:, :dump.shape[1] - 1]
    # print("W45", W45)
    ml_function.bias4 = 0

    dump = np.genfromtxt("matrix/W56.dat", delimiter='\t', dtype=np.float)
    ml_function.W56 = dump[:, :dump.shape[1] - 1]
    # print("W56", W56)

    dump = np.genfromtxt("matrix/bias6.dat", delimiter='\t', dtype=np.float)
    ml_function.bias5 = dump[:1]

    # print("bias 6", bias6)
    # ==========================

    # ========================== INN layers processing ==========================
    output_2_layer = sigmoid(np.array([snr_array]).dot(ml_function.W12) + ml_function.bias1)
    output_3_layer = sigmoid(output_2_layer.dot(ml_function.W23) + ml_function.bias2)
    output_4_layer = sigmoid(output_3_layer.dot(ml_function.W34) + ml_function.bias3)
    print(output_4_layer)
    output_5_layer = sigmoid(output_4_layer.dot(ml_function.W45) + ml_function.bias4)
    print(output_5_layer)
    output_final = (output_5_layer.dot(ml_function.W56) + ml_function.bias5)

    # print(output_2_layer.shape);
    # print(output_3_layer.shape);
    # print(output_4_layer.shape);
    # print(output_5_layer.shape);
    # print(output_final.shape);
    # print(output_final);
    # ==========================

    # smth = model.predict(np.array([snr_array]))
    # X_type = np.where(smth == np.amax(smth))
    # print ("output final:", output_final)

    return output_final


print(ml_function(par))
