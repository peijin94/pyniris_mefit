import numpy as np

import numba as nb

@nb.jit
def ch_voigt(a, u):
    # Constants for the calculation
    a0 = 122.607931777104326
    a1 = 214.382388694706425
    a2 = 181.928533092181549
    a3 = 93.155580458138441
    a4 = 30.180142196210589
    a5 = 5.912626209773153
    a6 = 0.564189583562615
    b0 = 122.60793177387535
    b1 = 352.730625110963558
    b2 = 457.334478783897737
    b3 = 348.703917719495792
    b4 = 170.354001821091472
    b5 = 53.992906912940207
    b6 = 10.479857114260399

    z = a-u*1j

    num=(((((a6*z+a5)*z+a4)*z+a3)*z+a2)*z+a1)*z+a0
    den =((((((z+b6)*z+b5)*z+b4)*z+b3)*z+b2)*z+b1)*z+b0
    fz=num/den

    vgt = fz.real
    dis = fz.imag

    # Derivatives calculation
    #numdz = ((((6*a6*z+5*a5)*z+4*a4)*z+3*a3)*z+2*a2)*z+a1
    #dendz =  (((((7*z+6*b6)*z+5*b5)*z+4*b4)*z+3*b3)*z+2*b2)*z+b1
    #fzdz = numdz / den - num * dendz / (den ** 2)
    #vgtda = fzdz.real
    #vgtdu = fzdz.imag
    #disda = vgtdu
    #disdu = -vgtda

    return vgt, dis #, vgtda, vgtdu, disda, disdu


@nb.njit
def niris_MEsinglet(x, B , theta ,chi ,eta0 ,dlambdaD ,a ,lambda0 ,B0 ,B1 ):
    geff = 3.0
    lambda_rest = 15648.5

    lambda_ = x
    v = (lambda_ - lambda0) / dlambdaD
    vb = geff * (4.67e-13 * lambda_rest ** 2 * B) / dlambdaD

    phib, psib = ch_voigt(a, v + vb)[0:2]
    phip, psip = ch_voigt(a, v)[0:2]
    phir, psir = ch_voigt(a, v - vb)[0:2]

    factor = 1. / np.sqrt(np.pi)
    phib *= factor
    psib *= factor
    phip *= factor
    psip *= factor
    phir *= factor
    psir *= factor

    st = np.sin(theta)
    st2 = st ** 2
    ct = np.cos(theta)

    etaI = 1 + 0.5 * eta0 * (phip * st2 + 0.5 * (phib + phir) * (1 + ct ** 2))
    etaQ = eta0 * 0.5 * (phip - 0.5 * (phib + phir)) * st2 * np.cos(2 * chi)
    etaU = eta0 * 0.5 * (phip - 0.5 * (phib + phir)) * st2 * np.sin(2 * chi)
    etaV = eta0 * 0.5 * (phir - phib) * ct

    rhoQ = eta0 * 0.5 * (psip - 0.5 * (psib + psir)) * st2 * np.cos(2 * chi)
    rhoU = eta0 * 0.5 * (psip - 0.5 * (psib + psir)) * st2 * np.sin(2 * chi)
    rhoV = eta0 * 0.5 * (psir - psib) * ct

    Delta = etaI ** 2 * (etaI ** 2 - etaQ ** 2 - etaU ** 2 - etaV ** 2 + rhoQ ** 2 + rhoU ** 2 + rhoV ** 2) - (etaQ * rhoQ + etaU * rhoU + etaV * rhoV) ** 2

    I = B0 + B1 * etaI * (etaI ** 2 + rhoQ ** 2 + rhoU ** 2 + rhoV ** 2) / Delta
    Q = -B1 * (etaI ** 2 * etaQ + etaI * (etaV * rhoU - etaU * rhoV) + rhoQ * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
    U = -B1 * (etaI ** 2 * etaU + etaI * (etaQ * rhoV - etaV * rhoQ) + rhoU * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
    V = -B1 * (etaI ** 2 * etaV + etaI * (etaU * rhoQ - etaQ * rhoU) + rhoV * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta

    # Creating the result array with the Stokes parameters
    f = np.concatenate((I, Q, U, V))

    return f

import numpy as np
from scipy.optimize import curve_fit

@nb.jit
def get_noise(data1d):
    # Perform convolution
    a = np.convolve(data1d, [-1, 2, -1], mode='valid')

    # Calculate and return the standard deviation
    return np.std(a) / np.sqrt(6)

def niris_mefit(x, data, par, function_to_fit):
    n = len(x) // 4

    # Assuming get_noise is a function that calculates the noise level
    weight = np.array([get_noise(data[:,i]) for i in range(4)])
    weight = 1.0 / (weight ** 2)

    par = np.array(par)
    # Parameter bounds and initial values
    lower_bounds = np.array([0, 0, 0, 1, 0, 1, -np.inf, 0, 0])- 1e-3
    upper_bounds = np.array([6000, np.pi, np.pi, 20, np.inf, 5, np.inf, np.inf, np.inf])+ 1e-3
    bounds = (lower_bounds, upper_bounds)


    par[np.where( par<lower_bounds)[0]] = lower_bounds[np.where( par<lower_bounds)[0]] + 1e-3
    par[np.where( par>upper_bounds)[0]] = upper_bounds[np.where( par>upper_bounds)[0]] - 1e-3

    dataIQUV = np.concatenate([data[:,i] for i in range(4)])
    result, covar = curve_fit(function_to_fit, x, dataIQUV
        , p0=par, bounds=bounds, maxfev=300, xtol=5e-5)

    return result

import numpy as np

@nb.jit
def niris_cogmag(wv, idata, vdata):
    geff = 3.0
    lambda_rest = 15648.5
    coeff = 4.67e-13 * lambda_rest**2 * geff

    weight1 = 1 - (idata + vdata)
    weight2 = 1 - (idata - vdata)


    wv2 = np.sum(weight1 * wv) / np.sum(weight1)
    wv1 = np.sum(weight2 * wv) / np.sum(weight2)

    B = (wv2 - wv1) / (2 * coeff)
    wlc = 0.5 * (wv2 + wv1)

    return B, wlc

# Example usage:
# wv = np.array([...])  # Wavelength array
# idata = np.array([...])  # Stokes I data
# vdata = np.array([...])  # Stokes V data
# B, wlc = niris_cogmag(wv, idata, vdata)


def init_par(x, data, dwl_line=1.5,dwl_core = 0.15, display=False):
    n = len(x) // 4

    # Calculate the continuum
    continuum = np.median(data[:][np.abs(x[:]) >= dwl_line])

    # Determine the line and core indices
    line_indices = np.where(np.abs(x[:]) <= dwl_line)
    core_indices = np.where(np.abs(x[:]) <= dwl_core)  # 
    # Assuming niris_cogmag is defined elsewhere
    blos, wlc = niris_cogmag(x[:][line_indices], data[:,0][line_indices] / continuum, data[:,3][line_indices] / continuum)
    blos = np.array(blos)

    # Core calculations
    icore = data[:,0][core_indices]
    qicore = data[:,1][core_indices] / icore
    uicore = data[:,2][core_indices] / icore
    btcoeff = 5000.
    bt = btcoeff * np.sqrt(np.sqrt(np.mean(qicore ** 2 + uicore ** 2)))
    
    # Magnetic field calculations
    Bfield = np.sqrt(blos ** 2 + bt ** 2)
    theta = np.arctan2(bt, blos)
    chi = 0.5 * np.median(np.arctan2(-uicore, -qicore) + 2 * np.pi * (np.arctan2(-uicore, -qicore) < 0))

    # Parameter calculations
    eta0 = 1.3
    dlambdaD = 0.15
    adamp = 1.0
    lambda0 = 0.0
    B1 = 2 * (continuum - np.min(icore))
    B0 = continuum - B1

    par = [Bfield, theta, chi, eta0, dlambdaD, adamp, lambda0, B0, B1]

    return par
