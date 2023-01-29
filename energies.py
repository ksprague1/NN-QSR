import numpy as np
class TFIM():
    @staticmethod
    def ground(N,h,J):
        h=h/J
        Pn = np.pi/N*(np.arange(-N+1,N,2))
        E0 = -1/N*np.sum(np.sqrt(1+h**2-2*h*np.cos(Pn)))
        return E0*J
class Rydberg():
    E={16:-0.4534,36:-0.4221,64:-0.40522,144:-0.38852,256:-0.38052,576:-0.3724,1024:-0.3687,2304:-0.3645}
    Err = {16: 0.0001,36: 0.0005,64: 0.0002, 144: 0.0002, 256: 0.0002, 576: 0.0006,1024: 0.0007,2304: 0.0007}