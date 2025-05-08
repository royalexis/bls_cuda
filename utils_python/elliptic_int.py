import numpy as np

def ellK(k):
    """
    Computes the elliptic integral of the first kind (Hasting's approximation)
    """
    m1 = 1 - k*k
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2 = (b0+m1*(b1+m1*(b2+m1*(b3+m1*b4)))) * np.log(m1)

    return ek1 - ek2

def ellE(k):
    """
    Computes the elliptic integral of the second kind (Hasting's approximation)
    """

    m1 = 1 - k*k
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    
    ee1 = 1 + m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2 = m1*(b1+m1*(b2+m1*(b3+m1*b4))) * np.log(m1)
    
    return ee1 - ee2

# import matplotlib.pyplot as plt
# import mpmath as mm

# x = np.linspace(0, 0.9, 100)
# y = np.zeros(len(x))
# y_test = np.zeros(len(x))
# for i, x_ in enumerate(x):
#     y[i] = mm.ellippi(0.25, x_)
#     y_test[i] = mm.ellipe(x_)/(1 - 0.25)

# plt.plot(x, y)
# plt.plot(x, y_test)
# plt.show()