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

def ellPi(n, k):
    """
    Computes the elliptic integral of the third kind
    """
    return ellK(k) + n/3 * Rj(0, 1-k*k, 1, 1 - n)

def Rc(x, y, tol=0.04):
    """
    R_C function for calculating elliptic integrals
    """

    if y > 0:
        xt = x
        yt = y
        w = 1
    else:
        xt = x - y
        yt = -y
        w = np.sqrt(x)/np.sqrt(xt)

    A0 = 1/3 * (xt + 2*yt)
    ave = A0
    s = 1
    n = 0
    while abs(s) > tol:
        alamb = 2 * np.sqrt(xt) * np.sqrt(yt) + yt
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        ave = 1/4 * (ave + alamb)
        s = (yt - A0)/(ave*4**n)
        n += 1

    return w * (ave)**(-1/2) * (1 + s*s*(0.3 + s*(1/7 + s*(3/8 + s*9/22))))

def Rf(x, y, z, tol=0.08):
    """
    R_F function for calculating elliptic integrals
    """
    xt, yt, zt = x, y, z

    delx, dely, delz = 1, 1 ,1
    while max(abs(delx), abs(dely), abs(delz)) > tol:
        sqrtx = np.sqrt(xt)
        sqrty = np.sqrt(yt)
        sqrtz = np.sqrt(zt)
        alamb = sqrtx*(sqrty + sqrtz) + sqrty*sqrtz
        xt = 0.25*(xt + alamb)
        yt = 0.25*(yt + alamb)
        zt = 0.25*(zt + alamb)
        ave = 1/3*(xt + yt + zt)
        delx = (ave - xt)/ave
        dely = (ave - yt)/ave
        delz = (ave - zt)/ave

    e2 = delx*dely - delz**2
    e3 = delx*dely*delz
    return (1 + (1/24*e2 - 0.1 - 3/44*e3)*e2 + 1/14*e3) / np.sqrt(ave)

def Rj(x, y, z, p, tol=0.05):
    """
    R_J function for calculating elliptic integrals
    """

    sum = 0
    fac = 1

    if p > 0:
        xt = x
        yt = y
        zt = z
        pt = p
    else:
        xt = min(x,y,z)
        zt = max(x,y,z)
        yt = x + y + z - xt - zt
        a = 1 / (yt - p)
        b = a*(zt - yt)*(yt - xt)
        pt = yt + b
        rho = xt*zt/yt
        tau = p*pt/yt
        rcx = Rc(rho, tau)

    delx, dely, delz, delp = 1, 1, 1, 1
    while max(abs(delx), abs(dely), abs(delz), abs(delp)) > tol:
        sqrtx = np.sqrt(xt)
        sqrty = np.sqrt(yt)
        sqrtz = np.sqrt(zt)
        alamb = sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
        alpha = (pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz)**2
        beta = pt*(pt+alamb)**2
        sum = sum + fac*Rc(alpha, beta)
        fac = 0.25 * fac
        xt = 0.25*(xt + alamb)
        yt = 0.25*(yt + alamb)
        zt = 0.25*(zt + alamb)
        pt = 0.25*(pt + alamb)
        ave = 0.2*(xt + yt + zt + 2*pt)
        delx = (ave-xt)/ave
        dely = (ave-yt)/ave
        delz = (ave-zt)/ave
        delp = (ave-pt)/ave

    ea = delx*(dely + delz) + dely*delz
    eb = delx*dely*delz
    ec = delp**2
    ed = ea - 3*ec
    ee = eb + 2*delp*(ea - ec)

    rj = 3*sum + fac*(1 + ed*(-3/14 + 9/88*ed - 9/52*ee) + eb*(1/6 + 
            delp*(-3/11 + delp*3/26)) + delp*ea*(1/3 - delp*3/22) - 1/3*delp*ec) / (ave*np.sqrt(ave))
    
    if p < 0:
        rj = a * (b * rj + 3*(rcx - Rf(xt,yt,zt)))

    return rj

# import matplotlib.pyplot as plt
# import mpmath as mm

# x = np.linspace(0, 0.9, 100)
# y = np.zeros(len(x))
# y_test = np.zeros(len(x))
# for i, x_ in enumerate(x):
#     y[i] = mm.ellippi(0.25, x_*x_)
#     y_test[i] = ellPi(0.25, x_)

# plt.plot(x, y)
# plt.plot(x, y_test)
# plt.plot(x, y - y_test)
# plt.show()