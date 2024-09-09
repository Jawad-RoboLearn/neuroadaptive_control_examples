import numpy as np
import matplotlib.pyplot as plt
# total sim time
t = 20

# dt 
dt = 0.1

# init conditions
x = 0
xr = 0
what = np.zeros(6)

# nominal control law alpha
alp = 2;

# nominal adaptive law gamma
gamm = 50;

# leakage term
sig = 0.2


# for data record
x_rec = []
xr_rec = []
u_rec = []
unc_rec = []
t_rec = []
ua_rec = []
unc_ua = []
index = 0

for i in np.arange(0.0, t + dt, dt):
    uncertainty = 1 + x**2 + np.sin(x) * x**3 + np.cos(x) * x**4
    
    if i <=5:
        c = 1
    if i > 5 and i <= 10:
        c = -1
    if i > 10 and i <= 15:
        c = 1
    if i > 15:
        c = -1
    
    theta = np.array([
    np.exp(-0.25 * (abs(x - 5) ** 2)),
    np.exp(-0.25 * (abs(x - 3) ** 2)),
    np.exp(-0.25 * (abs(x - 1) ** 2)),
    np.exp(-0.25 * (abs(x + 1) ** 2)),
    np.exp(-0.25 * (abs(x + 3) ** 2)),
    np.exp(-0.25 * (abs(x + 5) ** 2))
    ])
    
    un = -alp * (x - c)
    ua = np.dot(-what, theta) 
    u = un + ua
    xr = xr + dt * (-alp * (xr - c))
    what = what + dt * (gamm * (theta * (x-xr) - sig * what))
    x = x + dt * (uncertainty + u)
    
    
    x_rec.append(x)
    xr_rec.append(xr)
    u_rec.append(u)
    unc_rec.append(uncertainty)  # Assuming delta is defined elsewhere
    t_rec.append(i)
    ua_rec.append(ua)
    unc_ua.append(uncertainty + ua)


# Create the figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# First subplot
axs[0].plot(t_rec, x_rec, 'g', linewidth=4, label = 'x')  # x data
axs[0].plot(t_rec, xr_rec, 'r', linewidth=4, label = 'x_r')  # xr data
axs[0].set_xlabel(r'$t$ (sec)', fontsize=16)
axs[0].set_ylabel(r'$x(t)$', fontsize=16)
axs[0].grid()
axs[0].axis('tight')
axs[0].legend()

# First subplot
axs[1].plot(t_rec, u_rec, 'g', linewidth=4, label = 'u')  # x data
axs[1].plot(t_rec, ua_rec, 'r', linewidth=4, label = 'ua')  # xr data
axs[1].set_xlabel(r'$t$ (sec)', fontsize=16)
axs[1].set_ylabel(r'$u(t)$', fontsize=16)
axs[1].grid()
axs[1].axis('tight')
axs[1].legend()

# First subplot
axs[2].plot(t_rec,  unc_ua, 'g', linewidth=4, label = 'sigma_hat')  # x data
axs[2].set_xlabel(r'$t$ (sec)', fontsize=16)
axs[2].set_ylabel(r'$u(t)$', fontsize=16)
axs[2].grid()
axs[2].axis('tight')
axs[2].legend()
