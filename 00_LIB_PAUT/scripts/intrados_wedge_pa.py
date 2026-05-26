import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PA intrados
t = np.linspace(-50, 150, 1000)
x_pa = t
y_pa = 0.001808912565*t**2 - 0.1314751414*t + 17.33687396
z_pa = -0.003862177072*t**2 + 0.1391531218*t + 276.4371305

# Intrados
def intrados(x):
    return 0.001808914648*x**2 - 0.1314763917*x + 17.33694558

x_int = np.linspace(-50, 150, 100)
y_int = intrados(x_int)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_int, np.zeros_like(x_int), y_int, label='Intrados')
ax.plot(x_pa, y_pa, z_pa, label='PA Intrados', color='blue')
ax.view_init(elev=20, azim=-60)
plt.show()
