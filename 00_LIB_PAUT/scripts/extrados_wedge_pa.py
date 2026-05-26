import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PA
t = np.linspace(-30, 110, 1000)
x_pa = t
y_pa = -0.003609426882*t**2 + 0.3901521821*t + 19.70318543
z_pa = -0.005488853532*t**2 + 0.4063539903*t + 322.5637160

# Extrados
def extrados(x):
    return 0.001808914648*x**2 - 0.1314763917*x + 17.33694558

x_ext = np.linspace(-35, 115, 100)
y_ext = extrados(x_ext)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_ext, np.zeros_like(x_ext), y_ext, label='Extrados')
ax.plot(x_pa, y_pa, z_pa, label='PA', color='blue')
ax.view_init(elev=20, azim=-60)
plt.show()
