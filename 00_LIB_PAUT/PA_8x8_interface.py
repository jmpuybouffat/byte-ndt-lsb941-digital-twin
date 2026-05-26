import matplotlib.pyplot as plt
mport matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-30, 110, 100)
y = -0.003609404789*t**2 + 0.3901494991*t + 19.70324623
z = -0.005488828669*t**2 + 0.4063509956*t + 322.5637799

plt.plot(t, y, label='Y(t)')
plt.plot(t, z, label='Z(t)')
plt.xlabel('t')
plt.legend()
plt.show()

