import numpy as np
import matplotlib.pyplot as plt

y = np.zeros(20)+1


fig = plt.figure()
ax = fig.add_subplot(111)

n, bins, _ = ax.hist(y, color='#707070', bins=7, alpha=0.7, rwidth=0.85, density=True, align="left")

print(n)
print(bins)

plt.show()