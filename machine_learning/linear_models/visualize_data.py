#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X1 = np.random.rand(200) * 10
X2 = X1 + np.random.normal(0, 0.05, 200)
X3 = np.random.rand(200) * 5
X = np.column_stack([X1, X2, X3])
y = 4*X1 + 3*X3 + np.random.normal(0, 5, 200)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X1, X2, X3, c='blue', alpha=0.7, edgecolor='k')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('3D Scatter Plot of Features X1, X2, and X3')

plt.show()