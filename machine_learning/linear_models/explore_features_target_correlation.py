#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X1 = np.random.rand(200) * 10
X2 = X1 + np.random.normal(0, 0.05, 200)
X3 = np.random.rand(200) * 5
X = np.column_stack([X1, X2, X3])
y = 4*X1 + 3*X3 + np.random.normal(0, 5, 200)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X[:, 0], y, color='blue', alpha=0.6, edgecolors='k')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('y')
axes[0].set_title('X1 vs y')

axes[1].scatter(X[:, 1], y, color='green', alpha=0.6, edgecolors='k')
axes[1].set_xlabel('X2')
axes[1].set_ylabel('y')
axes[1].set_title('X2 vs y')

axes[2].scatter(X[:, 2], y, color='orange', alpha=0.6, edgecolors='k')
axes[2].set_xlabel('X3')
axes[2].set_ylabel('y')
axes[2].set_title('X3 vs y')

fig.suptitle('Exploring the Relationship Between Input Features (X1, X2, X3) and the Target Variable y', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()