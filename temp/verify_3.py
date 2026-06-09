import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("=== EXAM 3 ===")

# Q1 OLS by hand
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([3, 5, 6, 9, 11, 13])

x_bar = np.mean(X)
y_bar = np.mean(y)
print("x_bar:", x_bar, "y_bar:", y_bar)

S_xy = np.sum((X - x_bar) * (y - y_bar))
S_xx = np.sum((X - x_bar)**2)
print("S_xy:", S_xy, "S_xx:", S_xx)

b1 = S_xy / S_xx
b0 = y_bar - b1 * x_bar
print("b1:", b1, "b0:", b0)

pred = b0 + b1 * 8
print("y(8):", pred)

y_hat = b0 + b1 * X
e = y - y_hat
print("RSS:", np.sum(e**2))
print("sum(e):", np.sum(e))

# Q3 Naive Bayes
# Data sizes: 8 samples.
# Features: BP, Chol, Smoke, Heart
# 1: High, High, Yes, Yes
# 2: High, Norm, No, No
# 3: Norm, High, Yes, Yes
# 4: Norm, Norm, No, No
# 5: High, High, No, Yes
# 6: Norm, Norm, Yes, No
# 7: High, High, Yes, Yes
# 8: Norm, High, No, No

# counts
n = 8
n_yes = 4
n_no = 4

p_yes = 4/8
p_no = 4/8

# High BP | Yes: (1, 5, 7) -> 3. Total Yes = 4
# High BP | No: (2) -> 1. Total No = 4
# High Chol | Yes: (1, 3, 5, 7) -> 4.
# High Chol | No: (8) -> 1.
# No Smoke | Yes: (5) -> 1.
# No Smoke | No: (2, 4, 8) -> 3.

# with Laplace alpha=1, V=2 for each feature
p_bp_y = (3 + 1) / (4 + 2)
p_bp_n = (1 + 1) / (4 + 2)

p_chol_y = (4 + 1) / (4 + 2)
p_chol_n = (1 + 1) / (4 + 2)

p_smk_y = (1 + 1) / (4 + 2)
p_smk_n = (3 + 1) / (4 + 2)

print("\nLaplace Yes:", p_bp_y, p_chol_y, p_smk_y)
print("Laplace No:", p_bp_n, p_chol_n, p_smk_n)

pos_y = p_yes * p_bp_y * p_chol_y * p_smk_y
pos_n = p_no * p_bp_n * p_chol_n * p_smk_n

print("Pos Yes:", pos_y)
print("Pos No:", pos_n)

# Q4 Ridge
# beta = (3.0, -1.5, 0.8)
b = np.array([3.0, -1.5, 0.8])
lam = 0.2
L = 0.15 + lam * np.sum(b**2)
print("\nRidge L:", L)

# Q6 K-means Centroid
# Points: (1,3), (2,5), (4,1), (5,3)
pts = np.array([[1,3], [2,5], [4,1], [5,3]])
mu = np.mean(pts, axis=0)
print("\nmu:", mu)
J = np.sum((pts - mu)**2)
print("J:", J)

