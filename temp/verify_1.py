import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("=== EXAM 1 ===")
# Q1
X = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]])
y = np.array([1, 3, 4, 6, 8])
print("X^T X:\n", X.T @ X)
print("X^T y:", X.T @ y)
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Beta:", beta)
y_pred = X @ beta
print("y_pred:", y_pred)
e = y - y_pred
print("sum(e):", np.sum(e))
print("sum(x*e):", np.sum(X[:,1]*e))

# Q2
X_log = np.array([
    [1, 1.0, 0.5],
    [1, -0.5, 1.0],
    [1, 2.0, 1.5],
    [1, 0.0, 0.0],
    [1, 1.5, -1.0],
    [1, -1.0, 2.0],
    [1, 3.0, 0.5],
    [1, 0.5, 1.0]
])
y_log = np.array([1, 0, 1, 0, 0, 1, 1, 1])
beta_log = np.array([0.5, -1.2, 0.8])
z = X_log @ beta_log
p = sigmoid(z)
pred = (p >= 0.5).astype(int)
print("\nz:", z)
print("p:", p)
print("pred:", pred)
print("y:", y_log)
loss = -y_log * np.log(p) - (1 - y_log) * np.log(1 - p)
print("loss:", loss)
print("mean loss:", np.mean(loss))
print("accuracy:", np.mean(pred == y_log))

# Q4
x = np.array([0.5, -1.0, 2.0])
W1 = np.array([[0.3, -0.2, 0.5], [0.1, 0.4, -0.3]])
b1 = np.array([0.1, -0.1])
z1 = W1 @ x + b1
a1 = np.maximum(0, z1)
print("\nz1:", z1)
print("a1:", a1)
W2 = np.array([0.7, -0.5])
b2 = 0.2
z2 = W2 @ a1 + b2
p2 = sigmoid(z2)
print("z2:", z2)
print("p2:", p2)
loss_nn = -np.log(p2)
print("loss_nn:", loss_nn)
dz2 = p2 - 1
print("dz2:", dz2)
dW2 = dz2 * a1
db2 = dz2
print("dW2:", dW2)
print("db2:", db2)

# Q6
points = np.array([
    [1, 2], [2, 1], [2, 3],
    [8, 7], [9, 8], [8, 9]
])
mu1 = np.array([1, 2])
mu2 = np.array([9, 8])

def dist2(p, m):
    return np.sum((p - m)**2)

for i, p in enumerate(points):
    print(f"Point {i+1} dist to mu1:", dist2(p, mu1), "mu2:", dist2(p, mu2))

c1 = points[:3]
c2 = points[3:]
J = (sum(dist2(p, mu1) for p in c1) + sum(dist2(p, mu2) for p in c2)) / 6
print("J0:", J)
print("new mu1:", np.mean(c1, axis=0))
print("new mu2:", np.mean(c2, axis=0))

