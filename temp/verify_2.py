import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("=== EXAM 2 ===")
# Q2 GDA
X = np.array([1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0, 6.5])
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

mu0 = np.mean(X[Y==0])
mu1 = np.mean(X[Y==1])
print("mu0:", mu0, "mu1:", mu1)

var = np.mean((X[Y==0]-mu0)**2) * 0.5 + np.mean((X[Y==1]-mu1)**2) * 0.5
print("var (N div):", var)

var_unbiased = np.sum((X[Y==0]-mu0)**2) / 8 + np.sum((X[Y==1]-mu1)**2) / 8
print("var (N sum):", var_unbiased)

# Let's check calculation of variance in Exam 2 answer
sigma2 = (0.5**2 + 0 + 0.5**2)*2 / 8 
print("sigma2 my calc:", sigma2)

# Q3 Logistic Regression Gradient
X_log = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [1, 2, 1],
    [1, 1, 2],
    [1, 3, 1],
    [1, 0, 3]
])
y_log = np.array([1, 0, 1, 0, 1, 0])
p = np.array([0.5]*6)
error = p - y_log
print("\nerror:", error)
grad = X_log.T @ error / 6
print("grad:", grad)
beta = np.array([0,0,0]) - 0.5 * grad
print("beta:", beta)

# Q5 Backprop
x = np.array([2, -1])
W1 = np.array([[0.5, -0.3], [-0.2, 0.4]])
b1 = np.array([0.2, -0.1])
z1 = W1 @ x + b1
a1 = sigmoid(z1)
print("\nz1:", z1)
print("a1:", a1)

W2 = np.array([0.6, -0.4])
b2 = 0.3
z2 = W2 @ a1 + b2
y_hat = sigmoid(z2)
print("z2:", z2)
print("y_hat:", y_hat)

loss = 0.5 * (y_hat - 0)**2
print("loss:", loss)

d_o = (y_hat - 0) * y_hat * (1 - y_hat)
print("d_o:", d_o)

d_h = d_o * W2 * a1 * (1 - a1)
print("d_h:", d_h)

dW2 = d_o * a1
db2 = d_o
print("dW2:", dW2)

dW1 = np.outer(d_h, x)
print("dW1:\n", dW1)

# Q6 K-means
x_k = np.array([1, 2, 3, 10, 11, 12, 20, 21])
mu1, mu2 = 2.0, 11.0

# iter 1
d1 = (x_k - mu1)**2
d2 = (x_k - mu2)**2
c = np.where(d1 < d2, 0, 1)
print("\niter1 c:", c)
print("dist:", np.where(c==0, d1, d2))
J0 = np.mean(np.where(c==0, d1, d2))
print("J0:", J0)

mu1 = np.mean(x_k[c==0])
mu2 = np.mean(x_k[c==1])
print("new mu:", mu1, mu2)

# iter 2
d1 = (x_k - mu1)**2
d2 = (x_k - mu2)**2
c = np.where(d1 < d2, 0, 1)
print("iter2 c:", c)
print("dist:", np.where(c==0, d1, d2))
J1 = np.mean(np.where(c==0, d1, d2))
print("J1:", J1)
