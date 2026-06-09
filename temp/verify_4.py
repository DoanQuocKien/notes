import numpy as np

print("=== EXAM 4 ===")
# Q2 OLS
X = np.array([
    [1, 1, 2],
    [1, 2, 1],
    [1, 3, 3],
    [1, 4, 2],
    [1, 5, 4]
])
y = np.array([10, 12, 18, 20, 28])
print("X^T X:\n", X.T @ X)
print("X^T y:", X.T @ y)

beta = np.linalg.inv(X.T @ X) @ X.T @ y
print("beta:", beta)
pred = np.array([1, 3, 5]) @ beta
print("pred:", pred)

# Q4 Decision Tree Regression
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.5, 3.0, 4.5, 4.0, 8.0, 7.5, 9.0, 10.0])

mean_y = np.mean(y)
sse_total = np.sum((y - mean_y)**2)
print("\nsse total:", sse_total)

# split x <= 4
left = y[x <= 4]
right = y[x > 4]
ml = np.mean(left)
mr = np.mean(right)
ssel = np.sum((left - ml)**2)
sser = np.sum((right - mr)**2)
print("split <= 4: ml", ml, "mr", mr, "ssel", ssel, "sser", sser, "tot", ssel+sser)

# split x <= 3
left = y[x <= 3]
right = y[x > 3]
ml = np.mean(left)
mr = np.mean(right)
ssel = np.sum((left - ml)**2)
sser = np.sum((right - mr)**2)
print("split <= 3: ml", ml, "mr", mr, "ssel", ssel, "sser", sser, "tot", ssel+sser)

# Q5 Backprop 2-3-1
def sig(x): return 1/(1+np.exp(-x))
x_in = np.array([1, -1])
W1 = np.array([[0.2, -0.3], [0.4, 0.1], [-0.1, 0.5]])
z1 = W1 @ x_in
a1 = sig(z1)
print("\nz1:", z1)
print("a1:", a1)

W2 = np.array([0.5, -0.3, 0.4])
z2 = W2 @ a1
y_hat = sig(z2)
print("z2:", z2)
print("y_hat:", y_hat)

loss = -np.log(y_hat)
print("loss:", loss)

dz2 = y_hat - 1
print("dz2:", dz2)

dW2 = dz2 * a1
db2 = dz2
print("dW2:", dW2)

# Q6 GD with Ridge
x_gd = np.array([1, 2, 3, 4])
y_gd = np.array([3, 5, 7, 9])
beta = np.array([0.0, 0.0])

# ols grad at 0: -X^T y / n or sum? 
# Usually loss is 1/2n sum(e^2) or 1/n sum(e^2)
# Exam 04 says "loss OLS" "Ridge penalty \lambda ||\beta||^2"
grad_ols = np.array([-np.sum(y_gd)/4, -np.sum(x_gd * y_gd)/4])
print("\ngrad ols:", grad_ols)
# Wait, formula is X^T(Xb - y) / n? Yes. If sum, it's without /n. 
print("sum y:", np.sum(y_gd))
print("sum xy:", np.sum(x_gd * y_gd))

