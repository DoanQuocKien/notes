import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))

print("=== EXAM 5 ===")
# Q1 Logistic
X = np.array([
    [1, 2, 1],
    [1, 0, 2],
    [1, 1, 0],
    [1, 3, 2],
    [1, -1, 1],
    [1, 1, 3]
])
y = np.array([1, 0, 1, 1, 0, 0])
beta = np.array([-0.5, 1.0, -0.8])

z = X @ beta
p = sigmoid(z)
pred = (p >= 0.5).astype(int)
print("z:", z)
print("p:", p)
print("pred:", pred)
print("y:", y)
loss = -y * np.log(p) - (1-y) * np.log(1-p)
print("loss:", loss)
print("mean loss:", np.mean(loss))
print("acc:", np.mean(pred == y))

grad = X.T @ (p - y) / 6
print("\ngrad:", grad)
new_beta = beta - 0.5 * grad
print("new_beta:", new_beta)

# Q3 Decision Tree Entropy vs Gini
# A: [0, 0, 0, 1, 1, 1, 1, 0] -> 0: 4, 1: 4
# B: [0, 1, 0, 0, 1, 1, 0, 1]
# Y: [-, -, -, +, +, +, +, +] -> -: 3, +: 5
import math
def ent(pos, neg):
    t = pos+neg
    if t==0: return 0
    p = pos/t
    n = neg/t
    return - (p*math.log2(p) if p>0 else 0) - (n*math.log2(n) if n>0 else 0)

def gini(pos, neg):
    t = pos+neg
    if t==0: return 0
    return 1 - (pos/t)**2 - (neg/t)**2

e_root = ent(5, 3)
g_root = gini(5, 3)
print("\nRoot ent:", e_root, "gini:", g_root)

# A=0: (0,0,-), (0,1,-), (0,0,-), (0,1,+) -> pos=1, neg=3
# A=1: (1,0,+), (1,1,+), (1,1,+), (1,0,+) -> pos=4, neg=0
e_A = 4/8 * ent(1,3) + 4/8 * ent(4,0)
g_A = 4/8 * gini(1,3) + 4/8 * gini(4,0)
print("A split ent:", e_A, "gini:", g_A)
print("Gain A ent:", e_root - e_A, "gini:", g_root - g_A)

# B=0: (0,0,-), (0,0,-), (1,0,+), (1,0,+) -> pos=2, neg=2
# B=1: (0,1,-), (1,1,+), (1,1,+), (0,1,+) -> pos=3, neg=1
e_B = 4/8 * ent(2,2) + 4/8 * ent(3,1)
print("B split ent:", e_B)
print("Gain B ent:", e_root - e_B)

# Q4 Batch Norm
z_bn = np.array([2.0, 4.0, 6.0, 8.0])
mu = np.mean(z_bn)
var = np.var(z_bn) # N denominator!
print("\nBN mu:", mu, "var:", var)
z_hat = (z_bn - mu) / np.sqrt(var + 1e-8)
print("z_hat:", z_hat)
z_tilde = 2 * z_hat + 1
print("z_tilde:", z_tilde)

# Q5 K-means++
x_km = np.array([1, 3, 5, 15, 17])
mu1 = 1
D2 = (x_km - mu1)**2
print("\nD^2:", D2)
P = D2 / np.sum(D2)
print("P:", P)

mu2 = 17
D2_1 = (x_km - mu1)**2
D2_2 = (x_km - mu2)**2
D2_min = np.minimum(D2_1, D2_2)
print("new D^2:", D2_min)
P2 = D2_min / np.sum(D2_min)
print("new P:", P2)
