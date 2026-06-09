
# **ĐÁP ÁN ĐỀ THI MACHINE LEARNING — ĐỀ 01**

---

## Câu 1. OLS Linear Regression bằng Ma trận

### 1.1 Ma trận thiết kế

$$\mathbf{X} = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\ 3 \\ 4 \\ 6 \\ 8 \end{pmatrix}$$

### 1.2 Tính $\mathbf{X}^T \mathbf{X}$ và $\mathbf{X}^T \mathbf{y}$

$$\mathbf{X}^T \mathbf{X} = \begin{pmatrix} 5 & 10 \\ 10 & 30 \end{pmatrix}$$

Chi tiết: $\sum 1 = 5$, $\sum x_i = 0+1+2+3+4 = 10$, $\sum x_i^2 = 0+1+4+9+16 = 30$.

$$\mathbf{X}^T \mathbf{y} = \begin{pmatrix} 22 \\ 57 \end{pmatrix}$$

Chi tiết: $\sum y_i = 1+3+4+6+8 = 22$, $\sum x_i y_i = 0+3+8+18+32 = 61$.

*Sửa: $\sum x_i y_i = 0(1) + 1(3) + 2(4) + 3(6) + 4(8) = 0+3+8+18+32 = 61$*

$$\mathbf{X}^T \mathbf{y} = \begin{pmatrix} 22 \\ 61 \end{pmatrix}$$

### 1.3 Giải Normal Equations

$$\begin{pmatrix} 5 & 10 \\ 10 & 30 \end{pmatrix} \begin{pmatrix} \beta_0 \\ \beta_1 \end{pmatrix} = \begin{pmatrix} 22 \\ 61 \end{pmatrix}$$

Từ PT1: $5\beta_0 + 10\beta_1 = 22$ → $\beta_0 = \frac{22 - 10\beta_1}{5}$

Từ PT2: $10\beta_0 + 30\beta_1 = 61$

Thay PT1 vào PT2:

$$10 \cdot \frac{22 - 10\beta_1}{5} + 30\beta_1 = 61$$

$$2(22 - 10\beta_1) + 30\beta_1 = 61$$

$$44 - 20\beta_1 + 30\beta_1 = 61$$

$$10\beta_1 = 17 \implies \beta_1 = 1.7$$

$$\beta_0 = \frac{22 - 10(1.7)}{5} = \frac{22 - 17}{5} = 1.0$$

**Mô hình:** $\hat{y} = 1.0 + 1.7x$

### 1.4 Dự đoán

$$\hat{y}(6) = 1.0 + 1.7(6) = 1.0 + 10.2 = 11.2$$

### 1.5 Kiểm tra

Tính residual $e_i = y_i - \hat{y}_i$:

| $x$ | $y$ | $\hat{y}$ | $e$ |
| :--- | :--- | :--- | :--- |
| 0 | 1 | 1.0 | 0.0 |
| 1 | 3 | 2.7 | 0.3 |
| 2 | 4 | 4.4 | -0.4 |
| 3 | 6 | 6.1 | -0.1 |
| 4 | 8 | 7.8 | 0.2 |

$\sum e_i = 0.0 + 0.3 - 0.4 - 0.1 + 0.2 = 0.0$ ✓

$\sum x_i e_i = 0(0) + 1(0.3) + 2(-0.4) + 3(-0.1) + 4(0.2) = 0 + 0.3 - 0.8 - 0.3 + 0.8 = 0.0$ ✓

---

## Câu 2. Logistic Regression — Thay số

$\boldsymbol{\beta} = (0.5, -1.2, 0.8)$, $z = 0.5 - 1.2x_1 + 0.8x_2$

| STT | $z$ | $\hat{p}$ | Lớp dự đoán | $y$ | Đúng? | Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | $0.5-1.2+0.4=-0.3$ | 0.426 | 0 | 1 | Sai | $-\log(0.426) = 0.854$ |
| 2 | $0.5+0.6+0.8=1.9$ | 0.870 | 1 | 0 | Sai | $-\log(1-0.870) = 2.040$ |
| 3 | $0.5-2.4+1.2=-0.7$ | 0.332 | 0 | 1 | Sai | $-\log(0.332) = 1.103$ |
| 4 | $0.5+0+0=0.5$ | 0.622 | 1 | 0 | Sai | $-\log(1-0.622) = 0.973$ |
| 5 | $0.5-1.8-0.8=-2.1$ | 0.109 | 0 | 0 | Đúng | $-\log(1-0.109) = 0.115$ |
| 6 | $0.5+1.2+1.6=3.3$ | 0.964 | 1 | 1 | Đúng | $-\log(0.964) = 0.037$ |
| 7 | $0.5-3.6+0.4=-2.7$ | 0.063 | 0 | 1 | Sai | $-\log(0.063) = 2.765$ |
| 8 | $0.5-0.6+0.8=0.7$ | 0.668 | 1 | 1 | Đúng | $-\log(0.668) = 0.404$ |

**Cross-entropy trung bình:**

$$\bar{J} = \frac{0.854 + 2.040 + 1.103 + 0.973 + 0.115 + 0.037 + 2.765 + 0.404}{8} = \frac{8.291}{8} \approx 1.036$$

**Accuracy:**

$$\text{Accuracy} = \frac{3}{8} = 0.375$$

**Nhận xét:** Mô hình dự đoán sai 5/8 mẫu, đặc biệt sai nghiêm trọng ở mẫu 2 (predict Spam nhưng thật Not Spam) và mẫu 7 (predict Not Spam nhưng thật Spam). Hệ số $\beta_1 = -1.2$ làm $x_1$ lớn kéo xác suất xuống, mâu thuẫn với dữ liệu ở mẫu 7 ($x_1=3, y=1$). Mô hình cần được huấn luyện lại.

---

## Câu 3. Exponential Family và GLM

### 3.1 Chuẩn hóa Bernoulli

$$P(y \mid \phi) = \phi^y (1-\phi)^{1-y} = \exp\left(y \log\phi + (1-y)\log(1-\phi)\right)$$

$$= \exp\left(y \log\frac{\phi}{1-\phi} + \log(1-\phi)\right)$$

Đối chiếu $P(y \mid \eta) = b(y)\exp(\eta T(y) - a(\eta))$:

- $\eta = \log\frac{\phi}{1-\phi}$ (log-odds)
- $T(y) = y$
- $a(\eta) = -\log(1-\phi) = \log(1 + e^\eta)$
- $b(y) = 1$

### 3.2 Chứng minh $\phi = \sigma(\eta)$

Từ $\eta = \log\frac{\phi}{1-\phi}$:

$$e^\eta = \frac{\phi}{1-\phi}$$

$$e^\eta (1-\phi) = \phi$$

$$e^\eta - e^\eta \phi = \phi$$

$$e^\eta = \phi(1 + e^\eta)$$

$$\phi = \frac{e^\eta}{1 + e^\eta} = \frac{1}{1 + e^{-\eta}} = \sigma(\eta) \quad \blacksquare$$

### 3.3 Dự đoán với 3 GLM

$\eta = \boldsymbol{\beta}^T \mathbf{x} = 0.3(1) + 1.5(2) + (-0.5)(3) = 0.3 + 3.0 - 1.5 = 1.8$

- **Gaussian GLM:** $\hat{y} = \eta = 1.8$
- **Bernoulli GLM:** $P(y=1) = \sigma(1.8) = \frac{1}{1+e^{-1.8}} \approx 0.858$
- **Poisson GLM:** $\hat{\lambda} = e^{1.8} \approx 6.050$

---

## Câu 4. Neural Network Forward Pass — Mạng 3-2-1

### 4.1 Tầng ẩn

$$z_1^{[1]} = 0.3(0.5) + (-0.2)(-1.0) + 0.5(2.0) + 0.1 = 0.15 + 0.2 + 1.0 + 0.1 = 1.45$$

$$z_2^{[1]} = 0.1(0.5) + 0.4(-1.0) + (-0.3)(2.0) + (-0.1) = 0.05 - 0.4 - 0.6 - 0.1 = -1.05$$

ReLU:

$$a_1^{[1]} = \max(0, 1.45) = 1.45$$

$$a_2^{[1]} = \max(0, -1.05) = 0$$

### 4.2 Tầng output

$$Z^{[2]} = 0.7(1.45) + (-0.5)(0) + 0.2 = 1.015 + 0 + 0.2 = 1.215$$

$$\hat{y} = \sigma(1.215) = \frac{1}{1 + e^{-1.215}} \approx 0.771$$

### 4.3 Binary Cross-Entropy Loss

$$L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})] = -[1 \cdot \log(0.771) + 0] = -\log(0.771) \approx 0.260$$

### 4.4 Backprop tại output

Với sigmoid + binary cross-entropy, $dZ^{[2]} = \hat{y} - y$:

$$dZ^{[2]} = 0.771 - 1 = -0.229$$

$$dW^{[2]} = dZ^{[2]} \cdot (A^{[1]})^T = -0.229 \begin{pmatrix} 1.45 & 0 \end{pmatrix} = \begin{pmatrix} -0.332 & 0 \end{pmatrix}$$

$$db^{[2]} = dZ^{[2]} = -0.229$$

---

## Câu 5. Decision Tree — Gini Index

### 5.1 Gini Index ban đầu

10 mẫu: 6 Có, 4 Không.

$$G(S) = 1 - \left(\frac{6}{10}\right)^2 - \left(\frac{4}{10}\right)^2 = 1 - 0.36 - 0.16 = 0.48$$

### 5.2 Gini Gain — Thuộc tính Tuổi

- **Trẻ** (dòng 1,2,8,9): 1 Có, 3 Không → $G = 1 - (1/4)^2 - (3/4)^2 = 1 - 0.0625 - 0.5625 = 0.375$
- **Trung niên** (dòng 3,7): 2 Có, 0 Không → $G = 0$
- **Già** (dòng 4,5,6,10): 3 Có, 1 Không → $G = 1 - (3/4)^2 - (1/4)^2 = 0.375$

$$G_{split} = \frac{4}{10}(0.375) + \frac{2}{10}(0) + \frac{4}{10}(0.375) = 0.15 + 0 + 0.15 = 0.30$$

$$\text{Gain}_{\text{Tuổi}} = 0.48 - 0.30 = 0.18$$

### 5.3 Gini Gain — Thuộc tính Sinh viên

- **Có** (dòng 5,6,7,9,10): 4 Có, 1 Không → $G = 1 - (4/5)^2 - (1/5)^2 = 1 - 0.64 - 0.04 = 0.32$
- **Không** (dòng 1,2,3,4,8): 2 Có, 3 Không → $G = 1 - (2/5)^2 - (3/5)^2 = 1 - 0.16 - 0.36 = 0.48$

$$G_{split} = \frac{5}{10}(0.32) + \frac{5}{10}(0.48) = 0.16 + 0.24 = 0.40$$

$$\text{Gain}_{\text{Sinh viên}} = 0.48 - 0.40 = 0.08$$

### 5.4 Kết luận

| Thuộc tính | Gini Gain |
| :--- | :--- |
| **Tuổi** | **0.18** |
| Sinh viên | 0.08 |

Chọn **Tuổi** vì Gini Gain lớn hơn.

### 5.5 Cây sau bước chia

```
            [Tuổi]
         /    |     \
    Trẻ    Trung    Già
     |      niên     |
  [1C,3K]  → Có   [3C,1K]
  (tiếp           (tiếp
   chia)           chia)
```

---

## Câu 6. K-means Clustering — Trace

### 6.1 Iteration 0 — Gán cụm

$\mu_1 = (1,2)$, $\mu_2 = (9,8)$

| STT | $d^2(\cdot, \mu_1)$ | $d^2(\cdot, \mu_2)$ | Cụm |
| :--- | :--- | :--- | :--- |
| 1 | $(1-1)^2+(2-2)^2=0$ | $(1-9)^2+(2-8)^2=100$ | 1 |
| 2 | $(2-1)^2+(1-2)^2=2$ | $(2-9)^2+(1-8)^2=98$ | 1 |
| 3 | $(2-1)^2+(3-2)^2=2$ | $(2-9)^2+(3-8)^2=74$ | 1 |
| 4 | $(8-1)^2+(7-2)^2=74$ | $(8-9)^2+(7-8)^2=2$ | 2 |
| 5 | $(9-1)^2+(8-2)^2=100$ | $(9-9)^2+(8-8)^2=0$ | 2 |
| 6 | $(8-1)^2+(9-2)^2=98$ | $(8-9)^2+(9-8)^2=2$ | 2 |

$C_1 = \{(1,2),(2,1),(2,3)\}$, $C_2 = \{(8,7),(9,8),(8,9)\}$

**Distortion:**

$$J = \frac{1}{6}(0 + 2 + 2 + 2 + 0 + 2) = \frac{8}{6} \approx 1.333$$

### 6.2 Cập nhật tâm

$$\mu_1^{(1)} = \left(\frac{1+2+2}{3}, \frac{2+1+3}{3}\right) = \left(\frac{5}{3}, 2\right) \approx (1.667, 2.000)$$

$$\mu_2^{(1)} = \left(\frac{8+9+8}{3}, \frac{7+8+9}{3}\right) = \left(\frac{25}{3}, 8\right) \approx (8.333, 8.000)$$

### 6.3 Iteration 1 — Gán lại

Với tâm mới, tính khoảng cách. Vì hai nhóm cách nhau rất xa, nhãn không đổi.

$C_1 = \{(1,2),(2,1),(2,3)\}$, $C_2 = \{(8,7),(9,8),(8,9)\}$ — không thay đổi.

Tâm cập nhật lại cho cùng kết quả → **Hội tụ** ✓

### 6.4 Giải thích

K-means luôn hội tụ vì:
- Bước gán cụm giảm hoặc giữ nguyên $J$ (mỗi điểm chọn tâm gần nhất).
- Bước cập nhật tâm giảm hoặc giữ nguyên $J$ (trung bình là nghiệm tối ưu cho tổng bình phương khoảng cách cố định thành viên).
- $J \geq 0$ nên dãy giảm bị chặn dưới phải hội tụ.

Tuy nhiên, vì hàm mục tiêu không lồi, nghiệm cuối phụ thuộc vào khởi tạo và có thể là cực tiểu cục bộ. Giải pháp: chạy nhiều lần với K-means++ và chọn nghiệm có $J$ nhỏ nhất.

---

## Câu 7. Bias-Variance Decomposition

### 7.1 Phương trình

$$\text{Test MSE} = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$$

### 7.2 Chứng minh

Cho $y = f(\mathbf{x}) + \epsilon$ với $\mathbb{E}[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2$.

$$\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[(f + \epsilon - \hat{f})^2]$$

$$= \mathbb{E}[(f - \hat{f})^2 + 2\epsilon(f - \hat{f}) + \epsilon^2]$$

Vì $\epsilon$ độc lập với $\hat{f}$ và $\mathbb{E}[\epsilon] = 0$:

$$= \mathbb{E}[(f - \hat{f})^2] + \sigma^2$$

Thêm bớt $\bar{f} = \mathbb{E}[\hat{f}]$:

$$\mathbb{E}[(f - \hat{f})^2] = \mathbb{E}[((f - \bar{f}) + (\bar{f} - \hat{f}))^2]$$

$$= (f - \bar{f})^2 + 2(f - \bar{f})\mathbb{E}[\bar{f} - \hat{f}] + \mathbb{E}[(\bar{f} - \hat{f})^2]$$

Vì $\mathbb{E}[\bar{f} - \hat{f}] = 0$:

$$= (f - \bar{f})^2 + \mathbb{E}[(\hat{f} - \bar{f})^2]$$

$$= \text{Bias}^2 + \text{Variance}$$

**Kết luận:** $\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Var} + \sigma^2 \quad \blacksquare$

### 7.3 Chọn $\lambda$

| $\lambda$ | Trung bình validation MSE |
| :--- | :--- |
| 0.00 | $(1.50+1.65+1.40+1.55)/4 = 1.525$ |
| 0.05 | $(0.85+0.90+0.82+0.88)/4 = 0.8625$ |
| **0.50** | $(0.60+0.65+0.58+0.62)/4 = \mathbf{0.6125}$ |
| 5.00 | $(0.95+1.00+0.92+0.98)/4 = 0.9625$ |

**Chọn $\lambda = 0.50$** vì validation MSE trung bình nhỏ nhất. $\lambda = 0$ overfit, $\lambda = 5$ underfit.

---

## Câu 8. SGD và Khả Năng Tổng Quát Hóa (Nâng cao)

### 8.1 Tại sao Batch Size lớn tăng Overfitting
Tăng batch size làm gradient trơn tru và chính xác hơn trên tập huấn luyện. Tuy nhiên, chính sự "nhiễu" (stochastic noise) do mini-batch nhỏ tạo ra lại đóng vai trò như một cơ chế implicit regularization (điều chuẩn ngầm). Nhiễu này giúp quỹ đạo tối ưu "nảy" và thoát khỏi các cực tiểu cục bộ dốc (sharp minima - thường fit dữ liệu train rất khít nhưng generalize kém trên dữ liệu mới) để hội tụ về các cực tiểu bằng phẳng (flat minima - nơi hàm loss ít nhạy cảm với sự thay đổi nhỏ của dữ liệu, generalize tốt hơn). Khi batch size quá lớn, gradient mất đi độ nhiễu, mô hình dễ mắc kẹt ở sharp minima, dẫn đến overfitting trầm trọng hơn.

### 8.2 Điều chỉnh Learning Rate (Linear Scaling Rule)
Mức độ nhiễu trong bước cập nhật của SGD xấp xỉ tỷ lệ thuận với $\frac{\eta}{B}$ (trong đó $\eta$ là learning rate, $B$ là batch size).
Để duy trì mức độ nhiễu (và do đó giữ nguyên hiệu ứng implicit regularization), tỷ lệ này cần được giữ không đổi.
Do đó, nếu tăng batch size lên $k$ lần ($B \to kB$), ta nên tăng learning rate lên $k$ lần ($\eta \to k\eta$). Đây được gọi là "Linear Scaling Rule" phổ biến trong phân tán huấn luyện Deep Learning.

---

**HẾT ĐÁP ÁN**
