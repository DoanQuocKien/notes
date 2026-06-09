
# **ĐÁP ÁN ĐỀ THI MACHINE LEARNING — ĐỀ 02**

---

## Câu 1. MLE cho Linear Regression — Chứng minh OLS

### 1.1 Hàm mật độ

$$P(y_i \mid \mathbf{x}_i; \boldsymbol{\beta}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \boldsymbol{\beta}^T\mathbf{x}_i)^2}{2\sigma^2}\right)$$

### 1.2 Hàm Likelihood

$$L(\boldsymbol{\beta}) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \boldsymbol{\beta}^T\mathbf{x}_i)^2}{2\sigma^2}\right)$$

### 1.3 Chứng minh MLE ↔ OLS

Log-likelihood:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[\log\frac{1}{\sqrt{2\pi}\sigma} - \frac{(y_i - \boldsymbol{\beta}^T\mathbf{x}_i)^2}{2\sigma^2}\right]$$

$$= n\log\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \boldsymbol{\beta}^T\mathbf{x}_i)^2$$

Hạng đầu là hằng số theo $\boldsymbol{\beta}$. Hệ số $\frac{1}{2\sigma^2} > 0$.

$$\max_{\boldsymbol{\beta}} \ell \iff \min_{\boldsymbol{\beta}} \sum_{i=1}^{n}(y_i - \boldsymbol{\beta}^T\mathbf{x}_i)^2 \quad \blacksquare$$

### 1.4 Normal Equations

$$L = \frac{1}{2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

$$= \frac{1}{2}(\mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta})$$

Gradient:

$$\nabla_{\boldsymbol{\beta}} L = -\mathbf{X}^T\mathbf{y} + \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$

$$\implies \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

$$\implies \boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} \quad \blacksquare$$

---

## Câu 2. Gaussian Discriminant Analysis

### 2.1 Prior

$$P(y=0) = \frac{4}{8} = 0.5, \quad P(y=1) = \frac{4}{8} = 0.5$$

### 2.2 Mean

$$\mu_0 = \frac{1.0 + 1.5 + 2.0 + 2.5}{4} = \frac{7.0}{4} = 1.75$$

$$\mu_1 = \frac{5.0 + 5.5 + 6.0 + 6.5}{4} = \frac{23.0}{4} = 5.75$$

### 2.3 Variance chung

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_{y_i})^2$$

Lớp 0:
$(1.0-1.75)^2 + (1.5-1.75)^2 + (2.0-1.75)^2 + (2.5-1.75)^2 = 0.5625 + 0.0625 + 0.0625 + 0.5625 = 1.25$

Lớp 1:
$(5.0-5.75)^2 + (5.5-5.75)^2 + (6.0-5.75)^2 + (6.5-5.75)^2 = 0.5625 + 0.0625 + 0.0625 + 0.5625 = 1.25$

$$\sigma^2 = \frac{1.25 + 1.25}{8} = \frac{2.50}{8} = 0.3125$$

### 2.4 Phân loại $x = 3.5$

$$P(x=3.5 \mid y=0) = \frac{1}{\sqrt{2\pi(0.3125)}} \exp\left(-\frac{(3.5-1.75)^2}{2(0.3125)}\right) = \frac{1}{1.4005} \exp\left(-\frac{3.0625}{0.625}\right) = 0.7142 \cdot e^{-4.9} \approx 0.00532$$

$$P(x=3.5 \mid y=1) = \frac{1}{1.4005} \exp\left(-\frac{(3.5-5.75)^2}{0.625}\right) = 0.7142 \cdot e^{-8.1} \approx 0.000216$$

Nhân prior:

$$\text{score}_0 = 0.00532 \times 0.5 = 0.00266$$

$$\text{score}_1 = 0.000216 \times 0.5 = 0.000108$$

Posterior:

$$P(y=0 \mid x=3.5) = \frac{0.00266}{0.00266 + 0.000108} = \frac{0.00266}{0.00277} \approx 0.961$$

**Kết luận:** Phân loại $x = 3.5$ vào **lớp 0**.

### 2.5 Ranh giới quyết định

Vì prior bằng nhau, ranh giới là điểm cách đều hai mean:

$$x^* = \frac{\mu_0 + \mu_1}{2} = \frac{1.75 + 5.75}{2} = 3.75$$

(Khi prior bằng nhau và variance chung, ranh giới nằm giữa hai mean.)

---

## Câu 3. Logistic Regression — Gradient Descent

### 3.1 Xác suất ban đầu

Khi $\boldsymbol{\beta} = (0,0,0)$: $z_i = 0$ cho mọi mẫu, nên $\hat{p}_i = \sigma(0) = 0.5$ cho tất cả.

### 3.2 Gradient

Ma trận $\mathbf{X}$ (cột 1 cho bias):

$$\mathbf{X} = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \\ 1 & 3 & 1 \\ 1 & 0 & 3 \end{pmatrix}$$

Sai số: $\hat{p}_i - y_i$:

| STT | $\hat{p}$ | $y$ | $\hat{p} - y$ |
| :--- | :--- | :--- | :--- |
| 1 | 0.5 | 1 | -0.5 |
| 2 | 0.5 | 0 | 0.5 |
| 3 | 0.5 | 1 | -0.5 |
| 4 | 0.5 | 0 | 0.5 |
| 5 | 0.5 | 1 | -0.5 |
| 6 | 0.5 | 0 | 0.5 |

$$\nabla J = \frac{1}{6}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$$

Thành phần bias ($x_0 = 1$):

$$\frac{1}{6}\sum(\hat{p}_i - y_i) = \frac{1}{6}(-0.5+0.5-0.5+0.5-0.5+0.5) = 0$$

Thành phần $\beta_1$:

$$\frac{1}{6}\sum(\hat{p}_i - y_i)x_{i1} = \frac{1}{6}[(-0.5)(1)+(0.5)(0)+(-0.5)(2)+(0.5)(1)+(-0.5)(3)+(0.5)(0)]$$

$$= \frac{1}{6}(-0.5+0-1.0+0.5-1.5+0) = \frac{-2.5}{6} = -0.4167$$

Thành phần $\beta_2$:

$$\frac{1}{6}\sum(\hat{p}_i - y_i)x_{i2} = \frac{1}{6}[(-0.5)(0)+(0.5)(1)+(-0.5)(1)+(0.5)(2)+(-0.5)(1)+(0.5)(3)]$$

$$= \frac{1}{6}(0+0.5-0.5+1.0-0.5+1.5) = \frac{2.0}{6} = 0.3333$$

$$\nabla J = \begin{pmatrix} 0 \\ -0.4167 \\ 0.3333 \end{pmatrix}$$

### 3.3 Cập nhật

$$\boldsymbol{\beta}_{new} = \boldsymbol{\beta} - \alpha \nabla J = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} - 0.5 \begin{pmatrix} 0 \\ -0.4167 \\ 0.3333 \end{pmatrix} = \begin{pmatrix} 0 \\ 0.2083 \\ -0.1667 \end{pmatrix}$$

### 3.4 Ý nghĩa

- $\beta_1$ tăng (dương): $x_1$ lớn tương quan dương với $y=1$. Đúng vì các mẫu dương có $x_1$ lớn (1, 2, 3).
- $\beta_2$ giảm (âm): $x_2$ lớn tương quan dương với $y=0$. Đúng vì các mẫu $y=0$ có $x_2$ lớn (1, 2, 3).
- Bias không đổi vì số mẫu lớp 0 và lớp 1 bằng nhau.

---

## Câu 4. Decision Tree — Entropy

### 4.1 Entropy ban đầu

10 mẫu: 4 Cao, 6 Thấp.

$$H(S) = -\frac{4}{10}\log_2\frac{4}{10} - \frac{6}{10}\log_2\frac{6}{10}$$

$$= -0.4\log_2 0.4 - 0.6\log_2 0.6 = -0.4(-1.3219) - 0.6(-0.7370)$$

$$= 0.5288 + 0.4422 = 0.9710$$

### 4.2 Information Gain — Nợ

- **Nhiều** (dòng 1,4,7): 3 Cao, 0 Thấp → $H = 0$
- **Ít** (dòng 2,3,5,6,8,9,10): 1 Cao, 6 Thấp → $H = -\frac{1}{7}\log_2\frac{1}{7} - \frac{6}{7}\log_2\frac{6}{7} = 0.5917$

$$G_{split} = \frac{3}{10}(0) + \frac{7}{10}(0.5917) = 0 + 0.4142 = 0.4142$$

$$\text{Gain}_{\text{Nợ}} = 0.9710 - 0.4142 = 0.5568$$

### 4.3 Information Gain — Thu nhập

- **Thấp** (dòng 1,4,7,10): 3 Cao, 1 Thấp → $H = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} = 0.8113$
- **Trung bình** (dòng 2,6,9): 1 Cao, 2 Thấp → $H = -\frac{1}{3}\log_2\frac{1}{3} - \frac{2}{3}\log_2\frac{2}{3} = 0.9183$
- **Cao** (dòng 3,5,8): 0 Cao, 3 Thấp → $H = 0$

$$G_{split} = \frac{4}{10}(0.8113) + \frac{3}{10}(0.9183) + \frac{3}{10}(0)$$

$$= 0.3245 + 0.2755 + 0 = 0.6000$$

$$\text{Gain}_{\text{Thu nhập}} = 0.9710 - 0.6000 = 0.3710$$

### 4.4 Kết luận

| Thuộc tính | Information Gain |
| :--- | :--- |
| **Nợ** | **0.5568** |
| Thu nhập | 0.3710 |

**Chọn Nợ** vì Information Gain lớn nhất. Nhánh "Nhiều" nợ → thuần nhất lớp Cao (entropy = 0).

---

## Câu 5. Backpropagation — Mạng 2-2-1

### 5.1 Forward Propagation

$z_{h1} = 0.5(2) + (-0.3)(-1) + 0.2 = 1.0 + 0.3 + 0.2 = 1.5$

$h_1 = \sigma(1.5) = \frac{1}{1+e^{-1.5}} \approx 0.8176$

$z_{h2} = (-0.2)(2) + 0.4(-1) + (-0.1) = -0.4 - 0.4 - 0.1 = -0.9$

$h_2 = \sigma(-0.9) = \frac{1}{1+e^{0.9}} \approx 0.2891$

$z_o = 0.6(0.8176) + (-0.4)(0.2891) + 0.3 = 0.4906 - 0.1156 + 0.3 = 0.6750$

$\hat{y} = \sigma(0.6750) \approx 0.6625$

### 5.2 Loss

$$L = \frac{1}{2}(0.6625 - 0)^2 = \frac{1}{2}(0.4389) = 0.2195$$

### 5.3 Backpropagation

$\delta_o = (\hat{y} - y) \cdot \sigma'(z_o) = (0.6625 - 0) \cdot \hat{y}(1-\hat{y}) = 0.6625 \times 0.6625 \times 0.3375 = 0.1481$

$\delta_{h1} = \delta_o \cdot w_{h1 \to o} \cdot h_1(1-h_1) = 0.1481 \times 0.6 \times 0.8176 \times 0.1824 = 0.01325$

$\delta_{h2} = \delta_o \cdot w_{h2 \to o} \cdot h_2(1-h_2) = 0.1481 \times (-0.4) \times 0.2891 \times 0.7109 = -0.01217$

### 5.4 Gradient

| Trọng số | Gradient |
| :--- | :--- |
| $\partial L / \partial w_{h1 \to o} = \delta_o \cdot h_1$ | $0.1481 \times 0.8176 = 0.1211$ |
| $\partial L / \partial w_{h2 \to o} = \delta_o \cdot h_2$ | $0.1481 \times 0.2891 = 0.04282$ |
| $\partial L / \partial w_{x1 \to h1} = \delta_{h1} \cdot x_1$ | $0.01325 \times 2 = 0.02650$ |
| $\partial L / \partial w_{x2 \to h1} = \delta_{h1} \cdot x_2$ | $0.01325 \times (-1) = -0.01325$ |
| $\partial L / \partial w_{x1 \to h2} = \delta_{h2} \cdot x_1$ | $-0.01217 \times 2 = -0.02434$ |
| $\partial L / \partial w_{x2 \to h2} = \delta_{h2} \cdot x_2$ | $-0.01217 \times (-1) = 0.01217$ |

### 5.5 Cập nhật ($\eta = 0.1$)

| Kết nối | $w_{old}$ | Gradient | $w_{new} = w_{old} - 0.1 \times \text{grad}$ |
| :--- | :--- | :--- | :--- |
| $x_1 \to h_1$ | 0.500 | 0.02650 | 0.4974 |
| $x_2 \to h_1$ | -0.300 | -0.01325 | -0.2987 |
| $x_1 \to h_2$ | -0.200 | -0.02434 | -0.1976 |
| $x_2 \to h_2$ | 0.400 | 0.01217 | 0.3988 |
| $h_1 \to o$ | 0.600 | 0.1211 | 0.5879 |
| $h_2 \to o$ | -0.400 | 0.04282 | -0.4043 |

---

## Câu 6. K-means

### 6.1 Iteration 0

$\mu_1 = 2$, $\mu_2 = 11$

| $x$ | $d(x, \mu_1)$ | $d(x, \mu_2)$ | Cụm |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 100 | 1 |
| 2 | 0 | 81 | 1 |
| 3 | 1 | 64 | 1 |
| 10 | 64 | 1 | 2 |
| 11 | 81 | 0 | 2 |
| 12 | 100 | 1 | 2 |
| 20 | 324 | 81 | 2 |
| 21 | 361 | 100 | 2 |

$C_1 = \{1,2,3\}$, $C_2 = \{10,11,12,20,21\}$

$$J_0 = \frac{1}{8}(1+0+1+1+0+1+81+100) = \frac{185}{8} = 23.125$$

Cập nhật:

$$\mu_1^{(1)} = \frac{1+2+3}{3} = 2$$

$$\mu_2^{(1)} = \frac{10+11+12+20+21}{5} = \frac{74}{5} = 14.8$$

### Iteration 1

Với $\mu_1 = 2$, $\mu_2 = 14.8$:

| $x$ | $d^2(x,2)$ | $d^2(x,14.8)$ | Cụm |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 190.44 | 1 |
| 2 | 0 | 163.84 | 1 |
| 3 | 1 | 139.24 | 1 |
| 10 | 64 | 23.04 | 2 |
| 11 | 81 | 14.44 | 2 |
| 12 | 100 | 7.84 | 2 |
| 20 | 324 | 27.04 | 2 |
| 21 | 361 | 38.44 | 2 |

Không đổi: $C_1 = \{1,2,3\}$, $C_2 = \{10,11,12,20,21\}$

$$J_1 = \frac{1}{8}(1+0+1+23.04+14.44+7.84+27.04+38.44) = \frac{112.8}{8} = 14.1$$

$$\mu_1^{(2)} = 2, \quad \mu_2^{(2)} = 14.8 \quad \text{(hội tụ)}$$

### 6.2 Với $K = 3$

$C_1 = \{1,2,3\}$: $\mu_1 = 2$, distortion = $(1+0+1)/3 = 0.667$

$C_2 = \{10,11,12\}$: $\mu_2 = 11$, distortion = $(1+0+1)/3 = 0.667$

$C_3 = \{20,21\}$: $\mu_3 = 20.5$, distortion = $(0.25+0.25)/2 = 0.25$

$$J_{K=3} = \frac{1}{8}(1+0+1+1+0+1+0.25+0.25) = \frac{4.5}{8} = 0.5625$$

### 6.3 Elbow

| $K$ | Distortion |
| :--- | :--- |
| 1 | rất lớn |
| 2 | 14.1 |
| 3 | 0.5625 |

Giảm rất mạnh từ $K=2$ sang $K=3$ (từ 14.1 xuống 0.56). **Chọn $K = 3$** vì dữ liệu có 3 nhóm rõ ràng ({1,2,3}, {10,11,12}, {20,21}).

---

## Câu 7. Generative vs Discriminative Models (Nâng cao)

### 7.1 Khi $n \to \infty$ và giả định đúng
**Cả hai bằng nhau.** Khi $n \to \infty$ và dữ liệu thực sự thỏa mãn giả định Naive Bayes, cả hai mô hình đều hội tụ về cùng một bộ phân loại tối ưu (Bayes optimal classifier). Hệ số của Logistic Regression khi hội tụ sẽ hoàn toàn tương đương với các hệ số log-odds suy ra từ Naive Bayes.

### 7.2 Khi $n$ nhỏ
**Naive Bayes hội tụ nhanh hơn.** Naive Bayes tối đa hóa joint likelihood bằng cách ước lượng các tham số $\mu, \sigma$ riêng rẽ cho từng feature độc lập. Số lượng tham số hiệu dụng cần ước lượng ít và không phụ thuộc chéo vào nhau, dẫn đến tốc độ hội tụ nhanh (chỉ cần $O(\log p)$ dữ liệu). Logistic Regression phải tối ưu đồng thời tất cả các trọng số $\beta$ để tìm ranh giới phân biệt, cần nhiều dữ liệu hơn (cỡ $O(p)$) để tránh overfitting. Do đó khi dữ liệu ít, NB hoạt động tốt hơn LR.

### 7.3 Khi giả định độc lập bị vi phạm (Thực tế)
Đường cong Learning Curve của hai mô hình sẽ cắt nhau.
- **Khi $n$ nhỏ:** Naive Bayes vẫn có Test Error thấp hơn Logistic Regression do lợi thế hội tụ nhanh, trong khi LR dễ bị overfit.
- **Khi $n$ lớn:** LR sẽ vượt qua NB (có Test Error thấp hơn). Do giả định sai, Naive Bayes bị "bias" vĩnh viễn và tiến tới một mức Test Error tiệm cận cao. Ngược lại, Logistic Regression bỏ qua giả định phân phối, trực tiếp tối ưu ranh giới phân loại để cực tiểu hóa sai số dự đoán, nên khi đủ dữ liệu, nó sẽ tìm được ranh giới tốt hơn và hội tụ ở mức error thấp hơn.

---

**HẾT ĐÁP ÁN**
