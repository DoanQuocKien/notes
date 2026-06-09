
# **ĐÁP ÁN ĐỀ THI MACHINE LEARNING — ĐỀ 04**

---

## Câu 1. Logistic Regression — Chứng minh Tính Lồi

### 1.1 Gradient

Đặt $z^{(i)} = \boldsymbol{\beta}^T\mathbf{x}^{(i)}$, $\hat{y}^{(i)} = \sigma(z^{(i)})$.

Áp dụng chain rule cho từng mẫu:

$$\frac{\partial J^{(i)}}{\partial\boldsymbol{\beta}} = \frac{\partial J^{(i)}}{\partial\hat{y}^{(i)}} \cdot \frac{\partial\hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial\boldsymbol{\beta}}$$

**Thành phần Loss:**

$$\frac{\partial J^{(i)}}{\partial\hat{y}^{(i)}} = -\frac{y^{(i)}}{\hat{y}^{(i)}} + \frac{1-y^{(i)}}{1-\hat{y}^{(i)}} = \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1-\hat{y}^{(i)})}$$

**Thành phần Sigmoid:**

$$\frac{\partial\hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)}(1-\hat{y}^{(i)})$$

**Thành phần Tuyến tính:**

$$\frac{\partial z^{(i)}}{\partial\boldsymbol{\beta}} = \mathbf{x}^{(i)}$$

Nhân lại, mẫu số triệt tiêu:

$$\frac{\partial J^{(i)}}{\partial\boldsymbol{\beta}} = (\hat{y}^{(i)} - y^{(i)})\mathbf{x}^{(i)}$$

$$\boxed{\nabla J = \sum_{i=1}^{n}(\hat{y}^{(i)} - y^{(i)})\mathbf{x}^{(i)}}$$

### 1.2 Hessian

$$H = \frac{\partial}{\partial\boldsymbol{\beta}}\sum_{i=1}^{n}(\hat{y}^{(i)} - y^{(i)})\mathbf{x}^{(i)}$$

Vì $y^{(i)}$ là hằng số và $\frac{\partial\hat{y}^{(i)}}{\partial\boldsymbol{\beta}} = \hat{y}^{(i)}(1-\hat{y}^{(i)})\mathbf{x}^{(i)}$:

$$\boxed{H = \sum_{i=1}^{n}\hat{y}^{(i)}(1-\hat{y}^{(i)})\mathbf{x}^{(i)}(\mathbf{x}^{(i)})^T}$$

### 1.3 Chứng minh $H \succeq 0$

Với vector $\mathbf{v}$ bất kỳ:

$$\mathbf{v}^T H \mathbf{v} = \sum_{i=1}^{n}\hat{y}^{(i)}(1-\hat{y}^{(i)}) \cdot \mathbf{v}^T\mathbf{x}^{(i)}(\mathbf{x}^{(i)})^T\mathbf{v}$$

$$= \sum_{i=1}^{n}\hat{y}^{(i)}(1-\hat{y}^{(i)}) \cdot (\mathbf{v}^T\mathbf{x}^{(i)})^2$$

- $\hat{y}^{(i)} \in (0,1)$ → $\hat{y}^{(i)}(1-\hat{y}^{(i)}) > 0$
- $(\mathbf{v}^T\mathbf{x}^{(i)})^2 \geq 0$

Do đó $\mathbf{v}^T H \mathbf{v} \geq 0$ với mọi $\mathbf{v}$ → $H \succeq 0$ → $J$ lồi $\quad\blacksquare$

### 1.4 Hệ quả

Vì $J$ lồi, mọi cực tiểu cục bộ cũng là cực tiểu toàn cục. Gradient descent và Newton's method đều đảm bảo hội tụ đến nghiệm tối ưu duy nhất.

---

## Câu 2. OLS 2 biến

### 2.1

$$\mathbf{X} = \begin{pmatrix} 1&1&2\\ 1&2&1\\ 1&3&3\\ 1&4&2\\ 1&5&4 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 10\\12\\18\\20\\28 \end{pmatrix}$$

### 2.2

$$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 5 & 15 & 12 \\ 15 & 55 & 40 \\ 12 & 40 & 34 \end{pmatrix}$$

Chi tiết:
- $\sum 1 = 5$, $\sum x_1 = 15$, $\sum x_2 = 12$
- $\sum x_1^2 = 1+4+9+16+25 = 55$
- $\sum x_2^2 = 4+1+9+4+16 = 34$
- $\sum x_1 x_2 = 2+2+9+8+20 = 41$

*Sửa:* $\sum x_1 x_2 = 1(2)+2(1)+3(3)+4(2)+5(4) = 2+2+9+8+20 = 41$

$$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 5 & 15 & 12 \\ 15 & 55 & 41 \\ 12 & 41 & 34 \end{pmatrix}$$

$$\mathbf{X}^T\mathbf{y} = \begin{pmatrix} 88 \\ 318 \\ 242 \end{pmatrix}$$

Chi tiết: $\sum y = 88$, $\sum x_1 y = 10+24+54+80+140 = 308$

*Sửa:* $\sum x_1 y = 1(10)+2(12)+3(18)+4(20)+5(28) = 10+24+54+80+140 = 308$

$$\mathbf{X}^T\mathbf{y} = \begin{pmatrix} 88 \\ 308 \\ 242 \end{pmatrix}$$

$\sum x_2 y = 2(10)+1(12)+3(18)+2(20)+4(28) = 20+12+54+40+112 = 238$

$$\mathbf{X}^T\mathbf{y} = \begin{pmatrix} 88 \\ 308 \\ 238 \end{pmatrix}$$

### 2.3 Giải normal equations

Giải hệ (bằng máy tính hoặc elimination):

$$\begin{pmatrix} 5 & 15 & 12 \\ 15 & 55 & 41 \\ 12 & 41 & 34 \end{pmatrix} \boldsymbol{\beta} = \begin{pmatrix} 88 \\ 308 \\ 238 \end{pmatrix}$$

Giải ra: $\beta_0 \approx 2.073$, $\beta_1 \approx 3.219$, $\beta_2 \approx 1.461$

**Mô hình:** $\hat{y} = 2.073 + 3.219x_1 + 1.461x_2$

### 2.4

$$\hat{y}(3, 5) = 2.073 + 3.219(3) + 1.461(5) = 2.073 + 9.657 + 7.305 = 19.035$$

---

## Câu 3. GDA — Ranh giới Tuyến tính

### 3.1

$$P(y=1 \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y=1)P(y=1)}{P(\mathbf{x} \mid y=1)P(y=1) + P(\mathbf{x} \mid y=0)P(y=0)}$$

$$= \frac{1}{1 + \frac{P(\mathbf{x} \mid y=0)P(y=0)}{P(\mathbf{x} \mid y=1)P(y=1)}}$$

### 3.2

$$\log\frac{P(\mathbf{x}\mid y=1)}{P(\mathbf{x}\mid y=0)} = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_1)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}_1) + \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_0)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}_0)$$

Khai triển:

$$= -\frac{1}{2}\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - \frac{1}{2}\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - \boldsymbol{\mu}_0^T\boldsymbol{\Sigma}^{-1}\mathbf{x} + \frac{1}{2}\boldsymbol{\mu}_0^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0$$

Số hạng $\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$ **triệt tiêu** nhờ $\boldsymbol{\Sigma}$ chung:

$$= (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - \frac{1}{2}\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_0^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0$$

Đặt $z = \boldsymbol{\theta}^T\mathbf{x} + \theta_0$ với:

$$\boldsymbol{\theta} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$$

$$\theta_0 = -\frac{1}{2}\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_0^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0 + \log\frac{\phi}{1-\phi}$$

$$\therefore P(y=1 \mid \mathbf{x}) = \frac{1}{1+e^{-z}} = \sigma(\boldsymbol{\theta}^T\mathbf{x} + \theta_0) \quad\blacksquare$$

### 3.3

Nếu $\boldsymbol{\Sigma}_0 \neq \boldsymbol{\Sigma}_1$, các số hạng $\mathbf{x}^T\boldsymbol{\Sigma}_1^{-1}\mathbf{x}$ và $\mathbf{x}^T\boldsymbol{\Sigma}_0^{-1}\mathbf{x}$ **không triệt tiêu** nhau. Log-ratio chứa số hạng bậc hai theo $\mathbf{x}$, nên ranh giới quyết định là một mặt bậc hai (conic section: ellipse, parabola, hoặc hyperbola). Mô hình này gọi là **Quadratic Discriminant Analysis (QDA)**.

---

## Câu 4. Regression Tree

### 4.1 Không chia

$$\bar{y} = \frac{2.5+3.0+4.5+4.0+8.0+7.5+9.0+10.0}{8} = \frac{48.5}{8} = 6.0625$$

$$\text{SSE} = (2.5-6.0625)^2 + (3-6.0625)^2 + \cdots + (10-6.0625)^2$$

$$= 12.691 + 9.378 + 2.441 + 4.253 + 3.753 + 2.066 + 8.628 + 15.503 = 58.713$$

*Tính chi tiết:*

$$= (-3.5625)^2 + (-3.0625)^2 + (-1.5625)^2 + (-2.0625)^2 + (1.9375)^2 + (1.4375)^2 + (2.9375)^2 + (3.9375)^2$$

$$= 12.691 + 9.379 + 2.441 + 4.254 + 3.754 + 2.066 + 8.629 + 15.504 = 58.718$$

$$\text{MSE} = \frac{58.718}{8} = 7.340$$

### 4.2 Split $x \leq 4$

**Trái** ($x \leq 4$): $\{2.5, 3.0, 4.5, 4.0\}$

$$\bar{y}_L = \frac{2.5+3.0+4.5+4.0}{4} = 3.5$$

$$\text{SSE}_L = (2.5-3.5)^2+(3-3.5)^2+(4.5-3.5)^2+(4-3.5)^2 = 1+0.25+1+0.25 = 2.5$$

**Phải** ($x > 4$): $\{8.0, 7.5, 9.0, 10.0\}$

$$\bar{y}_R = \frac{8+7.5+9+10}{4} = 8.625$$

$$\text{SSE}_R = (8-8.625)^2+(7.5-8.625)^2+(9-8.625)^2+(10-8.625)^2 = 0.391+1.266+0.141+1.891 = 3.688$$

$$\text{Tổng SSE} = 2.5 + 3.688 = 6.188$$

### 4.3 Split $x \leq 3$

**Trái**: $\{2.5, 3.0, 4.5\}$, $\bar{y}_L = \frac{10}{3} = 3.333$

$$\text{SSE}_L = (2.5-3.333)^2+(3-3.333)^2+(4.5-3.333)^2 = 0.694+0.111+1.361 = 2.167$$

**Phải**: $\{4.0, 8.0, 7.5, 9.0, 10.0\}$, $\bar{y}_R = \frac{38.5}{5} = 7.7$

$$\text{SSE}_R = (4-7.7)^2+(8-7.7)^2+(7.5-7.7)^2+(9-7.7)^2+(10-7.7)^2$$

$$= 13.69+0.09+0.04+1.69+5.29 = 20.8$$

$$\text{Tổng SSE} = 2.167 + 20.8 = 22.967$$

### 4.4 Kết luận

| Split | Tổng SSE |
| :--- | :--- |
| **$x \leq 4$** | **6.188** |
| $x \leq 3$ | 22.967 |

**Chọn split $x \leq 4$** vì SSE nhỏ hơn rất nhiều.

---

## Câu 5. Backpropagation 2-3-1

### 5.1 Forward Pass

$$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]} = \begin{pmatrix} 0.2(1)+(-0.3)(-1) \\ 0.4(1)+0.1(-1) \\ -0.1(1)+0.5(-1) \end{pmatrix} = \begin{pmatrix} 0.5 \\ 0.3 \\ -0.6 \end{pmatrix}$$

$$\mathbf{A}^{[1]} = \sigma(\mathbf{Z}^{[1]}) = \begin{pmatrix} \sigma(0.5) \\ \sigma(0.3) \\ \sigma(-0.6) \end{pmatrix} \approx \begin{pmatrix} 0.6225 \\ 0.5744 \\ 0.3543 \end{pmatrix}$$

$$Z^{[2]} = 0.5(0.6225) + (-0.3)(0.5744) + 0.4(0.3543) + 0 = 0.3113 - 0.1723 + 0.1417 = 0.2807$$

$$\hat{y} = \sigma(0.2807) \approx 0.5697$$

### 5.2 Loss

$$L = -[1 \cdot \log(0.5697) + 0] = -\log(0.5697) \approx 0.5628$$

### 5.3

$$dZ^{[2]} = \hat{y} - y = 0.5697 - 1 = -0.4303$$

### 5.4

$$dW^{[2]} = dZ^{[2]} \cdot (\mathbf{A}^{[1]})^T = -0.4303 \begin{pmatrix} 0.6225 & 0.5744 & 0.3543 \end{pmatrix}$$

$$= \begin{pmatrix} -0.2679 & -0.2473 & -0.1525 \end{pmatrix}$$

$$db^{[2]} = dZ^{[2]} = -0.4303$$

---

## Câu 6. Gradient Descent với Ridge

### 6.1 Gradient OLS tại $\boldsymbol{\beta} = (0, 0)$

Mọi dự đoán = 0, residual = $y_i$. Gradient dạng "cộng":

$$\frac{\partial L}{\partial\beta_0} = -\frac{1}{4}\sum r_i = -\frac{1}{4}(3+5+7+9) = -\frac{24}{4} = -6$$

$$\frac{\partial L}{\partial\beta_1} = -\frac{1}{4}\sum r_i x_i = -\frac{1}{4}(3+10+21+36) = -\frac{70}{4} = -17.5$$

### 6.2 Gradient Ridge penalty

$$\nabla(\lambda\|\boldsymbol{\beta}\|_2^2) = 2\lambda\boldsymbol{\beta} = 2(0.1)\begin{pmatrix}0\\0\end{pmatrix} = \begin{pmatrix}0\\0\end{pmatrix}$$

(Tại $\boldsymbol{\beta} = 0$, penalty gradient = 0.)

### 6.3 Cập nhật

Gradient tổng = $(-6, -17.5) + (0, 0) = (-6, -17.5)$

$$\boldsymbol{\beta}_{new} = \begin{pmatrix}0\\0\end{pmatrix} - 0.1\begin{pmatrix}-6\\-17.5\end{pmatrix} = \begin{pmatrix}0.6\\1.75\end{pmatrix}$$

### 6.4 Nhận xét

Tại $\boldsymbol{\beta} = (0,0)$, Ridge penalty không ảnh hưởng vì gradient penalty = 0 ở gốc tọa độ. Sự khác biệt xuất hiện ở các bước sau: khi $\boldsymbol{\beta}$ xa 0, penalty đẩy hệ số về phía 0, co nhỏ chúng so với OLS thuần.

---

## Câu 7. Góc nhìn Bayes cho Regularization (Nâng cao)

### 7.1 Chứng minh Ridge Regression là MAP
Theo định lý Bayes: Posterior $\propto$ Likelihood $\times$ Prior
$$P(\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{X}) \propto P(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\beta}) \cdot P(\boldsymbol{\beta})$$
Lấy log hai vế để tìm giá trị tối đa (MAP):
$$\log \text{Posterior} = \log P(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\beta}) + \log P(\boldsymbol{\beta})$$
- Log-likelihood (nhiễu Gaussian $\sigma^2$): $-\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + C_1$
- Log-prior (Gaussian $\mathcal{N}(0, \tau^2\mathbf{I})$): $-\frac{1}{2\tau^2}\|\boldsymbol{\beta}\|_2^2 + C_2$
MAP tối đa hóa tổng hai đại lượng này, tương đương với **cực tiểu hóa** hàm mất mát bị đổi dấu:
$$L(\boldsymbol{\beta}) = \frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \frac{1}{2\tau^2}\|\boldsymbol{\beta}\|_2^2$$
Nhân toàn bộ với $2\sigma^2$, ta được hàm loss chính xác của Ridge Regression:
$$L_{ridge}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \frac{\sigma^2}{\tau^2}\|\boldsymbol{\beta}\|_2^2$$
Với $\lambda = \frac{\sigma^2}{\tau^2} \quad \blacksquare$

### 7.2 Prior của Lasso Regression
Lasso Regression tương ứng với giả định Prior của $\boldsymbol{\beta}$ là **phân phối Laplace** (Laplace distribution) có trung bình 0.
Hàm mật độ Laplace: $P(\beta) = \frac{1}{2b} \exp\left(-\frac{|\beta|}{b}\right)$. Lấy log sẽ tạo ra số hạng $-\frac{1}{b}|\beta|$, chính là gốc rễ của norm L1 ($\|\boldsymbol{\beta}\|_1$).

### 7.3 Giải thích tính thưa (Sparsity)
- **Phân phối Gaussian (Ridge):** Có dạng hình chuông mượt. Tại đỉnh $\beta=0$, đạo hàm bằng 0. Khi $\beta$ nhỏ dần về 0, lực đẩy của penalty cũng tiến dần về 0. Do đó, Ridge chỉ kéo hệ số "lại gần" 0 mà rất hiếm khi bằng chính xác 0 (lực cản bằng 0 trước khi chạm tới mốc).
- **Phân phối Laplace (Lasso):** Có một đỉnh nhọn (cusp) tại chính xác mốc $\beta=0$, nơi đạo hàm không liên tục. Xác suất Prior tập trung cực kì mạnh tại chính xác điểm 0. Lực cản của penalty L1 là một hằng số liên tục không suy giảm ngay cả khi $\beta$ cực kì gần 0. Điều này khiến quá trình tối ưu có xu hướng đẩy và "khóa" (truncate) hoàn toàn các trọng số không quan trọng về đúng giá trị 0, tạo ra sự thưa thớt (sparsity) và thực hiện Feature Selection tự động.

---

**HẾT ĐÁP ÁN**
