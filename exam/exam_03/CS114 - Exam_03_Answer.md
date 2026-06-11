---
title: "CS114 - Exam 03 Answer"
author: "Quoc Kien"
toc: true
toc-depth: 3
format:
  pdf:
    documentclass: scrartcl
    toc: true
    toc-depth: 3
    geometry:
      - margin=0.8in
    include-in-header:
      text: |
        \usepackage{fvextra}
        \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{array}
        \usepackage{enumitem}
        \setlist{nosep}
        \AtBeginDocument{\hypersetup{bookmarksopen=true,bookmarksnumbered=true,bookmarksdepth=3}}
---

# **ĐÁP ÁN ĐỀ THI MACHINE LEARNING — ĐỀ 03**

---

## Câu 1. Hồi quy Tuyến tính Đơn

### 1.1

$$\bar{x} = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5$$

$$\bar{y} = \frac{3+5+6+9+11+13}{6} = \frac{47}{6} \approx 7.833$$

### 1.2

| $x_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i-\bar{x})(y_i-\bar{y})$ | $(x_i-\bar{x})^2$ |
| :--- | :--- | :--- | :--- | :--- |
| 1 | -2.5 | -4.833 | 12.083 | 6.25 |
| 2 | -1.5 | -2.833 | 4.250 | 2.25 |
| 3 | -0.5 | -1.833 | 0.917 | 0.25 |
| 4 | 0.5 | 1.167 | 0.583 | 0.25 |
| 5 | 1.5 | 3.167 | 4.750 | 2.25 |
| 6 | 2.5 | 5.167 | 12.917 | 6.25 |

$$S_{xy} = 12.083+4.250+0.917+0.583+4.750+12.917 = 35.500$$

$$S_{xx} = 6.25+2.25+0.25+0.25+2.25+6.25 = 17.500$$

### 1.3

$$\beta_1 = \frac{S_{xy}}{S_{xx}} = \frac{35.5}{17.5} \approx 2.029$$

$$\beta_0 = \bar{y} - \beta_1\bar{x} = 7.833 - 2.029(3.5) = 7.833 - 7.101 = 0.733$$

**Mô hình:** $\hat{y} = 0.733 + 2.029x$

### 1.4

$$\hat{y}(8) = 0.733 + 2.029(8) = 0.733 + 16.229 = 16.962$$

### 1.5

| $x$ | $y$ | $\hat{y}$ | $e = y - \hat{y}$ | $e^2$ |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 3 | 2.762 | 0.238 | 0.057 |
| 2 | 5 | 4.790 | 0.210 | 0.044 |
| 3 | 6 | 6.819 | -0.819 | 0.671 |
| 4 | 9 | 8.848 | 0.152 | 0.023 |
| 5 | 11 | 10.876 | 0.124 | 0.015 |
| 6 | 13 | 12.905 | 0.095 | 0.009 |

$\sum e_i = 0.238+0.210-0.819+0.152+0.124+0.095 \approx 0.0$ ✓

$RSS = 0.057+0.044+0.671+0.023+0.015+0.009 = 0.819$

---

## Câu 2. Exponential Family

### 2.1 Gaussian dạng EFD ($\sigma^2 = 1$)

$$P(y \mid \mu) = \frac{1}{\sqrt{2\pi}} e^{-y^2/2} \cdot \exp\left(y\mu - \frac{\mu^2}{2}\right)$$

- $\eta = \mu$
- $T(y) = y$
- $a(\eta) = \frac{\eta^2}{2} = \frac{\mu^2}{2}$
- $b(y) = \frac{1}{\sqrt{2\pi}} e^{-y^2/2}$

### 2.2 Kiểm tra $\mathbb{E}[y]$

$$a'(\eta) = \frac{d}{d\eta}\frac{\eta^2}{2} = \eta = \mu \quad \checkmark$$

### 2.3 Kiểm tra $\text{Var}(y)$

$$a''(\eta) = \frac{d^2}{d\eta^2}\frac{\eta^2}{2} = 1 = \sigma^2 \quad \checkmark$$

### 2.4 Giải thích

- **Gaussian GLM:** $\eta = \mu$, link function là identity ($g(\mu) = \mu$), inverse link $g^{-1}(\eta) = \eta$. Dự đoán $\hat{y} = \boldsymbol{\beta}^T\mathbf{x}$ → **Linear Regression**.
- **Bernoulli GLM:** $\eta = \log\frac{\phi}{1-\phi}$, link function là logit, inverse link $\phi = \sigma(\eta)$. Dự đoán $P(y=1) = \sigma(\boldsymbol{\beta}^T\mathbf{x})$ → **Logistic Regression**.

Toàn bộ sự khác biệt giữa các GLM nằm ở inverse link function.

---

## Câu 3. Naive Bayes

### 3.1 Prior

4 mẫu "Có bệnh tim" (dòng 1,3,5,7), 4 mẫu "Không" (dòng 2,4,6,8).

$$P(\text{Có}) = P(\text{Không}) = 0.5$$

### 3.2 Conditional Probabilities (Laplace $\alpha = 1$)

**Lớp Có (4 mẫu):**

- Huyết áp=Cao: dòng 1,5,7 → count = 3 → $P = \frac{3+1}{4+2} = \frac{4}{6} = 0.667$
- Cholesterol=Cao: dòng 1,3,5,7 → count = 4 → $P = \frac{4+1}{4+2} = \frac{5}{6} = 0.833$
- Hút thuốc=Không: dòng 5 → count = 1 → $P = \frac{1+1}{4+2} = \frac{2}{6} = 0.333$

**Lớp Không (4 mẫu):**

- Huyết áp=Cao: dòng 2 → count = 1 → $P = \frac{1+1}{4+2} = \frac{2}{6} = 0.333$
- Cholesterol=Cao: dòng 8 → count = 1 → $P = \frac{1+1}{4+2} = \frac{2}{6} = 0.333$
- Hút thuốc=Không: dòng 2,4,8 → count = 3 → $P = \frac{3+1}{4+2} = \frac{4}{6} = 0.667$

### 3.3 Posterior

Email mới: Huyết áp=Cao, Cholesterol=Cao, Hút thuốc=Không.

$$\text{score}_{\text{Có}} = P(\text{Có}) \times 0.667 \times 0.833 \times 0.333 = 0.5 \times 0.185 = 0.0926$$

$$\text{score}_{\text{Không}} = P(\text{Không}) \times 0.333 \times 0.333 \times 0.667 = 0.5 \times 0.0740 = 0.03700$$

$$P(\text{Có} \mid \mathbf{x}) = \frac{0.0926}{0.0926 + 0.03700} = \frac{0.0926}{0.1296} \approx 0.714$$

### 3.4 Kết luận

Vì $P(\text{Có} \mid \mathbf{x}) \approx 0.714 > 0.5$, bệnh nhân mới được phân loại là **Có bệnh tim**.

---

## Câu 4. Ridge Regression

### 4.1 Gradient

$$\nabla L_{ridge} = -\frac{1}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + 2\lambda\boldsymbol{\beta}$$

### 4.2 Nghiệm đóng

Đặt gradient bằng 0:

$$-\frac{1}{n}\mathbf{X}^T\mathbf{y} + \frac{1}{n}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + 2\lambda\boldsymbol{\beta} = \mathbf{0}$$

$$\frac{1}{n}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + 2\lambda\boldsymbol{\beta} = \frac{1}{n}\mathbf{X}^T\mathbf{y}$$

$$\left(\frac{1}{n}\mathbf{X}^T\mathbf{X} + 2\lambda\mathbf{I}\right)\boldsymbol{\beta} = \frac{1}{n}\mathbf{X}^T\mathbf{y}$$

Nhân hai vế cho $n$:

$$(\mathbf{X}^T\mathbf{X} + 2n\lambda\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

$$\boxed{\boldsymbol{\beta}_{ridge} = (\mathbf{X}^T\mathbf{X} + 2n\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}} \quad \blacksquare$$

### 4.3 Ridge objective

$$\|\boldsymbol{\beta}\|_2^2 = 3.0^2 + (-1.5)^2 + 0.8^2 = 9 + 2.25 + 0.64 = 11.89$$

$$L_{ridge} = 0.15 + 0.2(11.89) = 0.15 + 2.378 = 2.528$$

### 4.4 Ridge vs Lasso

- **Ridge ($L_2$):** Co nhỏ tất cả hệ số một cách mượt và tỷ lệ. Không bao giờ đẩy hệ số về đúng 0. Phù hợp khi tất cả đặc trưng đều có ảnh hưởng.
- **Lasso ($L_1$):** Có thể đẩy một số hệ số về chính xác 0, thực hiện chọn đặc trưng tự động (feature selection). Nghiệm thưa (sparse). Phù hợp khi nhiều đặc trưng không liên quan.

---

## Câu 5. Neural Network — Kích thước

### 5.1 Tầng 1 ($n_0 = 4 \to n_1 = 5$)

- $\mathbf{W}^{[1]} \in \mathbb{R}^{5 \times 4}$
- $\mathbf{b}^{[1]} \in \mathbb{R}^{5 \times 1}$
- $\mathbf{Z}^{[1]} \in \mathbb{R}^{5 \times 10}$
- $\mathbf{A}^{[1]} \in \mathbb{R}^{5 \times 10}$

### 5.2 Tầng 2 ($n_1 = 5 \to n_2 = 3$)

- $\mathbf{W}^{[2]} \in \mathbb{R}^{3 \times 5}$
- $\mathbf{b}^{[2]} \in \mathbb{R}^{3 \times 1}$
- $\mathbf{Z}^{[2]} \in \mathbb{R}^{3 \times 10}$
- $\mathbf{A}^{[2]} \in \mathbb{R}^{3 \times 10}$

### 5.3 Tầng 3 ($n_2 = 3 \to n_3 = 2$)

- $\mathbf{W}^{[3]} \in \mathbb{R}^{2 \times 3}$
- $\mathbf{b}^{[3]} \in \mathbb{R}^{2 \times 1}$
- $\mathbf{Z}^{[3]} \in \mathbb{R}^{2 \times 10}$
- $\mathbf{A}^{[3]} \in \mathbb{R}^{2 \times 10}$

### 5.4 Tổng tham số

$$\text{Weights} = 5 \times 4 + 3 \times 5 + 2 \times 3 = 20 + 15 + 6 = 41$$

$$\text{Biases} = 5 + 3 + 2 = 10$$

$$\text{Tổng} = 41 + 10 = 51$$

### 5.5

$\mathbf{dW}^{[2]} = \frac{1}{m}\mathbf{dZ}^{[2]}(\mathbf{A}^{[1]})^T$

Kích thước: $(3 \times 10)(10 \times 5) = 3 \times 5$ = kích thước $\mathbf{W}^{[2]}$ ✓

### 5.6 ReLU vs Sigmoid

- **Sigmoid:** Đạo hàm $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$. Qua nhiều tầng, gradient nhân chuỗi nhiều số $< 0.25$ → tiệm cận 0 → **vanishing gradient**.
- **ReLU:** $g'(z) = 1$ khi $z > 0$, gradient không bị thu nhỏ → truyền ngược hiệu quả qua nhiều tầng.

---

## Câu 6. K-means — Chứng minh Tâm Cụm

### 6.1 Khai triển

$$f(\mu) = \sum_{i=1}^{m}\|x_i - \mu\|_2^2 = \sum_{i=1}^{m}(x_i - \mu)^T(x_i - \mu)$$

$$= \sum_{i=1}^{m}(\|x_i\|_2^2 - 2\mu^T x_i + \|\mu\|_2^2)$$

$$= \sum_{i=1}^{m}\|x_i\|_2^2 - 2\mu^T\sum_{i=1}^{m}x_i + m\|\mu\|_2^2 \quad \blacksquare$$

### 6.2 Gradient

$$\nabla_\mu f = -2\sum_{i=1}^{m}x_i + 2m\mu = \mathbf{0}$$

### 6.3 Nghiệm

$$2m\mu = 2\sum_{i=1}^{m}x_i \implies \mu^* = \frac{1}{m}\sum_{i=1}^{m}x_i \quad \blacksquare$$

### 6.4 Áp dụng

$$\mu^* = \frac{1}{4}[(1,3)+(2,5)+(4,1)+(5,3)] = \frac{(12,12)}{4} = (3, 3)$$

Distortion:

$$|(1,3)-(3,3)|^2 = 4+0 = 4$$
$$|(2,5)-(3,3)|^2 = 1+4 = 5$$
$$|(4,1)-(3,3)|^2 = 1+4 = 5$$
$$|(5,3)-(3,3)|^2 = 4+0 = 4$$

$$f(\mu) = (4+5+5+4) = 18$$

---

## Câu 7. Thử thách: Newton's Method cho Logistic Regression

### 7.1 Gradient

Đặt:

$$z_i=\mathbf{x}_i^T\boldsymbol{\beta},\quad p_i=\sigma(z_i).$$

Với một mẫu:

$$J_i=-y_i\log p_i-(1-y_i)\log(1-p_i).$$

Ta có:

$$\frac{\partial J_i}{\partial p_i}=-\frac{y_i}{p_i}+\frac{1-y_i}{1-p_i},\quad
\frac{\partial p_i}{\partial z_i}=p_i(1-p_i).$$

Nhân lại:

$$\frac{\partial J_i}{\partial z_i}
=\left(-\frac{y_i}{p_i}+\frac{1-y_i}{1-p_i}\right)p_i(1-p_i)
=p_i-y_i.$$

Vì $z_i=\mathbf{x}_i^T\boldsymbol{\beta}$:

$$\nabla_{\boldsymbol{\beta}}J_i=(p_i-y_i)\mathbf{x}_i.$$

Cộng toàn bộ mẫu:

$$\nabla J(\boldsymbol{\beta})=\mathbf{X}^T(\mathbf{p}-\mathbf{y}).$$

### 7.2 Hessian

Gradient là:

$$\nabla J=\sum_{i=1}^{m}(p_i-y_i)\mathbf{x}_i.$$

Vì:

$$\frac{\partial p_i}{\partial \boldsymbol{\beta}}=p_i(1-p_i)\mathbf{x}_i,$$

nên:

$$\nabla^2J=\sum_{i=1}^{m}p_i(1-p_i)\mathbf{x}_i\mathbf{x}_i^T.$$

Dạng ma trận:

$$\nabla^2J=\mathbf{X}^T\mathbf{R}\mathbf{X},$$

trong đó:

$$\mathbf{R}=\operatorname{diag}(p_1(1-p_1),\ldots,p_m(1-p_m)).$$

### 7.3 Tính lồi

Với mọi vector $\mathbf{v}$:

$$\mathbf{v}^T\nabla^2J\mathbf{v}
=\mathbf{v}^T\mathbf{X}^T\mathbf{R}\mathbf{X}\mathbf{v}
=(\mathbf{X}\mathbf{v})^T\mathbf{R}(\mathbf{X}\mathbf{v}).$$

Vì $p_i(1-p_i)\geq 0$, $\mathbf{R}$ là positive semidefinite. Do đó:

$$\mathbf{v}^T\nabla^2J\mathbf{v}\geq 0.$$

Suy ra $J$ là hàm lồi.

### 7.4 Cập nhật Newton

Newton's Method dùng:

$$\boldsymbol{\beta}_{new}
=\boldsymbol{\beta}-(\nabla^2J)^{-1}\nabla J.$$

Thay gradient và Hessian:

$$\boldsymbol{\beta}_{new}
=\boldsymbol{\beta}
-\left(\mathbf{X}^T\mathbf{R}\mathbf{X}\right)^{-1}\mathbf{X}^T(\mathbf{p}-\mathbf{y}).$$

### 7.5 So sánh với Gradient Descent

Gradient Descent dùng:

$$\boldsymbol{\beta}_{new}=\boldsymbol{\beta}-\alpha\mathbf{X}^T(\mathbf{p}-\mathbf{y}).$$

Nó rẻ hơn mỗi bước nhưng phụ thuộc learning rate $\alpha$ và thường cần nhiều vòng lặp. Newton dùng thông tin độ cong nên thường hội tụ rất nhanh gần nghiệm tối ưu, nhưng phải tạo và giải hệ tuyến tính với Hessian $p\times p$. Chi phí có thể lên tới $\mathcal{O}(p^3)$ nếu nghịch đảo trực tiếp, nên khó dùng khi số đặc trưng $p$ rất lớn.

---

**HẾT ĐÁP ÁN**
