---
title: "CS114 - Exam 04"
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

# **ĐỀ THI MACHINE LEARNING — ĐỀ 04**

* **Thời gian làm bài:** 90 phút
* **Hình thức:** Tự luận + tính toán
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

---


---

## Câu 1. Logistic Regression — Chứng minh Tính Lồi

Cho hàm mất mát Binary Cross-Entropy của Logistic Regression:

$$J(\boldsymbol{\beta}) = -\sum_{i=1}^{n}\left[y^{(i)}\log\sigma(\boldsymbol{\beta}^T\mathbf{x}^{(i)}) + (1-y^{(i)})\log(1-\sigma(\boldsymbol{\beta}^T\mathbf{x}^{(i)}))\right]$$

**Hãy:**

1. Tính gradient $\nabla J(\boldsymbol{\beta})$ bằng chain rule (trình bày 3 thành phần: Loss → Sigmoid → Tuyến tính).
2. Tính ma trận Hessian $H = \nabla^2 J(\boldsymbol{\beta})$.
3. Chứng minh $H \succeq 0$ (bán xác định dương), từ đó suy ra $J$ là hàm lồi.
4. Giải thích hệ quả thực tiễn: Logistic Regression không có cực tiểu cục bộ.

---


---

## Câu 2. OLS bằng Ma trận — Bộ dữ liệu 2 biến

Cho dữ liệu với mô hình $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$:

| STT | $x_1$ | $x_2$ | $y$ |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 2 | 10 |
| 2 | 2 | 1 | 12 |
| 3 | 3 | 3 | 18 |
| 4 | 4 | 2 | 20 |
| 5 | 5 | 4 | 28 |

**Hãy:**

1. Xây dựng $\mathbf{X}$ (có cột 1) và $\mathbf{y}$.
2. Tính $\mathbf{X}^T\mathbf{X}$ và $\mathbf{X}^T\mathbf{y}$.
3. Giải normal equations (có thể dùng elimination hoặc máy tính).
4. Dự đoán $\hat{y}$ khi $x_1 = 3, x_2 = 5$.

---


---

## Câu 3. GDA — Chứng minh Ranh giới Tuyến tính

Trong Gaussian Discriminant Analysis với $\mathbf{x} \mid y=0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma})$ và $\mathbf{x} \mid y=1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma})$ (ma trận hiệp phương sai chung).

**Hãy:**

1. Viết $P(y=1 \mid \mathbf{x})$ theo Định lý Bayes.
2. Chứng minh rằng $P(y=1 \mid \mathbf{x}) = \sigma(\boldsymbol{\theta}^T\mathbf{x} + \theta_0)$ bằng cách:
   - Tính log-ratio $\log\frac{P(\mathbf{x} \mid y=1)}{P(\mathbf{x} \mid y=0)}$.
   - Chỉ ra số hạng bậc hai $\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$ bị triệt tiêu nhờ $\boldsymbol{\Sigma}$ chung.
   - Xác định $\boldsymbol{\theta}$ và $\theta_0$.
3. Giải thích vì sao nếu $\boldsymbol{\Sigma}_0 \neq \boldsymbol{\Sigma}_1$, ranh giới quyết định sẽ là bậc hai (quadratic).

---


---

## Câu 4. Decision Tree — Regression Tree với MSE

Cho bảng dữ liệu hồi quy:

| STT | $x$ | $y$ |
| :--- | :--- | :--- |
| 1 | 1 | 2.5 |
| 2 | 2 | 3.0 |
| 3 | 3 | 4.5 |
| 4 | 4 | 4.0 |
| 5 | 5 | 8.0 |
| 6 | 6 | 7.5 |
| 7 | 7 | 9.0 |
| 8 | 8 | 10.0 |

**Hãy:**

1. Tính MSE khi không chia (dùng toàn bộ dữ liệu làm 1 lá, dự đoán = $\bar{y}$).
2. Xét split $x \leq 4$. Tính $\bar{y}_{left}$, $\bar{y}_{right}$, SSE trái, SSE phải, tổng SSE.
3. Xét split $x \leq 3$. Tính tương tự.
4. Chọn split tốt hơn theo tiêu chí SSE nhỏ hơn.

---


---

## Câu 5. Backpropagation qua Mạng 2-3-1

Cho mạng gồm:

* 2 input: $x_1 = 1$, $x_2 = -1$
* 1 hidden layer gồm 3 neuron, dùng sigmoid
* 1 output neuron, dùng sigmoid
* Loss: Binary cross-entropy $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$
* Nhãn thật: $y = 1$

**Trọng số tầng ẩn:**

$$\mathbf{W}^{[1]} = \begin{pmatrix} 0.2 & -0.3 \\ 0.4 & 0.1 \\ -0.1 & 0.5 \end{pmatrix}, \quad \mathbf{b}^{[1]} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$$

**Trọng số tầng output:**

$$\mathbf{W}^{[2]} = \begin{pmatrix} 0.5 & -0.3 & 0.4 \end{pmatrix}, \quad b^{[2]} = 0$$

**Hãy thực hiện:**

1. Forward pass: tính $\mathbf{Z}^{[1]}$, $\mathbf{A}^{[1]}$, $Z^{[2]}$, $\hat{y}$.
2. Tính loss.
3. Tính $dZ^{[2]}$ (biết sigmoid + BCE cho $dZ = \hat{y} - y$).
4. Tính $dW^{[2]}$ và $db^{[2]}$.

---


---

## Câu 6. Gradient Descent với Regularization

Cho mô hình tuyến tính $\hat{y} = \beta_0 + \beta_1 x$ với Ridge regularization. Bắt đầu từ $\beta_0 = 0$, $\beta_1 = 0$.

Dữ liệu:

| $x$ | $y$ |
| :--- | :--- |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |

**Hãy:**

1. Tính gradient của loss OLS (không có regularization) tại $\boldsymbol{\beta} = (0, 0)$.
2. Tính gradient của Ridge penalty $\lambda\|\boldsymbol{\beta}\|_2^2$ tại $\boldsymbol{\beta} = (0, 0)$ với $\lambda = 0.1$.
3. Tính gradient tổng và cập nhật $\boldsymbol{\beta}$ sau một bước với $\alpha = 0.1$.
4. So sánh $\boldsymbol{\beta}$ mới với trường hợp không regularization. Nhận xét.

---

## Câu 7. Thử thách: Batch Normalization Forward và Backward

Cho một mini-batch gồm 4 giá trị tiền kích hoạt của một neuron:

$$z=(1,3,5,7),\quad \epsilon=0,\quad \gamma=2,\quad \beta=-1.$$

Batch Normalization được định nghĩa bởi:

$$\mu_{\mathcal{B}}=\frac{1}{m}\sum_{i=1}^{m}z_i,\quad
\sigma_{\mathcal{B}}^2=\frac{1}{m}\sum_{i=1}^{m}(z_i-\mu_{\mathcal{B}})^2,$$

$$\hat{z}_i=\frac{z_i-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}},\quad
\tilde{z}_i=\gamma\hat{z}_i+\beta.$$

**Hãy:**

1. Tính $\mu_{\mathcal{B}}$, $\sigma_{\mathcal{B}}^2$, toàn bộ $\hat{z}_i$ và $\tilde{z}_i$.
2. Giả sử gradient đi từ tầng sau là $\frac{\partial L}{\partial \tilde{z}}=(1,-1,2,-2)$. Tính:

$$\frac{\partial L}{\partial \gamma}=\sum_{i=1}^{m}\frac{\partial L}{\partial \tilde{z}_i}\hat{z}_i,\quad
\frac{\partial L}{\partial \beta}=\sum_{i=1}^{m}\frac{\partial L}{\partial \tilde{z}_i}.$$

3. Chứng minh công thức ngắn gọn sau cho gradient đi về $z_i$:

$$\frac{\partial L}{\partial z_i}
=\frac{\gamma}{m\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}}
\left[
m g_i-\sum_{j=1}^{m}g_j-\hat{z}_i\sum_{j=1}^{m}g_j\hat{z}_j
\right],$$

trong đó $g_i=\frac{\partial L}{\partial \tilde{z}_i}$.
4. Tính $\frac{\partial L}{\partial z_i}$ cho cả 4 phần tử.
5. Giải thích tại sao kết quả gradient theo $z$ có tổng bằng 0 trong mini-batch này.

**HẾT**
