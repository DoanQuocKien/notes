---
title: "CS114 - Exam 03"
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

# **ĐỀ THI MACHINE LEARNING — ĐỀ 03**

* **Thời gian làm bài:** 90 phút
* **Hình thức:** Tự luận + tính toán
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

---


---

## Câu 1. Hồi quy Tuyến tính Đơn (OLS từ đầu)

Cho 6 điểm dữ liệu:

| STT | $x$ | $y$ |
| :--- | :--- | :--- |
| 1 | 1 | 3 |
| 2 | 2 | 5 |
| 3 | 3 | 6 |
| 4 | 4 | 9 |
| 5 | 5 | 11 |
| 6 | 6 | 13 |

**Hãy:**

1. Tính $\bar{x}$, $\bar{y}$.
2. Tính $S_{xy} = \sum(x_i - \bar{x})(y_i - \bar{y})$ và $S_{xx} = \sum(x_i - \bar{x})^2$.
3. Tính hệ số góc $\beta_1$ và hệ số chặn $\beta_0$.
4. Dự đoán $\hat{y}$ khi $x = 8$.
5. Tính RSS $= \sum(y_i - \hat{y}_i)^2$ và kiểm tra $\sum e_i = 0$.

---


---

## Câu 2. Chứng minh Hàm Sigmoid từ Exponential Family

**Hãy:**

1. Viết phân phối Gaussian $\mathcal{N}(\mu, \sigma^2)$ với $\sigma^2 = 1$ dưới dạng Exponential Family. Xác định $\eta$, $T(y)$, $a(\eta)$, $b(y)$.
2. Sử dụng tính chất $\mathbb{E}[y] = a'(\eta)$, kiểm tra rằng $\mathbb{E}[y] = \mu$ cho Gaussian ($\sigma^2 = 1$).
3. Sử dụng tính chất $\text{Var}(y) = a''(\eta)$, kiểm tra rằng $\text{Var}(y) = 1$ cho Gaussian ($\sigma^2 = 1$).
4. Giải thích vì sao Gaussian GLM cho ra Linear Regression (identity link) và Bernoulli GLM cho ra Logistic Regression (logit link).

---


---

## Câu 3. Naive Bayes với Laplace Smoothing

Cho bảng dữ liệu phân loại bệnh nhân có bệnh tim hay không:

| STT | Huyết áp | Cholesterol | Hút thuốc | Bệnh tim |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Cao | Cao | Có | Có |
| 2 | Cao | Bình thường | Không | Không |
| 3 | Bình thường | Cao | Có | Có |
| 4 | Bình thường | Bình thường | Không | Không |
| 5 | Cao | Cao | Không | Có |
| 6 | Bình thường | Bình thường | Có | Không |
| 7 | Cao | Cao | Có | Có |
| 8 | Bình thường | Cao | Không | Không |

Bệnh nhân mới: Huyết áp = Cao, Cholesterol = Cao, Hút thuốc = Không.

**Hãy:**

1. Tính prior probability.
2. Tính conditional probabilities cần thiết (có Laplace smoothing $\alpha = 1$).
3. Tính posterior cho hai lớp.
4. Kết luận bệnh nhân mới có bệnh tim hay không.

---


---

## Câu 4. Regularization và Ridge Regression

Cho mô hình hồi quy tuyến tính với hàm loss Ridge:

$$L_{ridge}(\boldsymbol{\beta}) = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \boldsymbol{\beta}^T\mathbf{x}_i)^2 + \lambda\|\boldsymbol{\beta}\|_2^2$$

**Hãy:**

1. Viết gradient $\nabla L_{ridge}$ theo $\boldsymbol{\beta}$.
2. Chứng minh nghiệm đóng Ridge Regression là: $\boldsymbol{\beta}_{ridge} = (\mathbf{X}^T\mathbf{X} + 2n\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$.
3. Cho $\boldsymbol{\beta} = (3.0, -1.5, 0.8)$, train loss $= 0.15$, $\lambda = 0.2$. Tính Ridge objective.
4. Giải thích sự khác biệt giữa Ridge ($L_2$) và Lasso ($L_1$) về mặt ảnh hưởng đến hệ số.

---


---

## Câu 5. Neural Network — Xác định Kích thước Ma trận

Cho mạng fully connected có kiến trúc: 4 input → 5 neuron ẩn (tầng 1) → 3 neuron ẩn (tầng 2) → 2 output. Mini-batch có $m = 10$ mẫu.

**Hãy:**

1. Liệt kê kích thước của $\mathbf{W}^{[1]}$, $\mathbf{b}^{[1]}$, $\mathbf{Z}^{[1]}$, $\mathbf{A}^{[1]}$.
2. Liệt kê kích thước của $\mathbf{W}^{[2]}$, $\mathbf{b}^{[2]}$, $\mathbf{Z}^{[2]}$, $\mathbf{A}^{[2]}$.
3. Liệt kê kích thước của $\mathbf{W}^{[3]}$, $\mathbf{b}^{[3]}$, $\mathbf{Z}^{[3]}$, $\mathbf{A}^{[3]}$.
4. Tổng số tham số (weights + biases) của mạng.
5. Với gradient $\mathbf{dW}^{[2]}$, xác nhận kích thước phải bằng kích thước $\mathbf{W}^{[2]}$.
6. Giải thích vì sao hidden layer hiện đại dùng ReLU thay vì Sigmoid.

---


---

## Câu 6. K-means — Chứng minh Tâm Cụm Là Trung Bình

Cho cụm $C_k = \{x_1, x_2, \ldots, x_m\}$. Hàm mục tiêu cụm:

$$f(\mu) = \sum_{i=1}^{m} \|x_i - \mu\|_2^2$$

**Hãy:**

1. Khai triển $f(\mu)$ dưới dạng $\sum \|x_i\|_2^2 - 2\mu^T\sum x_i + m\|\mu\|_2^2$.
2. Tính gradient $\nabla_\mu f$ và đặt bằng $0$.
3. Chứng minh $\mu^* = \frac{1}{m}\sum_{i=1}^{m} x_i$ (trung bình cộng).
4. Áp dụng: Cho cụm gồm 4 điểm $(1,3)$, $(2,5)$, $(4,1)$, $(5,3)$. Tính tâm cụm tối ưu và distortion.

---

## Câu 7. Thử thách: Newton's Method cho Logistic Regression

Cho Logistic Regression nhị phân với $m$ mẫu, ma trận thiết kế $\mathbf{X}\in\mathbb{R}^{m\times p}$, nhãn $\mathbf{y}\in\{0,1\}^m$, xác suất dự đoán:

$$\mathbf{p}=\sigma(\mathbf{X}\boldsymbol{\beta})$$

và negative log-likelihood:

$$J(\boldsymbol{\beta})=-\sum_{i=1}^{m}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].$$

**Hãy:**

1. Chứng minh $\nabla J(\boldsymbol{\beta})=\mathbf{X}^T(\mathbf{p}-\mathbf{y})$.
2. Chứng minh Hessian có dạng $\nabla^2J(\boldsymbol{\beta})=\mathbf{X}^T\mathbf{R}\mathbf{X}$, trong đó $\mathbf{R}$ là ma trận đường chéo với $R_{ii}=p_i(1-p_i)$.
3. Suy ra vì sao $J$ là hàm lồi.
4. Viết công thức cập nhật Newton:

$$\boldsymbol{\beta}_{new}=\boldsymbol{\beta}-\left(\mathbf{X}^T\mathbf{R}\mathbf{X}\right)^{-1}\mathbf{X}^T(\mathbf{p}-\mathbf{y}).$$

5. So sánh Newton's Method với Gradient Descent về tốc độ hội tụ và chi phí tính toán khi $p$ rất lớn.

**HẾT**
