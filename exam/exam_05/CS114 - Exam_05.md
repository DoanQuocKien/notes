---
title: "CS114 - Exam 05"
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

# **ĐỀ THI MACHINE LEARNING — ĐỀ 05**

* **Thời gian làm bài:** 90 phút
* **Hình thức:** Tự luận + tính toán
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

---


---

## Câu 1. Cross-Entropy Loss và Logistic Regression Hoàn chỉnh

Cho 6 mẫu dữ liệu huấn luyện và mô hình Logistic Regression ban đầu với $\boldsymbol{\beta} = (-0.5, 1.0, -0.8)$ (bias, $\beta_1$, $\beta_2$):

| STT | $x_1$ | $x_2$ | $y$ |
| :--- | :--- | :--- | :--- |
| 1 | 2 | 1 | 1 |
| 2 | 0 | 2 | 0 |
| 3 | 1 | 0 | 1 |
| 4 | 3 | 2 | 1 |
| 5 | -1 | 1 | 0 |
| 6 | 1 | 3 | 0 |

**Hãy:**

1. Tính $z_i$, $\hat{p}_i = \sigma(z_i)$, lớp dự đoán, cross-entropy loss cho từng mẫu.
2. Tính cross-entropy loss trung bình và accuracy.
3. Tính gradient $\nabla J = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$.
4. Cập nhật $\boldsymbol{\beta}$ sau một bước gradient descent với $\alpha = 0.5$.
5. Nhận xét $\boldsymbol{\beta}$ mới so với cũ.

---


---

## Câu 2. Naive Bayes — So sánh Có và Không có Laplace Smoothing

Cho bảng phân loại tin nhắn rác:

| STT | Từ "giảm giá" | Từ "miễn phí" | Từ "họp" | Từ "dự án" | Nhãn |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Có | Có | Không | Không | Rác |
| 2 | Có | Không | Không | Không | Rác |
| 3 | Không | Có | Không | Không | Rác |
| 4 | Không | Không | Có | Có | Bình thường |
| 5 | Không | Không | Có | Không | Bình thường |
| 6 | Không | Không | Không | Có | Bình thường |

Tin nhắn mới: "giảm giá" = Không, "miễn phí" = Có, "họp" = Không, "dự án" = Không.

**Hãy:**

1. **Không dùng** Laplace smoothing: tính $P(\text{Rác} \mid \mathbf{x})$. Nêu vấn đề gặp phải.
2. **Dùng** Laplace smoothing ($\alpha = 1$): tính lại $P(\text{Rác} \mid \mathbf{x})$ và $P(\text{Bình thường} \mid \mathbf{x})$.
3. Kết luận và giải thích tại sao Laplace smoothing quan trọng.

---


---

## Câu 3. Decision Tree — So sánh Entropy và Gini

Cho bảng dữ liệu nhị phân đơn giản:

| STT | A | B | Nhãn |
| :--- | :--- | :--- | :--- |
| 1 | 0 | 0 | - |
| 2 | 0 | 1 | - |
| 3 | 0 | 0 | - |
| 4 | 1 | 0 | + |
| 5 | 1 | 1 | + |
| 6 | 1 | 1 | + |
| 7 | 1 | 0 | + |
| 8 | 0 | 1 | + |

**Hãy:**

1. Tính Information Gain của thuộc tính A theo **Entropy**.
2. Tính Information Gain của thuộc tính A theo **Gini**.
3. Tính Information Gain của thuộc tính B theo **Entropy**.
4. So sánh và kết luận: A hay B tốt hơn? Hai metric có cho cùng kết luận không?

---


---

## Câu 4. Neural Network — Dropout và Batch Normalization

**Hãy:**

1. Cho tầng ẩn có 4 neuron với activation $\mathbf{a} = (0.8, 0.3, 0.6, 0.9)$. Nếu áp dụng Dropout với $p = 0.5$ và mask $\mathbf{M} = (1, 0, 1, 0)$, tính output sau Dropout (inverted dropout).
2. Cho $\mathbf{z} = (2.0, 4.0, 6.0, 8.0)$ trước khi đưa vào activation. Tính Batch Normalization:
   - Tính $\mu_{\mathcal{B}}$ và $\sigma_{\mathcal{B}}^2$.
   - Tính $\hat{z}_i$ (chuẩn hóa) với $\epsilon = 10^{-8}$.
   - Nếu $\gamma = 2$, $\beta = 1$, tính $\tilde{z}_i$.
3. Giải thích vì sao Dropout giống ensemble và vì sao Batch Normalization tăng tốc hội tụ.

---


---

## Câu 5. Dropout trong Neural Network

Trình bày kỹ thuật Inverted Dropout trong huấn luyện mạng Neural sâu. Tại sao Dropout thường được giải thích là có tác dụng tương tự như việc tạo ra một tập hợp (ensemble) gồm rất nhiều mạng con (sub-networks)?

---


---

## Câu 6. Chứng minh MLE của Exponential Family là Lồi

Cho phân phối thuộc Exponential Family:

$$P(y \mid \eta) = b(y)\exp(\eta T(y) - a(\eta))$$

**Hãy:**

1. Viết hàm Log-Likelihood $\ell(\eta)$ cho $n$ mẫu i.i.d.
2. Tính gradient $\nabla_\eta \ell$.
3. Tính Hessian $\nabla^2_\eta \ell$.
4. Chứng minh Hessian $\preceq 0$ (tức log-likelihood lõm, nên negative log-likelihood lồi).
5. Nêu ý nghĩa thực tiễn: MLE cho Exponential Family luôn có nghiệm tối ưu toàn cục duy nhất.

---

## Câu 7. Thử thách: Vanishing Gradient bằng Quy tắc Chuỗi

Xét một mạng neural sâu dạng vô hướng để đơn giản hóa:

$$a^{[0]}=x,\quad z^{[\ell]}=w_{\ell}a^{[\ell-1]},\quad a^{[\ell]}=\sigma(z^{[\ell]}),\quad \ell=1,\ldots,L,$$

với loss $L_{oss}=\frac{1}{2}(a^{[L]}-y)^2$ và sigmoid $\sigma(z)=\frac{1}{1+e^{-z}}$.

**Hãy:**

1. Chứng minh $\sigma'(z)=\sigma(z)(1-\sigma(z))\leq \frac{1}{4}$ với mọi $z$.
2. Dùng quy tắc chuỗi để viết $\left|\frac{\partial L_{oss}}{\partial w_1}\right|$ dưới dạng tích các đạo hàm tầng sau.
3. Giả sử $|w_\ell|\leq 1$ với mọi $\ell$, chứng minh gradient về tầng đầu bị chặn bởi một hằng số nhân với $\left(\frac{1}{4}\right)^{L-1}$.
4. Với $L=8$ và $|a^{[L]}-y|\leq 1$, ước lượng cận trên xấp xỉ của phần suy giảm gradient do các sigmoid tầng sau.
5. Giải thích vì sao ReLU, Batch Normalization và residual connection giúp giảm vấn đề này theo các cơ chế khác nhau.

**HẾT**
