

# **ĐỀ THI MACHINE LEARNING — ĐỀ 01**

* **Thời gian làm bài:** 90 phút
* **Hình thức:** Tự luận + tính toán
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

---

Câu 1\. OLS Linear Regression bằng Ma trận

Cho 5 điểm dữ liệu huấn luyện với mô hình $\hat{y} = \beta_0 + \beta_1 x$:

| STT | $x$ | $y$ |
| :--- | :--- | :--- |
| 1 | 0 | 1 |
| 2 | 1 | 3 |
| 3 | 2 | 4 |
| 4 | 3 | 6 |
| 5 | 4 | 8 |

**Hãy:**

1. Xây dựng ma trận thiết kế $\mathbf{X}$ (thêm cột 1 cho intercept) và vector $\mathbf{y}$.
2. Tính $\mathbf{X}^T \mathbf{X}$ và $\mathbf{X}^T \mathbf{y}$.
3. Giải hệ Normal Equations $\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$ để tìm $\beta_0, \beta_1$.
4. Dự đoán $\hat{y}$ khi $x = 6$.
5. Kiểm tra điều kiện $\mathbf{X}^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$.

---

Câu 2\. Logistic Regression — Thay số Tính Xác suất và Loss

Cho mô hình Logistic Regression có $\boldsymbol{\beta} = (0.5, -1.2, 0.8)$, phần tử đầu tiên là bias. Cho bảng 8 mẫu sau:

| STT | $x_1$ | $x_2$ | $y$ |
| :--- | :--- | :--- | :--- |
| 1 | 1.0 | 0.5 | 1 |
| 2 | -0.5 | 1.0 | 0 |
| 3 | 2.0 | 1.5 | 1 |
| 4 | 0.0 | 0.0 | 0 |
| 5 | 1.5 | -1.0 | 0 |
| 6 | -1.0 | 2.0 | 1 |
| 7 | 3.0 | 0.5 | 1 |
| 8 | 0.5 | 1.0 | 1 |

**Hãy:**

1. Tính điểm tuyến tính $z$, xác suất $\hat{p}$, lớp dự đoán cho từng mẫu.
2. Tính cross-entropy loss trung bình.
3. Tính accuracy.
4. Nêu nhận xét về mẫu nào mô hình dự đoán sai và giải thích.

---

Câu 3\. Họ Phân Phối Số Mũ (Exponential Family) và GLM

**Hãy:**

1. Viết phân phối Bernoulli $P(y \mid \phi) = \phi^y (1-\phi)^{1-y}$ dưới dạng chuẩn tắc Exponential Family. Xác định rõ $\eta$, $T(y)$, $a(\eta)$, $b(y)$.
2. Chứng minh rằng $\phi = \sigma(\eta) = \frac{1}{1 + e^{-\eta}}$ (tức hàm Sigmoid xuất hiện tự nhiên từ Exponential Family).
3. Cho $\boldsymbol{\beta} = (0.3, 1.5, -0.5)$ và $\mathbf{x} = (1, 2, 3)$. Tính dự đoán cho 3 trường hợp: Gaussian GLM, Bernoulli GLM, Poisson GLM.

---

Câu 4\. Neural Network Forward Pass — Mạng 3-2-1

Cho mạng gồm:

* 3 input: $x_1 = 0.5$, $x_2 = -1.0$, $x_3 = 2.0$
* 1 hidden layer gồm 2 neuron ($h_1, h_2$), dùng ReLU activation
* 1 output neuron ($o$), dùng sigmoid activation

**Trọng số:**

$$\mathbf{W}^{[1]} = \begin{pmatrix} 0.3 & -0.2 & 0.5 \\ 0.1 & 0.4 & -0.3 \end{pmatrix}, \quad \mathbf{b}^{[1]} = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}$$

$$\mathbf{W}^{[2]} = \begin{pmatrix} 0.7 & -0.5 \end{pmatrix}, \quad b^{[2]} = 0.2$$

**Hãy:**

1. Tính $\mathbf{Z}^{[1]}$, $\mathbf{A}^{[1]}$ (qua ReLU).
2. Tính $Z^{[2]}$ và xác suất output $\hat{y} = \sigma(Z^{[2]})$.
3. Nếu nhãn thật $y = 1$, tính binary cross-entropy loss.
4. Tính $dZ^{[2]}$, $dW^{[2]}$, $db^{[2]}$ (backprop bước đầu tiên tại output).

---

Câu 5\. Decision Tree — Gini Index

Cho bảng dữ liệu phân loại khách hàng mua sản phẩm:

| STT | Tuổi | Thu nhập | Sinh viên | Mua |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Trẻ | Cao | Không | Không |
| 2 | Trẻ | Cao | Không | Không |
| 3 | Trung niên | Cao | Không | Có |
| 4 | Già | Trung bình | Không | Có |
| 5 | Già | Thấp | Có | Có |
| 6 | Già | Thấp | Có | Không |
| 7 | Trung niên | Thấp | Có | Có |
| 8 | Trẻ | Trung bình | Không | Không |
| 9 | Trẻ | Thấp | Có | Có |
| 10 | Già | Trung bình | Có | Có |

**Hãy:**

1. Tính Gini Index ban đầu của tập dữ liệu.
2. Tính Gini Gain cho thuộc tính **Tuổi** và **Sinh viên**.
3. Chọn thuộc tính nào tốt hơn để chia tại node gốc.
4. Vẽ cây sau bước chia đầu tiên.

---

Câu 6\. K-means Clustering — Trace Thuật Toán

Cho 6 điểm dữ liệu trong không gian 2D:

| STT | $x_1$ | $x_2$ |
| :--- | :--- | :--- |
| 1 | 1 | 2 |
| 2 | 2 | 1 |
| 3 | 2 | 3 |
| 4 | 8 | 7 |
| 5 | 9 | 8 |
| 6 | 8 | 9 |

Cho $K = 2$, khởi tạo tâm cụm:

$$\mu_1^{(0)} = (1, 2), \quad \mu_2^{(0)} = (9, 8)$$

**Hãy:**

1. Thực hiện bước gán cụm (iteration 0) và tính distortion $J$.
2. Cập nhật tâm cụm.
3. Thực hiện bước gán cụm (iteration 1) và kiểm tra hội tụ.
4. Giải thích vì sao K-means luôn hội tụ nhưng có thể rơi vào cực tiểu cục bộ.

---

Câu 7\. Bias-Variance Decomposition

**Hãy:**

1. Viết phương trình phân rã Test MSE thành Bias², Variance, và Noise.
2. Chứng minh từng bước phân rã, bắt đầu từ $\mathbb{E}[(y - \hat{f})^2]$ với $y = f(\mathbf{x}) + \epsilon$, $\mathbb{E}[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2$.
3. Cho bảng cross-validation sau, chọn $\lambda$ tốt nhất:

| $\lambda$ | fold 1 | fold 2 | fold 3 | fold 4 |
| :--- | :--- | :--- | :--- | :--- |
| 0.00 | 1.50 | 1.65 | 1.40 | 1.55 |
| 0.05 | 0.85 | 0.90 | 0.82 | 0.88 |
| 0.50 | 0.60 | 0.65 | 0.58 | 0.62 |
| 5.00 | 0.95 | 1.00 | 0.92 | 0.98 |

---

Câu 8 (Nâng cao)\. SGD và Khả Năng Tổng Quát Hóa (Generalization)

Cho một mạng neural network được huấn luyện bằng mini-batch SGD. Một sinh viên nhận thấy mô hình bị overfitting (train loss rất thấp nhưng validation loss cao). Sinh viên quyết định tăng kích thước mini-batch (batch size) lên rất lớn (gần bằng toàn bộ tập dữ liệu) để tính gradient chính xác hơn, hy vọng giảm overfitting.

**Hãy:**

1. Giải thích tại sao việc tăng batch size lên quá lớn lại có thể làm *tăng* nguy cơ overfitting thay vì giảm. (Gợi ý: Liên hệ với tính ngẫu nhiên - stochasticity - và tính chất của các điểm cực tiểu cục bộ dốc/bằng phẳng).
2. Viết công thức xấp xỉ liên hệ giữa tỷ lệ $\frac{\text{learning\_rate}}{\text{batch\_size}}$ với mức độ nhiễu (noise) trong quá trình cập nhật SGD. Từ đó suy ra: nếu ta buộc phải tăng batch size lên $k$ lần, ta nên điều chỉnh learning rate như thế nào để cố gắng giữ nguyên hiệu ứng điều chuẩn (implicit regularization) của SGD?

**HẾT**
