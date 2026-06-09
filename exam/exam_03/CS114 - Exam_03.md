
# **ĐỀ THI MACHINE LEARNING — ĐỀ 03**

* **Thời gian làm bài:** 90 phút
* **Hình thức:** Tự luận + tính toán
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

---

Câu 1\. Hồi quy Tuyến tính Đơn (OLS từ đầu)

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

Câu 2\. Chứng minh Hàm Sigmoid từ Exponential Family

**Hãy:**

1. Viết phân phối Gaussian $\mathcal{N}(\mu, \sigma^2)$ với $\sigma^2 = 1$ dưới dạng Exponential Family. Xác định $\eta$, $T(y)$, $a(\eta)$, $b(y)$.
2. Sử dụng tính chất $\mathbb{E}[y] = a'(\eta)$, kiểm tra rằng $\mathbb{E}[y] = \mu$ cho Gaussian ($\sigma^2 = 1$).
3. Sử dụng tính chất $\text{Var}(y) = a''(\eta)$, kiểm tra rằng $\text{Var}(y) = 1$ cho Gaussian ($\sigma^2 = 1$).
4. Giải thích vì sao Gaussian GLM cho ra Linear Regression (identity link) và Bernoulli GLM cho ra Logistic Regression (logit link).

---

Câu 3\. Naive Bayes với Laplace Smoothing

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

Câu 4\. Regularization và Ridge Regression

Cho mô hình hồi quy tuyến tính với hàm loss Ridge:

$$L_{ridge}(\boldsymbol{\beta}) = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \boldsymbol{\beta}^T\mathbf{x}_i)^2 + \lambda\|\boldsymbol{\beta}\|_2^2$$

**Hãy:**

1. Viết gradient $\nabla L_{ridge}$ theo $\boldsymbol{\beta}$.
2. Chứng minh nghiệm đóng Ridge Regression là: $\boldsymbol{\beta}_{ridge} = (\mathbf{X}^T\mathbf{X} + 2n\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$.
3. Cho $\boldsymbol{\beta} = (3.0, -1.5, 0.8)$, train loss $= 0.15$, $\lambda = 0.2$. Tính Ridge objective.
4. Giải thích sự khác biệt giữa Ridge ($L_2$) và Lasso ($L_1$) về mặt ảnh hưởng đến hệ số.

---

Câu 5\. Neural Network — Xác định Kích thước Ma trận

Cho mạng fully connected có kiến trúc: 4 input → 5 neuron ẩn (tầng 1) → 3 neuron ẩn (tầng 2) → 2 output. Mini-batch có $m = 10$ mẫu.

**Hãy:**

1. Liệt kê kích thước của $\mathbf{W}^{[1]}$, $\mathbf{b}^{[1]}$, $\mathbf{Z}^{[1]}$, $\mathbf{A}^{[1]}$.
2. Liệt kê kích thước của $\mathbf{W}^{[2]}$, $\mathbf{b}^{[2]}$, $\mathbf{Z}^{[2]}$, $\mathbf{A}^{[2]}$.
3. Liệt kê kích thước của $\mathbf{W}^{[3]}$, $\mathbf{b}^{[3]}$, $\mathbf{Z}^{[3]}$, $\mathbf{A}^{[3]}$.
4. Tổng số tham số (weights + biases) của mạng.
5. Với gradient $\mathbf{dW}^{[2]}$, xác nhận kích thước phải bằng kích thước $\mathbf{W}^{[2]}$.
6. Giải thích vì sao hidden layer hiện đại dùng ReLU thay vì Sigmoid.

---

Câu 6\. K-means — Chứng minh Tâm Cụm Là Trung Bình

Cho cụm $C_k = \{x_1, x_2, \ldots, x_m\}$. Hàm mục tiêu cụm:

$$f(\mu) = \sum_{i=1}^{m} \|x_i - \mu\|^2$$

**Hãy:**

1. Khai triển $f(\mu)$ dưới dạng $\sum \|x_i\|^2 - 2\mu^T\sum x_i + m\|\mu\|^2$.
2. Tính gradient $\nabla_\mu f$ và đặt bằng $0$.
3. Chứng minh $\mu^* = \frac{1}{m}\sum_{i=1}^{m} x_i$ (trung bình cộng).
4. Áp dụng: Cho cụm gồm 4 điểm $(1,3)$, $(2,5)$, $(4,1)$, $(5,3)$. Tính tâm cụm tối ưu và distortion.

---

Câu 7 (Nâng cao)\. Inductive Bias và Biểu diễn Hàm

Bạn được giao một bộ dữ liệu phân loại nhị phân 2D. Các điểm thuộc Lớp 1 nằm gọn trong một hình tròn bán kính $R$ ở tâm hệ tọa độ, và các điểm thuộc Lớp 0 bao quanh nằm ngoài hình tròn đó. Không có nhiễu (noise) trong dữ liệu.

**Hãy:**

1. Mô hình Logistic Regression tiêu chuẩn (chỉ dùng các feature đầu vào là $x_1, x_2$) có thể phân loại đúng 100% bộ dữ liệu này không? Giải thích vì sao. Đề xuất một cách biến đổi dữ liệu (feature engineering) đơn giản để mô hình này giải quyết được bài toán.
2. Một Decision Tree tiêu chuẩn (các đường cắt luôn vuông góc với trục tọa độ) có thể phân loại tốt bộ dữ liệu này không? Mô tả hình dáng của ranh giới quyết định (decision boundary) mà cây sẽ tạo ra xung quanh hình tròn.
3. Nếu bài toán thay đổi: ranh giới thực sự của dữ liệu lại là đường chéo thẳng $x_1 + x_2 = 1$. So sánh độ phức tạp (số lượng tham số/node) cần thiết của Logistic Regression và Decision Tree để mô hình hóa chính xác ranh giới này.

**HẾT**
