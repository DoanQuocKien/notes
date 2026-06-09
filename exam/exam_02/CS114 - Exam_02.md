
# **ĐỀ THI MACHINE LEARNING — ĐỀ 02**

* **Thời gian làm bài:** 90 phút
* **Hình thức:** Tự luận + tính toán
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

---

Câu 1\. MLE cho Linear Regression — Chứng minh OLS

Cho tập dữ liệu $\{(x_i, y_i)\}_{i=1}^{n}$ với giả định nhiễu Gaussian:

$$y_i = \boldsymbol{\beta}^T \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

**Hãy:**

1. Viết hàm mật độ xác suất $P(y_i \mid \mathbf{x}_i; \boldsymbol{\beta})$ dưới dạng Gaussian.
2. Viết hàm Likelihood $L(\boldsymbol{\beta})$ cho toàn bộ tập dữ liệu (giả định i.i.d.).
3. Chứng minh rằng việc cực đại hóa Log-Likelihood tương đương với việc cực tiểu hóa tổng bình phương sai số (OLS).
4. Suy ra Normal Equations từ gradient của hàm loss ma trận $L = \frac{1}{2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$.

---

Câu 2\. Gaussian Discriminant Analysis (GDA)

Cho bảng dữ liệu 1 chiều ($p=1$) với nhãn nhị phân:

| STT | $x$ | $y$ |
| :--- | :--- | :--- |
| 1 | 1.0 | 0 |
| 2 | 1.5 | 0 |
| 3 | 2.0 | 0 |
| 4 | 2.5 | 0 |
| 5 | 5.0 | 1 |
| 6 | 5.5 | 1 |
| 7 | 6.0 | 1 |
| 8 | 6.5 | 1 |

**Hãy:**

1. Ước lượng prior $P(y=0)$, $P(y=1)$.
2. Ước lượng mean $\mu_0$, $\mu_1$.
3. Ước lượng variance chung $\sigma^2$ theo công thức GDA.
4. Phân loại điểm mới $x = 3.5$ bằng cách tính $P(y=0 \mid x)$ và $P(y=1 \mid x)$.
5. Tìm ranh giới quyết định (điểm $x^*$ mà tại đó $P(y=0 \mid x) = P(y=1 \mid x)$).

---

Câu 3\. Logistic Regression — Gradient Descent Một Bước

Cho mô hình Logistic Regression có $\boldsymbol{\beta}_{init} = (0, 0, 0)$ (bias, $\beta_1$, $\beta_2$). Cho 6 mẫu:

| STT | $x_1$ | $x_2$ | $y$ |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 2 | 1 | 1 |
| 4 | 1 | 2 | 0 |
| 5 | 3 | 1 | 1 |
| 6 | 0 | 3 | 0 |

**Hãy:**

1. Tính xác suất dự đoán ban đầu cho tất cả 6 mẫu (khi $\boldsymbol{\beta} = \mathbf{0}$).
2. Tính gradient $\nabla J = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$.
3. Cập nhật trọng số sau một bước gradient descent với $\alpha = 0.5$.
4. Giải thích ý nghĩa hướng cập nhật của từng hệ số.

---

Câu 4\. Decision Tree — Entropy và Information Gain

Cho bảng dữ liệu phân loại rủi ro tín dụng:

| STT | Trình độ | Nợ | Thu nhập | Rủi ro |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Cao | Nhiều | Thấp | Cao |
| 2 | Cao | Ít | Trung bình | Cao |
| 3 | Cao | Ít | Cao | Thấp |
| 4 | Trung bình | Nhiều | Thấp | Cao |
| 5 | Trung bình | Ít | Cao | Thấp |
| 6 | Trung bình | Ít | Trung bình | Thấp |
| 7 | Thấp | Nhiều | Thấp | Cao |
| 8 | Thấp | Ít | Cao | Thấp |
| 9 | Thấp | Ít | Trung bình | Thấp |
| 10 | Thấp | Ít | Thấp | Thấp |

**Hãy:**

1. Tính Entropy ban đầu $H(S)$.
2. Tính Information Gain cho thuộc tính **Nợ**.
3. Tính Information Gain cho thuộc tính **Thu nhập**.
4. Chọn thuộc tính tốt nhất để chia tại node gốc.

---

Câu 5\. Backpropagation — Mạng 2-2-1 với Squared Error

Cho mạng neural network:

* 2 input: $x_1 = 2$, $x_2 = -1$
* 1 hidden layer gồm 2 neuron, dùng sigmoid
* 1 output neuron, dùng sigmoid
* Loss: $L = \frac{1}{2}(\hat{y} - y)^2$
* Nhãn thật: $y = 0$

**Trọng số:**

| Kết nối | Trọng số |
| :--- | :--- |
| $x_1 \to h_1$ | 0.5 |
| $x_2 \to h_1$ | -0.3 |
| $x_1 \to h_2$ | -0.2 |
| $x_2 \to h_2$ | 0.4 |
| $h_1 \to o$ | 0.6 |
| $h_2 \to o$ | -0.4 |

**Bias:** $b_{h1} = 0.2$, $b_{h2} = -0.1$, $b_o = 0.3$

**Hãy thực hiện:**

1. Forward propagation: tính $h_1$, $h_2$, $\hat{y}$.
2. Tính loss $L$.
3. Tính $\delta_o$, $\delta_{h1}$, $\delta_{h2}$ (backpropagation).
4. Tính gradient cho tất cả 6 trọng số.
5. Cập nhật trọng số với $\eta = 0.1$.

*Biết rằng:* $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

---

Câu 6\. K-means — Chọn K và Distortion

Cho 8 điểm dữ liệu 1D:

$$x = \{1, 2, 3, 10, 11, 12, 20, 21\}$$

**Hãy:**

1. Với $K = 2$, khởi tạo $\mu_1 = 2$, $\mu_2 = 11$. Thực hiện 2 vòng lặp K-means. Tính distortion sau mỗi vòng.
2. Tính distortion cuối cùng cho $K = 3$ nếu các cụm tối ưu là $\{1,2,3\}$, $\{10,11,12\}$, $\{20,21\}$.
3. Dựa trên kết quả, vẽ đồ thị distortion theo $K$ và chọn $K$ hợp lý theo phương pháp elbow.

---

Câu 7 (Nâng cao)\. Generative vs Discriminative Models

Xét hai mô hình Naive Bayes (Generative) và Logistic Regression (Discriminative) được sử dụng để giải quyết cùng một bài toán phân loại nhị phân. Giả sử dữ liệu thực tế *hoàn toàn tuân theo* các giả định của Naive Bayes (các đặc trưng độc lập có điều kiện theo lớp, phân phối Gaussian).

**Hãy:**

1. Khi số lượng dữ liệu huấn luyện $n$ tiến tới vô cực ($n \to \infty$), mô hình nào sẽ có Test Error nhỏ hơn, hay cả hai bằng nhau? Vì sao?
2. Khi số lượng dữ liệu huấn luyện $n$ rất nhỏ, mô hình nào hội tụ đến Test Error tiệm cận của nó nhanh hơn (cần ít dữ liệu hơn)? Phân tích dựa trên số lượng tham số cần ước lượng.
3. Trong thực tế, giả định "độc lập có điều kiện" hiếm khi thỏa mãn. Khi đó, biểu đồ Learning Curve (Test Error theo $n$) của hai mô hình sẽ cắt nhau như thế nào khi $n$ tăng dần? Giải thích.

**HẾT**
