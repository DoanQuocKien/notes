
# **ĐỀ THI MACHINE LEARNING — ĐỀ 05**

* **Thời gian làm bài:** 90 phút
* **Hình thức:** Tự luận + tính toán
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

---

Câu 1\. Cross-Entropy Loss và Logistic Regression Hoàn chỉnh

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

Câu 2\. Naive Bayes — So sánh Có và Không có Laplace Smoothing

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

Câu 3\. Decision Tree — So sánh Entropy và Gini

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

Câu 4\. Neural Network — Dropout và Batch Normalization

**Hãy:**

1. Cho tầng ẩn có 4 neuron với activation $\mathbf{a} = (0.8, 0.3, 0.6, 0.9)$. Nếu áp dụng Dropout với $p = 0.5$ và mask $\mathbf{M} = (1, 0, 1, 0)$, tính output sau Dropout (inverted dropout).
2. Cho $\mathbf{z} = (2.0, 4.0, 6.0, 8.0)$ trước khi đưa vào activation. Tính Batch Normalization:
   - Tính $\mu_\mathcal{B}$ và $\sigma_\mathcal{B}^2$.
   - Tính $\hat{z}_i$ (chuẩn hóa) với $\epsilon = 10^{-8}$.
   - Nếu $\gamma = 2$, $\beta = 1$, tính $\tilde{z}_i$.
3. Giải thích vì sao Dropout giống ensemble và vì sao Batch Normalization tăng tốc hội tụ.

---

Câu 5\. K-means++ — Xác suất Chọn Tâm

Cho 5 điểm 1D: $x = \{1, 3, 5, 15, 17\}$.

Giả sử tâm đầu tiên được chọn là $\mu_1 = 1$.

**Hãy:**

1. Tính khoảng cách $D(x_i)$ từ mỗi điểm đến tâm $\mu_1$.
2. Tính $D(x_i)^2$.
3. Tính xác suất $P(x_i)$ để mỗi điểm được chọn làm tâm thứ hai.
4. Điểm nào có xác suất cao nhất? Giải thích trực giác.
5. Giả sử $\mu_2 = 17$. Cập nhật $D(x_i)$ (khoảng cách đến tâm gần nhất) và tính xác suất cho tâm thứ ba.

---

Câu 6\. Chứng minh MLE của Exponential Family là Lồi

Cho phân phối thuộc Exponential Family:

$$P(y \mid \eta) = b(y)\exp(\eta T(y) - a(\eta))$$

**Hãy:**

1. Viết hàm Log-Likelihood $\ell(\eta)$ cho $n$ mẫu i.i.d.
2. Tính gradient $\nabla_\eta \ell$.
3. Tính Hessian $\nabla^2_\eta \ell$.
4. Chứng minh Hessian $\preceq 0$ (tức log-likelihood lõm, nên negative log-likelihood lồi).
5. Nêu ý nghĩa thực tiễn: MLE cho Exponential Family luôn có nghiệm tối ưu toàn cục duy nhất.

---

Câu 7 (Nâng cao)\. K-means và "Lời nguyền chiều dữ liệu" (Curse of Dimensionality)

Thuật toán K-means tiêu chuẩn sử dụng khoảng cách Euclidean để đánh giá sự tương đồng. Tuy nhiên, khi áp dụng trên dữ liệu có số chiều $p$ rất lớn (ví dụ: ảnh siêu phân giải, vector văn bản), chất lượng phân cụm thường suy giảm nghiêm trọng do "Lời nguyền chiều dữ liệu".

**Hãy:**

1. Theo hiệu ứng của Lời nguyền chiều dữ liệu, khi số chiều $p \to \infty$, tỷ số chênh lệch giữa khoảng cách từ điểm xa nhất và điểm gần nhất trong bộ dữ liệu (ví dụ: $\frac{d_{max} - d_{min}}{d_{min}}$) sẽ có xu hướng tiến tới giá trị nào?
2. Giải thích vì sao hiện tượng thống kê này lại làm vô hiệu hóa cơ chế hoạt động của thuật toán K-means. 
3. Trong không gian có số chiều cực lớn, vị trí của "tâm cụm" (Centroid - trung bình cộng của các điểm) thường có tính chất gì bất lợi so với các điểm dữ liệu thực tế?
4. Đề xuất 2 kỹ thuật tiền xử lý (preprocessing) hoặc cải tiến phổ biến bắt buộc phải làm trước khi chạy K-means trên dữ liệu quá nhiều chiều.

**HẾT**
