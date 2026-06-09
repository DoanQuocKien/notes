
# **ĐÁP ÁN ĐỀ THI MACHINE LEARNING — ĐỀ 05**

---

## Câu 1. Logistic Regression Hoàn chỉnh

$\boldsymbol{\beta} = (-0.5, 1.0, -0.8)$, $z = -0.5 + 1.0x_1 - 0.8x_2$

### 1.1 Bảng tính

| STT | $z$ | $\hat{p}$ | Lớp | $y$ | Đúng? | Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | $-0.5+2-0.8 = 0.7$ | 0.668 | 1 | 1 | ✓ | $-\log(0.668) = 0.404$ |
| 2 | $-0.5+0-1.6 = -2.1$ | 0.109 | 0 | 0 | ✓ | $-\log(0.891) = 0.115$ |
| 3 | $-0.5+1-0 = 0.5$ | 0.622 | 1 | 1 | ✓ | $-\log(0.622) = 0.475$ |
| 4 | $-0.5+3-1.6 = 0.9$ | 0.711 | 1 | 1 | ✓ | $-\log(0.711) = 0.342$ |
| 5 | $-0.5-1-0.8 = -2.3$ | 0.091 | 0 | 0 | ✓ | $-\log(0.909) = 0.095$ |
| 6 | $-0.5+1-2.4 = -1.9$ | 0.130 | 0 | 0 | ✓ | $-\log(0.870) = 0.139$ |

### 1.2

$$\bar{J} = \frac{0.404+0.115+0.475+0.342+0.095+0.139}{6} = \frac{1.570}{6} = 0.262$$

$$\text{Accuracy} = \frac{6}{6} = 1.0 \quad (100\%)$$

### 1.3 Gradient

Sai số $\hat{p}_i - y_i$:

| STT | $\hat{p} - y$ | $x_0$ | $x_1$ | $x_2$ |
| :--- | :--- | :--- | :--- | :--- |
| 1 | -0.332 | 1 | 2 | 1 |
| 2 | 0.109 | 1 | 0 | 2 |
| 3 | -0.378 | 1 | 1 | 0 |
| 4 | -0.289 | 1 | 3 | 2 |
| 5 | 0.091 | 1 | -1 | 1 |
| 6 | 0.130 | 1 | 1 | 3 |

$$\nabla_{\beta_0} = \frac{1}{6}(-0.332+0.109-0.378-0.289+0.091+0.130) = \frac{-0.669}{6} = -0.1115$$

$$\nabla_{\beta_1} = \frac{1}{6}[(-0.332)(2)+(0.109)(0)+(-0.378)(1)+(-0.289)(3)+(0.091)(-1)+(0.130)(1)]$$

$$= \frac{1}{6}(-0.664+0-0.378-0.867-0.091+0.130) = \frac{-1.870}{6} = -0.3117$$

$$\nabla_{\beta_2} = \frac{1}{6}[(-0.332)(1)+(0.109)(2)+(-0.378)(0)+(-0.289)(2)+(0.091)(1)+(0.130)(3)]$$

$$= \frac{1}{6}(-0.332+0.218+0-0.578+0.091+0.390) = \frac{-0.211}{6} = -0.0352$$

### 1.4 Cập nhật

$$\boldsymbol{\beta}_{new} = \begin{pmatrix}-0.5\\1.0\\-0.8\end{pmatrix} - 0.5\begin{pmatrix}-0.1115\\-0.3117\\-0.0352\end{pmatrix} = \begin{pmatrix}-0.4443\\1.1559\\-0.7824\end{pmatrix}$$

### 1.5 Nhận xét

- Bias tăng: mô hình đẩy xác suất lớp 1 lên một chút.
- $\beta_1$ tăng: $x_1$ lớn tương quan mạnh hơn với $y=1$.
- $\beta_2$ tăng nhẹ (bớt âm): điều chỉnh nhẹ mối quan hệ âm với $x_2$.
- Accuracy đã 100% nên các thay đổi nhỏ, mô hình đang tinh chỉnh để tăng confidence.

---

## Câu 2. Naive Bayes — Laplace Smoothing

### 2.1 Không dùng Laplace

Prior: $P(\text{Rác}) = 3/6 = 0.5$, $P(\text{BT}) = 3/6 = 0.5$

**Lớp Rác (3 mẫu: dòng 1,2,3):**
- $P(\text{giảm giá=Không} \mid \text{Rác}) = 1/3$
- $P(\text{miễn phí=Có} \mid \text{Rác}) = 2/3$
- $P(\text{họp=Không} \mid \text{Rác}) = 3/3 = 1$
- $P(\text{dự án=Không} \mid \text{Rác}) = 3/3 = 1$

$$\text{score}_\text{Rác} = 0.5 \times \frac{1}{3} \times \frac{2}{3} \times 1 \times 1 = 0.1111$$

**Lớp BT (3 mẫu: dòng 4,5,6):**
- $P(\text{giảm giá=Không} \mid \text{BT}) = 3/3 = 1$
- $P(\text{miễn phí=Có} \mid \text{BT}) = \mathbf{0/3 = 0}$ ← **Vấn đề!**

$$\text{score}_\text{BT} = 0.5 \times 1 \times 0 \times \cdots = \mathbf{0}$$

**Vấn đề:** Xác suất bằng 0 triệt tiêu toàn bộ posterior. Mô hình luôn kết luận Rác dù "miễn phí" chỉ đơn giản chưa xuất hiện trong lớp BT ở tập huấn luyện.

### 2.2 Với Laplace ($\alpha = 1$)

Công thức: $P(w \mid c) = \frac{\text{count}(w, c) + 1}{N_c + 2}$

**Lớp Rác ($N_c = 3$):**
- $P(\text{giảm giá=Không} \mid \text{Rác}) = \frac{1+1}{3+2} = 2/5 = 0.4$
- $P(\text{miễn phí=Có} \mid \text{Rác}) = \frac{2+1}{3+2} = 3/5 = 0.6$
- $P(\text{họp=Không} \mid \text{Rác}) = \frac{3+1}{3+2} = 4/5 = 0.8$
- $P(\text{dự án=Không} \mid \text{Rác}) = \frac{3+1}{3+2} = 4/5 = 0.8$

$$\text{score}_\text{Rác} = 0.5 \times 0.4 \times 0.6 \times 0.8 \times 0.8 = 0.5 \times 0.1536 = 0.07680$$

**Lớp BT ($N_c = 3$):**
- $P(\text{giảm giá=Không} \mid \text{BT}) = \frac{3+1}{3+2} = 4/5 = 0.8$
- $P(\text{miễn phí=Có} \mid \text{BT}) = \frac{0+1}{3+2} = 1/5 = 0.2$
- $P(\text{họp=Không} \mid \text{BT}) = \frac{1+1}{3+2} = 2/5 = 0.4$
- $P(\text{dự án=Không} \mid \text{BT}) = \frac{1+1}{3+2} = 2/5 = 0.4$

$$\text{score}_\text{BT} = 0.5 \times 0.8 \times 0.2 \times 0.4 \times 0.4 = 0.5 \times 0.02560 = 0.01280$$

**Posterior:**

$$P(\text{Rác} \mid \mathbf{x}) = \frac{0.07680}{0.07680 + 0.01280} = \frac{0.07680}{0.08960} = 0.857$$

### 2.3 Kết luận

Phân loại: **Rác** (xác suất 85.7%).

Laplace smoothing cần thiết vì nó ngăn xác suất bằng 0 triệt tiêu toàn bộ tích xác suất. Nó "cho phép" các sự kiện chưa quan sát có một xác suất nhỏ hợp lý.

---

## Câu 3. Decision Tree — Entropy vs Gini

### Dữ liệu: 8 mẫu, 5 (+), 3 (-)

### 3.1 IG(A) theo Entropy

Entropy cha: $H = -\frac{5}{8}\log_2\frac{5}{8} - \frac{3}{8}\log_2\frac{3}{8}$

$= -0.625(-0.678) - 0.375(-1.415) = 0.424 + 0.531 = 0.954$

**A=0** (dòng 1,2,3,8): 1(+), 3(-) → $H = -\frac{1}{4}\log_2\frac{1}{4} - \frac{3}{4}\log_2\frac{3}{4} = 0.811$

**A=1** (dòng 4,5,6,7): 4(+), 0(-) → $H = 0$

$$IG_E(A) = 0.954 - \frac{4}{8}(0.811) - \frac{4}{8}(0) = 0.954 - 0.406 = 0.549$$

### 3.2 IG(A) theo Gini

Gini cha: $G = 1 - (5/8)^2 - (3/8)^2 = 1 - 0.391 - 0.141 = 0.469$

**A=0**: $G = 1 - (1/4)^2 - (3/4)^2 = 1 - 0.0625 - 0.5625 = 0.375$

**A=1**: $G = 0$

$$IG_G(A) = 0.469 - \frac{4}{8}(0.375) - \frac{4}{8}(0) = 0.469 - 0.188 = 0.281$$

### 3.3 IG(B) theo Entropy

**B=0** (dòng 1,3,4,7): 2(+), 2(-) → $H = 1.0$

**B=1** (dòng 2,5,6,8): 3(+), 1(-) → $H = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} = 0.811$

$$IG_E(B) = 0.954 - \frac{4}{8}(1.0) - \frac{4}{8}(0.811) = 0.954 - 0.500 - 0.406 = 0.049$$

### 3.4 So sánh

| Thuộc tính | IG (Entropy) | IG (Gini) |
| :--- | :--- | :--- |
| **A** | **0.549** | **0.281** |
| B | 0.049 | — |

**A tốt hơn B** theo cả hai metric. Entropy và Gini thường cho cùng kết luận về thứ tự, chỉ khác giá trị tuyệt đối.

---

## Câu 4. Dropout và Batch Normalization

### 4.1 Inverted Dropout

Dropout mask: $\mathbf{M} = (1, 0, 1, 0)$, $p = 0.5$

$$\mathbf{a}_{dropout} = \frac{\mathbf{a} \odot \mathbf{M}}{1 - p} = \frac{(0.8, 0, 0.6, 0)}{0.5} = (1.6, 0, 1.2, 0)$$

### 4.2 Batch Normalization

$\mathbf{z} = (2, 4, 6, 8)$

$$\mu_\mathcal{B} = \frac{2+4+6+8}{4} = 5$$

$$\sigma_\mathcal{B}^2 = \frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4} = \frac{9+1+1+9}{4} = 5$$

$$\hat{z}_i = \frac{z_i - 5}{\sqrt{5 + 10^{-8}}} \approx \frac{z_i - 5}{2.236}$$

| $z_i$ | $\hat{z}_i$ | $\tilde{z}_i = 2\hat{z}_i + 1$ |
| :--- | :--- | :--- |
| 2 | -1.342 | -1.683 |
| 4 | -0.447 | 0.106 |
| 6 | 0.447 | 1.894 |
| 8 | 1.342 | 3.683 |

### 4.3 Giải thích

**Dropout giống ensemble:** Mỗi bước huấn luyện, dropout tạo ra một mạng con (sub-network) khác nhau bằng cách tắt ngẫu nhiên các neuron. Khi test, sử dụng toàn bộ mạng tương đương với lấy trung bình dự đoán của exponentially nhiều sub-network → giống bagging/ensemble.

**Batch Normalization tăng tốc:** Chuẩn hóa input của mỗi tầng về $\mu = 0, \sigma = 1$ giảm hiện tượng internal covariate shift (phân phối input thay đổi liên tục khi train). Điều này cho phép dùng learning rate lớn hơn mà không gây dao động, tăng tốc hội tụ.

---

## Câu 5. Dropout trong Neural Network

Kỹ thuật Inverted Dropout ngẫu nhiên tắt đi (đặt giá trị bằng 0) một tỷ lệ $p$ các nơ-ron ở các tầng ẩn trong mỗi bước lặp khi huấn luyện. Cụ thể, vector kích hoạt $\mathbf{a}$ được nhân phần tử (element-wise) với một ma trận mặt nạ $\mathbf{M}$ (chứa 0 và 1) rồi chia cho $(1-p)$ để giữ nguyên cường độ tín hiệu (kỳ vọng toán học): $\mathbf{a}_{drop} = \frac{\mathbf{a} \odot \mathbf{M}}{1-p}$.

**Ý nghĩa Ensemble:** Ở mỗi vòng lặp, bằng cách tắt ngẫu nhiên các nơ-ron khác nhau, mô hình thực chất đang huấn luyện một mạng con (sub-network) khác nhau được trích xuất từ mạng toàn cục ban đầu. Khi bước vào giai đoạn dự đoán (inference/test mode) và không dùng Dropout, toàn bộ nơ-ron được giữ lại và tham gia tính toán. Hiệu ứng này tương đương về mặt toán học với việc lấy trung bình dự đoán (model averaging / bagging) của hàng tỷ mạng con đã được huấn luyện, giúp mô hình đạt được sự vững chãi (robust) và giảm thiểu Overfitting hiệu quả.

---

## Câu 6. MLE Exponential Family

### 6.1 Log-Likelihood

$$\ell(\eta) = \sum_{i=1}^{n}\log P(y^{(i)} \mid \eta) = \sum_{i=1}^{n}\left[\log b(y^{(i)}) + \eta T(y^{(i)}) - a(\eta)\right]$$

### 6.2 Gradient

$$\nabla_\eta \ell = \sum_{i=1}^{n}\left[T(y^{(i)}) - a'(\eta)\right] = \sum_{i=1}^{n}\left[T(y^{(i)}) - \mathbb{E}[T(y) \mid \eta]\right]$$

### 6.3 Hessian

$$\nabla^2_\eta \ell = -\sum_{i=1}^{n} a''(\eta) = -n \cdot a''(\eta)$$

### 6.4 Chứng minh

Ta biết $a''(\eta) = \text{Var}(T(y) \mid \eta)$ (tính chất chuẩn của Exponential Family).

Vì phương sai luôn $\geq 0$:

$$a''(\eta) = \text{Var}(T(y)) \geq 0$$

$$\implies \nabla^2_\eta \ell = -n \cdot a''(\eta) \leq 0$$

Tức Hessian $\preceq 0$ → log-likelihood là hàm **lõm** → negative log-likelihood là hàm **lồi** $\quad\blacksquare$

### 6.5 Ý nghĩa

Vì negative log-likelihood lồi, mọi bài toán MLE cho phân phối thuộc Exponential Family (Gaussian, Bernoulli, Poisson, ...) đều:
- Có tối đa **một** cực tiểu toàn cục (không có cực tiểu cục bộ).
- Gradient descent/Newton luôn hội tụ đến nghiệm tối ưu.
- Đây là lý do nền tảng mà Linear Regression (Gaussian) và Logistic Regression (Bernoulli) đều có hàm loss lồi.

---

## Câu 7. Vấn đề Triệt tiêu Đạo hàm (Vanishing Gradient) (Nâng cao)

Đạo hàm của hàm sigmoid là $\sigma'(z) = \sigma(z)(1-\sigma(z))$. 

Giá trị cực đại của đạo hàm này chỉ đạt được $0.25$ tại $z=0$, và nó hội tụ rất nhanh về $0$ khi $|z|$ tiến ra xa khỏi 0. Theo quy tắc chuỗi (Chain Rule) của thuật toán Backpropagation, gradient truyền về các tầng ẩn đầu tiên bằng tích của các đạo hàm cục bộ qua các tầng trên. 

Vì mỗi thành phần đạo hàm $\sigma'(z)$ luôn luôn $\le 0.25$, việc nhân chuỗi nhiều lần qua các tầng sẽ làm gradient nhỏ dần theo cấp số nhân (exponential decay). Khi mạng đủ sâu, tín hiệu gradient triệt tiêu hoàn toàn về 0 ở các tầng ẩn sâu nhất, khiến các trọng số không nhận được tín hiệu để cập nhật (Vanishing Gradient). Hậu quả là mạng bị tê liệt quá trình học, các tầng đầu tiên không thể rút trích được đặc trưng.

**HẾT ĐÁP ÁN**
