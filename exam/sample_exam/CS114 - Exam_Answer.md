# **ĐÁP ÁN ĐỀ THI MACHINE LEARNING**

BẢN ĐÁP ÁN CHI TIẾT

---

## Câu 1. Maximum Likelihood Estimation cho Logistic Regression

### 1.1 Hàm Likelihood

Với giả định các mẫu độc lập (i.i.d.), sử dụng công thức xác suất hợp nhất cho phân phối Bernoulli:

$$P(y_i \mid x_i; w, b) = p_i^{y_i}(1 - p_i)^{1 - y_i}$$

Hàm likelihood của toàn bộ tập dữ liệu:

$$L(w, b) = \prod_{i=1}^{n} p_i^{y_i}(1 - p_i)^{1 - y_i}$$

trong đó $p_i = \sigma(w^T x_i + b) = \frac{1}{1 + e^{-(w^T x_i + b)}}$.

### 1.2 Hàm Log-Likelihood

Lấy logarit tự nhiên:

$$\ell(w, b) = \log L(w, b) = \sum_{i=1}^{n} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

### 1.3 Chứng minh tương đương với Binary Cross-Entropy Loss

Binary cross-entropy loss được định nghĩa:

$$J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

Ta nhận thấy:

$$J(w, b) = -\frac{1}{n} \ell(w, b)$$

Do đó:

$$\max_{w,b} \ell(w,b) \iff \min_{w,b} \left( -\ell(w,b) \right) \iff \min_{w,b} J(w,b)$$

**Kết luận:** Cực đại hóa log-likelihood tương đương với cực tiểu hóa binary cross-entropy loss (khác nhau hệ số dương $\frac{1}{n}$ và đổi dấu, không ảnh hưởng đến điểm tối ưu).

### 1.4 Tính Gradient

**Tính chất quan trọng của sigmoid:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

Đặt $z_i = w^T x_i + b$, áp dụng chain rule:

$$\frac{\partial \ell}{\partial w} = \sum_{i=1}^{n} \frac{\partial \ell}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w}$$

Tính từng thành phần:

- $\frac{\partial \ell}{\partial p_i} = \frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i} = \frac{y_i - p_i}{p_i(1 - p_i)}$

- $\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$

- $\frac{\partial z_i}{\partial w} = x_i$

Nhân lại và triệt tiêu mẫu số:

$$\frac{\partial \ell}{\partial w} = \sum_{i=1}^{n} (y_i - p_i) x_i$$

Tương tự cho bias:

$$\frac{\partial \ell}{\partial b} = \sum_{i=1}^{n} (y_i - p_i)$$

### 1.5 Vì sao không có nghiệm đóng?

Đặt gradient bằng 0:

$$\sum_{i=1}^{n} (y_i - \sigma(w^T x_i + b)) x_i = 0$$

Do hàm sigmoid $\sigma(\cdot)$ là hàm phi tuyến, phương trình trên không thể giải đại số để rút $w$ và $b$ ra dạng tường minh. Khác với Linear Regression (nơi mô hình tuyến tính cho phương trình bậc nhất), Logistic Regression tạo ra hệ phương trình siêu việt (transcendental equations) không có nghiệm đóng.

Do đó, ta phải sử dụng các phương pháp tối ưu lặp:
- **Gradient Ascent/Descent:** Cập nhật $w \leftarrow w + \alpha \sum_i (y_i - p_i) x_i$, hội tụ tuyến tính.
- **Newton-Raphson:** Sử dụng Hessian $H = -\sum_i p_i(1-p_i) x_i x_i^T$, cập nhật $w \leftarrow w - H^{-1} \nabla \ell$, hội tụ bậc hai nhưng tốn $O(p^3)$ mỗi bước.

---

## Câu 2. (Chưa khôi phục được nội dung)

*Bỏ qua do đề gốc chưa khôi phục.*

---

## Câu 3. Backpropagation cho Mạng Neural Network Đơn Giản

### 3.1 Forward Propagation

**Tầng ẩn — Neuron $h_1$:**

$$z_{h1} = w_{x1 \to h1} \cdot x_1 + w_{x2 \to h1} \cdot x_2 + b_{h1}$$

$$z_{h1} = 0.1(1) + 0.2(2) + 0.1 = 0.1 + 0.4 + 0.1 = 0.6$$

$$h_1 = \sigma(0.6) = \frac{1}{1 + e^{-0.6}} \approx 0.6457$$

**Tầng ẩn — Neuron $h_2$:**

$$z_{h2} = 0.3(1) + 0.4(2) + 0.1 = 0.3 + 0.8 + 0.1 = 1.2$$

$$h_2 = \sigma(1.2) = \frac{1}{1 + e^{-1.2}} \approx 0.7685$$

**Tầng đầu ra — Neuron $o$:**

$$z_o = 0.5 \cdot h_1 + 0.6 \cdot h_2 + b_o = 0.5(0.6457) + 0.6(0.7685) + 0.1$$

$$z_o = 0.3229 + 0.4611 + 0.1 = 0.8840$$

$$\hat{y} = \sigma(0.8840) = \frac{1}{1 + e^{-0.8840}} \approx 0.7076$$

### 3.2 Tính Loss

$$L = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.7076 - 1)^2 = \frac{1}{2}(−0.2924)^2 = \frac{1}{2}(0.08550) \approx 0.04275$$

### 3.3 Tính Gradient (Backpropagation)

**Bước 1: Đạo hàm tại output**

$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 0.7076 - 1 = -0.2924$$

$$\sigma'(z_o) = \hat{y}(1 - \hat{y}) = 0.7076 \times 0.2924 = 0.2069$$

$$\delta_o = \frac{\partial L}{\partial z_o} = (\hat{y} - y) \cdot \sigma'(z_o) = (-0.2924)(0.2069) = -0.06050$$

**Bước 2: Gradient trọng số tầng output**

$$\frac{\partial L}{\partial w_{h1 \to o}} = \delta_o \cdot h_1 = (-0.06050)(0.6457) = -0.03907$$

$$\frac{\partial L}{\partial w_{h2 \to o}} = \delta_o \cdot h_2 = (-0.06050)(0.7685) = -0.04650$$

$$\frac{\partial L}{\partial b_o} = \delta_o = -0.06050$$

**Bước 3: Lan truyền ngược đến tầng ẩn**

$$\delta_{h1} = \delta_o \cdot w_{h1 \to o} \cdot h_1(1 - h_1)$$

$$= (-0.06050)(0.5)(0.6457)(1 - 0.6457)$$

$$= (-0.06050)(0.5)(0.2288) = -0.006921$$

$$\delta_{h2} = \delta_o \cdot w_{h2 \to o} \cdot h_2(1 - h_2)$$

$$= (-0.06050)(0.6)(0.7685)(1 - 0.7685)$$

$$= (-0.06050)(0.6)(0.1780) = -0.006459$$

**Bước 4: Gradient trọng số tầng ẩn**

$$\frac{\partial L}{\partial w_{x1 \to h1}} = \delta_{h1} \cdot x_1 = (-0.006921)(1) = -0.006921$$

$$\frac{\partial L}{\partial w_{x2 \to h1}} = \delta_{h1} \cdot x_2 = (-0.006921)(2) = -0.013842$$

$$\frac{\partial L}{\partial w_{x1 \to h2}} = \delta_{h2} \cdot x_1 = (-0.006459)(1) = -0.006459$$

$$\frac{\partial L}{\partial w_{x2 \to h2}} = \delta_{h2} \cdot x_2 = (-0.006459)(2) = -0.012918$$

### 3.4 Cập nhật trọng số ($\eta = 0.1$)

Công thức: $w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$

| Kết nối | $w_{old}$ | Gradient | $w_{new}$ |
| :---- | :---- | :---- | :---- |
| $x_1 \to h_1$ | 0.1 | -0.006921 | $0.1 - 0.1(-0.006921) = 0.10069$ |
| $x_2 \to h_1$ | 0.2 | -0.013842 | $0.2 - 0.1(-0.013842) = 0.20138$ |
| $x_1 \to h_2$ | 0.3 | -0.006459 | $0.3 - 0.1(-0.006459) = 0.30065$ |
| $x_2 \to h_2$ | 0.4 | -0.012918 | $0.4 - 0.1(-0.012918) = 0.40129$ |
| $h_1 \to o$ | 0.5 | -0.03907 | $0.5 - 0.1(-0.03907) = 0.50391$ |
| $h_2 \to o$ | 0.6 | -0.04650 | $0.6 - 0.1(-0.04650) = 0.60465$ |

**Bias cập nhật:**

- $b_{h1} = 0.1 - 0.1(-0.006921) = 0.10069$
- $b_{h2} = 0.1 - 0.1(-0.006459) = 0.10065$
- $b_o = 0.1 - 0.1(-0.06050) = 0.10605$

---

## Câu 4. Naive Bayes Classification

### 4.1 Prior Probability

Từ bảng dữ liệu: 5 mẫu Spam (dòng 1, 2, 3, 4, 10), 5 mẫu Not Spam (dòng 5, 6, 7, 8, 9).

$$P(\text{Spam}) = \frac{5}{10} = 0.5$$

$$P(\text{Not Spam}) = \frac{5}{10} = 0.5$$

### 4.2 Conditional Probabilities

**Lớp Spam (5 mẫu):**

| Từ | Số mẫu Spam có "Yes" | $P(\text{Yes} \mid \text{Spam})$ |
| :--- | :--- | :--- |
| free | 3 (dòng 1, 2, 4) | $\frac{3}{5} = 0.6$ |
| win | 3 (dòng 1, 3, 4) | $\frac{3}{5} = 0.6$ | [Sửa: dòng 10 cũng là Spam có win=Yes, tổng = 4] |
| meeting | 1 (dòng 4) | $\frac{1}{5} = 0.2$ |

Kiểm tra lại: Spam gồm dòng 1, 2, 3, 4, 10.
- free=Yes trong Spam: dòng 1(Yes), 2(Yes), 3(No), 4(Yes), 10(Yes) → 4 mẫu
- win=Yes trong Spam: dòng 1(Yes), 2(No), 3(Yes), 4(Yes), 10(Yes) → 4 mẫu
- meeting=Yes trong Spam: dòng 1(No), 2(No), 3(No), 4(Yes), 10(No) → 1 mẫu

$$P(\text{free=Yes} \mid \text{Spam}) = \frac{4}{5} = 0.8$$

$$P(\text{win=Yes} \mid \text{Spam}) = \frac{4}{5} = 0.8$$

$$P(\text{meeting=No} \mid \text{Spam}) = 1 - P(\text{meeting=Yes} \mid \text{Spam}) = 1 - \frac{1}{5} = 0.8$$

**Lớp Not Spam (5 mẫu: dòng 5, 6, 7, 8, 9):**

- free=Yes: dòng 5(No), 6(No), 7(Yes), 8(No), 9(No) → 1 mẫu
- win=Yes: dòng 5(No), 6(No), 7(No), 8(Yes), 9(No) → 1 mẫu
- meeting=Yes: dòng 5(Yes), 6(Yes), 7(Yes), 8(Yes), 9(No) → 4 mẫu

$$P(\text{free=Yes} \mid \text{Not Spam}) = \frac{1}{5} = 0.2$$

$$P(\text{win=Yes} \mid \text{Not Spam}) = \frac{1}{5} = 0.2$$

$$P(\text{meeting=No} \mid \text{Not Spam}) = 1 - \frac{4}{5} = 0.2$$

### 4.3 Tính Posterior

Email mới: free=Yes, win=Yes, meeting=No.

**Tính cho Spam:**

$$P(\mathbf{x} \mid \text{Spam}) \cdot P(\text{Spam}) = (0.8)(0.8)(0.8)(0.5) = 0.256$$

**Tính cho Not Spam:**

$$P(\mathbf{x} \mid \text{Not Spam}) \cdot P(\text{Not Spam}) = (0.2)(0.2)(0.2)(0.5) = 0.004$$

**Chuẩn hóa:**

$$P(\text{Spam} \mid \mathbf{x}) = \frac{0.256}{0.256 + 0.004} = \frac{0.256}{0.260} \approx 0.985$$

$$P(\text{Not Spam} \mid \mathbf{x}) = \frac{0.004}{0.260} \approx 0.015$$

### 4.4 Kết luận

Vì $P(\text{Spam} \mid \mathbf{x}) \approx 0.985 > 0.5$, email mới được phân loại là **Spam**.

### 4.5 Laplace Smoothing

Trong bài này không có xác suất nào bằng 0, nên Laplace smoothing không thay đổi kết luận. Nếu áp dụng Laplace smoothing ($\alpha = 1$) cho feature nhị phân:

$$P(x_j = 1 \mid c) = \frac{\text{count}(x_j = 1, c) + 1}{N_c + 2}$$

Ví dụ: $P(\text{meeting=Yes} \mid \text{Spam}) = \frac{1 + 1}{5 + 2} = \frac{2}{7} \approx 0.286$ thay vì $0.2$.

---

## Câu 5. Decision Tree với Information Gain dựa trên Entropy

### 5.1 Entropy ban đầu

Trong 14 mẫu: 9 mẫu Yes, 5 mẫu No.

$$Entropy(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14}$$

$$= -0.6429 \times (-0.6373) - 0.3571 \times (-1.4854)$$

$$= 0.4098 + 0.5306 = 0.9403$$

### 5.2 Information Gain cho từng thuộc tính

#### Outlook

- **Sunny** (dòng 1,2,8,9,11): 2 Yes, 3 No → $H = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} = 0.9710$
- **Overcast** (dòng 3,7,12,13): 4 Yes, 0 No → $H = 0$
- **Rain** (dòng 4,5,6,10,14): 3 Yes, 2 No → $H = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} = 0.9710$

$$Gain(S, \text{Outlook}) = 0.9403 - \frac{5}{14}(0.9710) - \frac{4}{14}(0) - \frac{5}{14}(0.9710)$$

$$= 0.9403 - 0.3468 - 0 - 0.3468 = 0.2467$$

#### Temperature

- **Hot** (dòng 1,2,3,13): 2 Yes, 2 No → $H = 1.0$
- **Mild** (dòng 4,8,10,11,12,14): 4 Yes, 2 No → $H = -\frac{4}{6}\log_2\frac{4}{6} - \frac{2}{6}\log_2\frac{2}{6} = 0.9183$
- **Cool** (dòng 5,6,7,9): 3 Yes, 1 No → $H = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} = 0.8113$

$$Gain(S, \text{Temperature}) = 0.9403 - \frac{4}{14}(1.0) - \frac{6}{14}(0.9183) - \frac{4}{14}(0.8113)$$

$$= 0.9403 - 0.2857 - 0.3936 - 0.2318 = 0.0292$$

#### Humidity

- **High** (dòng 1,2,3,4,8,12,14): 3 Yes, 4 No → $H = -\frac{3}{7}\log_2\frac{3}{7} - \frac{4}{7}\log_2\frac{4}{7} = 0.9852$
- **Normal** (dòng 5,6,7,9,10,11,13): 6 Yes, 1 No → $H = -\frac{6}{7}\log_2\frac{6}{7} - \frac{1}{7}\log_2\frac{1}{7} = 0.5917$

$$Gain(S, \text{Humidity}) = 0.9403 - \frac{7}{14}(0.9852) - \frac{7}{14}(0.5917)$$

$$= 0.9403 - 0.4926 - 0.2959 = 0.1518$$

#### Wind

- **Weak** (dòng 1,3,4,5,8,9,10,13): 6 Yes, 2 No → $H = -\frac{6}{8}\log_2\frac{6}{8} - \frac{2}{8}\log_2\frac{2}{8} = 0.8113$
- **Strong** (dòng 2,6,7,11,12,14): 3 Yes, 3 No → $H = 1.0$

$$Gain(S, \text{Wind}) = 0.9403 - \frac{8}{14}(0.8113) - \frac{6}{14}(1.0)$$

$$= 0.9403 - 0.4636 - 0.4286 = 0.0481$$

### 5.3 Chọn thuộc tính tốt nhất

| Thuộc tính | Information Gain |
| :--- | :--- |
| **Outlook** | **0.2467** |
| Humidity | 0.1518 |
| Wind | 0.0481 |
| Temperature | 0.0292 |

**Kết luận:** Chọn **Outlook** làm node gốc vì có Information Gain lớn nhất.

### 5.4 Cây quyết định sau bước chia đầu tiên

```
            [Outlook]
           /    |    \
    Sunny   Overcast   Rain
     |         |        |
  [2Y,3N]   [4Y,0N]  [3Y,2N]
  (tiếp      → Yes    (tiếp
   chia)              chia)
```

- Nhánh **Overcast** đã thuần nhất (4 Yes, 0 No) → lá = **Yes**
- Nhánh **Sunny** và **Rain** cần tiếp tục chia.

### 5.5 Giải thích

Thuộc tính có Information Gain lớn nhất được chọn vì nó **giảm entropy nhiều nhất** sau phép chia, nghĩa là nó phân tách dữ liệu thành các nhóm con đồng nhất nhất. Đây là chiến lược tham lam (greedy): tại mỗi bước, ta chọn phép chia tốt nhất cục bộ để nhanh nhất đạt được các nút lá thuần nhất.

---

## Câu 6. Chứng minh quan hệ khoảng cách trong cụm

### Chứng minh đẳng thức chính

Cần chứng minh:

$$\sum_{i=1}^{n} \|x_i - \mu\|^2 = \frac{1}{2n} \sum_{i=1}^{n} \sum_{j=1}^{n} \|x_i - x_j\|^2$$

**Bước 1: Khai triển vế phải**

$$\|x_i - x_j\|^2 = \|x_i\|^2 - 2x_i^T x_j + \|x_j\|^2$$

Tính tổng kép:

$$\sum_{i=1}^{n} \sum_{j=1}^{n} \|x_i - x_j\|^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} \left( \|x_i\|^2 - 2x_i^T x_j + \|x_j\|^2 \right)$$

Tách tổng:

$$= \sum_{i=1}^{n} \sum_{j=1}^{n} \|x_i\|^2 - 2\sum_{i=1}^{n} \sum_{j=1}^{n} x_i^T x_j + \sum_{i=1}^{n} \sum_{j=1}^{n} \|x_j\|^2$$

Mỗi hạng thứ nhất và hạng thứ ba đều bằng $n \sum_{i=1}^{n} \|x_i\|^2$:

$$= 2n \sum_{i=1}^{n} \|x_i\|^2 - 2\left(\sum_{i=1}^{n} x_i\right)^T \left(\sum_{j=1}^{n} x_j\right)$$

Vì $\sum_{i=1}^{n} x_i = n\mu$:

$$= 2n \sum_{i=1}^{n} \|x_i\|^2 - 2(n\mu)^T(n\mu)$$

$$= 2n \sum_{i=1}^{n} \|x_i\|^2 - 2n^2 \|\mu\|^2$$

Chia cho $2n$:

$$\frac{1}{2n} \sum_{i=1}^{n} \sum_{j=1}^{n} \|x_i - x_j\|^2 = \sum_{i=1}^{n} \|x_i\|^2 - n\|\mu\|^2$$

**Bước 2: Khai triển vế trái**

$$\sum_{i=1}^{n} \|x_i - \mu\|^2 = \sum_{i=1}^{n} \left( \|x_i\|^2 - 2x_i^T \mu + \|\mu\|^2 \right)$$

$$= \sum_{i=1}^{n} \|x_i\|^2 - 2\mu^T \sum_{i=1}^{n} x_i + n\|\mu\|^2$$

Thay $\sum_{i=1}^{n} x_i = n\mu$:

$$= \sum_{i=1}^{n} \|x_i\|^2 - 2\mu^T (n\mu) + n\|\mu\|^2$$

$$= \sum_{i=1}^{n} \|x_i\|^2 - 2n\|\mu\|^2 + n\|\mu\|^2$$

$$= \sum_{i=1}^{n} \|x_i\|^2 - n\|\mu\|^2$$

**Bước 3: So sánh**

Cả vế trái và vế phải đều bằng $\sum_{i=1}^{n} \|x_i\|^2 - n\|\mu\|^2$.

$$\therefore \sum_{i=1}^{n} \|x_i - \mu\|^2 = \frac{1}{2n} \sum_{i=1}^{n} \sum_{j=1}^{n} \|x_i - x_j\|^2 \quad \blacksquare$$

### Suy ra hệ quả

Chia cả hai vế cho $n$:

$$\frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^2 = \frac{1}{2n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} \|x_i - x_j\|^2 \quad \blacksquare$$

**Ý nghĩa:** Trung bình khoảng cách bình phương từ các điểm đến tâm cụm (vế trái — đại lượng K-means tối thiểu hóa) bằng đúng một nửa trung bình khoảng cách bình phương giữa tất cả các cặp điểm (vế phải). Điều này cho thấy việc tối thiểu hóa distortion trong K-means tương đương với việc tối thiểu hóa khoảng cách nội cụm trung bình giữa mọi cặp điểm.

---

**HẾT ĐÁP ÁN**
