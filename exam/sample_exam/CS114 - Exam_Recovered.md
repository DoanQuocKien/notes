

# **ĐỀ THI MACHINE LEARNING**

BẢN KHÔI PHỤC THEO MÔ TẢ

* **Thời gian làm bài:** 90 phút  
* **Hình thức:** Tự luận \+ tính toán  
* **Ghi chú:** Sinh viên được sử dụng máy tính cầm tay. Trình bày rõ công thức, các bước biến đổi và kết luận.

Câu 1\. Maximum Likelihood Estimation cho Logistic Regression  
Cho tập dữ liệu huấn luyện: $D={\\{(x\_{i},y\_{i})\\}\_{i=1}}^{n}$, với $y\_{i}\\in\\{0,1\\}$.  
Mô hình Logistic Regression được định nghĩa bởi:

$$P(y\_{i}=1|x\_{i};w,b)=\\sigma(w^{T}x\_{i}+b)$$

Trong đó:

* $\\sigma(z)=\\frac{1}{1+e^{-z}}$  
* $p\_{i}=P(y\_{i}=1|x\_{i};w,b)$

**Hãy:**

1. Viết hàm likelihood của toàn bộ tập dữ liệu.  
2. Viết hàm log-likelihood.  
3. Chứng minh rằng việc cực đại hóa log-likelihood tương đương với việc cực tiểu hóa binary cross-entropy loss.  
4. Tính gradient của log-likelihood theo $w$ và $b$.  
5. Nêu vì sao Logistic Regression thường không có nghiệm đóng cho MLE và cần dùng các phương pháp tối ưu lặp như Gradient Ascent, Gradient Descent hoặc Newton-Raphson.

Câu 2\. (Chưa khôi phục được nội dung)

*Phần này hiện chưa có đủ mô tả để khôi phục chính xác.*  
Câu 3\. Backpropagation cho mạng neural network đơn giản  
Cho một mạng neural network gồm:

* 2 input: $x\_{1}$, $x\_{2}$  
* 1 hidden layer gồm 2 neuron ($h\_1, h\_2$), dùng sigmoid activation  
* 1 output neuron ($o$), dùng sigmoid activation  
* Loss function (với một mẫu dữ liệu): Squared error $L=\\frac{1}{2}(\\hat{y}-y)^{2}$

Giả sử có một mẫu dữ liệu với: $x\_{1}=1$, $x\_{2}=2$, $y=1$.

**Các trọng số ban đầu:**

| Kết nối | Trọng số |
| :---- | :---- |
| $x\_{1} \\rightarrow h\_{1}$ | 0.1 |
| $x\_{2} \\rightarrow h\_{1}$ | 0.2 |
| $x\_{1} \\rightarrow h\_{2}$ | 0.3 |
| $x\_{2} \\rightarrow h\_{2}$ | 0.4 |
| $h\_{1} \\rightarrow o$ | 0.5 |
| $h\_{2} \\rightarrow o$ | 0.6 |

**Bias ban đầu:**

$$b\_{h1}=0.1, \\quad b\_{h2}=0.1, \\quad b\_{o}=0.1$$

**Hãy thực hiện:**

1. Forward propagation để tính $h\_{1}$, $h\_{2}$ và $\\hat{y}$.  
2. Tính loss $L$.  
3. Tính đạo hàm $\\frac{\\partial L}{\\partial w}$ đối với từng trọng số.  
4. Nếu learning rate $\\eta=0.1$, hãy cập nhật các trọng số sau một bước Gradient Descent.

*Biết rằng:* $\\sigma'(z) \= \\sigma(z)(1 \- \\sigma(z))$  
Câu 4\. Naive Bayes Classification  
Cho bảng dữ liệu huấn luyện sau về phân loại email là Spam hoặc Not Spam:

| STT | Có từ "free" | Có từ "win" | Có từ "meeting" | Nhãn |
| :---- | :---- | :---- | :---- | :---- |
| 1 | Yes | Yes | No | Spam |
| 2 | Yes | No | No | Spam |
| 3 | No | Yes | No | Spam |
| 4 | Yes | Yes | Yes | Spam |
| 5 | No | No | Yes | Not Spam |
| 6 | No | No | Yes | Not Spam |
| 7 | Yes | No | Yes | Not Spam |
| 8 | No | Yes | Yes | Not Spam |
| 9 | No | No | No | Not Spam |
| 10 | Yes | Yes | No | Spam |

Sử dụng thuật toán Naive Bayes với giả định các thuộc tính độc lập có điều kiện theo nhãn. Cho email mới cần dự đoán:

| Có từ "free" | Có từ "win" | Có từ "meeting" |
| :---- | :---- | :---- |
| Yes | Yes | No |

**Hãy:**

1. Tính prior probability $P(\\text{Spam})$ và $P(\\text{Not Spam})$.  
2. Tính các conditional probabilities cần thiết.  
3. Tính $P(\\text{Spam} | x)$ và $P(\\text{Not Spam} | x)$ theo Naive Bayes.  
4. Kết luận email mới thuộc lớp nào.  
5. Nếu có xác suất bằng 0, hãy áp dụng Laplace smoothing với $\\alpha=1$.

Câu 5\. Decision Tree với Information Gain dựa trên Entropy  
Cho bảng dữ liệu thời tiết dùng để quyết định có chơi tennis hay không:

| STT | Outlook | Temperature | Humidity | Wind | Play |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Sunny | Hot | High | Weak | No |
| 2 | Sunny | Hot | High | Strong | No |
| 3 | Overcast | Hot | High | Weak | Yes |
| 4 | Rain | Mild | High | Weak | Yes |
| 5 | Rain | Cool | Normal | Weak | Yes |
| 6 | Rain | Cool | Normal | Strong | No |
| 7 | Overcast | Cool | Normal | Strong | Yes |
| 8 | Sunny | Mild | High | Weak | No |
| 9 | Sunny | Cool | Normal | Weak | Yes |
| 10 | Rain | Mild | Normal | Weak | Yes |
| 11 | Sunny | Mild | Normal | Strong | Yes |
| 12 | Overcast | Mild | High | Strong | Yes |
| 13 | Overcast | Hot | Normal | Weak | Yes |
| 14 | Rain | Mild | High | Strong | No |

**Biết công thức tính:**

* **Entropy:**  
  $$Entropy(S) \= \-\\sum\_{c} p\_c \\log\_2(p\_c)$$

* **Information Gain** của thuộc tính $A$:  
  $$Gain(S,A) \= Entropy(S) \- \\sum\_{v \\in Values(A)} \\frac{|S\_v|}{|S|} \\cdot Entropy(S\_v)$$

**Hãy:**

1. Tính entropy ban đầu của tập dữ liệu $S$.  
2. Tính Information Gain cho các thuộc tính: Outlook, Temperature, Humidity, Wind.  
3. Chọn thuộc tính tốt nhất để làm node gốc của cây quyết định.  
4. Vẽ cây quyết định sau bước chia đầu tiên.  
5. Giải thích vì sao thuộc tính có Information Gain lớn nhất được chọn.

Câu 6\. Quan hệ giữa khoảng cách đến mean và khoảng cách giữa các cặp điểm trong cụm  
Cho một cụm dữ liệu $C \= \\{x\_{1}, x\_{2}, ..., x\_{n}\\}$ với mỗi $x\_{i} \\in \\mathbb{R}^{d}$. Gọi mean của cụm là:

$$\\mu \= \\frac{1}{n} \\sum\_{i=1}^{n} x\_{i}$$

**Hãy chứng minh đẳng thức sau:**

$$\\sum\_{i=1}^{n} ||x\_{i}-\\mu||^{2} \= \\frac{1}{2n} \\sum\_{i=1}^{n} \\sum\_{j=1}^{n} ||x\_{i}-x\_{j}||^{2}$$

**Từ đó suy ra:**

$$\\frac{1}{n} \\sum\_{i=1}^{n} ||x\_{i}-\\mu||^{2} \= \\frac{1}{2n^{2}} \\sum\_{i=1}^{n} \\sum\_{j=1}^{n} ||x\_{i}-x\_{j}||^{2}$$

**Nói cách khác:** Trung bình khoảng cách bình phương từ các điểm trong cụm đến mean có quan hệ trực tiếp với trung bình khoảng cách bình phương giữa tất cả các cặp điểm trong cụm.

*Lưu ý:* Trong công thức chuẩn của clustering/K-means, đại lượng đúng là khoảng cách bình phương Euclidean, không phải khoảng cách Euclidean thông thường.

**HẾT**