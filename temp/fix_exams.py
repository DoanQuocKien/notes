import re

def replace_file(filepath, old_q, new_q, old_a, new_a):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple search and split/replace logic
    # Find the header of the old question
    if old_q in content:
        # We assume the old question goes until "HẾT" or the end
        if "HẾT" in content:
            idx1 = content.find(old_q)
            idx2 = content.find("HẾT", idx1)
            content = content[:idx1] + new_q + "\n\n**HẾT**\n"
        else:
            idx1 = content.find(old_q)
            content = content[:idx1] + new_q + "\n"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated Q:", filepath)
    else:
        print("Could not find Q in", filepath)

    ans_path = filepath.replace(".md", "_Answer.md")
    with open(ans_path, 'r', encoding='utf-8') as f:
        content_a = f.read()
    
    if old_a in content_a:
        if "**HẾT ĐÁP ÁN**" in content_a:
            idx1 = content_a.find(old_a)
            idx2 = content_a.find("**HẾT ĐÁP ÁN**", idx1)
            content_a = content_a[:idx1] + new_a + "\n\n**HẾT ĐÁP ÁN**\n"
        else:
            idx1 = content_a.find(old_a)
            content_a = content_a[:idx1] + new_a + "\n"
        with open(ans_path, 'w', encoding='utf-8') as f:
            f.write(content_a)
        print("Updated A:", ans_path)
    else:
        print("Could not find A in", ans_path)

# EXAM 1
q1_old = "Câu 8. SGD và Khả Năng Tổng Quát Hóa (Nâng cao)"
q1_new = """Câu 8. Tính phi tuyến trong Mạng Neural (Nâng cao)

Vì sao ta không thể dùng hàm kích hoạt tuyến tính (linear activation, $g(z)=z$) cho tất cả các tầng ẩn trong một mạng Neural Network có nhiều tầng? Chứng minh bằng toán học."""
a1_old = "## Câu 8. SGD và Khả Năng Tổng Quát Hóa (Nâng cao)"
a1_new = """## Câu 8. Tính phi tuyến trong Mạng Neural (Nâng cao)

Nếu dùng hàm kích hoạt tuyến tính $g(z)=z$, phép tính ở mỗi tầng chỉ là một biến đổi tuyến tính:
Tầng 1: $\mathbf{A}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$
Tầng 2: $\mathbf{A}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]} = \mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = (\mathbf{W}^{[2]}\mathbf{W}^{[1]})\mathbf{x} + (\mathbf{W}^{[2]}\mathbf{b}^{[1]} + \mathbf{b}^{[2]})$

Đặt $\mathbf{W}' = \mathbf{W}^{[2]}\mathbf{W}^{[1]}$ và $\mathbf{b}' = \mathbf{W}^{[2]}\mathbf{b}^{[1]} + \mathbf{b}^{[2]}$, ta có $\mathbf{A}^{[2]} = \mathbf{W}'\mathbf{x} + \mathbf{b}'$. Đây lại là một phép biến đổi tuyến tính duy nhất.

Bất kể mạng có sâu bao nhiêu tầng, sự kết hợp của các phép biến đổi tuyến tính vẫn chỉ tạo ra một biến đổi tuyến tính. Mạng sẽ sụp đổ (collapse) thành một mô hình hồi quy tuyến tính/logistic thông thường, hoàn toàn mất đi khả năng học và mô phỏng các ranh giới phân loại phi tuyến phức tạp (như bài toán XOR)."""
replace_file(r"d:\CS114\notes\exam\exam_01\CS114 - Exam_01.md", q1_old, q1_new, a1_old, a1_new)

# EXAM 2
q2_old = "Câu 7 (Nâng cao). Generative vs Discriminative Models"
q2_new = """Câu 7 (Nâng cao). Hội tụ của thuật toán K-means

Dựa vào hàm mục tiêu Distortion (Sum of Squared Errors), hãy lập luận toán học để chứng minh vì sao thuật toán K-means luôn luôn hội tụ sau một số vòng lặp hữu hạn, nhưng không thể đảm bảo luôn đạt được cực tiểu toàn cục (global minimum)."""
a2_old = "## Câu 7. Generative vs Discriminative Models"
a2_new = """## Câu 7. Hội tụ của thuật toán K-means (Nâng cao)

Hàm mục tiêu của K-means là tổng bình phương khoảng cách: $J = \sum_{i=1}^m \|x_i - \mu_{c_i}\|^2 \ge 0$.
Thuật toán là quá trình Coordinate Descent với 2 bước:
1. **Gán cụm:** Chọn tâm cụm gần nhất cho mỗi điểm làm tổng khoảng cách $J$ giảm đi hoặc giữ nguyên.
2. **Cập nhật tâm:** Việc cập nhật tâm cụm bằng giá trị trung bình (mean) chính là nghiệm tối ưu giải bài toán cực tiểu hóa $J$ cho mỗi cụm, do đó $J$ tiếp tục giảm hoặc giữ nguyên.

Vì $J$ luôn giảm hoặc không đổi qua mỗi bước lặp và bị chặn dưới bởi 0, cộng với việc tập hợp các cách phân chia $m$ điểm vào $K$ nhóm là hữu hạn, nên thuật toán chắc chắn sẽ dừng và hội tụ.

Tuy nhiên, hàm $J$ không lồi hoàn toàn (non-convex). Do tính tham lam của quá trình lặp, thuật toán có thể bị kẹt ở các điểm cực tiểu cục bộ (local minimum) nếu các tâm cụm ban đầu được khởi tạo ở các vị trí không thuận lợi."""
replace_file(r"d:\CS114\notes\exam\exam_02\CS114 - Exam_02.md", q2_old, q2_new, a2_old, a2_new)

# EXAM 3
q3_old = "Câu 7 (Nâng cao). Cross-Entropy, K-L Divergence và Mối liên hệ PCA"
q3_new = """Câu 7 (Nâng cao). So sánh Gradient Descent và Newton's Method

Trong quá trình huấn luyện Logistic Regression, thuật toán tối ưu Gradient Descent và Newton's Method thường được cân nhắc sử dụng. Hãy trình bày sự khác biệt cơ bản về cơ chế tính toán, tốc độ hội tụ, và chi phí phần cứng giữa hai thuật toán này."""
a3_old = "## Câu 7. Cross-Entropy, K-L Divergence và Mối liên hệ PCA (Nâng cao)"
a3_new = """## Câu 7. So sánh Gradient Descent và Newton's Method (Nâng cao)

- **Cơ chế:** Gradient Descent là phương pháp tối ưu chỉ sử dụng đạo hàm bậc 1 ($\nabla J$). Newton's Method tối ưu mạnh mẽ hơn bằng cách sử dụng cả đạo hàm bậc 1 và độ cong bậc 2 của hàm số thông qua ma trận Hessian $H$. Công thức cập nhật của nó có chứa $H^{-1}\nabla J$.
- **Tốc độ hội tụ:** Nhờ có thông tin độ cong, Newton's Method có tốc độ hội tụ bậc 2 (Quadratic Convergence), chỉ cần số bước lặp rất ít và đặc biệt là **không cần điều chỉnh siêu tham số learning rate $\alpha$**. Trong khi đó, Gradient Descent hội tụ tuyến tính (Linear), rất chậm và phụ thuộc lớn vào việc chọn $\alpha$.
- **Chi phí phần cứng:** Dù Newton's method hội tụ trong ít bước lặp, chi phí ở mỗi bước lại vô cùng lớn. Việc tính và nghịch đảo ma trận Hessian $p \times p$ đòi hỏi chi phí thuật toán $\mathcal{O}(p^3)$. Nếu mô hình có số lượng thuộc tính $p$ rất lớn (hàng triệu), Newton's Method trở nên bất khả thi. Khi đó, Gradient Descent với chi phí $\mathcal{O}(p)$ ở mỗi vòng lặp là lựa chọn duy nhất."""
replace_file(r"d:\CS114\notes\exam\exam_03\CS114 - Exam_03.md", q3_old, q3_new, a3_old, a3_new)

# EXAM 4
q4_old = "Câu 7 (Nâng cao). Lời Nguyền Chiều Dữ Liệu và Sự Linh Hoạt Của SVM"
q4_new = """Câu 7 (Nâng cao). Batch Normalization trong Mạng Neural

Trình bày bài toán Internal Covariate Shift (hiện tượng dịch chuyển phân phối nội bộ) gặp phải trong việc huấn luyện các mạng Neural sâu. Kỹ thuật Batch Normalization đã giải quyết vấn đề này như thế nào?"""
a4_old = "## Câu 7. Lời Nguyền Chiều Dữ Liệu và Sự Linh Hoạt Của SVM (Nâng cao)"
a4_new = """## Câu 7. Batch Normalization trong Mạng Neural (Nâng cao)

**Internal Covariate Shift:** Trong một mạng neural sâu, mỗi khi trọng số ở các tầng trước được cập nhật, phân phối dữ liệu đầu ra của chúng (cũng là đầu vào của tầng sau) sẽ liên tục bị thay đổi và xô lệch. Các tầng ẩn sâu luôn phải chạy theo để học một phân phối mục tiêu đang biến động không ngừng, khiến quá trình hội tụ cực kỳ chậm và dễ bị kẹt.

**Giải pháp Batch Normalization:** Để khắc phục, kỹ thuật này chuẩn hóa trực tiếp điểm số tuyến tính $Z^{[l]}$ trên mỗi mini-batch $\mathcal{B}$ về trạng thái chuẩn (trung bình 0, phương sai 1):
$$\mu_{\mathcal{B}} = \frac{1}{m}\sum Z_i, \quad \sigma^2_{\mathcal{B}} = \frac{1}{m}\sum (Z_i-\mu_{\mathcal{B}})^2$$
$$\hat{Z}_i = \frac{Z_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$$
Sau đó, để không làm mất đi khả năng biểu diễn của tầng, giá trị này được scale và shift bằng hai tham số học được $\gamma$ và $\beta$: $\tilde{Z}_i = \gamma \hat{Z}_i + \beta$.
BN giúp phân phối dữ liệu truyền giữa các tầng ổn định, bôi trơn không gian lỗi, cho phép sử dụng learning rate lớn hơn và tăng tốc hội tụ lên gấp nhiều lần."""
replace_file(r"d:\CS114\notes\exam\exam_04\CS114 - Exam_04.md", q4_old, q4_new, a4_old, a4_new)

