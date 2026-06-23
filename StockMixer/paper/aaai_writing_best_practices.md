# Cẩm nang Viết Bài Top-Tier AI (AAAI, NeurIPS, ICML)

Dựa trên các hướng dẫn từ reviewer của NeurIPS/AAAI và các chuyên gia nghiên cứu hàng đầu, dưới đây là những "Best Practices" cốt lõi nhất để nâng tầm một bài báo từ mức "Tốt" lên mức "Top-Tier":

## 1. Tư duy cốt lõi (The Mindset)
- **Kể một câu chuyện kỹ thuật (Technical Narrative):** Bài báo không phải là nhật ký thí nghiệm. Hãy định nghĩa rõ: (1) Đâu là lỗ hổng chưa được giải quyết? (2) Ý tưởng của bạn giải quyết nó thế nào mà người khác chưa làm được? (3) Bằng chứng thực nghiệm nào chứng minh điều đó?
- **Sự rõ ràng (Clarity) đánh bại Sự phức tạp (Complexity):** Đừng cố nhồi nhét toán học rườm rà chỉ để tỏ ra "nguy hiểm". Reviewer đánh giá cao những ý tưởng phức tạp được giải thích một cách thanh lịch và dễ hiểu. Hãy giả định người đọc là chuyên gia trong ngành (ví dụ: biết RL là gì) nhưng không phải là chuyên gia trong ngách hẹp của bạn (ví dụ: không biết đặc thù nhiễu của Crypto).
- **Tính minh bạch và Trung thực (Honesty):** Đừng giấu giếm điểm yếu. Hãy tự làm "Red-team" bài báo của mình. Chỉ rõ các giả định (assumptions) và giới hạn (limitations) của phương pháp. Phương pháp thất bại ở đâu? Tại sao? Reviewer Top-tier cực kỳ ghét các bài báo "tô hồng" kết quả.

## 2. Cấu trúc bài báo chiến thắng
- **Abstract (Linh hồn của bài báo):** Gói gọn toàn bộ trong < 250 từ. Cấu trúc chuẩn: Bối cảnh $\rightarrow$ Vấn đề $\rightarrow$ Phương pháp đề xuất $\rightarrow$ Kết quả (Highlight con số ấn tượng nhất, ví dụ: Sharpe 3.511).
- **The "Teaser" Figure (Hình ảnh mồi nhử):** Các bài báo thành công thường có một hình minh họa (Figure 1) ngay trang 1 hoặc 2. Hình này phải tóm tắt trực quan cốt lõi ý tưởng của bạn (ví dụ: Hình ảnh `ContextNet` đang điều khiển hai phanh $\alpha_t$ và $\lambda_t$ dựa trên các đám mây Regime).
- **Methodology (Phương pháp):** Cung cấp đủ chi tiết để một người có chuyên môn có thể code lại được (Reproducibility). Sử dụng Pseudocode chuẩn chỉ. Các phương trình phải được giải thích rõ từng biến số.
- **Experiments (Thực nghiệm):**
  - *Strong Baselines:* Không chỉ so sánh với các baseline yếu, phải so với favorable performance in the tested setting hiện hành.
  - *Rigorous Evidence:* Dùng nhiều bộ dữ liệu (SP500, NASDAQ, Crypto là điểm cộng cực lớn cho bài của chúng ta vì tính đa dạng). Có kiểm định tính ổn định (Ablation studies, Multi-seed).

## 3. Lấy lòng Reviewer (Reviewer-Focused Strategies)
- **Hiểu Rubric chấm điểm:** Đọc kỹ "Reviewer Guidelines" của AAAI. Họ sẽ chấm dựa trên 4 tiêu chí: **Soundness (Độ tin cậy)**, **Clarity (Độ rõ ràng)**, **Significance (Tầm quan trọng)**, và **Novelty (Tính mới)**.
- **Dễ nhìn, Dễ đọc:** Dùng format chuyên nghiệp. Trong các bảng số liệu (Tables), dùng package `booktabs`, tránh dùng các đường kẻ dọc (vertical lines) gây rối mắt. Highlight các con số tốt nhất bằng chữ in đậm.
- **Tính lặp lại (Reproducibility):** Chắc chắn rằng code và dataset (hoặc link ẩn danh tới dataset) được đính kèm ở dạng Supplementary Material. Khai báo rõ cấu hình hyperparameter.

## 4. Workflow viết bài hiệu quả
- **Viết theo Vòng lặp (Iterative Writing):** Abstract $\rightarrow$ Outline (Gạch đầu dòng) $\rightarrow$ Introduction $\rightarrow$ Các phần còn lại $\rightarrow$ Tinh chỉnh.
- **Tránh HARKing (Hypothesizing After Results are Known):** Không viết theo kiểu bịa ra một giả thuyết sau khi đã thấy kết quả tốt. Hãy viết theo mạch logic: Bắt nguồn từ quan sát thị trường $\rightarrow$ Lập giả thuyết $\rightarrow$ Xây dựng mô hình $\rightarrow$ Kiểm chứng.

## 5. Áp dụng ngay vào bài báo "Unified Regime-Aware Optimization"
- **Nhấn mạnh vào "Sự chuyển dịch Regime":** Đây là Selling Point cực mạnh. Đừng chỉ khoe thuật toán RL, hãy bán câu chuyện "thị trường thay đổi liên tục, và AI của chúng tôi biết tự thích nghi".
- **Figure 1 của bài này:** Hãy vẽ một biểu đồ đường (Line chart) so sánh giá trị Portfolio của model tĩnh vs model của bạn trên đoạn thị trường Crypto bị sập mạnh. Bên dưới biểu đồ, vẽ đồ thị $\lambda_t$ (Downside penalty) vọt lên ngay trước cú sập. Bức hình này sẽ thuyết phục reviewer ngay từ trang cải tiến!
