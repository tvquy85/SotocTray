# Review Guide: TC-StockMixer (PRICAI 2026 Submission)

Chào ChatGPT / AI Reviewer,

Dự án này (**TC-StockMixer**) đang được chuẩn bị để nộp cho hội nghị **PRICAI 2026**. Mục tiêu của chúng tôi là đạt được đánh giá **"Strong Accept"**. 
Nhiệm vụ của bạn là đóng vai trò như một Domain Expert (Principal ML Scientist / Quant Researcher) để review toàn bộ dự án từ Storyline, Paper Draft đến Source Code. Hãy đánh giá tính mới (Novelty), tính ứng dụng (Applicability) và độ bền bỉ (Robustness) của framework.

Dưới đây là bản đồ hướng dẫn (Roadmap) để bạn bắt đầu quá trình review một cách hiệu quả nhất:

## Bước 1: Nắm bắt Ý tưởng Cốt lõi & Storyline (Rất quan trọng)
*Hãy bắt đầu đọc từ các file này để hiểu tại sao paper này lại giải quyết một "Pain Point" cực lớn trong AI Trading.*
1. `StockMixer/tonghop.md`: File tóm tắt quan trọng nhất. Nó chứa toàn bộ quá trình tiến hóa của mô hình từ PPO nguyên thủy đến V2 (Unified Regime-Aware Optimization), kết quả (cả thắng và thua), và lý luận tại sao nó phù hợp với tiêu chí của PRICAI 2026.
2. `MoTa.md`: Giải thích Intuition "đạp/nhả 2 chiếc phanh" ($\alpha_t$ và $\lambda_t$) thông qua mạng `ContextNet`.

## Bước 2: Đọc Bản thảo Bài báo (Paper Draft)
*Tiếp theo, hãy xem cách chúng tôi trình bày ý tưởng thành văn phong học thuật.*
- Nằm trong thư mục: `StockMixer/paper/`
- Trọng tâm review:
  - `03_Methodology.md`: Kiến trúc của `ContextNet` và hàm mục tiêu (NetRank Loss).
  - `04_Experiments_and_Results.md`: Kết quả thực chứng trên SP500, NASDAQ và Crypto. Đặc biệt chú ý đến sự bứt phá ở tập Crypto.

## Bước 3: Đánh giá Source Code (Triển khai thực tế)
*Chứng minh rằng ý tưởng không chỉ nằm trên giấy. Code được viết rõ ràng và có khả năng mở rộng (Generalization).*
- Nằm trong thư mục: `StockMixer/src_tc/` (Thư mục mã nguồn chính cho các mô hình Transaction-Cost Aware).
- Trọng tâm review:
  - Cách chúng tôi cài đặt hàm mục tiêu vi phân (differentiable portfolio objective) như NetRank Loss.
  - Cách kiến trúc `ContextNet` được xây dựng để xuất ra các dynamic penalties ($\alpha_t, \lambda_t$) dựa trên market regime.
  - **Lưu ý:** Thư mục `StockMixer/src/` là thư mục chứa mô hình Baseline cũ để đối chiếu.

## Bước 4: Kiểm chứng Số liệu & Căn nguyên Thất bại (Root Cause Analysis)
*Sự minh bạch trong việc phân tích cả những lúc mô hình thất bại (như trên vài Seed của NASDAQ) làm tăng tính thuyết phục.*
- Nằm trong thư mục: `StockMixer/reports/`
- Trọng tâm review:
  - `comprehensive_analysis.md`: Mổ xẻ nguyên nhân sâu xa vì sao PPO chết ở Bear Market, tại sao NetRank tĩnh lại thất bại trên NASDAQ.
  - `final_upgrade_statistics.md`: Số liệu chi tiết cho quá trình tiến hóa 5 giai đoạn.

---

## Tiêu chí Review (Yêu cầu dành cho ChatGPT)
Sau khi đọc xong các thành phần trên, hãy cung cấp cho chúng tôi một bản Review chi tiết dựa trên các câu hỏi sau:
1. **Novelty (Tính mới):** Việc chuyển từ Static Penalty sang Dynamic Meta-learning Penalty (ContextNet) đã đủ tầm để tạo ra contribution đột phá cho PRICAI chưa?
2. **Robustness (Tính bền bỉ):** Kết quả thử nghiệm dạng Stress-test trên Crypto (giảm MDD xuống -15.09% và đẩy Sharpe lên 3.511) có đủ sức thuyết phục hội đồng phản biện không?
3. **Weaknesses (Điểm yếu):** Dưới góc độ một Reviewer khó tính, bạn thấy lỗ hổng nào trong logic, phương pháp toán học, hoặc cách trình bày kết quả cần được vá ngay lập tức?
4. **Actionable Advice:** Những điểm cụ thể nào trong Paper Draft hoặc Source code cần tinh chỉnh để khóa chặt cơ hội nhận "Strong Accept"?

Hãy bắt đầu review theo Roadmap này. Cảm ơn bạn!
