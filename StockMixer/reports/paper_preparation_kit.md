# LOBExp / TC-StockMixer: Paper Preparation Kit (AAAI 2026 Target)

Tài liệu này tổng hợp toàn bộ các luận điểm khoa học, kết quả thực chứng và cốt truyện (Storyline) để viết bài báo đạt chuẩn Top-tier Conference (AAAI, NeurIPS, PRICAI).

## 1. Cốt truyện bài báo (Paper Storyline)
**Tiêu đề đề xuất (Proposed Title):** 
* "Unified Regime-Aware Optimization: Co-adapting Transaction Costs and Downside Risk via Meta-Learning in Portfolio Management" 
* (Hoặc) "Dynamic NetRank: A Context-Conditioned Approach for Robust Portfolio Optimization under High Market Volatility"

**Abstract Outline:**
1. **Vấn đề (Problem):** Tối ưu hóa danh mục đầu tư có tính đến chi phí giao dịch (Transaction-Cost-Aware Portfolio Optimization) thường sử dụng các hằng số phạt tĩnh. Điều này gây ra sự sụp đổ (Overfitting) khi triển khai trên các thị trường có độ nhiễu cao (NASDAQ, Crypto).
2. **Đề xuất (Method):** Bài báo đề xuất **Unified Regime-Aware Optimization (V2)**, tích hợp một mạng Meta-learning (`ContextNet`) vào mô hình MLP. Mạng này tự động nhận diện Market Regime để xuất ra ĐỒNG THỜI 2 tham số động: $\alpha_t$ (Phạt Turnover) và $\lambda_t$ (Phạt Rủi ro sụt giảm - Downside Variance Penalty).
3. **Kết quả (Results):** Thực nghiệm trên SP500, NASDAQ và Crypto chứng minh mô hình tự động cắt giảm Turnover trong thị trường nhiễu và bảo vệ danh mục xuất sắc. Trên Crypto, Net Sharpe đạt **3.511** (trung bình 3 seeds) và Max Drawdown nén xuống **-15.09%**.

## 2. Các Đóng góp Kỹ thuật (Technical Contributions)
Bài báo chia thành 3 phần Methodology chính:

- **Contribution 1 (Risk-Aware PPO / RL Baseline):** Chứng minh sự cần thiết của hàm Sortino và Drawdown Penalty thay vì chỉ tối ưu Gross Return. (Sử dụng số liệu SP500).
- **Contribution 2 (Rolling IC-Attention Ensemble):** Giải quyết nhược điểm của Naive Ensemble bằng cách gán trọng số theo sức mạnh dự báo quá khứ gần (Rolling IC 20 ngày), mô hình cắt giảm được Turnover không cần thiết. (Sử dụng số liệu NASDAQ).
- **Contribution 3 (Main Contribution - Co-adapting Meta-learning):** Đỉnh cao của bài báo. Trình bày mạng ContextNet xuất ra $\alpha_t$ và $\lambda_t$. Hàm Loss: 
  `Loss = - (Gross - cost * (alpha_t * Turnover) - lambda_t * DownsideVariance) + rho * Concentration`

## 3. Bằng chứng Thực chứng (Empirical Evidence)
Sử dụng dữ liệu từ `reports/final_upgrade_statistics.md` để lập các bảng (Tables):
* **Bảng 1 (SP500):** So sánh PPO Baseline vs Risk-Aware PPO.
* **Bảng 2 (NASDAQ):** So sánh Baseline vs Dynamic V1.
* **Bảng 3 (Crypto - OOD Test):** So sánh Baseline vs V1 vs V2 (Sharpe 3.511, MDD -15.09%). Đây là bảng chốt hạ Ablation Study.
