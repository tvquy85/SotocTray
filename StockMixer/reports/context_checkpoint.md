# Project Context Checkpoint (Dành cho AI Reload)

Khi tải lại (reload) project này vào phiên làm việc mới, hỡi AI, hãy đọc kỹ các thông tin sau để ngay lập tức tiếp tục công việc hỗ trợ User mà không bị mất ngữ cảnh (Context Loss):

## 1. Project Overview
- **Dự án:** LOBExp / StockTrader / TC-StockMixer
- **Mục tiêu:** Nâng cấp thuật toán Portfolio Optimization để submit bài báo đạt hạng "Strong Accept" tại AAAI / PRICAI 2026.
- **Tiến độ hiện tại:** Đã hoàn tất Phase 5. Thuật toán hiện tại (Version 2) đã được nâng cấp lên mức độ "Unified Regime-Aware Optimization", tự động co giãn chi phí giao dịch và rủi ro sụt giảm (Downside Risk).

## 2. File & Thư mục Trọng yếu (Important Files)
- `reports/final_upgrade_statistics.md`: Chứa TOÀN BỘ số liệu kết quả của các bài test trên SP500, NASDAQ, Crypto. (Đây là trái tim số liệu của bài báo).
- `reports/paper_preparation_kit.md`: Chứa cốt truyện và luận điểm để viết Paper.
- `src_tc/model_tc_v2.py`: Chứa kiến trúc `StockMixerBackboneV2` với mạng `ContextNet` xuất ra `alpha_t` và `lambda_t`.
- `src_tc/losses_v2.py`: Chứa hàm Loss V2 có tính đến `Downside Variance Penalty`.
- `configs/crypto_tc_v2.yaml`: File cấu hình mới nhất dùng để huấn luyện V2 trên Crypto.

## 3. Kết quả tốt nhất tính đến hiện tại (favorable performance in the tested setting in this Repo)
- **Mô hình:** `StockMixerBackboneV2` (Dynamic NetRank V2).
- **Thị trường thử nghiệm:** Crypto (117 tokens, cực kỳ nhiễu).
- **Kết quả trung bình (3 Seeds) Top-10:** Net Sharpe = 3.511, Turnover = 0.419, Max Drawdown = -15.09%.
- Các kết quả lưu tại: `results/crypto_tc_v2_seed1/`, `results/crypto_tc_v2_seed2/`, `results/crypto_tc_v2_seed3/`.

## 4. Next Steps (Hướng dẫn hành động tiếp theo)
Nếu User yêu cầu tiếp tục, hãy xem xét các hướng sau:
1. Vẽ biểu đồ (Equity Curve / Pareto Frontier) cho các file `metrics.json` của V2 để chuẩn bị chèn vào PDF của bài báo.
2. Viết mã nguồn LaTeX trực tiếp cho bài báo dựa trên `paper_preparation_kit.md`.
3. Chạy thêm Sensitivity Analysis (Thay đổi `eval_cost_bps` lên 30 hoặc 50) nếu User muốn tăng cường độ khó cho Robustness Test.
