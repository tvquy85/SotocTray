# Thuyết minh Dự án TC-StockMixer cho PRICAI 2026

File này mô tả lại một cách trực quan, dễ hiểu toàn bộ cách tiếp cận, điểm đóng góp (contribution), tiềm năng của bài báo tại hội nghị PRICAI 2026, cùng luồng thực nghiệm hiện tại. Bạn có thể dùng Domain Knowledge của mình để review và tinh chỉnh thêm các luận điểm.

---

## 1. Cách tiếp cận dễ hiểu (The Intuitive Approach)

### Vấn đề (Pain point)
Hầu hết các bài báo về AI dự đoán chứng khoán hiện nay (bao gồm cả mô hình StockMixer gốc) thường rơi vào một cái bẫy: **Chỉ tối ưu hóa độ chính xác dự đoán tĩnh (như sai số MSE hoặc hệ số tương quan Rank IC)**. 
Tuy nhiên, trong thực tế giao dịch, việc dự đoán đúng thứ hạng cổ phiếu chưa chắc đã sinh lời. Lý do lớn nhất là **Chi phí giao dịch (Transaction Cost)**. Nếu hôm nay mô hình bảo mua cổ phiếu A, ngày mai bảo bán A mua B, quá trình đảo danh mục liên tục này (Turnover cao) sẽ sinh ra chi phí giao dịch khổng lồ, "ăn mòn" toàn bộ lợi nhuận (Gross Return), dẫn đến lợi nhuận ròng (Net Return) bị âm.

### Giải pháp của TC-StockMixer
Chúng ta vẫn giữ nguyên kiến trúc mạng "Mixer" nhẹ, nhanh và hiệu quả của StockMixer (vốn bắt tương quan thị trường rất tốt). Nhưng ta **thay đổi não bộ (Hàm Loss) và cách ra quyết định** của nó:
1. **NetRank Loss**: Thay vì chỉ phạt mô hình khi nó dự đoán sai giá cổ phiếu, ta ép mô hình phải tính toán luôn cả *sự thay đổi danh mục so với ngày hôm qua*. Nếu việc đổi từ cổ phiếu A sang cổ phiếu B sinh ra lợi nhuận không đủ bù đắp chi phí giao dịch, hàm Loss sẽ phạt mô hình. Điều này ép mô hình học cách "chỉ giao dịch khi thực sự chắc chắn và lợi nhuận đủ lớn".
2. **Uncertainty-Gating (Kiểm soát rủi ro bằng độ bất định)**: Thị trường tài chính đầy nhiễu. Thay vì tin tưởng tuyệt đối vào 1 mô hình, ta dùng nhiều mô hình (ensemble các seed khác nhau). Chỉ những cổ phiếu nào mà cả 3 mô hình đều đồng thuận cao (độ bất định thấp) mới được đưa vào danh mục mua.

---

## 2. Điểm đóng góp chính (Contributions)

1. Đề xuất một hàm mục tiêu khả vi (Differentiable Portfolio Objective - **NetRank Loss**) kết hợp trực tiếp yếu tố Vòng quay danh mục (Turnover) và Chi phí giao dịch vào quá trình huấn luyện End-to-End.
2. Tích hợp cơ chế đo lường độ bất định (Uncertainty Estimation) thông qua Seed-Ensemble, cho phép mô hình có khả năng "từ chối giao dịch" (gating) khi tín hiệu dự đoán mâu thuẫn hoặc rủi ro cao.
3. Thiết lập một chuẩn mực Backtest chặt chẽ (Leakage-free) và cung cấp bộ kết quả chứng minh thực nghiệm rõ ràng: Việc hi sinh một phần nhỏ Rank IC (khả năng xếp hạng thuần túy) để đổi lấy sự ổn định của Turnover mang lại sự đột phá về Net Sharpe Ratio (lợi nhuận thực tế trên rủi ro).

---

## 3. Tại sao có khả năng cao được Accept tại PRICAI 2026?

1. **Tính Ứng Dụng & Giải quyết Đúng "Nỗi Đau":** PRICAI đánh giá cao các nghiên cứu AI ứng dụng giải quyết được rào cản trong thực tế. Đa số reviewer có kinh nghiệm AI for Finance đều biết rỗng (gap) giữa IC cao và lợi nhuận thực tế là rất lớn. Bài báo này giải quyết trực diện vấn đề đó.
2. **Độ sâu kĩ thuật (Technical Depth):** Việc đưa Turnover (vốn là hàm bậc thang/rời rạc) vào trong quá trình backward-propagation của PyTorch thông qua Softmax/Soft-turnover là một kĩ thuật khéo léo và có hàm lượng toán học tốt.
3. **Thực nghiệm vững chắc:** Không chỉ dừng ở lý thuyết, kết quả thực nghiệm trên SP500 và NASDAQ với mức phí giao dịch thực tế (10 bps) là minh chứng hùng hồn. Cải thiện Net Sharpe Ratio từ `0.090` lên `0.710` (SP500) là một con số cực kì thuyết phục đối với bất kì hội đồng nào.

---

## 4. Cách làm thí nghiệm từng bước (trường hợp thuận lợi Workflow)

Để chứng minh luận điểm, thí nghiệm được chạy qua các bước tiêu chuẩn như sau:

### Bước 1: Tiền xử lý và chia dữ liệu (Strict Temporal Split)
- **Mô tả:** Dữ liệu chứng khoán (như giá, khối lượng) được chia theo các mốc thời gian tăng dần để đảm bảo không bị "nhìn trộm tương lai" (Look-ahead bias).
- **Ví dụ minh họa:** Ta dùng dữ liệu từ `2010 - 2015` để học (Train), dùng `2016` để kiểm tra độ chính xác và điều chỉnh tham số (Valid), và dùng `2017 - 2020` làm thực tế để đánh giá (Test). 

### Bước 2: Tối ưu hoá mô hình với NetRank Loss
- **Mô tả:** Mô hình đưa ra điểm số (score) cho từng cổ phiếu để chọn Top 5 cổ phiếu tốt nhất mỗi ngày. Trong quá trình học, nó phải theo dõi danh mục ngày hôm trước (`prev_w`).
- **Ví dụ minh họa:** 
  - Hôm qua (T-1), danh mục Top 5 đang có Apple và Microsoft.
  - Hôm nay (T), mô hình dự đoán Tesla sẽ tăng thêm 0.05% và định bán Apple để mua Tesla. 
  - Hệ thống Loss sẽ tính: "Phí giao dịch bán Apple và mua Tesla tốn 0.1%. Đổi mã thế này lỗ mất 0.05%!". Hệ thống lập tức "phạt" (penalize) mô hình, ép nó tự học ra thói quen: **Nếu cổ phiếu mới không tốt vượt trội, thì cứ giữ nguyên danh mục cũ cho đỡ tốn phí.**

### Bước 3: Đánh giá bằng cơ chế Uncertainty-Gated Ensemble
- **Mô tả:** Thay vì chạy 1 lần, hệ thống chạy mô hình với 3 khởi tạo ngẫu nhiên khác nhau (Seed 1, 2, 3).
- **Ví dụ minh họa:**
  - Để xem ngày mai có nên mua mã Nvidia (NVDA) không.
  - Seed 1 bảo: "Tăng mạnh, mua!". Seed 2 bảo: "Tăng nhẹ, mua!". Seed 3 bảo: "Giảm mạnh, bán!".
  - Sự mâu thuẫn này sinh ra **độ bất định (Uncertainty) cao**. Hệ thống Gating sẽ đánh giá NVDA rủi ro và quyết định loại bỏ nó khỏi rổ mua, giúp danh mục an toàn hơn.

### Bước 4: Backtesting thực tế và Xuất báo cáo
- **Mô tả:** Chạy mô hình trên tập Test (2017-2020) với luật giao dịch Top-K Equal Weight, trừ đi đúng 10 bps (0.1%) cho mỗi lần thay đổi tỷ trọng. Tính các chỉ số thực chiến: Net Sharpe, Max Drawdown, Turnover. So sánh nó với Baseline (StockMixer gốc không có hàm ép phí giao dịch).
- **Ví dụ minh họa:** Tạo ra file `main_results.csv` và vẽ biểu đồ Equity Curve (Đường cong tài sản) so sánh tài khoản của người dùng TC-StockMixer so với StockMixer gốc sau 3 năm.

---

## 5. Hiện trạng thí nghiệm hiện tại

- **Đã hoàn thành 100% tiến trình trường hợp thuận lợi.**
- Hệ thống đã tự động chạy song song trót lọt trên 3 Seeds cho 2 tập dữ liệu lớn là **NASDAQ** và **SP500** (bỏ qua NYSE do thiếu dữ liệu data gốc).
- Các script tự động gộp kết quả (`aggregate_results.py`) và tạo báo cáo đồ thị (`generate_report.py`) đã xuất file thành công.
- **Thành quả**: Net Sharpe trên SP500 tăng từ mức cận biên `0.090` (của Baseline) lên `0.710`, với mức sụt giảm (Max Drawdown) được cải thiện an toàn hơn. Trên NASDAQ, từ chỗ bị lỗ nặng (`-0.398`), mô hình đã nén chi phí giao dịch cực kì hiệu quả để giữ được Net Sharpe dương.
- Tất cả hình ảnh và bảng biểu `.csv` đã nằm trong thư mục `tables/` và `figures/` sẵn sàng để bạn đưa vào bài paper PRICAI.
