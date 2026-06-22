# Phân tích Chuyên sâu (Comprehensive Analysis): TC-StockMixer & Các Baselines

Tài liệu này tổng hợp **toàn bộ** các chỉ số đo lường (thắng/thua/hòa) từ tất cả các thực nghiệm (Baseline StockMixer, S_CGRAN, PPO RL, Non-linear Impact, Market Regimes) trên 2 bộ dữ liệu (SP500, NASDAQ). Mục tiêu là mổ xẻ nguyên nhân gốc rễ (Root Causes) để tìm ra định hướng nâng cấp tiếp theo.

---

## 1. Kết quả Tổng quan: Baseline vs TC-StockMixer (Top 5 Portfolio, Cost: 10 bps)

### 1.1 SP500 (Thắng lợi lớn)
| Phương pháp | Seed/Ensemble | Net Sharpe | Max Drawdown | Avg Turnover |
| :--- | :--- | :--- | :--- | :--- |
| StockMixer (Baseline) | Seed 1 | 0.090 | -36.1% | 0.690 |
| **TC-StockMixer** | Seed 1 | 0.475 | -23.6% | 0.434 |
| **TC-StockMixer** | Seed 2 | 0.061 | -39.4% | 0.443 |
| **TC-StockMixer** | Seed 3 | 0.686 | -24.9% | 0.587 |
| **TC-StockMixer (Ensemble)** | $\kappa = 0.0$ | **0.710** | **-26.5%** | **0.436** |

**Nguyên nhân thành công:** SP500 có tính xu hướng (trend) dài và ít nhiễu. Hàm NetRank Loss hoạt động cực kỳ hiệu quả trong việc ổn định hóa thứ hạng, ngăn chặn mô hình đảo danh mục liên tục, qua đó giảm 37% Turnover và tăng Net Sharpe từ 0.09 lên 0.71.

### 1.2 NASDAQ (Kết quả hỗn hợp / Thua)
| Phương pháp | Seed/Ensemble | Net Sharpe | Max Drawdown | Avg Turnover |
| :--- | :--- | :--- | :--- | :--- |
| StockMixer (Baseline) | Seed 1 | -0.398 | -40.2% | 0.831 |
| **TC-StockMixer** | Seed 1 | -0.295 | -34.8% | 0.868 |
| **TC-StockMixer** | Seed 2 | **0.375** | **-19.4%** | **0.374** |
| **TC-StockMixer** | Seed 3 | -0.698 | -32.4% | 0.421 |
| **TC-StockMixer (Ensemble)** | $\kappa = 0.25$ | -1.225 | -36.3% | 0.536 |

**Nguyên nhân thất bại (Seed 3 & Ensemble):** NASDAQ hội tụ các cổ phiếu công nghệ siêu biến động. Việc cố gắng nén Turnover (thông qua NetRank) trong một thị trường xoay chiều cực nhanh đôi khi khiến mô hình bị "mắc kẹt" (hold) với các cổ phiếu đang giảm giá sâu (momentum đảo chiều mạnh). Seed 3 có Net Sharpe âm sâu (-0.698). Đặc biệt, khi Ensemble lại, kết quả càng tệ (-1.225), cho thấy các Seed dự đoán trái ngược nhau, dẫn đến nhiễu (noise) lấn át tín hiệu (signal).

---

## 2. Kiến trúc Khác: S_CGRAN vs TC-SCGRAN (Chứng minh tính tổng quát)
| Thị trường | Mô hình | Net Sharpe | Max Drawdown | Avg Turnover |
| :--- | :--- | :--- | :--- | :--- |
| SP500 | S_CGRAN (Gốc) | -0.701 | -52.7% | 1.440 |
| SP500 | **TC-SCGRAN** | **-0.137** | **-39.3%** | **1.366** |
| NASDAQ | S_CGRAN (Gốc) | -0.383 | -39.6% | 1.589 |
| NASDAQ | **TC-SCGRAN** | **0.467** | **-25.3%** | **0.362** |

**Nguyên nhân nâng cấp:** S_CGRAN gốc là mô hình có Turnover siêu cao (1.4 - 1.5), thay máu danh mục hàng ngày. Khi cấy NetRank loss vào, Turnover trên NASDAQ giảm ngoạn mục xuống 0.36, kéo Sharpe từ số âm lên số dương (0.467). Điều này xác nhận *NetRank loss hoạt động độc lập với kiến trúc mạng*.

---

## 3. RL Baseline: PPO (PPO vs TC-StockMixer)
| Dataset | Method | Net Sharpe | Turnover | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
| NASDAQ | ppo (Seed 2) | 0.939 | 0.110 | -31.9% |
| SP500 | ppo (Seed 1) | 1.443 | 0.071 | -17.8% |

**Nguyên nhân PPO chiến thắng về Net Sharpe:** 
PPO có khả năng tự do học một policy gần như "Buy & Hold" đối với các cổ phiếu an toàn nhất, thể hiện qua Turnover cực nhỏ (0.07). Trong môi trường Bull Market, chiến lược hold mang lại Net Sharpe cực cao (vì không tốn phí giao dịch). Tuy nhiên, PPO có rủi ro tiềm ẩn rất lớn ở Bear Market (xem phần 5).

---

## 4. Market Regime Analysis (Sức chống chịu trong Bull/Bear)
Phân tách hiệu suất trong giai đoạn thị trường tăng (Bull) và giảm mạnh (Bear - thị trường giảm >5% trong 60 ngày).

| Dataset | Method | Bull Ann Ret | Bull MDD | Bear Ann Ret | Bear MDD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NASDAQ | TC-StockMixer | -31.3% | -33.8% | N/A | N/A |
| NASDAQ | PPO | +31.1% | -20.8% | N/A | N/A |
| SP500 | Baseline | +18.1% | -36.0% | -328.5% | -16.5% |
| SP500 | TC-StockMixer | +38.8% | -26.5% | **-281.9%** | **-14.4%** |
| SP500 | PPO | +69.1% | -17.8% | -371.9% | -17.1% |

**Nguyên nhân sâu xa:**
1. **Thất bại của PPO trong Bear Market:** Mặc dù PPO lãi lớn ở Bull Market (+69.1%), nhưng khi Bear Market xảy ra (khủng hoảng), PPO thua lỗ nặng nề nhất (-371.9% return chuẩn hóa, MDD -17.1%). Vì PPO học policy Buy&Hold lười biếng, nó không kịp xả hàng khi thị trường sụp đổ.
2. **Chiến thắng của TC-StockMixer trong Bear Market:** TC-StockMixer vẫn giao dịch linh hoạt (Turnover ~0.4) nhưng có định hướng, giúp nó phòng thủ tốt nhất trong đợt sập của SP500 (MDD thấp nhất: -14.4%, và Return âm ít nhất: -281.9%).

---

## 5. Non-linear Market Impact (Trượt giá phi tuyến)
Khi khối lượng giao dịch lớn, chi phí không tỷ lệ thuận (linear) mà chịu trượt giá hàm căn bậc hai (Square-root Law).

| Dataset | Method | Turnover | Standard Sharpe | Non-linear Sharpe | Sharpe Drop |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SP500 | Baseline | 0.690 | 0.090 | -0.852 | **0.942** |
| SP500 | TC-StockMixer | 0.435 | 0.710 | 0.090 | **0.620** |
| SP500 | PPO | 0.071 | 1.443 | 1.341 | 0.102 |

**Phân tích nguyên nhân:** Baseline có Turnover quá cao, nên khi áp trượt giá, nó mua bán khối lượng lớn và gánh chịu tổn thất trượt giá khổng lồ (Drop 0.94). TC-StockMixer giữ lại được phần lớn lợi nhuận (chỉ rớt 0.62). PPO gần như không bị ảnh hưởng vì nó... không giao dịch.

---

## 6. Tổng kết Nguyên nhân Thất bại & Hướng nâng cấp tương lai (Next Steps)

### Vấn đề 1: Thất bại ở thị trường biến động mạnh (NASDAQ)
- **Triệu chứng:** TC-StockMixer Seed 3 và Ensemble bị âm Net Sharpe nặng.
- **Root Cause:** NetRank ép mô hình không thay đổi danh mục quá mức. Ở NASDAQ, các cổ phiếu (Tech) đảo chiều cực gắt. Việc "lười" đổi cổ phiếu khiến mô hình ôm bom rớt giá.
- **Giải pháp Nâng cấp:** Tích hợp cơ chế **Dynamic Gamma**: Cho phép `gamma_net` nhỏ đi khi thị trường có độ biến động cao (VIX cao), và tăng lên khi thị trường bình ổn.

### Vấn đề 2: PPO "lười biếng" nhưng Net Sharpe lại cực tốt trong dài hạn
- **Triệu chứng:** RL agent tìm ra lỗ hổng của reward function: Chỉ cần Buy & Hold nhóm cổ phiếu Bluechip, Turnover = 0, khỏi tốn phí, Net Sharpe tự nhiên cao.
- **Root Cause:** Thiếu constraint rủi ro trong PPO.
- **Giải pháp Nâng cấp:** Thiết kế một thuật toán lai (Hybrid RL + NetRank) hoặc thêm Downside Risk Penalty vào phần thưởng của PPO để ép nó cắt lỗ sớm thay vì hold qua mùa Bear Market.

### Vấn đề 3: Lỗi Ensemble
- **Triệu chứng:** NASDAQ trung bình các prediction lại làm kết quả tệ hơn từng Seed lẻ.
- **Root Cause:** Phân phối dự đoán (predictions) không đồng nhất.
- **Giải pháp Nâng cấp:** Sử dụng cơ chế chú ý (Attention-based Ensemble) thay vì trung bình trọng số đơn giản. Tự động loại bỏ các Seed có dự đoán nhiễu loạn trong quá trình Validation.
