# Goal.md — PRICAI 2026 Strong-Accept Upgrade Plan for TC-StockMixer

## Mission

Nâng cấp TC-StockMixer từ bản thử nghiệm “happy path” thành một submission PRICAI 2026 có logic phương pháp, code, backtest và kiểm định thống kê khớp nhau. Giai đoạn này **chỉ dùng bộ dữ liệu hiện có** đã chạy trước. Không mở rộng full-scale cho đến khi các gates dưới đây pass.

## Reviewer-fatal problems cần vá trước

1. `ContextNet` đang được mô tả như bộ nhận diện regime toàn thị trường, nhưng code hiện sinh `alpha_t` và `lambda_t` theo từng stock-row rồi lấy trung bình. Cần đổi thành **global market context encoder** có mask-aware averaging qua chiều asset.
2. Training loss dùng softmax differentiable portfolio, nhưng evaluation dùng Top-K equal-weight. Cần thêm **softmax backtest** làm primary evaluation; Top-K chỉ là secondary robustness check.
3. Kết quả hiện thiếu ablation, multi-seed, walk-forward, cost-grid và bootstrap confidence interval. Cần biến claim thành bằng chứng.

## Claim được phép dùng sau Phase 1

> TC-StockMixer introduces a context-conditioned adaptive regularizer for transaction-cost-aware and downside-risk-aware neural portfolio ranking. The context module estimates global market state from the current lookback window and modulates training-time penalties under a differentiable softmax portfolio objective.

Không dùng các cụm sau cho đến khi code thật sự chứng minh được:

- real-time controller;
- tự động cắt tỷ trọng tài sản rủi ro;
- framework đầu tiên;
- SOTA;
- hoàn hảo;
- không overfitting.

## Required acceptance gates

### Gate A — Code correctness

- `pytest` pass toàn bộ tests mới.
- `GlobalContextNet` output `alpha_t`, `lambda_t` dạng scalar `[1, 1]` cho mỗi rebalance date.
- `alpha_t`, `lambda_t` invariant với hoán vị thứ tự stock.
- Masked assets không ảnh hưởng đến market context.
- `evaluate_model` trả đủ `softmax`, `top5`, `top10`.
- Early stopping dùng `softmax.net_sharpe` hoặc metric được cấu hình rõ.

### Gate B — Current dataset first

Chạy trước trên `crypto_tc_v2.yaml` hoặc config ablation kế thừa nó. Chỉ sau khi pass mới mở rộng sang NASDAQ/SP500.

Minimum empirical target:

- dynamic-both > static-both về mean softmax net Sharpe qua ít nhất 10 seeds;
- dynamic-both không làm MDD xấu hơn static-both quá 2 điểm phần trăm tuyệt đối;
- turnover không tăng quá 10% nếu net return không cải thiện đủ;
- báo mean, std, CI 95% cho Sharpe, mean return, MDD, turnover.

### Gate C — Reviewer-grade evidence

- Ablation: no-net, static-turnover, static-downside, static-both, dynamic-alpha-only, dynamic-lambda-only, dynamic-both.
- Walk-forward trên dataset hiện tại.
- Cost grid: 5, 10, 15, 25, 50 bps.
- Paired stationary bootstrap trên daily return series.
- Simple baselines: equal-weight, buy-and-hold valid universe, random Top-K.

### Gate D — Paper readiness

- Methodology khớp code.
- Backtest protocol khớp training objective.
- Không còn overclaim.
- Có limitation về non-stationarity, data snooping, backtest overfitting, transaction-cost simplification.
- Tuân thủ double-anonymous PRICAI và ghi nhận LLM/AI tool nếu dùng ngoài copy-editing.

## Execution order for Antigravity

1. `01_Reproducibility_Baseline_Audit.md`
2. `02_Global_ContextNet_Regime.md`
3. `03_Softmax_Backtest_Alignment.md`
4. `04_Inference_Diagnostics.md`
5. `05_Ablation_Controls.md`
6. `06_MultiSeed_Runner_Aggregation.md`
7. `07_WalkForward_Current_Data.md`
8. `08_Statistical_Testing.md`
9. `09_Cost_Robustness.md`
10. `10_Baselines_Current_Data.md`
11. `11_Paper_Claim_Control.md`
12. `12_Final_Acceptance_Checklist.md`

Run one file at a time. After each file: run tests, commit, then move to next file.

## Trusted references

- PRICAI 2026 Call for Papers and Submission: https://2026.pricai.org/calls/call-for-papers ; https://2026.pricai.org/submission
- StockMixer AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/view/28681 ; https://github.com/SJTU-DMTai/StockMixer
- White (2000), Reality Check for Data Snooping: https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152
- Hansen (2005), Superior Predictive Ability test: https://www.tandfonline.com/doi/abs/10.1198/073500105000000063
- Politis and Romano (1994), Stationary Bootstrap: https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870
- Ledoit and Wolf (2008), robust Sharpe ratio testing: https://www.ledoit.net/jef_2008pdf.pdf
- Bailey and López de Prado (2014), Deflated Sharpe Ratio: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey et al. (2015), Probability of Backtest Overfitting: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
