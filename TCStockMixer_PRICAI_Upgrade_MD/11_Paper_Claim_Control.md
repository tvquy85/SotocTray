# 11 — Paper Claim Control and Reviewer-Safe Framing

## Objective
Rewrite the paper claims so they match the implementation and evidence. This reduces reject risk from overclaiming.

This task edits paper text only. Do not change model code here.

## Files to edit

```text
StockMixer/paper/01_Introduction.md
StockMixer/paper/02_Related_Work.md
StockMixer/paper/03_Methodology.md
StockMixer/paper/04_Experiments_and_Results.md
StockMixer/paper/05_Discussion.md
StockMixer/paper/06_Conclusion.md
StockMixer/tonghop.md
StockMixer/MoTa.md
```

## Step 1 — Replace forbidden phrases

Search for and remove these phrases:

```text
first framework
đầu tiên
SOTA
state-of-the-art
thắng tuyệt đối
perfect shield
tấm khiên hoàn hảo
siêu lợi nhuận
happy path
guaranteed
hoàn hảo
```

Replace with guarded language:

```text
context-conditioned adaptive regularization
risk-aware ranking framework
favorable performance in the tested setting
improved downside control under the current evaluation protocol
requires broader validation
```

## Step 2 — Correct the core contribution statement

Use this contribution statement unless later code truly implements portfolio-level controller execution:

```text
We propose TC-StockMixer, a transaction-cost-aware and downside-risk-aware extension of StockMixer. The method uses a market-regime encoder to condition the regularization strength of turnover and downside-risk penalties during training, thereby shaping a cross-asset neural ranking model for cost-aware portfolio selection.
```

Do not write that `alpha_t` and `lambda_t` directly control portfolio weights unless task 03 softmax execution is completed and reported.

## Step 3 — Add implementation clarification

In Methodology, add a subsection:

```text
Training-time adaptive regularization versus execution-time allocation
```

Required text:

```text
The context variables alpha_t and lambda_t are used to modulate the differentiable training objective. In Top-K evaluation, the learned score is used for ranking and portfolio construction is deterministic equal-weight Top-K. In the softmax execution variant, the same differentiable portfolio weights used during training are used for backtesting. We report both variants to separate ranking improvements from allocation-level effects.
```

Only include the final sentence if task 03 has been completed.

## Step 4 — Add ablation commitment

In Experiments, add:

```text
To identify whether gains come from market-conditioned regularization rather than additional parameters, we compare against no-penalty, static-penalty, single-dynamic-penalty, and random-context controls under the same dataset and transaction-cost assumptions.
```

## Step 5 — Add statistical reporting rule

In Experiments, use:

```text
Unless otherwise stated, all reported results are mean ± standard deviation across seeds. We also report bootstrap confidence intervals for Sharpe ratios and paired bootstrap comparisons against the strongest baseline available under the same data split.
```

Do not report only the best seed in the main table.

## Step 6 — Fix transaction-cost statement

Use one of the two options.

### Option A — same train and evaluation cost

```text
Both training and evaluation use a transaction cost of X bps.
```

### Option B — different train and evaluation cost

```text
The model is trained with X bps and evaluated with Y bps. We use the evaluation cost as the primary reporting cost and provide sensitivity analysis over 0, 5, 10, 15, 25, and 50 bps.
```

Do not mix 10 bps and 15 bps in different sections unless both are explicitly part of the cost grid.

## Step 7 — Add PRICAI compliance note

Add to the paper checklist, not necessarily the main paper:

```text
This submission is original, not under concurrent review, and anonymized for double-blind review. Any use of LLM-based assistance is documented according to the conference policy and does not replace the authors' scientific contribution, experiments, or analysis.
```

## Step 8 — Test case: claim audit script

Create `scripts/audit_paper_claims.py`.

```python
import argparse
import pathlib
import sys

FORBIDDEN = [
    "first framework",
    "đầu tiên",
    "sota",
    "state-of-the-art",
    "thắng tuyệt đối",
    "perfect shield",
    "tấm khiên hoàn hảo",
    "siêu lợi nhuận",
    "happy path",
    "guaranteed",
    "hoàn hảo",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="paper")
    args = parser.parse_args()

    bad = []
    for path in pathlib.Path(args.root).rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in FORBIDDEN:
            if phrase.lower() in text:
                bad.append((str(path), phrase))

    if bad:
        for path, phrase in bad:
            print(f"FORBIDDEN: {path}: {phrase}")
        sys.exit(1)
    print("Claim audit passed")


if __name__ == "__main__":
    main()
```

Run:

```bash
cd StockMixer
python scripts/audit_paper_claims.py --root paper
```

## Pass conditions

This step passes only if:

1. Forbidden overclaim phrases are removed from paper files.
2. The contribution statement matches actual code behavior.
3. Transaction-cost assumptions are consistent.
4. LLM assistance policy is documented outside the anonymized manuscript if required by submission system.
5. The claim audit script exits with code 0.

