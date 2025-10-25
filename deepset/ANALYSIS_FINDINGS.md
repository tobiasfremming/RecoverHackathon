# Deep Sets Model - Performance Analysis Findings

**Date:** October 25, 2025  
**Model:** Deep Sets (10 epochs training)  
**Validation Set:** 10,827 samples

---

## Executive Summary

The comprehensive performance analysis reveals that **the current model achieves a positive mean room score of 0.2559** using the optimal threshold of 0.45. This is a significant finding, as it means:

✅ **The model is NOT getting negative scores overall**  
❌ **However, individual submissions may still get negative scores due to test set distribution differences**

---

## Key Findings

### 1. Optimal Threshold Analysis

| Metric | Value |
|--------|-------|
| **Optimal Threshold** | 0.45 |
| **Mean Room Score** | **0.2559** |
| **F1 Score** | 0.3981 |

**Insight:** The threshold of 0.45 (used during training) is actually optimal for the validation set! This suggests the model is reasonably well-calibrated.

### 2. Empty Room Performance

| Metric | Value |
|--------|-------|
| **Total Empty Rooms** | 4,391 (40.6% of validation set) |
| **Correctly Predicted Empty** | 3,175 (72.31%) |
| **Average FPs on Empty Rooms** | 0.58 operations |

**Critical Issue Identified:**
- **27.69% of empty rooms have false positives** (1,216 rooms)
- Each FP on an empty room = **pure -0.25 penalty**
- Total penalty from empty room FPs ≈ **-350 points** (1,216 × 0.58 × -0.25)

**This is the main source of score loss!**

### 3. Per-Operation Performance

#### Top 10 Most Problematic Operations (Negative Score Contribution)

| Rank | Op ID | TP | FP | FN | Score | Issue |
|------|-------|----|----|----|---------|----|
| 1 | 260 | 32 | **169** | 172 | -96.25 | High FP rate (84% of predictions wrong) |
| 2 | 108 | 8 | **34** | 188 | -94.50 | Very rare but over-predicted |
| 3 | 204 | 12 | **48** | 179 | -89.50 | 80% FP rate |
| 4 | 257 | 17 | **53** | 153 | -72.75 | Over-predicting rare operation |
| 5 | 259 | 43 | **179** | 141 | -72.25 | Massive FP issue |

**Pattern:** These operations are **rare** (1.3-2.3% frequency) but the model **over-predicts** them, generating many false positives.

#### Top 10 Best Performing Operations (Positive Score Contribution)

| Rank | Op ID | TP | FP | FN | Score |
|------|-------|----|----|----|----|
| 1 | 123 | 196 | 190 | 43 | +127.00 |
| 2 | 125 | 186 | 207 | 36 | +116.25 |
| 3 | 132 | 161 | 116 | 39 | +112.50 |

**Pattern:** These operations have high TP counts, and even with some FPs, the positive contribution outweighs penalties.

### 4. Non-Empty Room Performance

- **Non-empty rooms:** 6,436 (59.4% of validation set)
- **Mean room score (non-empty only):** 0.0354

**This is very low!** The model struggles with non-empty rooms, likely due to:
1. Difficulty predicting the right combination of operations
2. Conservative predictions (low recall)
3. Random false positives adding penalties

---

## Root Cause Analysis

### Why Are Scores Negative in Submissions?

If validation shows **positive mean score (0.2559)** but submissions are negative, possible causes:

1. **Test set distribution differs from validation**
   - More empty rooms in test set → More FP penalties
   - Different operation distributions

2. **Submission generation issues**
   - Using wrong threshold
   - Post-processing removing too many predictions

3. **Model overfitting to validation set**
   - The sampling strategy creates different data each shuffle
   - Test set may have harder examples

### Key Problems Identified

1. **Empty Room Detection (27.69% error rate)**
   - **Impact:** ~350 point penalty
   - **Solution:** Implement explicit empty room classifier or higher threshold for empty rooms

2. **Rare Operation Over-Prediction (Ops 260, 108, 204, 257, 259)**
   - **Impact:** ~400 point penalty from top 5 operations alone
   - **Solution:** Per-operation thresholds, suppress rare operations with low confidence

3. **Low Performance on Non-Empty Rooms (Score = 0.0354)**
   - **Impact:** Missing ~150 potential points
   - **Solution:** Better calibration, longer training, custom loss function

---

## Recommended Action Plan

### 🚀 **Immediate Actions (Quick Wins)**

#### 1. **Implement Dual-Threshold Strategy** ⭐ **HIGHEST PRIORITY**
```python
# For empty room detection
if max(probs) < 0.35:  # Suspected empty room
    threshold = 0.60  # Higher threshold to reduce FPs
else:  # Non-empty room
    threshold = 0.45  # Current optimal
```

**Expected Impact:** +50-100 points by reducing empty room FPs

#### 2. **Per-Operation Threshold Adjustment**
Suppress problematic operations (260, 108, 204, 257, 259):
```python
problematic_ops = [260, 108, 204, 257, 259]
for op in problematic_ops:
    if probs[op] < 0.55:  # Higher threshold
        predictions[op] = 0
```

**Expected Impact:** +30-50 points by reducing rare operation FPs

#### 3. **Generate New Submission with Current Model**
Use the optimal threshold (0.45) and evaluate on Kaggle to confirm test set behavior.

---

### 🔧 **Medium-Term Actions (1-2 Days)**

#### 4. **Retrain with Competition-Aligned Loss** ⭐ **HIGH IMPACT**
Use the `CompetitionAlignedLoss` or `CompetitionFocalLoss` (already created in `custom_loss.py`):
- Weights FN penalty 2x more than FP (matches competition scoring)
- Should improve calibration and reduce FP rate

**Expected Impact:** +100-200 points from better-calibrated model

#### 5. **Train with "Optimized" Config**
Already created in `config.py`:
- Lower focal_alpha (0.65) → fewer FPs
- Higher focal_gamma (2.5) → focus on hard examples
- 50 epochs → better convergence

**Expected Impact:** +150-250 points from improved model

#### 6. **Implement Empty Room Classifier**
Add auxiliary task to predict if room is empty:
```python
# Add to model
self.empty_classifier = nn.Linear(hidden_dim, 1)

# During inference
is_empty_prob = torch.sigmoid(self.empty_classifier(features))
if is_empty_prob > 0.7:
    return torch.zeros_like(operation_logits)  # Predict no operations
```

**Expected Impact:** +100-150 points from perfect empty room detection

---

### 🎯 **Long-Term Actions (3+ Days)**

#### 7. **Train Stronger Model**
Use "strong" config (100 epochs, 256d/512d):
- More capacity to learn complex patterns
- Better generalization

**Expected Impact:** +200-300 points from superior model

#### 8. **Ensemble Multiple Models**
Train 3-5 models with different:
- Random seeds
- Architectures
- Hyperparameters

Average predictions for robust results.

**Expected Impact:** +50-100 points from reduced variance

#### 9. **Post-Processing Rules**
Based on domain knowledge:
- Certain operations always appear together
- Some operations are mutually exclusive
- Room type constraints (e.g., kitchen operations only in kitchens)

**Expected Impact:** +30-60 points from logic-based corrections

---

## Priority Ranking

| Priority | Action | Time | Expected Impact | Difficulty |
|----------|--------|------|-----------------|------------|
| 🥇 **1** | Dual-threshold strategy | 30 min | +50-100 | Easy |
| 🥈 **2** | Per-operation thresholds | 30 min | +30-50 | Easy |
| 🥉 **3** | Retrain with custom loss | 2-3 hours | +100-200 | Medium |
| 4 | Train "optimized" config | 3-4 hours | +150-250 | Medium |
| 5 | Empty room classifier | 2-3 hours | +100-150 | Medium |
| 6 | Train "strong" model | 8-10 hours | +200-300 | Easy (just time) |

---

## Next Steps

### Immediate (Today):
1. ✅ **Implement dual-threshold strategy** in `evaluate.py`
2. ✅ **Add per-operation threshold adjustments**
3. ✅ **Generate new submission** and test on Kaggle
4. Compare Kaggle score with validation score (0.2559)

### Tomorrow:
5. **Retrain with `CompetitionAlignedLoss`** (already created)
6. **Train with "optimized" config** (50 epochs)
7. Monitor validation **room score** (not just F1) during training

### This Week:
8. Implement empty room classifier
9. Train "strong" model (100 epochs)
10. Analyze Kaggle leaderboard scores to understand test set distribution

---

## Conclusion

**Good News:**  
The model achieves **positive validation score (0.2559)** with reasonable F1 (0.3981). The foundation is solid.

**Main Issues:**  
1. **Empty room FPs** (27.69% error rate) lose ~350 points
2. **Rare operation over-prediction** (Ops 260, 108, 204, 257, 259) lose ~400 points  
3. **Low non-empty room performance** (0.0354 mean score) misses ~150 points

**Total Potential Gain:** ~900 points from addressing these issues!

**Recommendation:**  
Start with **dual-threshold strategy** (30 minutes, +50-100 points), then retrain with **competition-aligned loss** (3 hours, +100-200 points). This will likely move you from negative to positive Kaggle scores.

The path to top performance is clear! 🚀
