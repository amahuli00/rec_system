# Two-Tower Recommendation Model - Performance Issues & Consultation Request

## Executive Summary

I've implemented a Two-Tower neural network for candidate generation in a movie recommendation system using the MovieLens dataset. The model is **underperforming a simple popularity baseline**, suggesting fundamental issues with the training strategy. I need help diagnosing the problem and implementing fixes.

**Critical Issue**: The learned model (Recall@10: 0.0180) performs worse than just recommending popular movies to everyone (Recall@10: 0.0227).

---

## Dataset Information

### Data Source
- **Dataset**: MovieLens (implicit feedback derived from explicit ratings)
- **Positive Threshold**: Ratings >= 4.0 (treated as positive interactions)
- **Train Size**: 463,020 positive interactions
- **Validation Size**: 12,528 positive interactions (490 users with positives)
- **Test Size**: 43,349 positive interactions

### Dataset Statistics
- **Number of users**: 5,400
- **Number of items**: 3,662
- **Sparsity**: ~98.4% (463,020 / (5,400 × 3,662))
- **Positive rate**: 57.9% of train ratings are >= 4
- **Time-based split**: Train/val/test split chronologically to prevent data leakage

### Data Characteristics
- Median positive interactions per user in train: ~62 items
- Long-tail distribution: some users have hundreds of ratings, others very few
- Item popularity follows power law distribution

---

## Model Architecture

### Two-Tower Design

**User Tower**:
```
Embedding(num_users=5,400, embedding_dim=128)
→ L2 Normalize
```

**Item Tower**:
```
Embedding(num_items=3,662, embedding_dim=128)
→ L2 Normalize
```

### Architecture Decisions
- **No MLP layers**: Just embedding + L2 normalization
- **Embedding dimension**: 128 (larger than typical since no MLP)
- **L2 Normalization**: Enables cosine similarity via dot product
- **Embedding initialization**: Normal(mean=0, std=0.1)
- **Total parameters**: ~1.16M parameters

### Scoring Function
```
score(user, item) = dot_product(user_embedding, item_embedding)
```
Since embeddings are L2-normalized, this equals cosine similarity.

---

## Training Strategy

### Loss Function
**Currently Using**: BPR (Bayesian Personalized Ranking)

```python
# For each training sample (user, positive_item):
neg_item = random_sample_from_all_items()
user_emb = UserTower(user)
pos_emb = ItemTower(positive_item)
neg_emb = ItemTower(neg_item)

pos_score = dot(user_emb, pos_emb)
neg_score = dot(user_emb, neg_emb)

loss = -log(sigmoid(pos_score - neg_score))
```

**Problem Identified**: Random negative sampling can include:
- Items the user actually liked (false negatives)
- Only 1 negative per positive (weak training signal)

### Hyperparameters
```
Batch size: 1024
Learning rate: 0.05
Optimizer: Adagrad
Epochs: 30
Weight decay: None
Dropout: 0.1 (defined but not used - Tower ignores it)
Gradient clipping: None
```

### Training Behavior
- **Epoch 1 loss**: 0.5317
- **Epoch 10 loss**: 0.3748
- **Epoch 30 loss**: 0.3723
- **Plateau**: Loss barely changes after epoch 10 (~0.0025 improvement over 20 epochs)

---

## Current Performance (VALIDATION SET)

### Two-Tower Model (Current)
- **Recall@10**: 0.0180 (1.8%)
- **Recall@50**: 0.1018 (10.2%)
- **Recall@100**: 0.1858 (18.6%)

### Popularity Baseline
- **Recall@10**: 0.0227 (2.3%) ← **BETTER**
- **Recall@50**: 0.1180 (11.8%) ← **BETTER**
- **Recall@100**: 0.1928 (19.3%) ← **BETTER**

### Random Baseline
- **Recall@10**: 0.0015 (0.15%)
- **Recall@50**: 0.0109 (1.1%)
- **Recall@100**: 0.0252 (2.5%)

**Interpretation**: Model beats random but loses to popularity. This suggests it's learning *something* but not meaningful user-item patterns.

---

## Code vs Documentation Mismatch

### What the Notebook Claims
The extensive markdown comments describe using **in-batch negatives (InfoNCE loss)**:
- "With batch size 1024, each sample gets ~1023 negatives for free"
- "Industry standard (YouTube, Google, Spotify)"
- "Dynamic negatives prevent overfitting"

### What the Code Actually Does
```python
USE_BPR = True  # ← This flag switches to BPR

if USE_BPR:
    neg_items = torch.randint(0, num_items, (len(user_ids),), device=device)
    loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
else:
    loss = in_batch_softmax_loss(user_emb, item_emb, TEMPERATURE)
```

The in-batch softmax code exists but isn't being used.

---

## My Hypotheses for Poor Performance

### Primary Suspects

1. **Weak Negative Sampling**
   - Random sampling can pick items the user rated >= 4 as "negatives"
   - Only 1 negative per positive (vs 1023 with in-batch)
   - No hard negative mining

2. **Aggressive Learning Rate**
   - Adagrad with LR=0.05 is very high for embedding models
   - Common practice: 0.001-0.01 with Adam
   - May cause unstable early training

3. **No Regularization**
   - No weight decay on embeddings
   - No dropout (defined but ignored)
   - Model may memorize training noise

4. **Insufficient Model Capacity**
   - No MLP layers (just embedding → normalize)
   - Can only learn linear relationships in embedding space
   - May need 1-2 hidden layers

5. **Early Plateau**
   - Loss plateaus around epoch 10
   - Suggests optimizer is stuck or LR is too low after adaptive adjustment

### Secondary Concerns

6. **Temperature Not Defined**
   - Code references `TEMPERATURE` variable for in-batch softmax
   - This variable is never set in the visible code

7. **Missing Best Practices**
   - No learning rate scheduling
   - No early stopping based on validation metrics
   - No gradient clipping
   - No warmup period

---

## What I Think Could Work

### High Priority Fixes

1. **Switch to In-Batch Softmax Loss**
   - Set `USE_BPR = False`
   - Define `TEMPERATURE = 0.05` (common value for contrastive learning)
   - Benefit: 1023 dynamic negatives per sample

2. **Fix Optimizer & Learning Rate**
   - Switch from Adagrad to Adam
   - Lower LR to 0.001 or 0.01
   - Add cosine annealing or step decay

3. **Add Regularization**
   - L2 regularization: `weight_decay=1e-6` in optimizer
   - Consider adding dropout=0.1 if we add MLP layers

### Medium Priority

4. **Improve BPR Negative Sampling** (if staying with BPR)
   - Sample negatives from items user hasn't interacted with
   - Use multiple negatives (4-10 per positive)
   - Consider hard negative mining after initial epochs

5. **Add MLP Layers**
   ```
   Embedding(128) → ReLU → Linear(128) → BatchNorm → Dropout
                  → ReLU → Linear(128) → L2 Normalize
   ```

6. **Learning Rate Warmup**
   - Linear warmup for first 1000-5000 steps
   - Then decay with cosine annealing

### Lower Priority

7. **Batch Size Experiments**
   - Try 2048 or 4096 for in-batch negatives (more negatives = better signal)
   - Requires more memory but may improve learning

8. **Embedding Dimension**
   - Try 64 or 256 to see if capacity is the issue
   - Monitor overfitting vs underfitting

---

## Specific Questions for ChatGPT

### Please search the internet and research:

1. **What are the standard hyperparameters for Two-Tower models in 2024-2025?**
   - Learning rates for Adam/AdamW
   - Typical embedding dimensions for datasets this size
   - Temperature values for contrastive learning
   - Batch sizes for in-batch negative sampling

2. **BPR vs In-Batch Softmax for implicit feedback?**
   - Which performs better for sparse datasets like MovieLens?
   - Are there recent papers comparing these approaches?
   - What do industry practitioners (Google, Spotify, Meta) recommend?

3. **Negative sampling best practices?**
   - How to avoid sampling false negatives in BPR?
   - Hard negative mining strategies
   - Popularity-biased negative sampling

4. **Why would popularity baseline beat a neural model?**
   - Is this a common problem with sparse implicit feedback?
   - What diagnostic tests can reveal the root cause?
   - Signs of data leakage vs model issues

5. **Regularization for embedding models?**
   - Typical weight decay values
   - Dropout rates if using MLP towers
   - Other regularization techniques (label smoothing, etc.)

### Specific Technical Advice Needed:

1. **Should I prioritize fixing negative sampling or switching to in-batch softmax?**
   - Which will have bigger impact?
   - Can I combine both approaches?

2. **What's a reasonable validation Recall@10 target for MovieLens?**
   - What do published papers achieve?
   - Is 0.02 (2%) reasonable or is something broken?

3. **Recommended training recipe**:
   - Step-by-step hyperparameter settings
   - Expected loss curves and convergence behavior
   - Validation metrics I should monitor

4. **Debugging checklist**:
   - How to verify no data leakage?
   - How to check if embeddings are learning meaningful patterns?
   - Diagnostic plots or tests to run?

---

## Code Availability

I can provide:
- Full Jupyter notebook with data loading, model definition, and training loop
- Dataset preprocessing code
- Current model weights
- Validation evaluation code

---

## Request to ChatGPT

**Please search the internet for recent best practices (2023-2025) on Two-Tower models, contrastive learning for recommendations, and implicit feedback learning.**

Based on the information above:

1. **Diagnose** what you think is the primary cause of poor performance
2. **Recommend** specific hyperparameter changes with justification
3. **Prioritize** the fixes (what to try first, second, third)
4. **Provide** a concrete training recipe I should try
5. **Share** any relevant papers, blog posts, or GitHub repos that address similar issues
6. **Suggest** diagnostic tests to verify the fixes are working

Please be specific with numbers (learning rates, batch sizes, etc.) and explain the reasoning behind your recommendations.

---

## Timeline & Context

- **Project Goal**: Learning-focused recommendation system (prioritize understanding over SOTA)
- **Background**: Strong software engineering, refreshing ML knowledge
- **Timeline**: Prototype phase, can iterate quickly
- **Resources**: Local MacBook (CPU only), can use Colab for GPU if needed
- **Next Steps**: Once candidate generation works, move to ranking stage

Thank you for your help!
