# Naive Bayes: Interview Revision Notes

## 1. Conditional Probability

Probability of A **given** B has already happened:

```
P(A | B) = P(A ∩ B) / P(B)
```

Two quick cases to remember:

- **Independent events:** `P(A ∩ B) = P(A) × P(B)` (knowing B tells you nothing about A)
- **Mutually exclusive events:** `P(A ∩ B) = 0` (they can't happen together)

**Chain rule of conditional probability** (always true, no assumptions):

```
P(x1, x2, ..., xn) = P(x1 | x2..xn) · P(x2 | x3..xn) · ... · P(xn)
```

---

## 2. Bayes Theorem

Flip the direction of a conditional probability:

```
              P(B | A) · P(A)
P(A | B)  =  -----------------
                   P(B)
```

| Term | Name | Meaning |
|------|------|---------|
| `P(A \| B)` | **Posterior** | What we want: belief in A after seeing B |
| `P(B \| A)` | **Likelihood** | How likely is this evidence if A is true |
| `P(A)` | **Prior** | Belief in A before seeing any evidence |
| `P(B)` | **Evidence** | How likely the evidence is overall |

**For classification:** A = class `C_k`, B = features `X = (x1, x2, ..., xn)`

```
              P(X | C_k) · P(C_k)
P(C_k | X) = ---------------------
                    P(X)
```

---

## 3. The Problem: Why We Can't Use Bayes Directly

Using the chain rule on the likelihood:

```
P(X | C_k) = P(x1 | x2, x3, ..., xn, C_k) · P(x2 | x3, ..., xn, C_k) · ... · P(xn | C_k)
```

Each term is conditioned on **many other features**. To estimate this we would need
data for every combination of feature values, which explodes exponentially
(curse of dimensionality). Not practical.

---

## 4. Why It's Called "Naive"

We make one **big simplifying assumption: conditional independence**.

> Given the class `C_k`, all features are independent of each other.
> `x1, x2, x3 ...` don't depend on one another, they only depend on the class `C_k`.

So every term simplifies:

```
P(x1 | x2, x3, ..., xn, C_k)  →  P(x1 | C_k)
```

This is "naive" because real features are usually correlated
(e.g. in spam detection, "free" and "offer" often appear together).
But the assumption is wrong in theory and still works well in practice.

With the assumption:

```
P(X | C_k) = P(x1 | C_k) · P(x2 | C_k) · ... · P(xn | C_k)
```

---

## 5. Final Formula

```
                    1                 n
P(C_k | X)  =  ------- · P(C_k) · ∏  P(x_i | C_k)
                    Z                i=1
```

where `Z = P(X)` is the evidence (a normalizing constant).

In words: **posterior ∝ prior × product of per-feature likelihoods**

**Training = building a lookup table.** Just count and store `P(C_k)` and
`P(x_i | C_k)` from the data. No iterative optimization, which is why it is very fast.

---

## 6. Why `argmax`?

We want the **most likely class** for a given input:

```
ŷ = argmax    P(C_k) · ∏ P(x_i | C_k)
    k ∈ {1..K}           i=1..n
```

Reasons:

1. **Goal is classification, not exact probability.** We only need to know which class wins.
2. **`Z = P(X)` is the same for every class.** It doesn't depend on `C_k`, so it can't
   change which class has the highest score. We drop it and skip computing it.
3. This is called the **MAP (Maximum A Posteriori) rule**: pick the class with the
   maximum posterior probability.

---

## 7. Handling Numerical (Continuous) Data

For categorical features we can just count to get `P(x_i | C_k)`.
For numerical features, counting doesn't work (too many unique values), so we
**assume the feature follows a known distribution within each class**, estimate its
parameters from training data, and plug the value into that distribution's formula.

**Default for continuous data: assume Normal (Gaussian) distribution.**

For each feature `i` and class `C_k`, compute from training data:
- `μ_ik` = mean of feature `i` among samples of class `C_k`
- `σ_ik` = standard deviation of feature `i` among samples of class `C_k`

Then:

```
                       1                 (x_i - μ_ik)²
P(x_i | C_k)  =  ---------------- · exp( - ----------------- )
                  √(2π) · σ_ik              2 · σ_ik²
```

This is **Gaussian Naive Bayes**. (Strictly this is a density, not a probability,
but it is fine for comparing classes.)

**The distribution depends on the dataset.** Pick the one that matches the feature type:

| Data type | Distribution | Variant | Typical use |
|-----------|--------------|---------|-------------|
| Continuous numbers (height, temperature) | Normal | Gaussian NB | Sensor / tabular data |
| Counts (how many times a word appears) | Multinomial | Multinomial NB | Text classification |
| Binary (word present or not, yes/no) | Bernoulli (binomial with 1 trial) | Bernoulli NB | Short text, spam flags |

The rest of the algorithm stays the same: only the way we compute `P(x_i | C_k)` changes.

---

## 8. Interview Quick Hits

- **Why naive?** Assumes features are conditionally independent given the class.
- **Why does it still work?** We need the correct class *ranking*, not exact probabilities.
- **Zero-probability problem:** if a feature value never appears with a class, the whole
  product becomes 0. Fix: **Laplace (add-one) smoothing**.
- **Numerical underflow:** multiplying many small probabilities → near 0. Fix: use
  **log probabilities** and add instead of multiply.
- **Variants:** Gaussian NB (continuous features), Multinomial NB (word counts, text),
  Bernoulli NB (binary features).
- **Strengths:** fast, works with small data, good baseline for text classification and spam filtering.
- **Weakness:** the independence assumption; probability outputs are poorly calibrated.
