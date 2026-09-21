# Deep Neural Networks (AMLSIZG511) — Master Study Index & Formula Sheet

**Program:** M.Tech, BITS Pilani WILP — Semester 1
**Scope of these notes:** Modules 1–5 of the course (Fundamentals → CNNs), built directly from the course's own slide decks, numerical worked-example sheets, and Jupyter notebooks (see "Source Attribution" below). Modules 6–10 (Sequence Models/RNN, Attention/Transformers, NAS, Time-Series, Other Learning Techniques) are **not yet covered** — no source PDFs for these were found in the workspace at the time of writing; add them here once the corresponding slide decks/materials are available.

---

## 1. Table of Contents

| # | File | Topic | Status |
|---|---|---|---|
| 01 | `01_Fundamentals_of_Neural_Networks.md` | AI/ML/DL landscape, the Perceptron, XOR problem, MLPs, Universal Approximation, depth vs. width | ✅ Complete |
| 02 | `02_Deep_Feedforward_Networks.md` | Computation graphs, forward/backward propagation, activation functions, softmax+cross-entropy, full hand-derived numerical example, vanishing gradients, network sizing | ✅ Complete |
| 03 | `03_Optimization_of_Deep_Models.md` | Saddle points/plateaus, GD variants, Momentum/Nesterov, AdaGrad/RMSprop/Adam/AdamW (all with numeric traces), LR schedules, gradient clipping, batch-size effects | ✅ Complete |
| 04 | `04_Regularization_for_Deep_Models.md` | Bias-variance, L1/L2/Elastic-Net, Dropout, Xavier/He init, BatchNorm/LayerNorm, Early Stopping, Data Augmentation, 15 fully worked numericals | ✅ Complete |
| 05 | `05_Convolutional_Networks.md` | Convolution math, stride/padding/output-size formula, parameter counting (AlexNet), pooling, receptive fields, cross-correlation, transposed/dilated conv | ✅ Complete |

---

## 2. Prerequisites Map

```
Linear Algebra + Calculus (chain rule) + Probability
        │
        ▼
01 Fundamentals (Perceptron → MLP → Universal Approximation)
        │
        ▼
02 Deep Feedforward Networks (forward/backward prop, activations, loss functions)
        │
        ├──────────────┬───────────────┐
        ▼              ▼               ▼
03 Optimization   04 Regularization  05 CNNs
(how weights      (how to prevent    (specialized architecture built
 are updated)      overfitting)       on top of 02's backprop machinery)
```
Each later module *assumes* the backprop/computation-graph mechanics of Module 2, and CNNs (05) reuse regularization techniques (BatchNorm, Dropout/DropBlock, data augmentation) from Module 4.

---

## 3. Consolidated Formula Sheet

### 3.1 Fundamentals / Feedforward (Modules 1–2)
| Concept | Formula |
|---|---|
| Perceptron output | $y = \text{step}(w^\top x + b)$ |
| Neuron pre-activation | $z^{(l)} = W^{(l)}a^{(l-1)}+b^{(l)}$ |
| Sigmoid | $\sigma(z)=\dfrac{1}{1+e^{-z}}$, $\sigma'(z)=\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z)=\dfrac{e^z-e^{-z}}{e^z+e^{-z}}$, derivative $1-\tanh^2(z)$ |
| ReLU | $\text{ReLU}(z)=\max(0,z)$; derivative $1$ if $z>0$ else $0$ |
| Softmax | $\text{softmax}(z)_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ |
| Binary Cross-Entropy | $L=-\big[y\log\hat y+(1-y)\log(1-\hat y)\big]$ |
| Categorical Cross-Entropy | $L=-\sum_i y_i\log\hat y_i$ |
| Backprop (chain rule, general layer) | $\dfrac{\partial L}{\partial W^{(l)}} = \dfrac{\partial L}{\partial z^{(l)}}\cdot a^{(l-1)\top}$ |

### 3.2 Optimization (Module 3)
| Optimizer | Update rule |
|---|---|
| Gradient Descent | $w \leftarrow w-\eta\nabla L(w)$ |
| Momentum | $v\leftarrow \beta v-\eta\nabla L(w)$; $w\leftarrow w+v$ |
| Nesterov | $v\leftarrow\beta v-\eta\nabla L(w+\beta v)$; $w\leftarrow w+v$ |
| AdaGrad | $G\leftarrow G+g^2$; $w\leftarrow w-\dfrac{\eta}{\sqrt{G+\epsilon}}g$ |
| RMSprop | $E[g^2]\leftarrow \rho E[g^2]+(1-\rho)g^2$; $w\leftarrow w-\dfrac{\eta}{\sqrt{E[g^2]+\epsilon}}g$ |
| Adam | $m\leftarrow\beta_1m+(1-\beta_1)g$; $v\leftarrow\beta_2v+(1-\beta_2)g^2$; bias-correct $\hat m,\hat v$; $w\leftarrow w-\dfrac{\eta}{\sqrt{\hat v}+\epsilon}\hat m$ |
| AdamW | Same as Adam, but weight decay applied **directly** to weights, decoupled from the gradient-based update: $w\leftarrow w-\eta\Big(\dfrac{\hat m}{\sqrt{\hat v}+\epsilon}+\lambda w\Big)$ |
| LR schedules | Step decay, exponential decay $\eta_t=\eta_0e^{-kt}$, cosine annealing |
| Gradient clipping | $g \leftarrow g\cdot\min\!\Big(1,\dfrac{\text{threshold}}{\|g\|}\Big)$ |

### 3.3 Regularization (Module 4)
| Concept | Formula |
|---|---|
| L2 (Ridge) penalty | $\lambda\sum_i w_i^2$ ; update: $w\leftarrow(1-2\eta\lambda)w-\eta\nabla L$ |
| L1 (Lasso) penalty | $\lambda\sum_i\lvert w_i\rvert$ ; update: $w\leftarrow w-\eta(\nabla L+\lambda\,\text{sign}(w))$ |
| Elastic Net | $\lambda[\alpha\sum\lvert w_i\rvert+(1-\alpha)\sum w_i^2]$ |
| Xavier/Glorot init | $W\sim\mathcal N\big(0,\ 2/(n_{in}+n_{out})\big)$ |
| He init | $W\sim\mathcal N(0,\ 2/n_{in})$ |
| Dropout (inverted) | keep w.p. $1-p$, scale kept units by $1/(1-p)$ during training |
| Batch Norm | $\hat x=\dfrac{x-\mu_{\mathcal B}}{\sqrt{\sigma_{\mathcal B}^2+\epsilon}}$; $y=\gamma\hat x+\beta$ (normalize across the **batch**) |
| Layer Norm | Same formula, but normalize across **features** of one sample |
| Vanishing/exploding gradient condition | $\left\|\dfrac{\partial h^{(l)}}{\partial h^{(l-1)}}\right\| \lessgtr 1 \Rightarrow$ vanish/explode across depth $L$ |

### 3.4 CNNs (Module 5)
| Concept | Formula |
|---|---|
| Output size (conv or pool) | $O=\left\lfloor\dfrac{n+2p-f}{s}\right\rfloor+1$ |
| Same padding | $p=\dfrac{f-1}{2}$ |
| Conv layer parameters | $P_c = \underbrace{K\times K\times C\times N}_{\text{weights}} + \underbrace{N}_{\text{biases}}$ |
| Pooling layer parameters | $0$ (hyperparameters only) |
| Receptive field (layer $k$) | $R_k = 1+\sum_{j=1}^{k}\big[(F_j-1)\textstyle\prod_{i<j}S_i\big]$ |
| Dilated conv effective kernel size | $K_{\text{eff}}=K+(K-1)(d-1)$ |

---

## 4. Concept Map (cross-module connections)

```
                         ┌─────────────────────────┐
                         │  01 Perceptron → MLP     │
                         │  (Universal Approx.)     │
                         └───────────┬─────────────┘
                                     │ stack layers, need training
                                     ▼
                         ┌─────────────────────────┐
                         │ 02 Forward/Backward Prop │
                         │ activations, loss fns    │
                         └───────────┬─────────────┘
                     ┌───────────────┼───────────────────┐
                     ▼               ▼                   ▼
        ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │ 03 Optimization  │ │ 04 Regularization │ │ 05 CNNs          │
        │ SGD→Momentum→    │ │ L1/L2, Dropout,   │ │ conv+pool as a   │
        │ Adam/AdamW,      │ │ BatchNorm/LayerNorm,│ │ structured, weight-│
        │ LR schedules,    │ │ Xavier/He init,   │ │ shared special case│
        │ grad clipping    │ │ Early Stopping    │ │ of 02's FF network │
        └────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
                  │  weight decay ≠ L2 for Adam →  AdamW      │
                  └──────────────────┬─────────────────────────┘
                                     ▼
                     BatchNorm placement (Conv→BN→ReLU),
                     DropBlock (CNN-specific dropout),
                     data augmentation (Mixup/CutMix) — all reused directly in 05
```

**Key cross-cutting themes:**
1. **The chain rule** (Module 2) underlies backprop everywhere — optimizers (03) just change *how* the gradient is used; CNNs (05) just change *how* the gradient is computed (via shared/sparse weights).
2. **Vanishing/exploding gradients** appear in Module 2 (deep FF nets) and are formally diagnosed & treated in Module 4 (init schemes, BatchNorm, gradient clipping from Module 3).
3. **Parameter efficiency** is a recurring design principle — deep-narrow vs. shallow-wide (Module 1), weight sharing in CNNs (Module 5), and weight decay as *implicit* capacity control (Module 4) all reduce effective model complexity relative to raw parameter count.

---

## 5. Master Practice-Problem Index
- **Module 1:** perceptron weight update by hand; XOR-solvability proof; universal approximation intuition — see file 01, §Practice Problems.
- **Module 2:** full 2-layer forward+backward pass numeric trace (ReLU+Sigmoid+BCE); vanishing-gradient magnitude calculation; parameter-budget sizing — see file 02, §Practice Problems.
- **Module 3:** GD/Momentum/RMSprop/Adam traced on $L(w_1,w_2)=w_1^2+4w_2^2$ starting at $(4,2)$; LR schedule computation; gradient-clipping numeric example — see file 03, §Practice Problems.
- **Module 4:** **15 fully worked numericals** (L1/L2/Ridge/Lasso/Elastic-Net losses, weight updates, regularized cost functions) — see file 04, §10; plus 5 additional self-check problems in §14.
- **Module 5:** convolution/pooling output-size and parameter-count problems (incl. full AlexNet layer-by-layer worked example); receptive-field computation; dilated-conv equivalence — see file 05, §11 and §15.

---

## 6. Source Attribution
All notes are grounded in this course's own materials (extracted to `extracted\*.txt` via PyMuPDF), not generic textbook paraphrasing:
- `AMLSIZG511_Deep Neural Networks (1).pdf` — official course handout/curriculum (10-module structure, textbook refs T1=Zhang et al. *Dive into Deep Learning*, R1=Goodfellow et al. *Deep Learning*).
- `CS-1 and 2-DNN.pdf`, `CS03_DNN.pdf` — Fundamentals lecture decks (Module 1).
- `DNN_M5_DFNN_Numerical.pdf` — richest source for Module 2; complete worked forward/backward-prop numericals.
- `DNN_CS04_Optimizers.pdf` — richest source for Module 3; full numeric optimizer traces on a shared toy problem.
- `DNN_CS05_Regularization.pdf` — Module 4 source deck (60 slides).
- `Regularization practice numbericals.pdf` + `regularization_15_solutions_line_by_line.pdf` — the 15 fully-solved regularization numericals reproduced in file 04.
- `S7-CNNs.pdf` — Module 5 source deck (Dr. S. Suresh Kumar), incl. AlexNet worked examples and practice questions with answers.
- Jupyter notebooks in `Class ppt and Other material\Webinar-1\` — AND/OR/XOR gate implementations and house-price backprop, used as real, attributed code in file 01.

---

## 7. How to Use These Notes
1. Read modules **in order** (01→05) — each depends on concepts from the previous ones (see Concept Map above).
2. Each file is self-contained with: Core Concept → Math Foundations (derivations) → Diagrams (ASCII) → Algorithms/Code → Worked Numericals → Connections → Key Takeaways → Common Mistakes → Practice Problems.
3. Use **this index** as a quick-reference formula sheet before exams; use the individual files for derivations and worked examples when a formula needs to be *re-derived* or understood conceptually.
4. **Gap to fill later:** Modules 6–10 (RNN/Sequence Models, Attention/Transformers, Neural Architecture Search, Time-Series applications, Other Learning Techniques) require additional source slide decks not present in the workspace at the time of writing — extend this index and add `06_...md` through `10_...md` once those materials are available.
