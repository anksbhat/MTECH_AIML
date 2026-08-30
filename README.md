# Machine Learning (AIML ZG565) — Study Repository
**BITS Pilani WILP · M.Tech AIML · Semester 1**

Beginner-friendly, deep-dive study guides for the ML course — with plain-English explanations, **mental maps & diagrams**, detailed math, fully worked examples, and runnable Python for every topic.

---

## 📁 Repository structure

```
Machine Learning/
├── README.md                         ← you are here (index)
├── main.py                           ← scratch file
│
├── 01_Course_Slides_and_Handout/     ← original course PDFs
│   ├── AMLSIZG565_ Machine Learning COURSE HANDOUT (1).pdf
│   ├── AIML ZG565(ML)_Module 1.pptx.pdf
│   ├── AIML ZG565(ML)_Module 2.pdf
│   └── AIML ZG565(ML)_Module 3.pdf
│
├── 02_Reference_Textbooks/           ← prescribed books
│   ├── MachineLearningTomMitchell.pdf                 (T1)
│   └── Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf  (R1)
│
└── 03_Study_Guides/                  ← the guides (one folder per module)
    ├── Module1_Introduction/
    ├── Module2_Workflow_and_Mathematics/
    ├── Module3_Linear_Regression/
    └── Module4_Linear_Classification/
```

Each module folder contains:
- `*_Guide.md` — the study guide (open in any Markdown viewer / PyCharm preview)
- `Module<N>_examples.py` — runnable, verified worked examples
- `images/` — all diagrams & **mental maps** used by the guide

---

## 📚 Study guides (click to open)

| # | Module | Guide | Covers |
|---|--------|-------|--------|
| 1 | Introduction | [Module1_Introduction_Guide.md](03_Study_Guides/Module1_Introduction/Module1_Introduction_Guide.md) | What/Why ML, ⟨T,P,E⟩, designing a learner, types of ML, workflow |
| 2a | ML Workflow | [Module2_MachineLearningWorkflow_Guide.md](03_Study_Guides/Module2_Workflow_and_Mathematics/Module2_MachineLearningWorkflow_Guide.md) | Data & attribute types, data quality, preprocessing, sampling, feature engineering, metrics |
| 2b | Math Preliminaries | [Module2_MathematicalPreliminaries_Guide.md](03_Study_Guides/Module2_Workflow_and_Mathematics/Module2_MathematicalPreliminaries_Guide.md) | Linear algebra, calculus, probability, decision theory, information theory |
| 3 | Linear Regression | [Module3_LinearModelsForRegression_Guide.md](03_Study_Guides/Module3_Linear_Regression/Module3_LinearModelsForRegression_Guide.md) | Least squares, normal equation, gradient descent, R², polynomial/basis, bias-variance, regularization |
| 4 | Linear Classification | [Module4_LinearModelsForClassification_Guide.md](03_Study_Guides/Module4_Linear_Classification/Module4_LinearModelsForClassification_Guide.md) | Discriminant functions, decision theory, logistic regression, log-loss, softmax, ROC/AUC |

> Each guide opens with a **🗺️ bird's-eye mental map** summarising the whole module.

---

## ▶️ Running the code examples

The examples use `numpy`, `pandas`, `scikit-learn` (and `matplotlib` for regenerating diagrams).

```powershell
# one-time install
pip install numpy pandas scikit-learn matplotlib

# run a module's worked examples (from its folder)
cd "03_Study_Guides\Module3_Linear_Regression"
python Module3_examples.py
```

Every printed number in the examples matches the worked examples inside the corresponding guide.

---

## 🗺️ Course roadmap (from the handout)

| Module | Topic | Status |
|--------|-------|--------|
| M1 | Introduction | ✅ Guide ready |
| M2 | ML Workflow (+ Math Preliminaries) | ✅ Guide ready |
| M3 | Linear Models for Regression | ✅ Guide ready |
| M4 | Linear Models for Classification | ✅ Guide ready |
| M5 | Decision Trees | ⬜ next |
| M6 | Instance-Based Learning | ⬜ |
| M7 | Support Vector Machines | ⬜ |
| M8 | Bayesian Learning | ⬜ |
| M9 | Ensemble Learning | ⬜ |
| M10 | Unsupervised Learning | ⬜ |
| M11 | Model Evaluation / Comparison | ⬜ |

**Textbooks:** T1 = Mitchell (1997); R1 = Bishop (2006); R2 = Tan, Steinbach & Kumar.
