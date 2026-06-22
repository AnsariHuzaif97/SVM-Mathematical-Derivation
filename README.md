<!-- PROJECT BANNER -->
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00f2fe&height=250&section=header&text=Decoding%20SVM%20Mathematics&fontSize=60&animation=fadeIn&fontAlignY=38&desc=A%20Complete%20End-to-End%20Mathematical%20Derivation%20of%20Support%20Vector%20Machines&descAlignY=60&descAlign=50" alt="SVM Banner" />
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Mathematics-Linear_Algebra-013243?style=for-the-badge&logo=math&logoColor=white" alt="Math" />
  <img src="https://img.shields.io/badge/Optimization-Lagrange_Multipliers-D00000?style=for-the-badge" alt="Optimization" />
  <img src="https://img.shields.io/badge/Machine_Learning-Core_Algorithms-F7931E?style=for-the-badge" alt="ML" />
  <img src="https://img.shields.io/badge/Difficulty-Advanced-8957e5?style=for-the-badge" alt="Advanced" />
</div>

<br/>

## 📖 Abstract & Motivation

While most Data Scientists implement Support Vector Machines (SVMs) via a single line of code (`from sklearn.svm import SVC`), truly world-class Machine Learning Engineers understand the underlying mathematics. 

This repository contains my complete, end-to-end mathematical derivation of SVMs. It decodes the "magic" behind the algorithm, proving how we transition from a simple geometric hyperplane objective to a complex dual optimization problem using **Lagrange Multipliers** and **Slack Variables**.

---

## 🧮 Core Mathematical Highlights

### 1. The Hyperplane & The Margin
The objective of an SVM is to find a hyperplane that correctly classifies data points while maximizing the distance (margin) between the closest data points (Support Vectors) and the boundary.

The equation of our n-dimensional hyperplane is given by:

```math
\mathbf{w} \cdot \mathbf{x} + b = 0
```

Where $\mathbf{w}$ is the weight vector (perpendicular to the hyperplane) and $b$ is the bias. The total margin distance between the positive ($H_1$) and negative ($H_{-1}$) hyperplanes is $\frac{2}{\|\mathbf{w}\|}$. 

To maximize the margin, we must minimize $\|\mathbf{w}\|$. For mathematical convenience, we minimize:

```math
f(\mathbf{w}) = \frac{1}{2} \|\mathbf{w}\|^2
```

### 2. Handling Real-World Data (Soft-Margin)
Real-world datasets are messy and rarely linearly separable. To accommodate outliers, we introduce **Slack Variables ($\xi_i$)** to penalize misclassified points.

Our new objective function becomes:

```math
\min_{\mathbf{w}, b, \xi} \left( \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \xi_i \right)
```

Subject to the constraints:

```math
y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i
```
```math
\xi_i \geq 0
```

### 3. The Lagrangian Formulation
To solve this constrained inequality optimization problem, we utilize the **Lagrange Multiplier Method**. The primal Lagrangian $\mathcal{L}$ is formulated as:

```math
\mathcal{L}(\mathbf{w}, b, \xi, \alpha, \mu) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \xi_i - \sum_{i=1}^{m} \alpha_i \left[ y_i (\mathbf{w} \cdot \mathbf{x}_i + b) - 1 + \xi_i \right] - \sum_{i=1}^{m} \mu_i \xi_i
```

### 4. Dual Optimization 
By differentiating $\mathcal{L}$ with respect to $\mathbf{w}$, $b$, and $\xi$, and equating to zero, we derive the critical constraints:

```math
\mathbf{w} = \sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i
```
```math
\sum_{i=1}^{m} \alpha_i y_i = 0
```

By substituting these back into the Lagrangian, the optimizer solves the **Dual Problem** to find the optimal multipliers $\alpha_i$. Once found, the optimal hyperplane is perfectly constructed.

---

## 🚀 Why This Matters

Understanding the deep mathematics behind SVMs allows me to:
1. Make highly informed decisions when selecting hyperparameter $C$ (the penalty term).
2. Understand exactly how different Kernel functions map data to higher dimensions.
3. Debug production-level classification models when convergence fails.

---
<div align="center">
  <b>Authored by Md Huzaifa Ansari</b><br>
  <i>Top 1.98% Kaggle Expert | Persuing Bachelors in CS @ IIT Patna</i>
</div>
