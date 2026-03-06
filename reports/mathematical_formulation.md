# Mathematical Formulation of the Split-Stream Hybrid Language Model

## The Master Equation

The entire model in one line:

$$\hat{P}(x_{t+1} \mid x_{\leq t}) \;=\; \mathrm{softmax}\!\Bigg(\mathbf{E}^{\!\top} \;\mathrm{LN}\!\bigg(\,\phi_{\mathrm{KAN}}\!\Big(\,\mathcal{L}_{w}^{\,L_L}\!(\mathbf{h}^{(0)}) \;\Big\|\; \mathcal{S}^{\,L_S}\!(\mathbf{h}^{(0)})\Big) \;+\; \mathbf{h}^{(0)}\bigg)_{\!t}\Bigg)$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{E}$ | Shared token embedding (weight-tied with output head) |
| $\mathbf{h}^{(0)}$ | Input representation (embedding + position) |
| $\mathcal{L}_{w}^{L_L}$ | $L_L$-layer **local** stream (sliding-window attention, window $w$) |
| $\mathcal{S}^{L_S}$ | $L_S$-layer **global** stream (selective state-space model) |
| $\phi_{\mathrm{KAN}}$ | KAN fusion (nonlinear, radial-basis learned merge) |
| $\|$ | Concatenation |
| $\mathrm{LN}$ | Layer normalization |
| $+ \;\mathbf{h}^{(0)}$ | Residual skip from input |

---

## 1. Input Representation

$$\mathbf{h}^{(0)}_t = \mathrm{Dropout}\!\Big(\mathbf{E}[x_t] + \mathbf{P}[t]\Big)$$

- $\mathbf{E} \in \mathbb{R}^{V \times d}$: token embedding ($V = 257$, $d = 256$)
- $\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}$: learned positional embedding

---

## 2. Local Stream — Sliding-Window Attention

Each of $L_L = 3$ layers applies windowed self-attention plus a feed-forward network:

$$\mathbf{h}_{L}^{(\ell)} = \mathbf{h}_{L}^{(\ell-1)} + \mathrm{SWA}_{w}\!\Big(\mathrm{LN}\!\big(\mathbf{h}_{L}^{(\ell-1)}\big)\Big)$$

$$\mathbf{h}_{L}^{(\ell)} \leftarrow \mathbf{h}_{L}^{(\ell)} + \mathrm{FFN}\!\Big(\mathrm{LN}\!\big(\mathbf{h}_{L}^{(\ell)}\big)\Big)$$

**Sliding-window attention** restricts the receptive field to the $w$ most recent tokens:

$$\mathrm{SWA}_{w}(\mathbf{z})_t = \sum_{j \,\in\, \mathcal{W}(t)} \alpha_{t,j}\;\mathbf{V}_j$$

$$\alpha_{t,j} = \frac{\exp\!\big(\mathbf{Q}_t^{\top} \mathbf{K}_j \,/\, \sqrt{d_h}\big)}{\displaystyle\sum_{k \,\in\, \mathcal{W}(t)} \exp\!\big(\mathbf{Q}_t^{\top} \mathbf{K}_k \,/\, \sqrt{d_h}\big)}$$

$$\mathcal{W}(t) = \big\{\,j : \max(0,\; t - w + 1) \leq j \leq t\,\big\}$$

**Why:** Captures precise local syntax and short-range patterns with $O(Tw)$ cost instead of $O(T^2)$.

---

## 3. Global Stream — Selective State-Space Model

Each of $L_S = 3$ SSM layers applies a gated recurrence that selectively remembers or forgets:

$$\mathbf{h}_{S}^{(\ell)} = \mathbf{h}_{S}^{(\ell-1)} + \mathrm{SSM}^{(\ell)}\!\Big(\mathbf{h}_{S}^{(\ell-1)}\Big)$$

The SSM core at each time step $t$:

$$\boldsymbol{\alpha}_t = \sigma\!\big(\mathbf{W}_{\delta}\,\mathbf{u}_t\big)$$

$$\mathbf{s}_t = \big(1 - \boldsymbol{\alpha}_t\big) \odot \mathbf{s}_{t-1} \;+\; \boldsymbol{\alpha}_t \odot \mathbf{b}_t$$

$$\mathbf{r}_t = \mathbf{s}_t \odot \tanh\!\big(\mathbf{c}_t\big)$$

where $[\boldsymbol{\delta}_t,\;\mathbf{b}_t,\;\mathbf{c}_t] = \mathbf{W}_{\mathrm{in}}\;\mathrm{LN}(\mathbf{h})_t$ projects into gate, input, and output components, and $\mathbf{s}_t \in \mathbb{R}^{N}$ ($N = 64$) is the hidden state.

Output with skip connection:

$$\mathrm{SSM}(\mathbf{h})_t = \mathbf{W}_{\mathrm{out}}\,\mathbf{r}_t + \mathbf{W}_{\mathrm{skip}}\,\mathbf{h}_t$$

**Why:** The gated recurrence compresses long-range context into a fixed-size state $\mathbf{s}_t$, giving the model a "global memory" the local window cannot provide.

---

## 4. KAN Fusion — Nonlinear Stream Merge

The two streams are concatenated and merged via a Kolmogorov-Arnold Network using radial basis functions:

$$\mathbf{z}_t = \Big[\,\mathbf{h}_{L}^{(L_L)}\;\Big\|\;\mathbf{h}_{S}^{(L_S)}\Big]_t \;\in\; \mathbb{R}^{2d}$$

$$\phi_k(z) = \exp\!\left(-\frac{(z - \mu_k)^2}{2\,\sigma^2}\right), \qquad k = 1, \ldots, K$$

$$\phi_{\mathrm{KAN}}(\mathbf{z})_j = \sum_{i=1}^{2d}\;\sum_{k=1}^{K} w_{j,i,k}\;\phi_k(z_i) \;+\; b_j$$

- $K = 8$ basis functions with learned width $\sigma$ and fixed centers $\mu_k$
- $\mathbf{W} \in \mathbb{R}^{d \times 2d \times K}$: learnable coefficients

**Why:** KAN can learn nonlinear, input-dependent blending of the two streams — strictly more expressive than linear concatenation or summation.

---

## 5. Output and Training Objective

$$\mathbf{y}_t = \mathrm{LN}\!\Big(\phi_{\mathrm{KAN}}(\mathbf{z}_t) + \mathbf{h}^{(0)}_t\Big)$$

$$\hat{P}(x_{t+1} \mid x_{\leq t}) = \mathrm{softmax}\!\big(\mathbf{E}^{\top}\,\mathbf{y}_t\big)$$

Trained by minimizing cross-entropy (equivalent to minimizing perplexity):

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log \hat{P}\!\big(x_t \mid x_{<t}\big)$$

$$\mathrm{PPL} = \exp(\mathcal{L})$$

---

## 6. Mathematical Defense — Why Every Component Matters

### 6a. Ablation Inequality (experimentally verified)

$$\mathrm{PPL}_{\text{full KAN}} \;<\; \mathrm{PPL}_{\text{sum fusion}} \;<\; \mathrm{PPL}_{\text{SSM only}} \;<\; \mathrm{PPL}_{\text{local only}}$$

$$7.10 \;<\; 9.98 \;<\; 14.73 \;<\; 16.15 \qquad (\text{ctx}=1024)$$

Each removal **strictly worsens** performance, proving every component contributes:

| What you remove | PPL goes from → to | What it proves |
|-----------------|---------------------|----------------|
| KAN fusion → sum fusion | 7.10 → 9.98 (+41%) | Nonlinear fusion matters |
| Local stream → SSM only | 7.10 → 14.73 (+107%) | Local attention matters |
| SSM stream → local only | 7.10 → 16.15 (+127%) | Global memory matters |

### 6b. Quality Gain Under Fair Comparison

Parameter fairness constraint:

$$\frac{\big|\,|\Theta_{\text{hybrid}}| - |\Theta_{\text{baseline}}|\,\big|}{|\Theta_{\text{baseline}}|} < 0.5\%$$

Quality improvement metric:

$$\mathcal{Q} = 1 - \frac{\mathrm{PPL}_{\text{hybrid}}}{\mathrm{PPL}_{\text{baseline}}}$$

| Context | Baseline PPL | Hybrid PPL | $\mathcal{Q}$ |
|---------|-------------|-----------|----------------|
| 256 | 14.10 ± 0.33 | 8.50 ± 0.27 | **39.7%** |
| 512 | 13.04 ± 0.30 | 7.63 ± 0.15 | **41.5%** |
| 1024 | 13.11 ± 0.05 | 7.10 ± 0.04 | **45.9%** |

Replicated across 3 seeds with 95% CIs.

### 6c. Why the Architecture Works — Intuition in One Equation

A standard transformer must pack **both** local syntax and long-range context into a single residual stream:

$$\mathbf{h}_{\text{baseline}} = f_{\text{attn}}^{(1:N)}(\mathbf{h}^{(0)})$$

The split-stream decomposes this into two **specialized** subproblems:

$$\mathbf{h}_{\text{hybrid}} = \phi_{\mathrm{KAN}}\!\Big(\underbrace{\mathcal{L}_{w}^{L_L}(\mathbf{h}^{(0)})}_{\text{local patterns}} \;\Big\|\; \underbrace{\mathcal{S}^{L_S}(\mathbf{h}^{(0)})}_{\text{global memory}}\Big)$$

Each stream can **specialize** on its own task without interference, and the KAN learns **how much** of each to use at every position. This is why the hybrid achieves lower perplexity with the same parameter budget.

---

## 7. Integral Derivation of the SSM Recurrence

The gated recurrence in our code comes from solving a differential equation.

### Start: the ODE

The SSM hidden state follows:

$$\frac{d\mathbf{s}}{dt} = -\lambda(t)\,\mathbf{s}(t) + \lambda(t)\,\mathbf{b}(t)$$

where $\lambda(t)$ is a learned decay rate and $\mathbf{b}(t)$ is the input.

### Solve: integrate both sides

$$\boxed{\mathbf{s}(t) = e^{-\int_0^t \lambda\,d\tau}\;\mathbf{s}(0) \;+\; \int_0^t e^{-\int_\tau^t \lambda\,du}\;\lambda(\tau)\,\mathbf{b}(\tau)\;d\tau}$$

The state at time $t$ = decayed initial state + weighted integral of all past inputs.

### Discretize: one step at a time

Set $\alpha_t = \sigma(\delta_t) \approx 1 - e^{-\lambda_t}$, and the integral over one step becomes:

$$\mathbf{s}_t = (1 - \alpha_t)\,\mathbf{s}_{t-1} + \alpha_t\,\mathbf{b}_t$$

This is exactly line 36 of `ssm_block.py`. The sigmoid gate is the exponential decay of the continuous ODE.
