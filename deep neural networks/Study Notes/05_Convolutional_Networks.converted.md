# Convolutional Neural Networks (CNNs)

**Course:** AMLSIZG511 – Deep Neural Networks (BITS Pilani WILP)
**Sources:** Session 7 slide deck `S7-CNNs.pdf` (Dr. S. Suresh Kumar), Course Handout (T1 Ch.9; R1 Ch.9 Goodfellow et al.)

---

## Table of Contents (this module)
1. Motivation — Computer Vision Challenges & Why CNNs
2. The Convolution Operation (1D → 2D → volumes)
3. Stride, Padding & Output-Size Formula
4. Parameter Counting (with full AlexNet worked example)
5. Pooling
6. Receptive Fields, Sparse Connectivity & Parameter Sharing
7. Cross-Correlation vs. True Convolution
8. Transposed & Dilated Convolutions
9. The Full CNN Pipeline & CNN as a Special MLP
10. Code Implementation (from-scratch NumPy conv/pool)
11. Worked Numerical Practice Problems (from the slides, with full solutions)
12. Connections, Key Takeaways, Common Mistakes, Extra Practice

---

## 1. Motivation — Computer Vision Challenges & Why CNNs

### 1.1 Representing images
- **Grayscale image:** a matrix of pixel intensities `I(i,j)∈[0,255]`.
- **Color image:** each pixel is a 3-vector (R, G, B), each channel `∈[0,255]` → `256³ = 16,777,216` representable colors. `I(i,j) = [R(i,j), G(i,j), B(i,j)]`.

### 1.2 Why a plain fully-connected (FC) network struggles with images
- **Scale:** a `50×50` image needs 2,500 input dimensions; an HD `1920×1080` image needs ~2 million inputs — an FC layer connecting to even a modest hidden layer would need billions of parameters.
- **Classic challenges:** viewpoint variation, illumination, deformation, occlusion, background clutter, inter-class vs. intra-class variation.
- **Invariance needs:**
  - **Translation invariance** — an object should be detected wherever it appears in the image (achieved mainly via **max pooling**).
  - **Equivariance** — `f(gx) = g(f(x))`: shifting the input should shift the output correspondingly (the **convolution operation itself is translation-equivariant**).

### 1.3 Why CNNs work for images (3 core properties, from the slides)
1. **Some patterns are much smaller than the whole image** — a neuron doesn't need to see the whole image to detect a small pattern (e.g., a "beak" detector only needs a small receptive field).
2. **The same pattern can appear in different regions** — detectors ("upper-left beak", "middle beak") can share the *same* set of parameters (**parameter sharing**).
3. **Subsampling pixels does not change the object** — we can shrink the image (via pooling) without losing the essential content, reducing computation.

CNNs are simply **neural networks that use the convolution operation** in place of general matrix multiplication in at least one layer. Origin: inspired by studies of the visual cortex → Neocognitron (1980s) → LeNet-5 (LeCun, 1998), which introduced convolution + pooling layers together.

---

## 2. The Convolution Operation

### 2.1 Intuition: 1-D convolution (noisy sensor smoothing)
Suppose we track an object's position with a noisy sensor at discrete times. We want a weighted average that gives more weight to recent measurements:

> **`s(t) = ∫ x(a) w(t-a) da (continuous) s(t)=Σ(a) x(a) w(t-a) (discrete)`**

where `x(t)` = input (position), `w(a)` = kernel/filter (weighting by age of measurement `a`), `s(t)` = smoothed output (feature map). We only sum over a small window (the filter), and **slide** it over the input.

Example (from slides): for kernel weights `w₀,…,w₆`,

> **`s_i = x_0w₆+x_1w₅+x_2w₄+x_3w₃+x_4w₂+x_5w₁+x_6w₀`**

### 2.2 2-D discrete convolution (images)
For kernel of height `m`, width `n`:

> **`S_(ij) = Σ(k=1..m)Σ(l=1..n) I_(i+k-1, j+l-1) K_(k,l)`**

i.e., place the kernel at the top-left corner, do an element-wise product with the overlapping image patch, sum all values → **one output pixel**. Slide the kernel across (and down) the image to fill the entire output — called the **feature map** or **activation map**.

**Key facts:** kernels are usually square and **odd-sized** (3×3, 5×5, 7×7) so they can be centered symmetrically on a pixel; a `5×5` image convolved with a kernel (no padding) shrinks to `3×3` — the output is smaller than the input; **kernel values are the learnable network parameters**.

### 2.3 Spatial invariance & multiple filters
Convolving the *same* filter across every spatial location means CNNs "compute the same features... across all spatial areas" — the same pattern detector fires wherever the pattern occurs. Using **multiple filters** (e.g., Filter 1, Filter 2, …) in one layer lets the network detect multiple different small patterns simultaneously, each producing its own feature map; stacking these feature maps gives the layer's output **volume** (depth = number of filters).

### 2.4 Convolution over volumes (color/RGB images)
When the input has multiple channels (e.g., 3 for RGB), **the filter must also have that many channels** — a `3×3×3` filter cube slides over the `H× W×3` image; at each position, all `27` values (`3×3×3`) are multiplied element-wise with the corresponding 27 image values and summed into a **single scalar** — repeating over all spatial positions yields a single 2-D output map (not 3-D) per filter. With `N` filters, you get `N` such maps stacked as the output volume.

### 2.5 Activation after convolution
After filtering, feature maps pass through an activation function — most commonly **ReLU**, which zeroes out negative responses ("discards" them), keeping only the pixels that strongly matched the filter's pattern.

---

## 3. Stride, Padding & Output-Size Formula

### 3.1 Stride
Instead of sliding the kernel by 1 pixel each step, we can slide by `s` pixels — the **stride**. Larger stride → smaller, coarser output, less computation.

### 3.2 Padding
**Problem:** without padding, (a) the image shrinks with every conv layer, and (b) corner/edge pixels are used far fewer times than center pixels (their information is under-represented / "thrown away").
**Solution:** pad the input with zeros around the border before convolving.

| Padding type | Formula | Effect |
|---|---|---|
| **Valid** | `p=0` | No padding; output `<` input |
| **Same** | `p=(f-1)/(2)` | Output size = input size |

Common padding amounts: kernel `3×3⇒ p=1`; kernel `5×5⇒ p=2`; kernel `7×7⇒ p=3`.

### 3.3 General output-size formula
For input size `n× n`, kernel size `f× f`, stride `s`, padding `p`:

> **`O = ⌊(n+2p-f)/(s)⌋ + 1 (applied to both height and width)`**

**Worked example (from the slides — AlexNet's first conv layer):** input `227×227×3`, kernel `11×11×3`, `N=96` filters, `s=4`, `p=0`:

> **`O = ⌊(227-11+0)/(4)⌋+1 = ⌊(216)/(4)⌋+1 = 54+1 = 55`**

→ output feature map is `55×55×96`.

---

## 4. Parameter Counting

For a convolution layer with kernel size `K`, `N` filters, and `C` input channels (**every kernel's depth always equals the number of input channels**, so each kernel has `K× K× C` parameters plus 1 bias):

> **`W_c = K× K× C× N B_c = N P_c = W_c + B_c`**

**Worked example (AlexNet's first conv layer):** `K=11, C=3, N=96`:

> **`W_c = 11×11×3×96 = 34,848 B_c = 96 P_c = 34,848+96=boxed34,944`**

**Pooling layers have NO learnable parameters** — pool size, stride, and padding are hyperparameters only (this is a key structural difference from conv/FC layers).

### 4.1 Full worked example: INPUT → CONV → RELU → POOL → FC (from the slides)
An RGB image of size `227×227×3` is processed by: Conv1 (96 filters, `11×11`, stride 4, valid padding, ReLU) → MaxPool (pool `3×3`, stride 2, valid padding) → FC (10-class output).

| Layer | Output shape | Filter count | Filter size | Weights | Biases | Total params |
|---|---|---|---|---|---|---|
| INPUT | `227×227×3` | – | – | – | – | – |
| CONV + RELU | `55×55×96` | 96 | `11×11×3` | `34,848` | `96` | `34,944` |
| POOL | `27×27×96` | N/A | `3×3` | `0` | `0` | `0` |
| FC | `1×1×10` | – | – | – | – | – |

Pooling output size check: `n=55, f=3, s=2, p=0`: `O=⌊(55-3)/2⌋+1 = 26+1=27` ✓ (matches slide: `27×27×96`).

**Why this matters:** compare to a *fully connected* layer mapping the same `227×227×3` input directly to a 96-unit layer — that would require `227×227×3×96≈ 14.1` **million** weights, vs. only `34,944` for the conv layer. This ~400× reduction comes from **parameter sharing** and **sparse (local) connectivity** (§6).

---

## 5. Pooling

**Purpose:** achieve (further) scale/translation invariance, subsample/reduce representation size, speed up computation. If the input has multiple channels, pooling is applied independently on each channel — number of output channels = number of input channels (pooling never changes depth).

### 5.1 Max Pooling — worked example
Input `4×4`, pool `f=2`, stride `s=2`:

```
[2, 4, 7, 4; 6, 6, 9, 8; 3, 4, 8, 3; 7, 5, 3, 6]\ xrightarrowmax pool 2×2, s=2\ [6, 9; 7, 8]
```

(top-left `2×2` block `{2,4,6,6}→max=6`; top-right `{7,4,9,8}→9`; bottom-left `{3,4,7,5}→7`; bottom-right `{8,3,3,6}→8`.)

### 5.2 Average Pooling — same input

> **`[4.5, 7; 4.75, 5]`**

(top-left average `=(2+4+6+6)/4=4.5`; top-right `=(7+4+9+8)/4=7`; bottom-left `=(3+4+7+5)/4=4.75`; bottom-right `=(8+3+3+6)/4=5`.)

### 5.3 Pooling output size
Same general formula as conv (with no learnable filter, just a pool window): `O=⌊(n-f)/s⌋+1` for valid padding. **Worked example (AlexNet pooling):** input `28×28×8`, stride 2, filter `2×2`, valid padding: `O=⌊(28-2)/2⌋+1 = 13+1=14` → output `14×14×8`.

---

## 6. Receptive Fields, Sparse Connectivity & Parameter Sharing

### 6.1 Receptive field
For a convolution with kernel size `K`, each output element depends on (only "sees") a `K× K` receptive field in the input. **Across multiple layers**, the receptive field grows: with filter sizes `F_j` and strides `S_i` (convention `S₀=1`) at layer `k`:

> **`R_k = 1 + Σ(j=1..k)[(F_j-1)Π(i=0..j-1)S_i]`**

**Worked example (from slides):** `F₁=F₂=3`, `S₁=S₂=1`, at layer `k=2`: `R₂ = 1+2(1)+2(1) = 5` — a `5×5` region of the original input influences one unit at layer 2.

### 6.2 Sparse connectivity vs. fully connected
In an FC layer, every output unit depends on **every** input unit. In a CNN, each output unit depends only on a small local **receptive field** (its kernel footprint) — most "weights" in the equivalent (sparse) weight matrix are effectively zero. This is what makes CNNs far cheaper: "each neuron on the first layer receives every input `x₁` to `x₉`, but weights for many inputs are effectively 0" — the CNN's implicit weight matrix `W₁` is simply "a sparse version of the filter."

### 6.3 Parameter sharing
The **same kernel** (same set of weights) is reused at every spatial location — dramatically reducing the number of free parameters vs. an FC layer that would need independent weights per location (§4.1 numeric comparison). This is the key structural property enabling CNNs to scale to large images.

### 6.4 CNN training is still backprop on an MLP
Because the convolution is just a (very structured, sparse, weight-shared) linear operation followed by a nonlinearity, **CNNs are trained exactly like ordinary feedforward networks** — gradients of the loss w.r.t. weights are computed by the same backpropagation chain rule as Module 2, just respecting the weight-sharing constraint (gradients from all shared positions are summed before the weight update). "Essentially, CNN is also a MLFF (multi-layer feedforward) Network."

---

## 7. Cross-Correlation vs. True Convolution

- **True convolution** flips the kernel (180°) before sliding and taking the dot product.
- **Cross-correlation** is simply the sliding dot product — **no flipping**.
- For **symmetric kernels**, the two operations give identical results.
- **In practice:** almost all deep learning frameworks implement **cross-correlation** but still call it "convolution," because the kernel is *learned* during training — whether or not it is pre-flipped makes no difference to what the network can learn (it would just learn the flipped version of the kernel instead).

---

## 8. Transposed & Dilated Convolutions

### 8.1 Transposed convolution ("deconvolution", fractionally-strided convolution)
**Purpose:** upsampling — produce an output feature map **larger** than the input (opposite of standard conv). Instead of sliding the kernel over the input, a transposed conv effectively **slides the input over the kernel**. If `f` is a conv layer with `Y=f(X)`, the transposed layer `g` (with the same hyperparameters) produces `g(Y)` with the **same shape as `X`**. **Applications:** image generation, super-resolution, semantic segmentation (upsampling back to full image resolution), generating images from noise vectors (GANs).

### 8.2 Dilated (atrous) convolution
**Purpose:** cheaply increase the receptive field of output units **without** increasing the kernel size or the number of parameters. The kernel is "inflated" by inserting `d-1` empty spaces between kernel elements (`d` = dilation rate; `d=1` = ordinary convolution). Output size formula (per the slides): `O = ⌊(I-K+2p)/s⌋+1`, with the **effective** kernel size increased to `K_(eff) = K + (K-1)(d-1)` for a dilation rate `d`.

---

## 9. The Full CNN Pipeline

```
INPUT → [CONV → ReLU → POOL] × several times → FLATTEN → FC → softmax → class probabilities
                (repeatable block)
```
- **Convolution** layers extract local features (edges → textures → parts → objects, hierarchically, as depth increases).
- **Pooling** layers subsample, reducing spatial size and adding local translation invariance.
- **Flatten** converts the final (small) spatial feature volume into a 1-D vector.
- **Fully connected** layers combine the extracted features for final classification (e.g., "cat", "dog", …).
- Standard losses/optimizers apply as in Module 2/3: Binary/Multi-class/Class-weighted Cross-Entropy, L1(MAE)/L2(MSE) for regression heads; Dropout and BatchNorm (Module 4) are used the same way; Adam/RMSProp (Module 3) are the typical optimizers; Sigmoid (binary) or Softmax (multi-class) at the output layer.

---

## 10. Code Implementation

### 10.1 From-scratch 2-D convolution (cross-correlation, as frameworks do it)
```python
# Topic: 2D convolution (cross-correlation) from scratch
# Purpose: Reproduce the "2D Discrete Convolution Operation" (§2.2) and output-size formula (§3.3)
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode="constant")
    n = image.shape[0]
    f = kernel.shape[0]
    out_size = (n - f) // stride + 1
    output = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            row, col = i * stride, j * stride
            patch = image[row:row+f, col:col+f]
            output[i, j] = np.sum(patch * kernel)   # element-wise product + sum
    return output

# Example: reproduce the pooling-adjacent "complete example" style conv
image = np.array([
    [1,0,0,0,0,1],
    [0,1,0,0,1,0],
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [0,1,0,0,1,0],
    [1,0,0,0,0,1],
])
sobel_vertical = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
print(conv2d(image, sobel_vertical, stride=1, padding=0).shape)   # (4, 4) -> matches 6x6 in, 3x3 kernel, valid pad
```

### 10.2 Max & Average pooling
```python
# Topic: Pooling layer forward pass
# Purpose: Reproduce the §5.1/§5.2 worked pooling example
import numpy as np

def pool2d(x, f=2, stride=2, mode="max"):
    n = x.shape[0]
    out_size = (n - f) // stride + 1
    out = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            row, col = i * stride, j * stride
            patch = x[row:row+f, col:col+f]
            out[i, j] = patch.max() if mode == "max" else patch.mean()
    return out

x = np.array([
    [2,4,7,4],
    [6,6,9,8],
    [3,4,8,3],
    [7,5,3,6],
])
print("Max pool:\n", pool2d(x, mode="max"))       # -> [[6,9],[7,8]]
print("Avg pool:\n", pool2d(x, mode="average"))   # -> [[4.5,7],[4.75,5]]
```

### 10.3 Output-size and parameter-count helpers (verify AlexNet numbers)
```python
def conv_output_size(n, f, s=1, p=0):
    return (n + 2*p - f) // s + 1

def conv_param_count(k, c, num_filters):
    weights = k * k * c * num_filters
    biases = num_filters
    return weights, biases, weights + biases

# AlexNet layer 1 checks
print(conv_output_size(227, 11, s=4, p=0))          # -> 55
print(conv_param_count(11, 3, 96))                  # -> (34848, 96, 34944)
print(conv_output_size(55, 3, s=2, p=0))             # -> 27  (maxpool after conv1)
```

### 10.4 A minimal CNN in a DL framework (illustrative, PyTorch-style pseudocode)
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Linear(27*27*96, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
```

---

## 11. Worked Numerical Practice Problems (from the slide deck, with full solutions)

**Q1.** Input `5×5` convolved with filter `3×3`, **same** padding, stride 1. Find `p`.
*Solution:* Same padding requires output size = input size. `p=(f-1)/(2)=(3-1)/(2)=[1]`.

**Q2.** Input `5×6` convolved with filter `3×3`, same padding, stride 1. Show the padded input size.
*Solution:* `p=1` on all sides → padded input becomes `(5+2)×(6+2) = [7×8]`.

**Q3.** Input `5×6` convolved with filter `3×3`, **valid** padding, stride 2. Find output size.
*Solution:* `p=0`. Height: `⌊(5-3)/2⌋+1=1+1=2`. Width: `⌊(6-3)/2⌋+1=1+1=2`. Output `=[2×2]`.

**Q4.** Convolution on input `5×5×16`, filter `5×5`, stride 1, valid padding, 80 filters. Find output shape.
*Solution:* `p=0`: `O=⌊(5-5)/1⌋+1=1`. Output shape `=[1×1×80]` (depth = number of filters).

**Q5.** Max pooling on input `28×28×8`, stride 2, filter `2×2`, valid padding. Find output shape.
*Solution:* `O=⌊(28-2)/2⌋+1=13+1=14`. Output shape `=[14×14×8]` (channels unchanged by pooling).

**Q6 (full pipeline, AlexNet-style, worked in §4.1).** RGB image `227×227×3` → Conv(96 filters, `11×11`, `s=4`, valid) → ReLU → MaxPool(`3×3`, `s=2`, valid) → FC(10 classes). Find each layer's output shape and total learnable parameters.
*Solution:*
- Conv output: `O=⌊(227-11)/4⌋+1=54+1=55` → `55×55×96`; params `=11×11×3×96+96=34,944`.
- Pool output: `O=⌊(55-3)/2⌋+1=26+1=27` → `27×27×96`; params `=0`.
- FC output: `1×1×10` (10-class scores).
- **Total learnable parameters (conv+pool shown) `=boxed34,944`** (FC layer's own params depend on its unspecified width, computed separately as `27×27×96×10 + 10` if fully connecting the flattened pooled output directly to 10 outputs).

---

## 12. Connections to Other Topics
- **Module 2 (Feedforward Networks):** a CNN is trained via the exact same backpropagation/chain-rule machinery — convolution is just a structured (sparse, weight-shared) linear layer.
- **Module 3 (Optimization):** Adam/RMSProp/momentum apply unchanged; batch-size effects (Module 3 §7) are especially relevant since CNNs on images are usually memory-bound.
- **Module 4 (Regularization):** Dropout (esp. **DropBlock**/**SpatialDropout** variants), **Batch Normalization** (placed after conv, before activation), and data augmentation (rotation/flip/crop/Mixup/CutMix) are the dominant regularizers for CNNs.
- **Parameter-efficiency argument (§4.1, §6.3)** directly parallels why deep narrow networks (Module 1 §depth-vs-width) are preferred over shallow wide ones — both exploit compositional/hierarchical structure to save parameters.

## 13. Key Takeaways
1. Convolution replaces a dense matrix multiply with a **local, weight-shared, sparse** linear operation — dramatically reducing parameters (AlexNet conv1: 34,944 vs. ~14M for an equivalent FC layer) while preserving translation equivariance.
2. **Output size** for any conv or pooling layer: `O=⌊(n+2p-f)/s⌋+1` — memorize this single formula; it drives all shape/parameter bookkeeping.
3. **Parameter count** for a conv layer: `P_c = (K× K× C× N) + N` (weights + biases); **pooling layers have zero learnable parameters**.
4. **Max pooling** contributes translation invariance and reduces spatial size cheaply; **average pooling** is used less often but is smoother.
5. Frameworks implement **cross-correlation**, not flipped convolution — doesn't matter in practice since kernels are learned.
6. The **receptive field** grows with depth and with each layer's kernel size/stride — deeper layers "see" progressively larger regions of the original input, enabling hierarchical feature learning (edges → parts → objects).
7. Transposed convolution upsamples (segmentation, generation); dilated convolution grows the receptive field without adding parameters.

## 14. Common Mistakes & Misconceptions
- **Mistake:** believing convolution kernels must be flipped in code, matching the mathematical definition. **Correction:** virtually all frameworks use cross-correlation; this only matters for interoperability with hand-derived signal-processing convolution formulas.
- **Mistake:** forgetting that a kernel's depth must equal the input's channel count. **Correction:** a `K× K` "filter" applied to a `C`-channel input is really `K× K× C`, producing **one** 2-D output map per filter (not `C` maps).
- **Mistake:** assuming pooling layers have learnable parameters like conv layers. **Correction:** pooling has **zero** parameters — only hyperparameters (pool size, stride, padding).
- **Mistake:** using "same" padding but forgetting the padding amount depends on kernel size (`p=(f-1)/2`, only exact for odd `f` and stride 1). **Correction:** always re-derive `p` from the formula for the specific `f` and `s` in use.
- **Mistake:** confusing the *output number of channels* (always equals the number of filters `N` used, regardless of input channel count `C`) with the *input* channel count. **Correction:** input channels only affect the kernel's *depth*, not the output's depth.

## 15. Additional Practice Problems
1. An input of size `32×32×3` is convolved with 64 filters of size `5×5`, stride 1, same padding. Find the output shape and the number of learnable parameters.
   *Solution:* Same padding with `f=5,s=1⇒ p=2`; output height `=⌊(32+4-5)/1⌋+1=32` → output `32×32×64`. Params `=5×5×3×64+64=4800+64=4864`.
2. A `64×64×32` feature map is max-pooled with `f=2,s=2`, valid padding. Find the output shape.
   *Solution:* `O=⌊(64-2)/2⌋+1=32` → output `32×32×32` (channels unchanged).
3. Compute the receptive field at layer 3 for three stacked `3×3`, stride-1 conv layers (`F₁=F₂=F₃=3`, `S₁=S₂=S₃=1`).
   *Solution:* Using `R_k=1+Σ(j)(F_j-1)Π(i<j)S_i`: `R₃ = 1+(3-1)+(3-1)+(3-1)=1+2+2+2=7` — a `7×7` receptive field, illustrating how stacking small kernels cheaply grows the receptive field (the VGG-style design principle).
4. Why does using three stacked `3×3` conv layers (receptive field 7×7, per Q3) have fewer parameters than one `7×7` conv layer, for `C` input/output channels?
   *Solution:* Three `3×3` layers: `3×(3×3× C× C)=27C²` parameters (ignoring biases); one `7×7` layer: `7×7× C× C=49C²` parameters. The stacked small kernels achieve the same receptive field with **~45% fewer parameters** and an extra 2 nonlinearities (more expressive), which is exactly the design insight behind VGG-style architectures.
5. A dilated convolution uses a `3×3` kernel with dilation rate `d=2`. What is its effective kernel size, and what would the equivalent (non-dilated) kernel size need to be to achieve the same receptive field?
   *Solution:* `K_(eff) = K+(K-1)(d-1) = 3+(2)(1)=5`. So a `3×3` kernel with `d=2` has the receptive field of a plain `5×5` kernel, but uses only `9` parameters instead of `25`.
