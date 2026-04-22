# 🎯 Focused Interview Prep — Based on HR Feedback

> **HR Said:** Mostly situational. Improve Python & ML (weakness from previous rounds). Watch out for Computer Vision questions.

**Strategy:** This guide addresses your exact gaps. Study this BEFORE the 200-question file.

---

# PART 1: PYTHON — Nail the Fundamentals

> These are the questions that trip up candidates in technical rounds. Know these cold.

---

## 1.1 Core Python Concepts They'll Test

### Q1. What happens when you do `a = [1,2,3]; b = a; b.append(4)`? What is `a` now?
**A:** `a` is `[1,2,3,4]`. Both `a` and `b` point to the **same list object** in memory. Python variables are references, not containers. This is called **aliasing**. To create an independent copy: `b = a.copy()` (shallow) or `b = copy.deepcopy(a)` (deep for nested structures).

### Q2. What is the output?
```python
def f(x, lst=[]):
    lst.append(x)
    return lst

print(f(1))  # [1]
print(f(2))  # [1, 2]  — NOT [2]!
```
**A:** The default mutable argument `lst=[]` is created **once** when the function is defined, not on each call. All calls share the same list. **Fix:** Use `lst=None` and `if lst is None: lst = []` inside the function. This is one of the most common Python gotchas.

### Q3. Explain `is` vs `==`.
**A:** `==` checks **value equality** (do they contain the same data?). `is` checks **identity** (are they the exact same object in memory?).
```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b  # True  (same values)
a is b  # False (different objects)

x = None
x is None  # True — always use 'is' for None comparison
```

### Q4. What are Python's data structures and their time complexities?
**A:**
| Structure | Lookup | Insert | Delete | Use Case |
|-----------|--------|--------|--------|----------|
| `list` | O(n) | O(1) amortized (end) | O(n) | Ordered, mutable sequences |
| `dict` | O(1) avg | O(1) avg | O(1) avg | Key-value mapping |
| `set` | O(1) avg | O(1) avg | O(1) avg | Membership testing, dedup |
| `tuple` | O(n) | Immutable | Immutable | Fixed data, dict keys |
| `deque` | O(n) | O(1) both ends | O(1) both ends | Queues, sliding windows |

**Critical:** If they ask "how would you check if an element exists in a large collection?" — the answer is `set` or `dict`, NOT `list`. `list` is O(n), `set` is O(1).

### Q5. What is a dictionary comprehension? Give an ML example.
**A:**
```python
# Create a mapping of class index to class name
class_names = ['cat', 'dog', 'bird']
idx_to_class = {i: name for i, name in enumerate(class_names)}
# {0: 'cat', 1: 'dog', 2: 'bird'}

# Filter metrics above a threshold
metrics = {'accuracy': 0.95, 'recall': 0.72, 'f1': 0.88}
good_metrics = {k: v for k, v in metrics.items() if v > 0.8}
# {'accuracy': 0.95, 'f1': 0.88}
```

### Q6. Explain Python decorators with a real example.
**A:** A decorator wraps a function to add behavior without modifying it:
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.2f}s")
        return result
    return wrapper

@timer
def train_model(epochs):
    # training logic here
    pass

train_model(10)  # Prints: train_model took 45.23s
```
**Real uses:** `@torch.no_grad()` for inference, `@app.route()` in Flask, `@staticmethod`, `@property`.

### Q7. What is the GIL and how does it affect ML code?
**A:** The Global Interpreter Lock allows only one thread to execute Python bytecode at a time. Impact on ML:
- **CPU-bound** (data preprocessing): Threading won't help → use `multiprocessing` or `joblib`
- **I/O-bound** (loading files, API calls): Threading works fine → use `threading` or `asyncio`
- **NumPy/PyTorch operations**: They **release the GIL** internally → already parallelized in C/CUDA
- **GPU operations**: Not affected by GIL at all

### Q8. What are generators? Why are they critical for ML?
**A:**
```python
# BAD: Loads ALL images into memory
images = [load_image(path) for path in all_paths]  # 100GB in RAM!

# GOOD: Generator — loads one at a time
def image_generator(paths):
    for path in paths:
        yield load_image(path)  # Only one image in memory

# PyTorch DataLoader uses this concept internally
for batch in image_generator(all_paths):
    model(batch)
```
Generators use `yield` instead of `return` — they produce values lazily, one at a time, using O(1) memory regardless of dataset size.

### Q9. Explain `map`, `filter`, `lambda` — and when to use list comprehension instead.
**A:**
```python
# Lambda: anonymous function
square = lambda x: x**2

# map: apply function to every element
squared = list(map(lambda x: x**2, [1,2,3]))  # [1, 4, 9]

# filter: keep elements where function returns True
positives = list(filter(lambda x: x > 0, [-1, 2, -3, 4]))  # [2, 4]

# PREFER list comprehension — more Pythonic and readable:
squared = [x**2 for x in [1,2,3]]
positives = [x for x in [-1,2,-3,4] if x > 0]
```

### Q10. What is `__init__`, `__repr__`, `__str__`, `__len__` in a class?
**A:** These are **dunder (magic) methods** — they define how Python objects behave:
```python
class Dataset:
    def __init__(self, data):      # Constructor — called on Dataset(data)
        self.data = data
    
    def __len__(self):              # len(dataset) — PyTorch requires this
        return len(self.data)
    
    def __getitem__(self, idx):     # dataset[0] — PyTorch requires this
        return self.data[idx]
    
    def __repr__(self):             # Debugging representation
        return f"Dataset(n={len(self.data)})"
    
    def __str__(self):              # print(dataset) — human readable
        return f"Dataset with {len(self.data)} samples"
```
**PyTorch connection:** Every custom Dataset MUST implement `__len__` and `__getitem__`.

### Q11. Explain exception handling best practices.
**A:**
```python
# BAD: Catching everything
try:
    result = model.predict(image)
except:  # Never do this — hides bugs
    pass

# GOOD: Specific exceptions with context
try:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    result = model.predict(image)
except FileNotFoundError:
    logger.error(f"Image not found: {path}")
    return None
except ValueError as e:
    logger.error(f"Invalid image: {e}")
    return None
except RuntimeError as e:
    logger.error(f"Model inference failed: {e}")
    raise  # Re-raise unexpected errors
```

### Q12. How does Python manage memory? What is reference counting?
**A:** Python uses **reference counting** — each object tracks how many variables point to it. When count reaches 0, memory is freed immediately. A **cyclic garbage collector** handles cases where objects reference each other (e.g., A→B→A). For ML: large arrays (numpy/torch tensors) use memory from the underlying C/CUDA allocators. Use `del tensor` and `torch.cuda.empty_cache()` to explicitly free GPU memory.

---

## 1.2 Python Coding Patterns for ML

### Q13. How do you read a large CSV file efficiently?
**A:**
```python
# BAD: Loads entire file
df = pd.read_csv('huge_file.csv')

# GOOD: Read in chunks
for chunk in pd.read_csv('huge_file.csv', chunksize=10000):
    process(chunk)

# BETTER for specific columns:
df = pd.read_csv('huge_file.csv', usecols=['image_path', 'label'])
```

### Q14. How would you parallelize data preprocessing?
**A:**
```python
from multiprocessing import Pool
import cv2

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    return img

# Process 1000 images using all CPU cores
with Pool(processes=8) as pool:
    processed = pool.map(preprocess_image, image_paths)

# In PyTorch — DataLoader does this automatically:
loader = DataLoader(dataset, num_workers=4, pin_memory=True)
```

### Q15. Write a Python class for a simple ML pipeline.
**A:**
```python
class ImageClassifier:
    def __init__(self, model_path, classes, device='cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.classes = classes
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, path):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, len(self.classes))
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model.to(self.device)
    
    def predict(self, image):
        """Returns predicted class and confidence."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, idx = probs.max(dim=1)
        return self.classes[idx.item()], confidence.item()
```

---

# PART 2: ML FUNDAMENTALS — Strengthen Your Weak Areas

> Focus on concepts you might have struggled to explain clearly in previous rounds.

---

## 2.1 Training Concepts (Explain Like You Truly Understand)

### Q16. Walk through what happens in ONE training iteration.
**A:**
```
1. FORWARD PASS: Input → Model → Predictions
   - Data batch goes through layers sequentially
   - Each layer applies: output = activation(weights × input + bias)

2. LOSS COMPUTATION: Predictions vs Ground Truth → Loss value
   - E.g., CrossEntropyLoss for classification
   - Single scalar number that says "how wrong are we"

3. BACKWARD PASS: Compute gradients via backpropagation
   - loss.backward() — chain rule through every layer
   - Every weight gets a gradient: "which direction reduces loss"

4. PARAMETER UPDATE: optimizer.step()
   - weight_new = weight_old - learning_rate × gradient
   - Then optimizer.zero_grad() to clear gradients for next iteration
```
**In code:**
```python
for images, labels in train_loader:
    outputs = model(images)           # Forward
    loss = criterion(outputs, labels)  # Loss
    loss.backward()                    # Backward
    optimizer.step()                   # Update
    optimizer.zero_grad()              # Reset
```

### Q17. What is learning rate and why is it the most important hyperparameter?
**A:** Learning rate controls step size during gradient descent. 
- **Too high:** Loss oscillates or diverges — model overshoots optimal weights
- **Too low:** Training is extremely slow, may get stuck in local minima
- **Just right:** Loss decreases steadily and converges

**Advanced strategies I use:**
- **Learning rate warmup:** Start very small, gradually increase (prevents instability early)
- **Cosine annealing:** Gradually decrease LR following a cosine curve
- **ReduceLROnPlateau:** Automatically reduce LR when validation loss plateaus
- Typical starting values: 1e-3 for Adam, 1e-2 for SGD

### Q18. Explain the difference between loss functions: MSE, CrossEntropy, BCE.
**A:**
| Loss | Formula | Use Case |
|------|---------|----------|
| **MSE** | mean((y - ŷ)²) | Regression (predicting continuous values) |
| **CrossEntropy** | -Σ y·log(ŷ) | Multi-class classification (one correct class) |
| **BCE** | -[y·log(ŷ) + (1-y)·log(1-ŷ)] | Binary classification or multi-label (each label independent) |

**In my multilabel project:** I used **BCE with class weights** because each image can have multiple labels independently, and some labels are rare.

### Q19. What is batch size's effect on training?
**A:**
- **Large batch:** More stable gradients (less noise), faster per epoch (GPU utilization), but may generalize worse, needs larger LR
- **Small batch:** Noisier gradients (acts as regularization), better generalization often, but slower per epoch, more iterations
- **Sweet spot:** Usually 16, 32, or 64. Limited by GPU memory for large models
- **If GPU memory is the bottleneck:** Use gradient accumulation — simulate large batch by accumulating gradients over multiple small batches before updating

### Q20. What is the difference between Adam, SGD, and AdamW?
**A:**
- **SGD:** Simple: w = w - lr × gradient. Needs careful LR tuning. With momentum: remembers previous gradient direction for smoother updates. Often generalizes best at convergence
- **Adam:** Adaptive per-parameter LR + momentum. Converges fast, less tuning needed. Can generalize slightly worse than well-tuned SGD
- **AdamW:** Adam with decoupled weight decay (proper L2 regularization). Currently the standard for training transformers/BERT

**My practice:** Adam/AdamW for quick experiments and transformers, SGD with momentum for CNNs when I have time to tune.

### Q21. Explain dropout. How does it work during training vs inference?
**A:**
- **Training:** Randomly sets neuron outputs to 0 with probability p (e.g., 0.5). Remaining outputs are scaled by 1/(1-p) to maintain expected values. Forces the network to learn redundant representations — no single neuron can be relied upon
- **Inference:** Dropout is **disabled** (all neurons active). This is why `model.eval()` is critical before inference — forgetting this was a bug in my early projects
- **Typical values:** 0.1-0.3 for transformers, 0.5 for fully connected layers in CNNs, not used in convolutional layers (use BatchNorm instead)

### Q22. What are the different types of regularization?
**A:**
1. **L1/L2 regularization** — penalize large weights in loss function
2. **Dropout** — randomly deactivate neurons during training
3. **Batch Normalization** — normalize layer inputs (mild regularization effect)
4. **Data augmentation** — create training variations (rotation, flipping, etc.)
5. **Early stopping** — stop when validation loss stops improving
6. **Weight decay** — directly decay weights toward zero each step

**Key insight:** You almost never use just one. A typical setup: data augmentation + dropout + weight decay + early stopping.

### Q23. Explain underfitting vs overfitting. How do you diagnose each?
**A:**
```
UNDERFITTING:                      OVERFITTING:
- Train accuracy: LOW              - Train accuracy: HIGH
- Val accuracy: LOW                - Val accuracy: LOW (big gap)
- Model too simple                 - Model too complex
                                   
Fix: More complex model,           Fix: More data, augmentation,
more features, train longer,       regularization (dropout, L2),
reduce regularization              simpler model, early stopping
```
**Diagnosis:** Plot training loss AND validation loss. If both are high → underfitting. If training loss is low but validation loss is high → overfitting. If both are low → good fit.

### Q24. What is the difference between epoch, batch, and iteration?
**A:**
- **Epoch:** One complete pass through the ENTIRE training dataset
- **Batch:** A subset of the dataset processed together (e.g., 32 images)
- **Iteration:** Processing one batch (forward + backward + update)
- **Relationship:** If dataset has 1000 samples, batch size is 100 → 10 iterations per epoch
- **Typical training:** 10-100 epochs depending on dataset size and convergence

---

## 2.2 Model Evaluation (Where Candidates Often Stumble)

### Q25. You have 1000 positive and 100,000 negative samples. Your model gets 99% accuracy. Is it good?
**A:** **No.** A model that predicts "negative" for everything achieves 100000/101000 = 99.01% accuracy. This is the **accuracy paradox** with imbalanced data.

**Better metrics:**
- **Precision:** Of predicted positives, how many are correct?
- **Recall:** Of actual positives, how many did we find?
- **F1-score:** Harmonic mean of precision and recall
- **PR-AUC:** Better than ROC-AUC for imbalanced data
- **Confusion matrix:** Shows the full picture

### Q26. When would you optimize for precision vs recall?
**A:**
- **Precision (minimize false positives):** Spam filtering (don't put real email in spam), recommending products (don't annoy users with bad recommendations)
- **Recall (minimize false negatives):** Disease detection (don't miss cancer), fraud detection (don't miss fraud), safety systems (don't miss defects)
- **In CV at AIMonk:** Depends on the application — defect detection needs high recall; automated decisions need high precision

### Q27. Explain k-fold cross-validation step by step.
**A:**
```
Dataset: [A B C D E]  (5-fold example)

Fold 1: Train=[B C D E]  Val=[A]  → Score: 0.85
Fold 2: Train=[A C D E]  Val=[B]  → Score: 0.88
Fold 3: Train=[A B D E]  Val=[C]  → Score: 0.83
Fold 4: Train=[A B C E]  Val=[D]  → Score: 0.87
Fold 5: Train=[A B C D]  Val=[E]  → Score: 0.86

Final Score: Mean ± Std = 0.858 ± 0.018
```
**When to use:** Small datasets where train/val split would waste data. **When NOT to use:** Large datasets (single split is fine), time-series (use temporal split).

---

# PART 3: COMPUTER VISION — Deep Preparation

> HR specifically warned about CV questions. Know these extremely well.

---

## 3.1 CNN Fundamentals

### Q28. Explain convolution operation step by step. What does a filter learn?
**A:**
```
Input (5×5):          Filter (3×3):         Output (3×3):
1 0 1 0 1             1 0 1                 4 3 4
0 1 0 1 0             0 1 0                 3 4 3
1 0 1 0 1      ×      1 0 1          =      4 3 4
0 1 0 1 0
1 0 1 0 1

Computation: Slide filter across input, multiply element-wise, sum.
Position (0,0): 1×1+0×0+1×1 + 0×0+1×1+0×0 + 1×1+0×0+1×1 = 4
```
**What filters learn:**
- Early layers: Edges (horizontal, vertical, diagonal), color gradients
- Middle layers: Textures, patterns, shapes
- Deep layers: Object parts (eyes, wheels, text characters)
- This hierarchy is learned automatically — we don't design the filters

### Q29. Calculate the output size of a convolution.
**A:**
```
Output size = (Input - Filter + 2×Padding) / Stride + 1

Example: Input=224×224, Filter=3×3, Padding=1, Stride=1
Output = (224 - 3 + 2×1) / 1 + 1 = 224  (same padding)

Example: Input=224×224, Filter=7×7, Padding=0, Stride=2
Output = (224 - 7 + 0) / 2 + 1 = 109
```
**Common gotcha:** If the calculation doesn't give an integer, the configuration is invalid (floor is often applied but can cause information loss at edges).

### Q30. How many parameters does a Conv2D layer have?
**A:**
```
Parameters = (Filter_H × Filter_W × Input_Channels + 1) × Output_Channels
                                                    ↑ bias

Example: Conv2d(in_channels=3, out_channels=64, kernel_size=3)
Parameters = (3 × 3 × 3 + 1) × 64 = 28 × 64 = 1,792

Example: Conv2d(in_channels=64, out_channels=128, kernel_size=3)
Parameters = (3 × 3 × 64 + 1) × 128 = 577 × 128 = 73,856
```
**Key insight:** Parameters DON'T depend on input spatial size — this is why CNNs work on different image sizes (unlike fully connected layers).

### Q31. What is 1×1 convolution and why is it useful?
**A:** A 1×1 conv operates on each pixel independently across channels. It:
1. **Reduces/increases channel dimensions** — e.g., 256 channels → 64 channels (compression)
2. **Adds non-linearity** — it's like a per-pixel fully connected layer with activation
3. **Used in:** ResNet bottleneck blocks (reduce→compute→expand), Inception (channel reduction before expensive 3×3/5×5 convs), MobileNet (pointwise conv)

Think of it as "mixing channels" at each spatial position without looking at neighboring pixels.

### Q32. Explain depthwise separable convolution (MobileNet).
**A:** Splits standard convolution into two steps:
1. **Depthwise conv:** Apply ONE filter PER input channel (C filters of size K×K×1)
2. **Pointwise conv:** 1×1 conv to combine channels (M filters of size 1×1×C)

**Computational saving:**
- Standard: K² × C × M × H × W multiplications
- Separable: K² × C × H × W + C × M × H × W
- **Reduction factor: ~K²** (for K=3, ~9× fewer operations)

**Why it matters for AIMonk:** Edge deployment, mobile CV, real-time processing.

### Q33. Compare pooling types and Global Average Pooling.
**A:**
- **Max Pooling:** Takes max in each window → captures strongest activations. Most common
- **Average Pooling:** Takes mean → smoother, preserves more info
- **Global Average Pooling (GAP):** Average over ENTIRE feature map → one value per channel
  - Replaces fully connected layers at the end of CNNs
  - Eliminates most parameters (ResNet: 512 channels → 512-dim vector → classification)
  - Acts as structural regularization
  - Makes model accept any input size

---

## 3.2 Key CNN Architectures (Know the Evolution)

### Q34. Trace the evolution: LeNet → AlexNet → VGG → ResNet → EfficientNet.
**A:**
```
LeNet (1998): 5 layers, handwritten digits. Proved CNNs work.
      ↓
AlexNet (2012): 8 layers, ReLU, dropout, GPU training. Won ImageNet.
      ↓
VGG (2014): 16-19 layers, only 3×3 filters. Simple and deep.
      ↓ Problem: Going deeper stopped helping → degradation problem
ResNet (2015): 152 layers with SKIP CONNECTIONS. Revolution.
      ↓
EfficientNet (2019): Compound scaling (depth × width × resolution).
      Systematically optimized CNN architecture.
```
**What to say:** "ResNet's key insight was that skip connections allow identity mapping, solving the degradation problem. Before ResNet, deeper models were paradoxically harder to train than shallow ones."

### Q35. Explain ResNet's residual blocks in detail.
**A:**
```
            Input (x)
              |
              ├── Conv → BN → ReLU → Conv → BN → F(x)
              |                                    |
              └────────────── Skip ───────────────→ +
                                                   |
                                                 ReLU
                                                   |
                                              Output = F(x) + x
```
**Why it works:**
1. If F(x) = 0 (identity is optimal), the block naturally passes input through
2. Gradients flow directly through skip connection — no vanishing gradient through 100+ layers
3. **Bottleneck variant (ResNet-50+):** 1×1(reduce) → 3×3(compute) → 1×1(expand) — much fewer params

### Q36. What is EfficientNet's compound scaling?
**A:** Previous work scaled networks in only one dimension (deeper OR wider OR higher resolution). EfficientNet scales all three simultaneously with fixed ratios:
```
depth   = α^φ
width   = β^φ  
resolution = γ^φ

Where α·β²·γ² ≈ 2 (FLOPS roughly double when φ increases by 1)
```
**Result:** Better accuracy with fewer parameters than manually designed architectures. EfficientNet-B0 is a great baseline for transfer learning.

---

## 3.3 Object Detection & Segmentation

### Q37. Explain YOLO (You Only Look Once) architecture.
**A:**
```
1. Divide image into S×S grid (e.g., 7×7)
2. Each grid cell predicts:
   - B bounding boxes (x, y, w, h, confidence) 
   - C class probabilities
3. Output tensor: S × S × (B×5 + C)
4. Single forward pass → all detections at once (hence "Only Look Once")

YOLOv1 → v2 (anchor boxes) → v3 (multi-scale) → v5 (CSP backbone) → v8 (anchor-free)
```
**Key advantage:** Real-time speed (30-60+ FPS). Trade: slightly lower accuracy on small objects compared to two-stage detectors. **Relevance to AIMonk:** Most production CV systems need real-time or near-real-time.

### Q38. What is Non-Maximum Suppression (NMS)? Walk through an example.
**A:**
```
Detections for "car":
Box A: confidence=0.95, position=[100,100,200,200]
Box B: confidence=0.88, position=[110,105,210,205]  ← overlaps A heavily
Box C: confidence=0.70, position=[400,300,500,400]  ← separate car

Step 1: Sort by confidence → A(0.95), B(0.88), C(0.70)
Step 2: Keep A. Compute IoU(A,B)=0.85 > threshold(0.5) → Remove B
Step 3: Keep C. IoU(A,C)=0.0 < threshold → Keep C

Final: Box A (first car) and Box C (second car)
```

### Q39. Explain Intersection over Union (IoU) with code.
**A:**
```python
def calculate_iou(box1, box2):
    """boxes as [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2-x1) * max(0, y2-y1)
    
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```
**IoU thresholds:** 0.5 (standard detection), 0.75 (strict), 0.5:0.95 (COCO mAP averages across thresholds).

### Q40. What is mAP (mean Average Precision)? How is it calculated?
**A:**
1. For each class, rank all detections by confidence
2. At each detection, compute precision and recall
3. Plot Precision-Recall curve
4. AP (Average Precision) = area under the PR curve
5. mAP = mean of AP across all classes

```
COCO mAP: Average over IoU thresholds 0.5 to 0.95 (step 0.05)
Pascal VOC mAP: Single IoU threshold of 0.5
```
**Know this:** mAP@0.5 is lenient, mAP@0.75 is strict, mAP@[.5:.95] is the standard benchmark.

### Q41. Explain semantic segmentation vs instance segmentation.
**A:**
```
SEMANTIC SEGMENTATION (e.g., U-Net, DeepLab):
- Labels every pixel with a class
- Two adjacent cars → both labeled "car" (no distinction)
- Architecture: Encoder-Decoder with skip connections

INSTANCE SEGMENTATION (e.g., Mask R-CNN):
- Labels every pixel AND distinguishes individual instances
- Two adjacent cars → "car_1" and "car_2" (separate masks)
- Architecture: Detection head + mask prediction branch

PANOPTIC SEGMENTATION:
- Combines both: instances for "things" (cars, people) 
  + semantic for "stuff" (sky, road)
```

### Q42. Explain the U-Net architecture. Why is it good for segmentation?
**A:**
```
Encoder (Contracting):     Decoder (Expanding):
Input 572×572             
  ↓ Conv+Pool → 64ch       ↑ UpConv + Skip → 64ch → Output 388×388
  ↓ Conv+Pool → 128ch      ↑ UpConv + Skip → 128ch
  ↓ Conv+Pool → 256ch      ↑ UpConv + Skip → 256ch
  ↓ Conv+Pool → 512ch      ↑ UpConv + Skip → 512ch
  ↓            1024ch ───→  ↑ (Bottleneck)
```
**Key feature: Skip connections** from encoder to decoder — they pass high-resolution spatial information directly, allowing precise localization. Without them, the decoder must reconstruct spatial details from compressed features.

**Use cases at AIMonk:** Medical imaging, defect detection, document layout analysis.

---

## 3.4 Image Preprocessing & OpenCV

### Q43. What OpenCV operations would you use for preprocessing in your license plate project?
**A:**
```python
import cv2
import numpy as np

# 1. Read and convert color space
img = cv2.imread('plate.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Note: BGR not RGB!

# 2. Noise removal
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# 3. Adaptive thresholding (handles varying lighting)
thresh = cv2.adaptiveThreshold(blurred, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# 4. Morphological operations (clean up)
kernel = np.ones((3,3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 5. Find contours (potential plate regions)
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

# 6. Filter by aspect ratio (plates are wider than tall)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    if 2.0 < aspect_ratio < 6.0:  # Typical plate ratio
        plate_region = img[y:y+h, x:x+w]
```

### Q44. What is the difference between BGR and RGB? Why does it matter?
**A:** OpenCV loads images in **BGR** (Blue, Green, Red) order. Most other libraries (PIL, matplotlib, PyTorch) use **RGB**. Mixing them up causes subtle color errors that are hard to debug — the model trains on wrong color information.
```python
# OpenCV → PyTorch/PIL:
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# PIL → OpenCV:
bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
```
**This was a real bug in my license plate project** — inconsistent color spaces caused accuracy drops in certain lighting.

### Q45. Explain image augmentation techniques for computer vision.
**A:**
```python
from torchvision import transforms

train_transform = transforms.Compose([
    # Spatial transforms
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    
    # Color transforms
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    
    # Advanced
    transforms.RandomErasing(p=0.1),  # Cutout-style
    
    # Standard normalization
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])  # ImageNet stats
])
```
**Key principle:** Augmentations should be realistic — don't vertically flip if objects are always upright. For license plates: rotation (slight), perspective transform, brightness changes. NOT vertical flip (plates don't appear upside down in practice).

### Q46. What is transfer learning for CV? Show the code.
**A:**
```python
import torchvision.models as models
import torch.nn as nn

# 1. Load pretrained model
model = models.resnet50(pretrained=True)

# 2. Freeze early layers (generic features)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace final layer for our task
num_classes = 10  # Our specific classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
# ^ This layer's params require_grad=True by default

# 4. Fine-tune with smaller LR
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# 5. Optionally unfreeze later layers for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Low LR
    {'params': model.fc.parameters(), 'lr': 1e-3}       # Higher LR
])
```

---

## 3.5 Advanced CV Topics (Bonus — Impress the Interviewer)

### Q47. What is attention in computer vision (Vision Transformer)?
**A:** ViT (Vision Transformer) applies the Transformer's self-attention to images:
1. Split image into patches (16×16 pixels each)
2. Flatten each patch and project to an embedding
3. Add positional embeddings
4. Pass through Transformer encoder blocks (self-attention + FFN)
5. Classification from [CLS] token

**Key advantage:** Global receptive field from the first layer (vs CNN which builds it gradually). **Trade-off:** Needs more data than CNNs (less inductive bias about spatial locality).

### Q48. What is CLIP and how does it connect vision and language?
**A:** CLIP (Contrastive Language-Image Pre-training) trains two encoders:
- Image encoder (CNN or ViT)
- Text encoder (Transformer)

Trained with contrastive loss: matching image-text pairs should have high similarity, mismatched pairs should have low similarity. Trained on 400M image-text pairs from the internet.

**Use cases:** Zero-shot classification (no task-specific training needed), image search, multimodal AI. **Relevance to AIMonk:** Building flexible CV systems that can handle new categories without retraining.

### Q49. How do you handle class imbalance in object detection?
**A:**
1. **Focal Loss** (used in RetinaNet): Down-weights easy negatives, focuses on hard examples
   - FL = -α(1-p)^γ · log(p), where γ=2 focuses on hard examples
2. **Hard negative mining:** Select the most difficult negative examples for training
3. **Oversampling:** Repeat images with rare objects
4. **Data augmentation:** Create synthetic variations of rare classes
5. **SMOTE for detection:** Paste rare objects onto backgrounds synthetically

### Q50. What is Feature Pyramid Network (FPN)?
**A:** FPN builds multi-scale feature maps for detecting objects of different sizes:
```
                    Predict small objects
Top-down pathway:  ← P5 (1/32 scale, 256ch)
                    ← P4 (1/16 scale, 256ch) + upsampled P5
                    ← P3 (1/8 scale, 256ch)  + upsampled P4
                    ← P2 (1/4 scale, 256ch)  + upsampled P3
                    Predict large objects

Bottom-up pathway (ResNet):
C2 → C3 → C4 → C5
```
**Key insight:** Low-level features (high resolution) detect small objects, high-level features (semantic) detect large objects. FPN combines both. Used in Faster R-CNN, Mask R-CNN, and YOLO v3+.

---

# PART 4: CRITICAL SITUATIONAL QUESTIONS

> The HR said this round is MOSTLY situational. Prepare these answers well.

---

### Q51. "Tell me about a time you were stuck on a technical problem for days. What did you do?"
**A (STAR):**
- **Situation:** In my license plate project, accuracy was stuck at 75% despite trying multiple model architectures for 4 days
- **Task:** I needed to reach at least 80% for the project to be viable
- **Action:** I stopped changing the model and instead systematically analyzed failures. I created a spreadsheet categorizing each failure: lighting (35%), angle (25%), occluded plates (20%), non-standard fonts (20%). This revealed the problem was preprocessing, not the model. I implemented adaptive thresholding and lighting-specific preprocessing pipelines
- **Result:** Accuracy jumped to 83%. Lesson: when stuck, analyze the errors — don't blindly try things

### Q52. "You're given a project with unclear requirements. How do you proceed?"
**A:**
1. **Ask specific questions:** "What does success look like? What's the accuracy target? What's the latency requirement?" — not vague questions
2. **Document assumptions explicitly:** Shared doc or email saying "I'm assuming X, Y, Z — please correct me if wrong"
3. **Build the smallest possible prototype first** — it forces clarity because stakeholders react to concrete work more than abstract specifications
4. **Show progress early** — demo after 2-3 days, not 2-3 weeks. Course-correct based on feedback
5. **Real example:** In NyayaConnect, the initial requirement was "build an AI legal assistant." I clarified: what types of questions? Which laws? What response time? This turned a vague idea into specific, implementable features

### Q53. "How do you handle a situation where you're falling behind on a deadline?"
**A:**
1. **Communicate immediately** — don't wait until the deadline to say "it's not ready"
2. **Assess what's left and provide a realistic new ETA** — "I need 3 more days because X took longer than expected due to Y"
3. **Propose solutions:**
   - "I can deliver a reduced scope by the original date" (what can we cut?)
   - "I can bring in help if someone can take task Z"
   - "I need 2 extra days for the full scope"
4. **Learn for next time:** What was the estimation gap? (usually: underestimating data issues or integration complexity)

### Q54. "Your model works great in testing but a client reports it's failing in production. Walk me through your debugging process."
**A:**
```
Hour 1: TRIAGE
- Get specific examples of failures from the client
- Check production logs for errors and input data
- Compare production input distribution with training data

Hour 2-4: DIAGNOSE
- Run the failing examples through the model locally — can I reproduce?
- Check preprocessing pipeline — is it identical to training?
- Check model version — is the right model deployed?
- Check data pipeline — is data being corrupted in transit?

Hour 4-6: FIX
- If data distribution shift → communicate timeline for retraining
- If bug in pipeline → hotfix and deploy
- If model version mismatch → deploy correct version

Always: Document the root cause and add monitoring to catch it earlier next time
```

### Q55. "How do you prioritize between writing tests vs shipping features quickly?"
**A:** They're not in conflict — untested code ships "fast" but creates tech debt that slows you later. My approach:
- **Always test:** Critical paths, data preprocessing, model input/output shapes
- **Sometimes skip:** UI tweaks, one-off scripts, exploration code
- **Rule of thumb:** If the code will be production or if someone else will modify it — test it
- At AIMonk, I'd follow the team's testing culture and contribute to raising the bar

### Q56. "Tell me about a time you had to make a decision with incomplete information."
**A:** When choosing the embedding model for my RAG pipeline — there were dozens of options (sentence-transformers, OpenAI embeddings, E5, etc.) with no clear winner for my use case. I didn't have time for exhaustive benchmarking.
- **What I did:** Picked 3 representative candidates, tested on 50 real queries (not benchmarks), measured retrieval quality with a quick relevance score
- **Decision:** Chose sentence-transformers/all-MiniLM-L6-v2 — best balance of quality and speed
- **Outcome:** It performed well. Later validation on the full dataset confirmed the choice was right
- **Lesson:** Make decisions quickly with enough information, not all information. Perfect is the enemy of done

### Q57. "A senior engineer disagrees with your approach. They want to use method A, you believe method B is better. What do you do?"
**A:**
1. **Listen first** — they may have context I lack (production experience, past failures with method B)
2. **Present my reasoning with evidence** — "I believe B is better because: data point 1, benchmark 2, paper 3"
3. **Propose a test** — "Can we prototype both in a timeboxed experiment?"
4. **If they still prefer A, commit to it** — disagree and commit. They're senior for a reason. Executing well on a "wrong" approach is better than poorly executing the "right" one
5. **Never backstab** — "I told you so" destroys trust

### Q58. "How do you handle criticism of your code in a review?"
**A:** I welcome it — code review is one of the best learning opportunities. When I get feedback:
1. I assume positive intent — the reviewer is trying to improve the code, not attack me
2. If I disagree, I explain my reasoning (code comments in the PR), not ego
3. If they're right (usual case), I thank them, fix it, and learn for next time
4. I specifically look for patterns in feedback — if multiple reviewers mention the same thing, it's a skill gap to address

### Q59. "You discover another team's code has a bug that affects your work. What do you do?"
**A:**
1. **Verify it's actually a bug** — maybe I'm using their API incorrectly
2. **Document the issue clearly** — reproduction steps, expected vs actual behavior, impact
3. **Report it to the team** — message/Slack, then JIRA ticket. Be respectful, not accusatory
4. **Propose a workaround** in my code while they fix it — don't block on their timeline
5. **Offer to help** — "I've debugged it to this function, happy to submit a PR if that helps"

### Q60. "How would you explain your license plate recognition project to a client who has no technical background?"
**A:** "Imagine you're at a toll booth. Our system is like having a very focused worker who looks at every car, finds the license plate, and reads the number — but does it in less than a second. 

It works in three steps: First, it finds where the plate is in the photo (like your eye finding text on a page). Then it cleans up the image so the text is clear (like adjusting brightness on your phone). Finally, it reads the characters one by one.

Right now, it reads correctly 83% of the time. The remaining 17% are mostly nighttime conditions and badly angled plates — and we have a clear plan to improve those."

---

## Quick Reference: Top 10 Things to Review Night Before

1. **Python mutable default arguments bug** (Q2)
2. **Training loop: forward → loss → backward → step → zero_grad** (Q16)
3. **Conv output size formula** (Q29)
4. **ResNet skip connections — why they work** (Q35)
5. **IoU calculation** (Q39)
6. **mAP explanation** (Q40)
7. **BGR vs RGB** (Q44)
8. **Transfer learning code** (Q46)
9. **STAR format for situational** (Q51-60)
10. **Your license plate project pipeline end-to-end** (Q43)
