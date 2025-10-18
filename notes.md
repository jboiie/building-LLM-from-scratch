# Build LLMs from Scratch – Detailed Notes (20-Minute Recap, Vizuara)

## Introduction and Philosophy
- **Focus:** Teaching fundamental engineering and research principles in ML/LLMs, not just running applications.
- **Reference Book:** Closely follows *Build a Large Language Model from Scratch* (Manning Publications, Sebastian Raschka).
- **Goal:** Enable students to *understand, build, and innovate* LLMs from the ground up.
- **Approach:** Deep-dive, step-by-step coding and theory, with whiteboard explanations and hands-on projects.

---

## Stages of Building an LLM

### 1. Stage One: Foundation
- **Data Preparation and Sampling**
  - Gather large datasets (documents, sentences).
  - Tokenize texts into tokens → token IDs.
- **Preprocessing Pipeline**
  - Token IDs → Token Embeddings (high-dimensional vector space, e.g., dimension 768 for GPT-2).
  - Add Positional Embeddings (to capture token order/position information).
  - **Result:** Input Embeddings (token embeddings + positional embeddings).
- **Objective:** Transform raw text into model-ready input embeddings.

---

### 2. Stage Two: Attention Mechanism (Engine of LLMs)
- **Purpose:** Capture relationships and context between all tokens in a sequence.
- **Steps:**
  - Convert input embeddings to context vectors using Query, Key, and Value matrices (all trainable).
  - Compute Attention Scores: \( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \)
    - Scale by sqrt of key dimension.
    - Apply softmax for normalization.
    - Apply mask (for next-token prediction tasks).
  - Combine multiple Attention Heads → context vector matrix (for long-range and parallel dependencies).
- **Output:** Richer context vectors (embedding both semantic meaning & contextual relationships).

---

### 3. Stage Three: LLM Architecture

#### Transformer Block Composition
- **Components Inside Each Transformer Block:**
  - **Input Layers:** Token embedding, positional embedding, dropout.
  - **Transformer Core:** 
    - Layer normalization
    - Multi-head masked self-attention
    - Dropout
    - Shortcut/residual connections
    - Feed-forward neural network
    - Additional normalization and dropout
- **Stacking:**
  - E.g., GPT-2: 12–24 transformer blocks, each with multiple (12–24) attention heads.
- **Final Layers:**
  - Layer normalization
  - Output neural network → logits for token prediction.

#### Full Model Flow
```
Raw Text → Tokenization → Token IDs → Token Embedding → Positional Embedding → Input Embedding
→ [Transformer Blocks × N]
→ Output Logits → Predicted Next Token
```

---

## Training: Pre-Training Loop

1. **Forward Pass:**
   - Feed sequences into the model, get predictions for next tokens.
   - Compute loss (typically cross-entropy between predicted and actual next token).

2. **Backward Pass / Gradient Calculation:**
   - Calculate gradients for every trainable parameter: token embeddings, positional embeddings, normalization, query/key/value matrices, feed-forward layers, output layers.
   - Example parameter update:  
     \( w_{i+1} = w_i - \alpha \frac{\partial \text{Loss}}{\partial w} \)
     (\(\alpha\) = learning rate)

3. **Update Weights:**
   - Vanilla gradient descent or Adam/AdamW.

4. **Scale and Feasibility:**
   - Full LLM pre-training (GPT-2+) requires large-scale compute (costs $1M+), but small-scale pre-training and architecture understanding are feasible on personal hardware.

---

## Next Steps: Loading Pretrained Weights & Token Prediction

- Can use public pretrained weights (e.g., GPT-2) to skip heavy pre-training.
- Run model inference for next-token prediction using custom or pre-trained weights.

---

## Fine-Tuning (Personalization & Specialization)

### Projects Built:
- **Email Spam Classifier:** LLM fine-tuned on email classification (spam/non-spam).
- **Instruction-following Chatbot:** Fine-tuned LLM to act as an assistant (e.g., converting active to passive sentences).

#### Methods:
- Prepare domain/task-specific datasets.
- Run training/fine-tuning loops using base LLM code.

---

## Evaluation Methods

1. **MMLU Benchmark:**
   - *Massive Multi-task Language Understanding*
   - 57 test sets; evaluates broad LLM capabilities.

2. **Human Evaluation:**
   - Manual comparison of outputs versus ground truth answers.

3. **LLM-Based Evaluation:**
   - Use strong models (e.g., Llama-3-8B Instruct) to score model outputs.
   - Assigns numeric score (0–100) based on similarity to the true output.

---

## Practical Coding & Research Approach

- **Codebase:** Modular, editable building blocks.
- **Suggested Experiments:**
  - Modify hyperparameters (learning rate, depth, number of blocks).
  - Try different optimizers.
  - Change evaluation methods.
  - Research alternatives for architecture and process.


## Final Takeaways

- *Understand* the nuts and bolts—become an LLM researcher/engineer, not just a user.
- The course and recap make you strong in engineering and scientific thinking around LLMs.
- Next steps: Dive into research, innovate, and contribute to advancing LLM technology.

---

> *“The real revolution in LLMs is understanding and building every stage yourself—from text to embeddings, attention to architecture, training to evaluation.”* [1]


