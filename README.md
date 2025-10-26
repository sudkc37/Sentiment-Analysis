 <div align="center">
  <h2>FinBERT Sentiment Analysis: Mathematical Foundations</h2>
  <p><i>A Deep Dive into Transformer Architecture and Financial NLP</i></p>
  <p><i>Sudip Khadka</i></p>
</div>

<hr>

<h3>1. Introduction</h3>
<p>
  Every day, thousands of financial news articles flood the market. Traders need to quickly gauge: Is the sentiment positive, negative, or neutral? Machine learning classifiers can automate this task but financial text presents unique challenges such as domain-specific jargon, context-dependent phrases like "beating expectations," and words like "sell" that shift meaning based on context. 
Traditional approaches like Naive Bayes or logistic regression provide a solid foundation but they struggle with the complexity of financial language. This is where FinBERT comes in. FinBERT is a specialized BERT model pre-trained on financial text that captures these contextual subtleties. This project explores both the implementation of FinBERT and the mathematical intuition behind transformer-based models, revealing how modern NLP transforms raw text into actionable sentiment predictions. To demonstrate its real world applicability, I built an end-to-end pipeline that automatically collects financial news and articles from Yahoo Finance. The system is complemented by an interactive front end featuring a semantic meter that dynamically visualizes market sentiment for any chosen company.
</p>




<p align="center">
<img src="https://github.com/user-attachments/assets/f17aaa86-e154-43d7-acc5-66522ac4a71c" alt="Screenshot 2024-12-09 at 2 17 15 PM" width="950" height="400">
 <br>
  <em>Figure: Market Sentiment Meter with Key Real-Time Financial Indicators”</em>
</p>



<h2>2. Transformer Architecture: Mathematical Foundation</h2>

The Transformer architecture forms the backbone of today’s Large Language Models (LLMs). It is built two key components that are  encoder and the decoder which work together to process information effectively. It is Designed for handling sequential data to capture complex relationships and contextual meaning across long sequences. Encoder transforms an input sequence such as words into numerical vectors that capture the underlying meaning and contextual relationships within the data. The architecture is composed of multiple identical layers with each layer containing a self-attention mechanism and a feed-forward network to refine these representations.

<p align="center">
<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="FinBERT Diagram" width="500" height="500">
 <br>
  <em>Figure: Transformer architecture from the paper “Attention is All You Need”</em>
</p>




<h3>2.1 Self-Attention Mechanism</h3>
<p>
The core innovation of transformer models is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when processing each word. The attention mechanism computes a weighted sum of values based on the similarity between queries and keys. The self-attention mechanism is defined as
</p>

 
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$


Where:<br>
• Q (Query) = input sequence transformed by weight matrix W<sub>Q</sub><br>
• K (Key) = input sequence transformed by weight matrix W<sub>K</sub><br>
• V (Value) = input sequence transformed by weight matrix W<sub>V</sub><br>
• dₖ = dimension of key vectors (used for scaling)<br>
• softmax normalizes attention weights to sum to 1
</div>

<p>
The scaling factor √dₖ prevents the dot products from growing too large in magnitude, which would push the softmax into regions with extremely small gradients. This scaling is crucial for stable training.
</p>

<h3>2.2 Multi-Head Attention</h3>
<p>
Instead of using a single attention function, transformer models employ multiple attention heads that can learn different aspects of the relationships between words. Each head uses different learned linear projections of the queries, keys, and values.
</p>


$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W_O
$$

$$
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

<div>
• h = number of attention heads (typically 8 or 12)<br>
• W<sub>O</sub> = output projection matrix<br>
• Each head learns different attention patterns
</div>

<h3>2.3 Positional Encodings</h3>
<p>
Transformers process all positions simultaneously (unlike recurrent networks) therefore they need explicit position information. BERT uses learned position embeddings, while the original Transformer used sinusoidal position encodings. Sinusoidal Position Encodings are:
</p>

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}}\right)
$$

Where:<br>
• pos = position in the sequence<br>
• i = dimension index<br>
• d<sub>model</sub> = model dimension (typically 768 for BERT-base)
</div>

<h2>3. BERT Architecture and Pre-training</h2>

<h3>3.1 Bidirectional Context</h3>
<p>
Unlike traditional language models that process text left to right or right to left, BERT uses bidirectional context by predicting masked tokens based on both left and right context. This is formalized through the Masked Language Modeling (MLM) objective:
</p>

$$
P(x_i \mid x_{-i}) = \text{softmax}(W \cdot h_i + b)
$$

<div>
Where:<br>
• xᵢ = masked token at position i<br>
• x₍₋ᵢ₎ = all other tokens in the sequence<br>
• hᵢ = hidden state representation of position i<br>
• W and b = learned parameters for the MLM head
</div>

</p>
During pre-training, approximately 15% of input tokens are masked, and the model learns to predict them based on the surrounding context. This forces the model to develop deep bidirectional representations.
</p>

<h3>3.2 Model Architecture Specifications</h3>

<table>
  <tr>
    <th>Parameter</th>
    <th>BERT-Base</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Layers (L)</td>
    <td>12</td>
    <td>Number of transformer encoder layers</td>
  </tr>
  <tr>
    <td>Hidden Size (H)</td>
    <td>768</td>
    <td>Dimension of hidden states</td>
  </tr>
  <tr>
    <td>Attention Heads (A)</td>
    <td>12</td>
    <td>Number of self-attention heads</td>
  </tr>
  <tr>
    <td>Feed-forward Size</td>
    <td>3072</td>
    <td>Dimension of feed-forward layer (4×H)</td>
  </tr>
  <tr>
    <td>Total Parameters</td>
    <td>~110M</td>
    <td>Total trainable parameters</td>
  </tr>
  <tr>
    <td>Vocabulary Size</td>
    <td>30,522</td>
    <td>WordPiece vocabulary size</td>
  </tr>
  <tr>
    <td>Max Sequence Length</td>
    <td>512</td>
    <td>Maximum input token length</td>
  </tr>
</table>




<h2>4. Fine-tuning for Sentiment Classification</h2>
<h3>4.1 Classification Objective</h3>
<div class="paragraph">
            For sentiment classification, we add a classification layer on top of BERT's [CLS] token representation. The model is fine-tuned using cross-entropy loss. The cross-entropy loss is given by:

$$
L = - \sum_{i=1}^{C} y_i \, \log(\hat{y}_i)
$$

where:
- $C$ = number of classes (e.g., 3 for positive/negative/neutral)  
- $y_i$ = true label (one-hot encoded)  
- $\hat{y}_i$ = predicted probability for class $i$  
- The loss measures the divergence between predicted and true distributions  

<h3>4.2 Fine-tuning Proces</h3>
<div class="paragraph">
            Starting from pre-trained weights, we optimize the model parameters through gradient descent. The fine-tuning objective can be expressed as:
</div>

$$
\theta^* = \arg\min_{\theta} L(\theta; D_\text{financial})
$$

using gradient descent update rule yields;

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)
$$

- $\theta$ = model parameters  
- $\eta$ = learning rate (typically $2\times10^{-5}$ to $5\times10^{-5}$ for BERT)  
- $\nabla_\theta L$ = gradient of loss with respect to parameters  
- $D_\text{financial}$ = financial sentiment dataset  

<h2>5. Optimization Strategy</h2>
<h3>5.1 AdamW Optimizer</h3>
<div class="paragraph">
            The finBERT uses AdamW (Adam with decoupled weight decay), which combines adaptive learning rates with proper weight decay regularization. The update rules are:
</div>


$$
\begin{aligned}
m_t &= \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t &\text{(first moment)}\\
v_t &= \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2 &\text{(second moment)}\\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} &\text{(bias-corrected first moment)}\\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} &\text{(bias-corrected second moment)}
\end{aligned}
$$

where:
- $g_t$ = gradient at time step $t$  
- $\beta_1 = 0.9$ (first moment decay rate)  
- $\beta_2 = 0.999$ (second moment decay rate)  
- $\epsilon = 1\times 10^{-8}$ (numerical stability term)  
- $\lambda$ = weight decay coefficient (typically 0.01)  
- $\eta$ = learning rate 


<h3>5.2 Learning Rate Scheduling</h3>

<div>
The training employs a learning rate schedule with linear warm-up followed by linear decay. This helps stabilize early training and gradually reduces the learning rate. Learning Rate Schedule is given as:
 </div>

For warm-up phase:

$$
\text{step} \le \text{warmup\-steps}: \quad
\eta(\text{step}) = \eta_\text{max} \times \frac{\text{step}}{\text{warmup\-steps}}
$$

For Decay phase:

$$
\text{step} > \text{warmup\-steps}: \quad
\eta(\text{step}) = \eta_\text{max} \times \frac{\text{total\-steps} - \text{step}}{\text{total\-steps} - \text{warmup\-steps}}
$$

<div class="equation-block">
            Where:<br>
            • η_max = maximum learning rate (e.g., 2e-5)<br>
            • warmup_steps = number of warm-up steps (typically 10% of total)<br>
            • total_steps = total number of training steps
 </div>

 <h3>5.3 Gradient Clipping</h3>
<div class="paragraph">
            To prevent exploding gradients, gradient clipping is applied:
</div>

$$
\text{If } \|\nabla_\theta\| > \text{max\-norm}, \quad
\nabla_\theta = \nabla_\theta \times \frac{\text{max\-norm}}{\|\nabla_\theta\|}
$$


<div>
            Where:<br>
            • ||∇θ|| = L2 norm of gradient<br>
            • max_norm = clipping threshold (typically 1.0)
</div>

<h2>6. Evaluation Metrics</h2>
<h3>6.1 Classification Metrics</h3>
The model performance is evaluated using multiple metrics that capture different aspects of classification quality.

Key Metrics:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

</div>
           Where:<br>
            • TP = True Positives<br>
            • TN = True Negatives<br>
            • FP = False Positives<br>
            • FN = False Negatives
</div>


<h3>6.2 Multi-class Extensions</h3>
For multi-class sentiment classification (positive, negative, neutral), we use macro and weighted averaging.

Macro-averaged F1:

$$
F1_\text{macro} = \frac{1}{C} \sum_{i=1}^{C} F1_i
$$

Weighted F1:

$$
F1_\text{weighted} = \sum_{i=1}^{C} \frac{n_i}{N} \, F1_i
$$

where;

- $C$ = number of classes  
- $F1_i$ = F1 score for class $i$  
- $n_i$ = number of samples in class $i$  
- $N$ = total number of samples




**Note: This article is for educational purposes only and is not intended to provide financial advice. It aims to demonstrate how language models can be utilized in financial applications.**



**Deployment:**

- Create a docker file

- Build a docker image:
 
    - docker build -t username/appname:latest .

- Check locally before pushing it to docker hub:
  
    - docker container run -d -p port:imageport username/appname:latest

- Verify:
  
    - docker container ls

localhost:yourport

- Push image to docker hub:
  
    - docker push username/appname


**Note: You can run the application locally. The app makes several API requests and has a size larger than the free deployment limits provided by some cloud platforms.**
