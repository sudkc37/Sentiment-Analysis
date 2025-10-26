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






<h2>2. Transformer Architecture: Mathematical Foundation</h2>

The Transformer architecture forms the backbone of today’s Large Language Models (LLMs). It is built two key components that are  encoder and the decoder which work together to process information effectively. It is Designed for handling sequential data to capture complex relationships and contextual meaning across long sequences. Encoder transforms an input sequence such as words into numerical vectors that capture the underlying meaning and contextual relationships within the data. The architecture is composed of multiple identical layers with each layer containing a self-attention mechanism and a feed-forward network to refine these representations.

<p align="center">
<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="FinBERT Diagram" width="500" height="500">
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



























# Leveraging Large Language Model (LLM) for Sentiment Analysis

**Note: This article is for educational purposes only and is not intended to provide financial advice. It aims to demonstrate how language models can be utilized in financial applications.**

This article offers a comprehensive understanding of how fine-tuned language models can be leveraged with advanced data science tools, including web scraping, visualization, modeling, and APIs, to build real-world financial applications. These applications aim to facilitate effective and informed decision-making.

The resulting application features a dashboard with a user-friendly interface that includes:

- Key Real-Time Financial Indicators
- Market Sentiment Meter
- Price Chart of a Selected Portfolio Ticker
  <img width="1438" alt="Screenshot 2024-12-09 at 2 17 15 PM" src="https://github.com/user-attachments/assets/f17aaa86-e154-43d7-acc5-66522ac4a71c">
<img width="1438" alt="Screenshot 2024-12-09 at 2 17 34 PM" src="https://github.com/user-attachments/assets/a02f5b89-8fd0-4dbc-9695-fc12a9f088bf">


The article also provides a simple breakdown and architecture of the model, illustrating the workflow and integration of various components. 

**Scraper.py**

It uses a Yahoo Finance API with a built-in scraping method to extract the news feed from the Yahoo Finance webpage. The API returns financial news as an RSS feed in XML format, fetching the latest financial news published on Yahoo Finance in real-time. We are only using the summary content of each news feed. For simplicity, I have directly fed the summary content into our language model (FinBERT). However, to achieve more robust analysis, we could further leverage large language models (LLMs) like GPT to summarize the news before feeding it into our custom language model. That said, using GPT incurs additional costs.

**LanguageModel.py**

This module incorporates FinBERT, a specialized Natural Language Processing (NLP) model fine-tuned from the pre-trained BERT model. FinBERT is specifically designed for analyzing sentiment in financial texts. It has been further trained on a large corpus of financial data, making it highly effective for sentiment classification tasks within the financial domain.
The AutoTokenize, automatically loads the appropriate tokenizer for the FinBERT model. The tokenizer converts raw text into input tokens that the model can process. Then AutoModelForSequenceClassification loads the pre-trained FinBERT model tailored for sequence classification tasks such as sentiment analysis. And finally,The  embedding  function returns both the tokenizer and model, making them accessible for downstream tasks like processing financial text and performing sentiment analysis.

**SentimentScore.py**

It analyzes financial news summaries to determine overall market sentiment. It uses our  language model to classify each news summary into Positive, Neutral, or Negative categories based on the content. The classifications are stored in the dataframe, and the most frequent sentiment (mode) across all summaries is identified as the overall market sentiment. This provides a quick, automated way to gauge market sentiment from financial news.

**Detail.py**

The ticker detail function retrieves detailed financial information about a specific stock ticker using the “yfinance” library. It fetches various key metrics such as current price, open price, bid/ask prices, market cap, and industry, organizing them into a dictionary. Leveraging “yfinance.Ticker.info”, we can further model other key indicators like moving averages (MA), stochastic oscillation, implied volatility, gamma, and vega for advanced financial analysis.

**Graph.py**

The plot price function generates a time-series plot of minute-by-minute adjusted closing prices for a given stock ticker over the past day. It calculates the time range by setting the end date to the current date and time and the start date to one day earlier, ensuring it fetches intraday data using an interval of one minute. Using “yfinance”, it retrieves this high-frequency data, and matplotlib is used to create and save the plot, providing a detailed visualization of short-term price movements.

**app2.py**

The application allows users to input a company's stock ticker to analyze recent news articles, assess market sentiment, and access detailed company information alongside a graph of the stock's price trends. By leveraging multiple modules, it provides a streamlined solution for stock market insights.

Under the hood, the app uses a news scraper to fetch the latest articles about the company. A language model then performs sentiment analysis on the scraped content, categorizing the sentiment as positive, neutral, or negative. Additional modules retrieve stock-specific details and generate a graph of the stock's recent price movements. This combination of tools ensures that the app delivers a comprehensive overview of the company's market position.
When users open the app, it displays analysis for a default stock ticker. They can enter a different ticker to dynamically update the sentiment analysis, company details, and stock graph. The results include a sentiment score with a color-coded indicator (green for positive, yellow for neutral, and red for negative), a summary of company-specific data, and a visual representation of the stock price trends.

The application includes robust error handling to manage scenarios like missing data or processing failures. In such cases, placeholders like "N/A" are displayed, ensuring the app remains functional and user-friendly. Additionally, the app dynamically processes user inputs without requiring page reloads, enhancing the overall interactivity.
Built with the Flask framework, the application is designed to run on a server, providing an accessible and efficient web-based interface for users. This architecture makes it well-suited for real-time stock market analysis and sentiment assessment.


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
