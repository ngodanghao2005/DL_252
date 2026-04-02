# Image Dataset Classification

**Group:** LTH (252)  
**Course:** Deep Learning (CO3021)  
**Institution:** Ho Chi Minh City University of Technology (HCMUT)

---

## 1. Dataset Exploration & Preprocessing

### 📊 Dataset Overview: CIFAR10

The project utilizes the **CIFAR10 (Canadian Institute For Advanced Research - 10)** dataset, a standard benchmark for image classification tasks. The primary objective is to evaluate and compare the performance of CNN and Vision Transformer (ViT) architectures.

- **Data Source:** [Hugging Face - CIFAR10 Dataset](https://huggingface.co/datasets/uoft-cs/cifar10)
- **Problem Type:** Image classification

#### Key Statistics:

| Metric                      | Value                              |
| :-------------------------- | :--------------------------------- |
| **Total Samples**           | 60,000 (50,000 train + 10,000 test)|
| **Unique Classes (Genres)** | 10                                 |
| **Modalities**              | Visual Information (Image)         |

#### Data Characteristics:

- **Visual Data:** The `img` attribute consists of color images with a native resolution of $32 \times 32$ pixels, representing 10 distinct object categories.
- **Label Distribution:** The dataset is perfectly balanced, with each of the 10 classes containing exactly 6,000 samples (5,000 for training and 1,000 for testing).
- **Color Profile:** Pixel intensity analysis shows a non-uniform distribution across channels, specifically a high frequency in the Blue channel (~70 intensity), likely due to sky or water backgrounds.

---

### 🖼️ Dataset Preview

The MMIMDb dataset structure combines textual content with visual imagery:

![Dataset Sample Preview](./public/multimodal/dataset.png)  
_Figure 1: Preview of the MMIMDb dataset structure displaying plot summaries and corresponding movie posters._

---

## 2. In-depth Exploratory Data Analysis (EDA)

To understand the complexity of the MMIMDb dataset, we conducted a comprehensive analysis focusing on label distribution, multi-label characteristics, and textual features.

### 📊 Genre Distribution & Class Imbalance

Analysis of the 26 unique genres reveals a significant **Long-tail distribution**. Dominant genres comprise the majority of the dataset, while niche genres like _Film-Noir_ have significantly fewer samples, presenting a challenge for model convergence.

![Genre Distribution Chart](./public/multimodal/genre_distribution.png)  
_Figure 2: Distribution of samples across 26 movie genres, highlighting the class imbalance._

### 🏷️ Multi-label Characteristics

MMIMDb is inherently multi-label. Our analysis shows that most movies are associated with **2 to 3 genres** simultaneously. This overlap requires the model to capture complex relationships between different categories.

![Genres per Movie Chart](./public/multimodal/labels_per_movie.png)  
_Figure 3: Frequency of the number of labels assigned per movie._

### 🔗 Genre Correlation (Co-occurrence)

The correlation matrix visualizes how genres frequently appear together (e.g., _Action_ often co-occurs with _Adventure_). Understanding these dependencies is crucial for the **Joint Embedding** strategy.

![Co-occurrence Heatmap](./public/multimodal/correlation_matrix.png)  
_Figure 4: Heatmap illustrating the co-occurrence patterns between genres._

### 📝 Textual Feature Analysis

We analyzed the word count distribution of the movie plots. This guided our decision on the `max_length` parameter for the DistilBERT tokenizer to ensure context isn't truncated.

![Text Length Distribution](./public/multimodal/text_length.png)  
_Figure 5: Distribution of word counts in movie plot summaries._

---

[⬅️ Back to README](./README.md)
