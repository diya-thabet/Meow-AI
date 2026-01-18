<div align="center">
 

<img src="ReadmeImg/logo.png" alt="Meow AI Logo" width="200" height="auto" />

# Meow AI
### Vision-LLM for Compound Facial Emotion Recognition (FER-CE)

<p>
<img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/Docker-Enabled-2496ED.svg" alt="Docker">
<img src="https://img.shields.io/badge/Model-Vision--LLM-orange.svg" alt="Model">
<img src="https://img.shields.io/badge/Dataset-RAF--CE-green.svg" alt="Dataset">
</p>

</div>

---

## 📄 About The Project

**Meow AI** goes beyond traditional emotion recognition. While standard AI can tell if someone is just "Happy" or "Sad," humans are more complex. We often feel **Compound Emotions** (e.g., *Happily Surprised* or *Sadly Angry*).

This project uses **Vision-LLMs** (Large Vision-Language Models) to:
1.  **See** the face.
2.  **Classify** the complex emotion (14 different types).
3.  **Explain** *why* (e.g., *"The person looks happily surprised because their mouth is smiling and eyebrows are raised"*).

It unifies computer vision with linguistic reasoning to provide a deeper understanding of human facial expressions.

### 🌟 Key Features
* **Compound Emotion Recognition:** Detects 14 mixed emotional states (RAF-CE Dataset).
* **Textual Explanations:** Generates natural language descriptions of the emotion.
* **Explainable AI (XAI):** Uses Heatmaps and Grad-CAM to show which parts of the face (eyes, mouth, etc.) led to the decision.
* **Model Comparison:** Benchmarks Vision-LLMs (BLIP-2, LLaVA, Qwen-VL) against traditional models (ResNet, ViT).
* **Dockerized:** Easy to deploy and run using Docker containers.

---

## 📊 The Dataset: RAF-CE

We utilize the **RAF-CE (Real-world Affective Faces - Compound Emotions)** dataset.
* **Source:** [RAF-CE Official Website](http://whdeng.cn/RAF/model4.html)
* **Content:** 4,549 real-world images with compound labels and Action Unit (AU) annotations.

**The 14 Compound Classes:**
| ID | Emotion Label | ID | Emotion Label |
|:---:|:---|:---:|:---|
| 0 | Happily Surprised | 7 | Fearfully Surprised |
| 1 | Happily Disgusted | 8 | Fearfully Disgusted |
| 2 | Sadly Fearful | 9 | Angrily Surprised |
| 3 | Sadly Angry | 10 | Angrily Disgusted |
| 4 | Sadly Surprised | 11 | Disgustedly Surprised |
| 5 | Sadly Disgusted | 12 | Happily Fearful |
| 6 | Fearfully Angry | 13 | Happily Sad |

> **⚠️ Note:** This dataset is for non-commercial research purposes only. You must apply for a password from the authors to download the images.

---

## 🛠️ Methodology

Our pipeline consists of three main layers:

1.  **Data Preparation:** * Face detection, cropping, and alignment.
    * Data augmentation (lighting, rotation) to solve class imbalance.
2.  **Vision-LLM Training:** * We fine-tune models like **BLIP-2** or **LLaVA**.
    * **Input:** Image + Text Prompt (e.g., *"Describe the emotional state..."*).
    * **Output:** The emotion label + a text explanation.
3.  **Interpretation:** * Comparing the text explanation with visual heatmaps (faithfulness check).

---

## 🚀 Getting Started

You can run this project locally using Python or easily via Docker.

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop) (Recommended)
* **OR** Python 3.9+ and CUDA (if running locally without Docker)

### 🐳 Option 1: Run with Docker (Recommended)

We have containerized the application to ensure it works on any machine.

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/diya-thabet/Meow-AI.git
    cd Meow-Ai
    ```

2.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```
    
3.  **Access the App:**
    Open your browser and go to `http://localhost:8501` (if using the Streamlit interface).

### 💻 Option 2: Manual Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the training script:**
    ```bash
    python src/train.py --model blip2 --epochs 10
    ```

---

## 📂 Project Structure

```text
meow-ai/
├── data/                  # RAF-CE Dataset (images not included in repo)
├── docker/                # Docker configuration files
├── notebooks/             # Jupyter notebooks for EDA and testing
├── ReadmeImg/             # Images for documentation
│   └── logo.png
├── src/                   # Source code
│   ├── data_loader.py     # Preprocessing and loading
│   ├── models.py          # Vision-LLM and Baseline definitions
│   ├── train.py           # Training loop
│   └── evaluate.py        # Metrics (F1-Score, BLEU, Confusion Matrix)
├── app.py                 # Streamlit Interface (optional)
├── Dockerfile
├── requirements.txt
└── README.md
