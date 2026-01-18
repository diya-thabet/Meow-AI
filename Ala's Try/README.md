# **🎭 Multimodal Facial Emotion Recognition (Vision-LLM & XAI)**

A deep learning system that goes beyond simple classification. This project not only **predicts** compound facial emotions but also **visually explains** its decision (XAI) and **verbally describes** the facial features using a Vision-LLM.

## **🚀 Project Overview**

Standard emotion recognition models give you a label (e.g., "Happy"). This project implements a **3-Stage Pipeline** to create a "Holistic Vision System":

1. **The Brain:** Classifies the specific compound emotion (e.g., "Happily Surprised").  
2. **The Eye:** Visualizes which parts of the face triggered the decision (Interpretability).  
3. **The Voice:** Generates a natural language description of the facial state (Open/Closed mouth, Eyes, etc.).

---

## **🏗️ Architecture**

The system is built on three distinct modules working in parallel:

| Component | Role | Model Used | Function |
| :---- | :---- | :---- | :---- |
| **🧠 The Brain** | **Classification** | **ResNet-18** | Serves as the Vision Baseline. Extracts features and predicts one of **14 Compound Emotions** (RAF-DB). |
| **👁️ The Eye** | **Explainability (XAI)** | **Grad-CAM** | Generates Heatmaps to validate anatomical focus (e.g., verifying the model looks at the mouth for "Surprise"). |
| **🗣️ The Voice** | **Vision-LLM** | **BLIP-VQA** | A Visual Question Answering model that answers prompts like *"Describe the state of the eyes"* or *"What is the facial expression?"* |

---

## **📊 Dataset: RAF-DB (Compound)**

We utilized the **Real-world Affective Faces Database (RAF-DB)**, specifically the **Compound Emotions** subset. Unlike basic datasets (Happy/Sad), this dataset contains complex emotional states:

* **14 Classes:** *Happily Surprised, Happily Disgusted, Sadly Fearful, Angrily Surprised, etc.*  
* **Challenge:** High inter-class similarity makes this a difficult classification task compared to basic emotions.

---

## **📈 Results & Performance**

### **1\. Classification (ResNet-18)**

* **Training Accuracy:** \~99% (Model successfully learned the dataset features).  
* **Test Accuracy:** \~51-52%.  
  * *Note:* While seemingly low compared to basic emotion tasks (usually \~80%), 52% is significantly higher than random chance (7%) for 14 complex compound classes. The gap indicates overfitting due to the dataset size, which is common in deep learning on smaller datasets.

Confusion Matrix:

(Insert your Heatmap image here)

\!\[Confusion Matrix\](images/confusion\_matrix.png)

### **2\. Visual Explanation (Grad-CAM)**

We used Grad-CAM to inspect the final convolutional layer of ResNet-18.

* **Observation:** The heatmap successfully focuses on key facial landmarks (lips, eyes, nasolabial folds) rather than the background.

(Insert a Grad-CAM example image here)

\!\[Grad-CAM Result\](images/gradcam\_example.png)

### **3\. Textual Description (BLIP)**

Using **Salesforce BLIP-VQA**, the system answers specific questions about the face:

* *Q: "What is the state of the mouth?"* \-\> **A: "Open"**  
* *Q: "What is the emotion?"* \-\> **A: "Surprise"**

---

## **🛠️ Tech Stack**

* **Deep Learning:** PyTorch, Torchvision  
* **Models:** ResNet-18 (Pretrained & Fine-tuned), BLIP (Hugging Face Transformers)  
* **Visualization:** Matplotlib, Seaborn, Grad-CAM (pytorch-grad-cam)  
* **Environment:** Google Colab (T4 GPU)

## **💻 How to Run**

1. **Install Dependencies:**  
```

pip install torch torchvision transformers accelerate bitsandbytes grad-cam
```
 
2. **Load the Model:**  
```
model \= models.resnet18(weights=None)  
model.fc \= nn.Linear(model.fc.in\_features, 14)  
model.load\_state\_dict(torch.load('best\_model\_night.pth'))
``` 
3. Inference:  
   Pass an image through the pipeline to get the Class Label (ResNet), the Heatmap (Grad-CAM), and the Description (BLIP).

---

## **🔮 Future Improvements**

* **Data Augmentation:** To reduce overfitting and bridge the gap between Train (99%) and Test (52%) accuracy.  
* **Large Multi-Modal Models (LMM):** Replace BLIP with **LLaVA** or **GPT-4o** for more nuanced psychological analysis of the facial expressions.  
* **Real-Time Inference:** Optimization for webcam feed processing.

---

### **👤 Authors**

* **\[Ala Eddine Madani\]** \- *My try at Meow AI Project*

