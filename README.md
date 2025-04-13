# Brain-Stroke-Detection-using-YOLOv8-Classification
Brain Stroke Classification


This project is an AI-based GUI application that classifies brain CT images into stroke types using the **YOLOv8 classification model**.

---

## 📊 Dataset

- **Source**: [Kaggle - Brain Stroke CT Images](https://www.kaggle.com/datasets/your-dataset-link-here)  
- The dataset consists of **3 distinct classes**:
  - 🩸 **Bleeding** (Hemorrhagic Stroke)
  - 🚫 **Ischemia** (Ischemic Stroke)
  - ✅ **Normal** (No Stroke)

Images are organized into `train`, `val`, and `test` folders for model training and evaluation.

---

## 👨‍💻 About the Application

- Built with **Python + Tkinter** for a clean and interactive GUI
- Designed to be **user-friendly** and **visually minimal**
- Simply select an image → preview appears in the interface
- Prediction is displayed in a **separate popup window** with color-coded results

---

## 📂 Project Structure

| Bleeding (Hemorrhagic Stroke) | Ischemia (Ischemic Stroke) | Normal (Healthy Brain) |
|-------------------------------|-----------------------------|-------------------------|
| ![Bleeding](docs/bleeding_sample.png) | ![Ischemia](docs/ischemia_sample.png) | ![Normal](docs/normal_sample.png) |
