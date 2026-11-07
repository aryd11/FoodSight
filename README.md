# 🍲 FoodSight: Food-101 Classification & Calorie Tracker

FoodSight is an end-to-end Deep Learning application that identifies 101 different types of food from a photo and estimates its calorie content using the USDA Food Database. 

🔗 **Live Demo:** [https://foodsight-track.streamlit.app]

## 📌 Problem Statement
The **Food-101** dataset is a challenging Computer Vision benchmark. It contains 101,000 images across 101 food categories. The main challenge is that many food classes are visually identical (e.g., `steak` vs. `filet_mignon`, or `chocolate_mousse` vs. `chocolate_cake`). Furthermore, identifying the food is only half the problem; users want to know the nutritional value of their meal.

## 🧠 Model & Approach
*   **Architecture:** EfficientNet-B3 (pre-trained on ImageNet-1k).
*   **Training Strategy:** Gradual Unfreezing in 3 phases:
    1.  Train only the classifier head (Feature Extraction).
    2.  Unfreeze the last convolutional block (`blocks[6]`).
    3.  Unfreeze the middle block (`blocks[5]`).
*   **Data Augmentation:** `RandomResizedCrop`, `RandomHorizontalFlip`, and `RandAugment` to prevent overfitting.
*   **Regularization:** Label Smoothing (0.1), Dropout (0.4), and BatchNorm Freezing to preserve ImageNet statistics.

## 📊 Results
The model was evaluated on the **Official Food-101 Test Set** (25,250 unseen images).

| Model | Parameters | Validation Accuracy | Test Accuracy |
|-------|------------|---------------------|---------------|
| ResNet50 (Baseline) | 25.6M | 74.90% | - |
| **EfficientNet-B3** | **12M** | **77.50%** | **82.21%** |

*EfficientNet-B3 outperformed the baseline ResNet50 while using less than half the parameters.*

### 🔍 Confusion Matrix Insights
Analysis of the confusion matrix revealed that the model rarely makes random mistakes. The remaining errors are due to visual ambiguity:
*   `steak` is often confused with `filet_mignon` or `pork_chop`.
*   `tuna_tartare` is confused with `beef_tartare`.
*   `chocolate_mousse` is confused with `chocolate_cake`.

## ⚙️ Features
1.  **Food Classification:** Upload an image to get the Top-3 predictions with confidence scores.
2.  **Calorie Estimation:** Integrates with a local USDA dataset to calculate calories based on predicted food and portion size.
3.  **Human-in-the-Loop:** If the prediction is wrong, users can correct it. The image is automatically saved into a `retraining_data` folder for future model fine-tuning.
4.  **Meal History:** Logs daily meals and allows downloading the history as a CSV file.

## 💻 Tech Stack
*   **Deep Learning:** PyTorch, timm
*   **Web App:** Streamlit
*   **Data Processing:** Pandas, NumPy
*   **Dataset:** Food-101, USDA FoodData Central

