# 🏠 House Price Prediction with Multimodal Features

This project explores the impact of combining **structured property metadata** with **image features** extracted from various Convolutional Neural Networks (CNNs) for predicting house prices. We benchmark the performance of 21 popular CNN architectures (AlexNet, VGG, ResNet, EfficientNet, etc.) as feature extractors to identify the most effective visual representation for this task.

## 📊 Project Goal

The primary goal is to determine whether image-derived features significantly improve the predictive accuracy (measured by $R^2$ and RMSE) of a regression model compared to using structured text data alone.

## 🛠️ Setup and Prerequisites

### 1\. Environment

This project was executed in a **Google Colab** environment, leveraging its GPU access for efficient CNN feature extraction.

### 2\. Dependencies

All required libraries can be installed using the following command:

```bash
!pip install torch torchvision scikit-learn pandas matplotlib pillow tqdm
```

### 3\. Data Structure

Ensure your Google Drive is mounted correctly and the following file structure is in place, as referenced in the notebook:

| Path Variable | Location | Description |
| :--- | :--- | :--- |
| `csv_path` | `/content/drive/MyDrive/property_prices.csv` | The structured dataset containing property features (area, bedrooms, location, etc.) and the target **price**. |
| `img_dir` | `/content/drive/My Drive/Images` | The folder containing all property images. Images are assumed to be named sequentially (e.g., `0001.jpg`, `0002.jpg`) corresponding to the row index in the CSV. |

-----

## 💻 Methodology

### 1\. Text Data Preprocessing (Structured Features)

  * **Cleaning:** Dropped irrelevant columns (`detail_url`, `image_filenames`), handled duplicates, and imputed missing values (median for numeric, mode for categorical).
  * **Encoding & Scaling:** Categorical features were processed using **`LabelEncoder`**, and all numerical features were standardized using **`StandardScaler`**.

### 2\. Image Feature Extraction (Visual Features)

The core of the analysis involves using **21 different CNNs** pretrained on **ImageNet** as fixed feature extractors.

  * Each model's final classification layer is removed, and the output from the final convolutional/pooling layer is used as the feature vector.
  * Images are resized to $224 \times 224$ and normalized before input.
  * The extracted features for each model are saved as separate NumPy files (e.g., `alexnet_features.npy`) to prevent repeated extraction.

### 3\. Combined Modeling

  * The structured **text features (`X_scaled`)** and the extracted **image features (`img_features`)** for a given CNN model are **concatenated** horizontally ($\text{combined} = [\mathbf{X}_{\text{scaled}}, \mathbf{X}_{\text{img}}]$).
  * The combined feature set is split into training and testing sets ($\text{test\_size}=0.2$).
  * A **Random Forest Regressor** (`n_estimators=80`) is trained on the combined data for prediction.
  * Performance is evaluated using **Root Mean Squared Error (RMSE)** and **$R^2$ Score**.

-----

## 📈 Results and Analysis

**NOTE:** The initial results show extremely poor performance with highly negative $R^2$ scores. This indicates a critical issue in the data pipeline or model setup.

### 1\. Summary of Initial Poor Performance

| Metric | Range | Implication |
| :--- | :--- | :--- |
| **$R^2$ Score** | $-4.98 \times 10^{10}$ to $-1.35 \times 10^{11}$ | The model performs substantially worse than predicting the mean, suggesting a major data integrity or scaling error. |
| **RMSE** | $5.56 \times 10^{11}$ to $9.14 \times 10^{11}$ | Error values are extremely large, confirming prediction instability. |
| **"Best" Model** | **InceptionResNetV2** | Showed the 'least worst' $R^2$ of $-4.98 \times 10^{10}$. |

### 2\. Recommended Next Steps (Debugging)

To achieve meaningful results, the following issues must be resolved:

  * **Re-evaluate Feature Scaling (Critical):** The **`price`** column ($y$) was not scaled or normalized before being used as the target variable for the regression model, while the feature matrix ($X$) was scaled. **Mixing scaled features with an unscaled target variable (especially one with large values) often leads to large loss values, numerical instability, and models that fail to converge, resulting in the observed poor $R^2$ and high RMSE.**
  * **Scale the Target Variable:** Before splitting the data, the `y` (price) variable should be scaled using **`StandardScaler`** (or $\log$-transformed) and then inverse-transformed after prediction.
  * **Verify Image Feature Dimensionality:** Ensure the fallback vector size matches the actual output dimension of each CNN's feature extractor. For example, the `feature_dim` for **AlexNet** is typically 4096, but you used 256. This discrepancy might cause issues if not corrected.

-----

## 🚀 Corrective Action (The Fix)

To address the $R^2$ issue, the target variable $y$ must be scaled. Here is the revised approach before training:

```python
# Create a separate scaler for the target price
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Use y_scaled instead of raw y in the train_test_split
Xtr, Xte, ytr, yte = train_test_split(combined, y_scaled, test_size=0.2, random_state=42)

# Train and Predict (as before)
model.fit(Xtr, ytr)
preds_scaled = model.predict(Xte)

# IMPORTANT: Inverse transform predictions to get real prices
preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
yte_original = y_scaler.inverse_transform(yte.reshape(-1, 1)).flatten()

# Calculate metrics using the original, unscaled prices
rmse = np.sqrt(mean_squared_error(yte_original, preds))
r2 = r2_score(yte_original, preds)
```
