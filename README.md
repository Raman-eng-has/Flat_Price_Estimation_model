# 🏠 Flat Price Estimator for UrbanNest Realtors

A machine learning-based tool to estimate flat prices in urban areas using features like area, distance to metro, amenities, and more.

## 🔧 Features

- Train a model using your own dataset (`flat_price` as target)
- Load existing `.pkl` model
- Predict flat prices using:
  - Manual user input
  - Bulk predictions from CSV file
- Works on Google Colab or any local Python environment

## 🧠 Technologies Used

- Python
- pandas, numpy
- scikit-learn (RandomForestRegressor)
- joblib for model saving/loading

## 📁 File Structure

```
flat-price-estimator-urbannest/
├── flat_price_estimator.py
├── flat_price_model.pkl (optional)
├── sample_training.csv
├── sample_bulk_input.csv
├── Flat_Price_Estimator_Project_Report.docx
└── README.md
```

## 💡 Dataset Format

The training CSV should have the following columns:

```
area_sqft,bedrooms,distance_to_metro_km,age_of_flat_years,amenities_score,flat_price
```

For bulk prediction (CSV input), use the same format **without `flat_price`** column.

## 🚀 How to Use

### ▶️ Option 1: Run on Google Colab

1. Open [Colab](https://colab.research.google.com/)
2. Upload `flat_price_estimator.py`
3. Run the script and follow prompts to upload your dataset or model

### 🖥️ Option 2: Run Locally

```bash
pip install pandas numpy scikit-learn joblib
python flat_price_estimator.py
```

## 📈 Model Evaluation

During training, the script prints:

- R² Score (how well it fits)
- MAE (mean absolute error)
- RMSE (root mean square error)

## 📩 Author

## 📩 Author

**Raman Kashyap**  
[GitHub: @Raman-eng-has](https://github.com/raman-sharma)  
Email: kashyapraman190@gmail.com  | Developed as a data science project using Python
