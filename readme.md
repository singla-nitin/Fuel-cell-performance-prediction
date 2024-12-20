# Fuel Cell Performance Prediction

This repository evaluates different machine learning models to predict fuel cell performance using the `Fuel_cell_performance_data-Full.csv` dataset.

## Project Structure

```
.
├── main.py                 # Main script for model training and evaluation
├── Fuel_cell_performance_data-Full.csv # Dataset file
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and folders
├── model_evaluation_metrics.csv # Output metrics
```

## Workflow

1. Load the dataset and split into training and testing sets.
2. Train models:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. Evaluate models using R-squared, MSE, and MAE.
4. Save results to `model_evaluation_metrics.csv`.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python main.py
   ```
3. View the results in `model_evaluation_metrics.csv`.

## Dependencies

- pandas
- scikit-learn
- numpy

Refer to `requirements.txt` for exact versions.
