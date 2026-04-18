# Neural-Networks---Medical-Insurance
Medical Insurance Cost Prediction Using Neural Networks


Project Overview

This project uses a deep learning regression model built with TensorFlow/Keras to predict individual medical insurance costs based on demographic and lifestyle factors such as age, BMI, smoking status, and number of dependents.

The goal is to demonstrate an end-to-end machine learning workflow including data preprocessing, model development, evaluation, and interpretation.



Objective
Predict medical insurance charges (charges)
Build a neural network regression model
Compare baseline and improved architectures
Evaluate model performance using MAE and MSE
Visualize predictions and training behavior

Dataset

The dataset includes the following features:

age: Age of the individual
sex: Gender
bmi: Body Mass Index
children: Number of dependents
smoker: Smoking status
region: Residential region
charges: Medical insurance cost (target variable)


Technologies Used
Python
Pandas, NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib, Seaborn

 Data Preprocessing
One-hot encoding applied to categorical variables
Standardization applied to numerical features
Train/validation/test split (70/15/15)
Target variable scaled for stable neural network training

Model Architecture
Baseline Model:
Dense(64) → Dense(32) → Dense(1)
Activation: ReLU
Loss: MSE
Optimizer: Adam
Improved Model:
Added Dropout layers
Increased network depth

 Results
Model	MAE (USD)
Baseline	~$2,586
Improved	~$2,668

 Final selected model: Baseline (better generalization)

Key Visualizations
Training vs Validation Loss

Shows convergence and generalization behavior.

Predicted vs Actual

Visual comparison of model accuracy against ideal predictions.

Key Insights
Smoking status is strongly associated with higher insurance costs
BMI and age also show positive correlation with charges
Simpler neural network performed better than more complex architecture
Dataset does not require deep model complexity
 Conclusion

This project demonstrates that in structured tabular data problems, simpler neural networks can outperform more complex architectures when the dataset is relatively small. Proper preprocessing and validation strategy were critical in achieving stable performance.

 Future Improvements
Compare with tree-based models (Random Forest, XGBoost)
Apply feature importance analysis (SHAP)
Perform hyperparameter tuning
Experiment with log-transformed target variable

How to Run
pip install -r requirements.txt
python main.py
