# Heart_Disease_Prediction
Heart disease Prediction using Machine Learning Models

This project focuses on building and evaluating machine learning models to classify a dataset. The project includes data preprocessing, feature engineering, visualization, and training various classification models to compare their performance.

Dataset: This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

Project Workflow

Data Exploration & Preprocessing: Loaded dataset using pandas
Checked for missing values and handled them
Performed exploratory data analysis (EDA) with visualizations
Converted categorical variables into numerical features using one-hot encoding
Scaled numerical features using StandardScaler

Data Visualization: Plotted histograms for numerical features
Used boxplots to compare numerical variables by the target variable
Visualized feature correlations using a heatmap

Feature Engineering: One-hot encoding was applied to categorical features
Standardization applied to numerical columns (age, trestbps, chol, thalach, oldpeak)

Model Training & Evaluation
The following machine learning models were implemented and evaluated:

K-Nearest Neighbors (KNN): Used KNeighborsClassifier with k=3

Decision Tree Classifier: Used DecisionTreeClassifier with different max depths

Random Forest Classifier: Used RandomForestClassifier with different numbers of estimators

Logistic Regression: Trained a LogisticRegression model with max_iter=1000

Support Vector Machine (SVM):Used SVC with a linear kernel

Neural Network: Built a Sequential model using TensorFlow/Keras, Included hidden layers with ReLU activation and a sigmoid output layer

Dependencies
To run this project, install the following libraries:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
