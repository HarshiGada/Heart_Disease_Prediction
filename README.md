# Heart_Disease_Prediction
Heart disease Prediction using Machine Learning Models

This project focuses on building and evaluating machine learning models to classify a dataset. The project includes data preprocessing, feature engineering, visualization, and training various classification models to compare their performance.

Project Workflow

Data Exploration & Preprocessing

Loaded dataset using pandas

Checked for missing values and handled them

Performed exploratory data analysis (EDA) with visualizations

Converted categorical variables into numerical features using one-hot encoding

Scaled numerical features using StandardScaler

Data Visualization

Plotted histograms for numerical features

Used boxplots to compare numerical variables by the target variable

Visualized feature correlations using a heatmap

Feature Engineering

One-hot encoding was applied to categorical features

Standardization applied to numerical columns (age, trestbps, chol, thalach, oldpeak)

Model Training & Evaluation
The following machine learning models were implemented and evaluated:

K-Nearest Neighbors (KNN)

Used KNeighborsClassifier with k=3

Achieved accuracy using accuracy_score

Decision Tree Classifier

Used DecisionTreeClassifier with different max depths

Found optimal depth using cross-validation

Random Forest Classifier

Used RandomForestClassifier with different numbers of estimators

Determined best number of estimators based on accuracy

Logistic Regression

Trained a LogisticRegression model with max_iter=1000

Evaluated using accuracy, precision, recall, and confusion matrix

Support Vector Machine (SVM)

Used SVC with a linear kernel

Evaluated using accuracy and classification report

Neural Network

Built a Sequential model using TensorFlow/Keras

Included hidden layers with ReLU activation and a sigmoid output layer

Evaluated using accuracy and classification report

Dependencies
To run this project, install the following libraries:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

Running the Project

Load the dataset in a Jupyter Notebook or Colab.

Execute the preprocessing and visualization steps.

Train and evaluate different models.

Compare the accuracy and performance metrics.

Results

The accuracy of different models was compared.

The best-performing model can be chosen based on evaluation metrics.
