# Predicting-Building-Damage-Grade-by-Earthquake

# Building Structure Damage Prediction

This project aims to predict the damage grade of building structures based on various features using machine learning models. The analysis involves data loading, preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Project Steps

1.  **Data Loading and Initial Inspection:**
    *   Loads the building structure data from a CSV file.
    *   Performs initial inspection of data types and identifies missing values.

2.  **Data Cleaning:**
    *   Explicitly converts specific columns to the 'object' data type.
    *   Removes rows with missing values.

3.  **Exploratory Data Analysis (EDA):**
    *   Visualizes the distribution of the target variable ('damage_grade').
    *   Analyzes the relationship between 'damage_grade' and various categorical features such as district, land surface condition, foundation type, roof type, ground floor type, other floor type, position, plan configuration, condition after earthquake, and proposed technical solutions.
    *   Analyzes the relationship between 'damage_grade' and numerical features such as number of floors, building height, age, and plinth area.
    *   Visualizes the proportion of different superstructure types across damage grades using radar charts.

4.  **Feature Engineering:**
    *   Creates new features representing the difference in floors and height before and after the earthquake.

5.  **Data Splitting:**
    *   Splits the dataset into training and testing sets using stratified splitting to maintain the proportion of damage grades.

6.  **Target Variable Encoding:**
    *   Encodes the categorical target variable ('damage_grade') into numerical labels.

7.  **Data Preprocessing:**
    *   Defines a preprocessing pipeline using `ColumnTransformer` to handle different feature types:
        *   Converting specific columns to object type.
        *   Handling outliers in 'age_building' using `RobustScaler`.
        *   Scaling numerical features using `StandardScaler` and `MinMaxScaler`.
        *   One-hot encoding categorical features.
        *   Removing features with low variance.
    *   Applies the preprocessing steps to the training and testing data.

8.  **Model Training and Evaluation:**
    *   Initializes K-fold cross-validation.
    *   Trains and evaluates the following machine learning models:
        *   **Random Forest Classifier:** Trains the model, makes predictions, calculates accuracy and Mean Squared Error (MSE), and displays a confusion matrix. Saves the actual test data and predictions to Excel files.
        *   **ElasticNet Regression:** Trains the model, makes predictions, calculates accuracy and MSE, and saves predictions to an Excel file.
        *   **Decision Tree Regressor:** Trains the model, makes predictions, calculates accuracy and MSE, and saves predictions to an Excel file.

## Libraries Used

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `sklearn` (for various preprocessing techniques, model selection, models, and metrics)
*   `xlsxwriter` (for writing to Excel files)

## Data Source

The dataset used in this project is assumed to be in a CSV file named `csv_building_structure.csv` located in a Google Drive folder accessible at `/content/drive/MyDrive/2072dataset/`.

## How to Run the Code

1.  Ensure you are running the code in a Google Colab environment or a Jupyter Notebook with access to your Google Drive.
2.  Mount your Google Drive in the Colab environment (Cell 2 in the provided code).
3.  Make sure the dataset file (`csv_building_structure.csv`) is located at the specified path in your Google Drive.
4.  Run the cells sequentially in the notebook.

## Output Files

The code generates the following output files in the Colab environment:

*   `y-test-RF.xlsx`: Contains the actual target values for the test set.
*   `y_pred_RF.xlsx`: Contains the predictions from the Random Forest Classifier.
*   `y_pred_Elactic.xlsx`: Contains the predictions from the ElasticNet Regression model.
*   `y_pred_DT.xlsx`: Contains the predictions from the Decision Tree Regressor.

# Second part of this project Support Vector Regression (SVR) Analysis on Hitters Dataset

This project performs a Support Vector Regression (SVR) analysis on the Hitters dataset to predict player salaries. The analysis follows these steps:

1.  **Load Libraries and Data:**
    *   Essential libraries for data manipulation, visualization, and machine learning are imported.
    *   The Hitters dataset is loaded from Google Drive.

2.  **Data Preprocessing:**
    *   Rows with missing values in the 'Salary' column are removed.
    *   Categorical features ('League', 'Division', 'NewLeague') are one-hot encoded.
    *   The original categorical columns and the 'Salary' column are dropped from the feature set (X).
    *   The target variable (y) is defined as the 'Salary' column.

3.  **Data Splitting:**
    *   The dataset is split into training and testing sets.

4.  **Model Training (Initial):**
    *   A Support Vector Regressor with a Radial Basis Function (RBF) kernel is initialized and trained on the training data.

5.  **Model Evaluation (Initial):**
    *   The model's performance is evaluated on both the training and testing sets using Root Mean Squared Error (RMSE) and R-squared scores.

6.  **Hyperparameter Tuning:**
    *   A grid search with cross-validation is performed to find the optimal 'C' parameter for the SVR model.

7.  **Model Training (Tuned):**
    *   An SVR model is trained using the best 'C' parameter found during tuning.

8.  **Model Evaluation (Tuned):**
    *   The performance of the tuned model is evaluated on the training and testing sets using R-squared scores and RMSE.

## Results

The initial SVR model showed the following performance:

*   **Train RMSE:** {{np.sqrt(mean_squared_error(y_train,SVR_Radial_Basis.predict(X_train)))}}
*   **Train R-squared:** {{r2_score(y_train,SVR_Radial_Basis.predict(X_train))}}
*   **Test RMSE:** {{np.sqrt(mean_squared_error(y_test,SVR_Radial_Basis.predict(X_test)))}}
*   **Test R-squared:** {{r2_score(y_test,SVR_Radial_Basis.predict(X_test))}}

After hyperparameter tuning, the model with the optimal 'C' parameter ({{svr_cv_model.best_params_['C']}}) achieved the following performance:

*   **Tuned Train RMSE:** {{np.sqrt(mean_squared_error(y_train,svr_tuned.predict(X_train)))}}
*   **Tuned Train R-squared:** {{r2_score(y_train,svr_tuned.predict(X_train))}}
*   **Tuned Test RMSE:** {{np.sqrt(mean_squared_error(y_test,svr_tuned.predict(X_test)))}}
*   **Tuned Test R-squared:** {{r2_score(y_test,svr_tuned.predict(X_test))}}

The tuned model shows improved R-squared scores on both the training and testing sets compared to the initial model, indicating a better fit to the data. The test RMSE also decreased after tuning.
