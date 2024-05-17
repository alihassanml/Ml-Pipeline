# Machine Learning Pipeline with Titanic Dataset

This repository contains a machine learning pipeline for the Titanic dataset, demonstrating a complete end-to-end data preprocessing and model training workflow using scikit-learn. The pipeline includes data cleaning, feature engineering, and model training using a decision tree classifier.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Description](#pipeline-description)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to predict the survival of passengers on the Titanic using a machine learning pipeline. This repository demonstrates how to set up a pipeline that includes data preprocessing steps such as handling missing values, encoding categorical features, scaling numerical features, and training a machine learning model.

## Dataset
The dataset used in this project is the Titanic dataset, which contains information about the passengers on the Titanic, such as age, sex, passenger class, and whether they survived the disaster.

You can download the dataset from [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data).

## Requirements
- Python 3.6+
- scikit-learn
- pandas
- numpy

## Installation
1. Clone the repository:
    ```bash
    https://github.com/alihassanml/Ml-Pipeline.git
    ```
2. Navigate to the project directory:
    ```bash
    cd titanic-ml-pipeline
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Ensure you have the Titanic dataset in the project directory.
2. Run the pipeline script:
    ```bash
    python pipeline.py
    ```

## Pipeline Description
The pipeline is implemented using scikit-learn's `Pipeline` and `ColumnTransformer` classes to handle preprocessing and model training in a streamlined manner.

1. **Data Loading**:
    ```python
    import pandas as pd

    data = pd.read_csv('train.csv')
    X = data.drop(columns='Survived')
    y = data['Survived']
    ```

2. **Preprocessing**:
    - **Numerical Features**:
        - Imputation of missing values using `SimpleImputer`
        - Scaling using `MinMaxScaler`
    - **Categorical Features**:
        - Imputation of missing values using `SimpleImputer`
        - Encoding using `OneHotEncoder`

    ```python
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    numeric_features = ['Age', 'Fare']
    categorical_features = ['Embarked', 'Sex', 'Pclass']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    ```

3. **Model Training**:
    - Decision Tree Classifier

    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier())
    ])

    pipeline.fit(X_train, y_train)
    ```

4. **Model Evaluation**:
    - Evaluate the trained model on the test set.

    ```python
    from sklearn.metrics import accuracy_score

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize and expand this README to better suit your project's specific needs. Happy coding!
