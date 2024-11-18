# Fraud Detection with Vertex AI

This project aims to build an effective **Fraud Detection Model** using machine learning techniques and deploy it on **Google Cloud's Vertex AI**. The goal is to detect fraudulent transactions from a set of transactional data, allowing businesses to prevent fraud in real-time.

## Project Overview

The project follows a **data science pipeline** to address the problem of **fraud detection**. We focus on leveraging **imbalanced datasets** and apply various **modeling techniques** to optimize performance on fraud detection.

The workflow is divided into the following stages:
1. **Data Collection and Preprocessing**: 
   - Raw transactional data is cleaned and preprocessed to remove noise and prepare it for model training.
   - The dataset is split into a training set and a test set, and imbalanced classes are handled using techniques like **SMOTE** and **Tomek Links**.

2. **Model Development**:
   - Several machine learning models are trained, evaluated, and compared using metrics like **Precision**, **Recall**, **F1-score**, and **ROC-AUC** to optimize the model for fraud detection.

3. **Model Deployment**:
   - The trained model is deployed to **Google Cloud's Vertex AI**, which allows for real-time predictions and scalability.

4. **Model Monitoring**:
   - Model monitoring is implemented to track the model's performance over time, detect **drift**, and ensure that the model continues to perform well as new data is encountered.

## Resources and Services Used

The following resources and services were utilized throughout the project:

### **Google Cloud Platform (GCP)**:
- **Vertex AI**: Used for training, deploying, and monitoring the machine learning model.
- **Google Cloud Storage**: Used to store raw and preprocessed datasets.
- **BigQuery**: Used for querying large datasets for initial exploration and analysis (if applicable).
- **Google Cloud SDK**: Used to interact with GCP services, including creating and managing AI models on Vertex AI.

### **Machine Learning & Data Science Libraries**:
- **TensorFlow 2.x**: Used to build and train the fraud detection model.
- **Scikit-learn**: Used for model evaluation, SMOTE, and data preprocessing.
- **Pandas**: Used for data manipulation and exploration.
- **Matplotlib/Seaborn**: Used for data visualization and model evaluation plots.
- **Imbalanced-learn**: Used for balancing the dataset with SMOTE and Tomek Links.

### **Google Cloud Services**:
- **Google Cloud AI Platform (Vertex AI)**: For model deployment and inference.
- **Cloud Storage**: To store preprocessed datasets and model artifacts.
- **Monitoring and Logging**: Integrated with Vertex AI to monitor model predictions and performance over time.

## Key Techniques and Methods Used

1. **Data Preprocessing**:
   - **Handling Missing Values**: Missing data were filled using statistical imputation or dropped if the missing rate was high.
   - **Feature Engineering**: Generated new features based on the existing dataset, such as aggregating transaction amounts, frequency of user activities, etc.
   - **Imbalanced Data Handling**: Used **SMOTE** (Synthetic Minority Over-sampling Technique) and **Tomek Links** to address class imbalance.

2. **Model Development**:
   - Trained several machine learning models, including **Logistic Regression**, **Random Forest**, **Gradient Boosting**, and **Neural Networks**.
   - Tuned the models using **GridSearchCV** to optimize hyperparameters.
   - Evaluated the models using metrics such as **Precision**, **Recall**, **F1-Score**, **AUC-ROC**, and **Confusion Matrix**.

3. **Model Evaluation**:
   - Used **cross-validation** to ensure model performance is consistent across different subsets of the data.
   - Model performance was evaluated for its ability to detect **fraudulent transactions** (high recall for fraud class).

4. **Model Deployment**:
   - Deployed the model to **Google Vertex AI** using its **model deployment** functionality.
   - Exposed the model via an API for real-time inference (fraud prediction).

5. **Model Monitoring**:
   - Set up **model monitoring** on Vertex AI to track prediction quality and identify potential data drift or model degradation.
   - **Logging** was configured to capture model inputs and outputs for analysis.

## Project Structure

### **File Structure**:

- `data/`: Contains the raw and preprocessed datasets used for training and evaluation.
- `notebooks/`: Jupyter notebooks for data exploration, model training, evaluation, deployment, and monitoring.
- `src/`: Python scripts for core functionalities including data preprocessing, model training, deployment, and monitoring.
- `Dockerfile`: For containerizing the application (if needed for cloud deployment).
- `requirements.txt`: List of Python dependencies for the project.
- `requirements.yml`: Conda environment file to set up the project environment.
- `.gitignore`: Specifies files and directories to ignore in version control.

### **Notebooks**:
- `1_data_preprocessing.ipynb`: Data cleaning and preprocessing.
- `2_model_training.ipynb`: Model training and evaluation.
- `3_model_deployment.ipynb`: Deployment of the model on Vertex AI.
- `4_model_monitoring.ipynb`: Setting up monitoring and logging for model predictions.

### **Python Scripts**:
- `data_preprocessing.py`: Functions for loading and cleaning the data.
- `model_training.py`: Functions for training and tuning the model.
- `model_deployment.py`: Functions for deploying the model on Vertex AI.
- `model_monitoring.py`: Functions for monitoring the deployed model.
