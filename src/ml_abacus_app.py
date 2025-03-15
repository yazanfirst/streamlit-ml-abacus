
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as XGBClassifier
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="ML Abacus - Machine Learning App",
    page_icon="📊",
    layout="wide"
)

# Main app header
st.title("ML Abacus: Machine Learning Made Easy")
st.markdown("### Upload your CSV, analyze data, and train ML models in minutes!")

# Initialize session state for maintaining app state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = None
if 'numerical_columns' not in st.session_state:
    st.session_state.numerical_columns = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Step 1: File Upload Section
st.header("📁 Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        
        # Display basic info
        st.success(f"✅ Dataset successfully loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # Display first 5 rows
        st.subheader("Preview of Your Dataset")
        st.dataframe(data.head())
        
        # Get data info
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # Step 2: Data Cleaning
        st.header("🧹 Step 2: Data Cleaning")
        
        # Show missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            st.subheader("Missing Values")
            st.write(missing_values[missing_values > 0])
            
            if st.button("Drop Missing Values"):
                data = data.dropna()
                st.session_state.data = data
                st.success(f"✅ Missing values dropped. New shape: {data.shape}")
                st.dataframe(data.head())
        else:
            st.success("✅ No missing values found in the dataset.")
        
        # Step 3: Exploratory Data Analysis (EDA)
        st.header("📊 Step 3: Exploratory Data Analysis")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())
        
        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.session_state.numerical_columns = numeric_cols
        st.session_state.categorical_columns = categorical_cols
        
        st.subheader("Column Types")
        st.write(f"Numeric columns: {', '.join(numeric_cols)}")
        st.write(f"Categorical columns: {', '.join(categorical_cols)}")
        
        # Correlation matrix for numeric data
        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # Distribution plots for numeric columns
        st.subheader("Distribution of Numeric Features")
        selected_numeric = st.selectbox("Select a numeric feature", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[selected_numeric], kde=True, ax=ax)
        plt.title(f'Distribution of {selected_numeric}')
        st.pyplot(fig)
        
        # Bar plots for categorical columns
        if categorical_cols:
            st.subheader("Distribution of Categorical Features")
            selected_categorical = st.selectbox("Select a categorical feature", categorical_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            data[selected_categorical].value_counts().plot(kind='bar', ax=ax)
            plt.title(f'Count of {selected_categorical}')
            st.pyplot(fig)
        
        # Step 4: Feature Selection and Target Variable
        st.header("🎯 Step 4: Select Features and Target Variable")
        
        # Select target variable
        target_variable = st.selectbox("Select the target variable", data.columns)
        
        if st.button("Confirm Target Selection"):
            # Prepare X and y
            X = data.drop(target_variable, axis=1)
            y = data[target_variable]
            
            st.session_state.X = X
            st.session_state.y = y
            
            st.success(f"✅ Target variable set to '{target_variable}'")
            st.write(f"Features shape: {X.shape}")
            st.write(f"Target shape: {y.shape}")
            
            # Display class distribution for classification problems
            if len(y.unique()) < 10:  # Assume classification if < 10 unique values
                st.subheader("Target Variable Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                y.value_counts().plot(kind='bar', ax=ax)
                plt.title(f'Distribution of {target_variable}')
                st.pyplot(fig)
        
        # Step 5: Model Selection
        st.header("🤖 Step 5: Select and Configure ML Model")
        
        models = {
            "Logistic Regression": LogisticRegression,
            "Random Forest": RandomForestClassifier,
            "Support Vector Machine": SVC,
            "K-Nearest Neighbors": KNeighborsClassifier,
            "XGBoost": XGBClassifier
        }
        
        selected_model = st.selectbox("Choose a model", list(models.keys()))
        
        # Hyperparameter configuration based on selected model
        st.subheader("Hyperparameter Configuration")
        
        model_params = {}
        
        if selected_model == "Logistic Regression":
            penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
            C = st.slider("C (Regularization strength)", 0.01, 10.0, 1.0)
            max_iter = st.slider("Maximum Iterations", 100, 2000, 100)
            model_params = {"penalty": penalty, "C": C, "max_iter": max_iter}
            
        elif selected_model == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 500, 100)
            max_depth = st.slider("Maximum Depth", 1, 50, 10)
            min_samples_split = st.slider("Minimum Samples to Split", 2, 20, 2)
            model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split}
            
        elif selected_model == "Support Vector Machine":
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            C = st.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
            gamma = st.selectbox("Gamma", ["scale", "auto"])
            model_params = {"kernel": kernel, "C": C, "gamma": gamma}
            
        elif selected_model == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            weights = st.selectbox("Weight Function", ["uniform", "distance"])
            model_params = {"n_neighbors": n_neighbors, "weights": weights}
            
        elif selected_model == "XGBoost":
            n_estimators = st.slider("Number of Trees", 10, 500, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            max_depth = st.slider("Maximum Depth", 1, 15, 6)
            model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}
        
        # Step 6: Train/Test Split and Model Training
        st.header("🔄 Step 6: Train/Test Split and Model Training")
        
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        random_state = st.slider("Random State", 0, 100, 42)
        
        if st.session_state.X is not None and st.session_state.y is not None:
            if st.button("Train Model"):
                X = st.session_state.X
                y = st.session_state.y
                
                # Identify numeric and categorical features again
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Create preprocessing pipelines
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                # Combine preprocessing steps
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])
                
                # Create and train the model with the selected parameters
                model_class = models[selected_model]
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model_class(**model_params))
                ])
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )
                
                # Display split information
                st.write(f"Training set: {X_train.shape[0]} samples")
                st.write(f"Testing set: {X_test.shape[0]} samples")
                
                # Train the model with progress bar
                with st.spinner('Training model... This may take a moment.'):
                    model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                
                # Display results for classification
                if len(np.unique(y)) < 10:  # Classification task
                    # Accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    st.subheader("Model Performance")
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    # Classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.pyplot(fig)
                
                # Feature importance for applicable models
                if selected_model in ["Random Forest", "XGBoost"]:
                    st.subheader("Feature Importance")
                    
                    # Get feature names after preprocessing
                    feature_names = []
                    for name, trans, cols in preprocessor.transformers_:
                        if name != 'remainder':
                            if name == 'cat' and len(cols) > 0:
                                # Get the one-hot encoded feature names
                                encoder = trans.named_steps['onehot']
                                feature_names.extend(encoder.get_feature_names_out(cols))
                            else:
                                feature_names.extend(cols)
                    
                    # Extract feature importances
                    if selected_model == "Random Forest":
                        importances = model.named_steps['model'].feature_importances_
                        
                        # Only keep top 20 features if there are many
                        if len(feature_names) > 20:
                            indices = np.argsort(importances)[-20:]
                            plt.figure(figsize=(12, 8))
                            plt.title('Top 20 Feature Importances')
                            plt.barh(range(20), importances[indices], align='center')
                            plt.yticks(range(20), [feature_names[i] for i in indices])
                            plt.gca().invert_yaxis()
                        else:
                            plt.figure(figsize=(12, 8))
                            plt.title('Feature Importances')
                            plt.barh(range(len(feature_names)), importances, align='center')
                            plt.yticks(range(len(feature_names)), feature_names)
                            plt.gca().invert_yaxis()
                        
                        st.pyplot(plt)
                
                # Option to download the trained model
                model_name = selected_model.replace(" ", "_").lower()
                
                # Mark that model has been trained successfully
                st.session_state.model_trained = True
                st.success("✅ Model trained successfully!")
                
        else:
            st.warning("Please select a target variable before training the model.")
        
        # Add download button for trained model if available
        if st.session_state.model_trained:
            st.header("📥 Download Trained Model")
            st.info("You can download your trained model as a pickle file.")
            # Placeholder for download functionality
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Sample data option
    if st.button("Use Sample Dataset"):
        # Create a simple sample dataset
        np.random.seed(42)
        sample_data = {
            'age': np.random.randint(18, 90, 100),
            'income': np.random.randint(20000, 150000, 100),
            'education_years': np.random.randint(8, 20, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 100),
            'loan_approved': np.random.choice([0, 1], 100)
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Save to session state
        st.session_state.data = sample_df
        
        st.success("✅ Sample dataset loaded!")
        st.dataframe(sample_df.head())

# Footer
st.markdown("---")
st.markdown("**ML Abacus** | Made with ❤️ using Streamlit | © 2023")
