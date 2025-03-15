
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
from datetime import datetime
import io
import base64
import re

# Allow embedding in iframes
# This must be the first Streamlit command
st.set_page_config(
    page_title="ML Abacus - Machine Learning App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

# Set custom headers to allow embedding
def set_embed_headers():
    import streamlit.components.v1 as components
    from streamlit.web.server.server import Server
    from streamlit.components.v1.components import html

    # Only need to modify this header at this point
    st.markdown("""
        <style>
            header {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

set_embed_headers()

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
if 'data_quality_report' not in st.session_state:
    st.session_state.data_quality_report = None
if 'data_issues_detected' not in st.session_state:
    st.session_state.data_issues_detected = False

# Data quality detection functions
def detect_missing_values(df):
    """Detect missing values in the dataframe"""
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Values': missing.values,
        'Percentage (%)': missing_percent.values
    })
    return missing_df[missing_df['Missing Values'] > 0].reset_index(drop=True)

def detect_outliers(df, numeric_cols):
    """Detect outliers using Z-score and IQR methods"""
    outliers_report = {}
    
    for col in numeric_cols:
        # Skip if column has missing values
        if df[col].isnull().any():
            continue
            
        # Z-score method
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        z_outliers = df.index[z_scores > 3].tolist()
        
        # IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = df.index[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].tolist()
        
        # Combine both methods
        combined_outliers = list(set(z_outliers + iqr_outliers))
        
        if combined_outliers:
            outliers_report[col] = {
                'z_score_count': len(z_outliers),
                'iqr_count': len(iqr_outliers),
                'combined_count': len(combined_outliers),
                'combined_percent': (len(combined_outliers) / len(df)) * 100,
                'indices': combined_outliers[:5]  # Store first 5 indices for example
            }
    
    return outliers_report

def detect_duplicates(df):
    """Detect duplicate rows"""
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        return {
            'count': dup_count,
            'percent': (dup_count / len(df)) * 100,
            'first_few': df[df.duplicated(keep='first')].head(5).index.tolist()
        }
    return None

def detect_inconsistent_formats(df, cols=None):
    """Detect inconsistent formats in text and date columns"""
    if cols is None:
        cols = df.select_dtypes(include=['object']).columns.tolist()
    
    inconsistencies = {}
    
    for col in cols:
        # Check for date format inconsistencies
        date_patterns = []
        text_cases = {}
        
        for value in df[col].dropna().unique():
            # Try to detect date formats
            try:
                # Check common date formats
                if re.match(r'\d{4}-\d{2}-\d{2}', str(value)):
                    date_patterns.append('YYYY-MM-DD')
                elif re.match(r'\d{2}/\d{2}/\d{4}', str(value)):
                    date_patterns.append('MM/DD/YYYY')
                elif re.match(r'\d{2}-\d{2}-\d{4}', str(value)):
                    date_patterns.append('DD-MM-YYYY')
                
                # Check text case inconsistencies
                if isinstance(value, str):
                    if value.isupper():
                        text_cases['uppercase'] = text_cases.get('uppercase', 0) + 1
                    elif value.islower():
                        text_cases['lowercase'] = text_cases.get('lowercase', 0) + 1
                    elif value[0].isupper():
                        text_cases['capitalized'] = text_cases.get('capitalized', 0) + 1
                    else:
                        text_cases['mixed'] = text_cases.get('mixed', 0) + 1
            except:
                continue
        
        # Report inconsistencies
        if len(set(date_patterns)) > 1:
            inconsistencies[col] = {'type': 'date_format', 'formats': list(set(date_patterns))}
        elif len(text_cases) > 1:
            inconsistencies[col] = {'type': 'text_case', 'cases': text_cases}
    
    return inconsistencies

def detect_incorrect_data_types(df):
    """Detect columns where data types might be incorrect"""
    issues = {}
    
    # Check numeric values stored as strings
    for col in df.select_dtypes(include=['object']).columns:
        # Skip columns with too many unique values (likely not numeric)
        if df[col].nunique() > min(100, len(df) * 0.5):
            continue
            
        # Try to convert to numeric
        numeric_conversion = pd.to_numeric(df[col], errors='coerce')
        if numeric_conversion.notna().sum() / len(df) > 0.8:  # If >80% can be converted to numeric
            issues[col] = {
                'current_type': 'object',
                'suggested_type': 'numeric',
                'convertible_percent': numeric_conversion.notna().sum() / len(df) * 100
            }
    
    # Check categorical columns stored as numeric
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        # If a numeric column has few unique values, it might be categorical
        if df[col].nunique() < min(10, len(df) * 0.05):
            issues[col] = {
                'current_type': df[col].dtype.name,
                'suggested_type': 'categorical',
                'unique_values': df[col].nunique()
            }
    
    return issues

def detect_invalid_values(df, numeric_cols):
    """Detect invalid values like negatives in positive-only fields"""
    issues = {}
    
    # Columns that typically shouldn't have negative values
    potential_positive_only = [
        col for col in numeric_cols if any(kw in col.lower() for kw in 
        ['age', 'salary', 'income', 'price', 'cost', 'quantity', 'experience', 'years'])
    ]
    
    for col in potential_positive_only:
        if (df[col] < 0).any():
            neg_count = (df[col] < 0).sum()
            issues[col] = {
                'issue': 'negative_values',
                'count': neg_count,
                'percent': neg_count / len(df) * 100,
                'examples': df.loc[df[col] < 0, col].head(5).tolist()
            }
        
        # Check for zero values where inappropriate
        if col.lower() in ['age', 'experience', 'years']:
            if (df[col] == 0).any():
                zero_count = (df[col] == 0).sum()
                if col in issues:
                    issues[col]['zero_count'] = zero_count
                    issues[col]['zero_percent'] = zero_count / len(df) * 100
                else:
                    issues[col] = {
                        'issue': 'zero_values',
                        'count': zero_count,
                        'percent': zero_count / len(df) * 100
                    }
    
    return issues

def check_column_mismatch(df, expected_columns=None):
    """Check if required columns exist"""
    # This function would be more useful with domain-specific expected columns
    # Without that, we'll just check for common column names
    common_columns = [
        'id', 'name', 'date', 'age', 'gender', 'email', 'address', 'phone',
        'salary', 'income', 'price', 'category', 'status'
    ]
    
    if expected_columns is None:
        expected_columns = common_columns
    
    missing_columns = [col for col in expected_columns if col.lower() in [c.lower() for c in expected_columns] and col.lower() not in [c.lower() for c in df.columns]]
    unexpected_columns = []  # This would require domain knowledge to implement effectively
    
    return {'missing': missing_columns, 'unexpected': unexpected_columns}

def generate_data_quality_report(df):
    """Generate a comprehensive data quality report"""
    report = {}
    
    # Detect numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Missing values
    report['missing_values'] = detect_missing_values(df)
    
    # 2. Outliers
    report['outliers'] = detect_outliers(df, numeric_cols)
    
    # 3. Duplicates
    report['duplicates'] = detect_duplicates(df)
    
    # 4. Inconsistent formats
    report['inconsistent_formats'] = detect_inconsistent_formats(df)
    
    # 5. Incorrect data types
    report['incorrect_data_types'] = detect_incorrect_data_types(df)
    
    # 6. Invalid values
    report['invalid_values'] = detect_invalid_values(df, numeric_cols)
    
    # 7. Column mismatch
    report['column_mismatch'] = check_column_mismatch(df)
    
    # Check if any issues were detected
    has_issues = (
        not report['missing_values'].empty or
        bool(report['outliers']) or
        bool(report['duplicates']) or
        bool(report['inconsistent_formats']) or
        bool(report['incorrect_data_types']) or
        bool(report['invalid_values']) or
        (len(report['column_mismatch']['missing']) > 0)
    )
    
    return report, has_issues

# Handle data quality fixes
def fix_missing_values(df, strategy):
    """Fix missing values based on the selected strategy"""
    if strategy == 'drop_rows':
        return df.dropna(), "Rows with missing values dropped"
    elif strategy == 'drop_columns':
        return df.dropna(axis=1), "Columns with missing values dropped"
    elif strategy == 'impute_mean':
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        return df, "Missing numeric values imputed with mean"
    elif strategy == 'impute_median':
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        return df, "Missing numeric values imputed with median"
    elif strategy == 'impute_mode':
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        return df, "Missing values imputed with mode"
    return df, "No changes made"

def fix_outliers(df, column, strategy):
    """Fix outliers in the specified column"""
    if strategy == 'remove':
        # Calculate boundaries
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)], f"Outliers removed from {column}"
    
    elif strategy == 'cap':
        # Calculate boundaries
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        df_copy = df.copy()
        df_copy.loc[df_copy[column] < lower_bound, column] = lower_bound
        df_copy.loc[df_copy[column] > upper_bound, column] = upper_bound
        return df_copy, f"Outliers capped in {column}"
    
    elif strategy == 'log_transform':
        # Apply log transformation
        df_copy = df.copy()
        if (df_copy[column] > 0).all():  # Check if all values are positive
            df_copy[column] = np.log(df_copy[column])
            return df_copy, f"Log transformation applied to {column}"
        else:
            min_val = df_copy[column].min()
            if min_val <= 0:
                df_copy[column] = np.log(df_copy[column] - min_val + 1)
                return df_copy, f"Log transformation (shifted) applied to {column}"
    
    return df, "No changes made"

def fix_duplicates(df):
    """Remove duplicate rows"""
    return df.drop_duplicates(), "Duplicate rows removed"

def fix_data_types(df, column, new_type):
    """Convert column to the correct data type"""
    df_copy = df.copy()
    if new_type == 'numeric':
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        return df_copy, f"{column} converted to numeric"
    elif new_type == 'categorical':
        df_copy[column] = df_copy[column].astype('category')
        return df_copy, f"{column} converted to categorical"
    elif new_type == 'datetime':
        df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
        return df_copy, f"{column} converted to datetime"
    return df, "No changes made"

def standardize_text_case(df, column, case_type):
    """Standardize text case in a column"""
    df_copy = df.copy()
    if case_type == 'lower':
        df_copy[column] = df_copy[column].str.lower()
        return df_copy, f"{column} converted to lowercase"
    elif case_type == 'upper':
        df_copy[column] = df_copy[column].str.upper()
        return df_copy, f"{column} converted to uppercase"
    elif case_type == 'title':
        df_copy[column] = df_copy[column].str.title()
        return df_copy, f"{column} converted to title case"
    return df, "No changes made"

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
        
        # Step 2: Data Quality Assessment
        st.header("🔍 Step 2: Data Quality Assessment")
        
        # Generate data quality report
        if st.button("Analyze Data Quality"):
            with st.spinner("Analyzing data quality..."):
                report, has_issues = generate_data_quality_report(data)
                st.session_state.data_quality_report = report
                st.session_state.data_issues_detected = has_issues
        
        # Display data quality report
        if st.session_state.data_quality_report is not None:
            report = st.session_state.data_quality_report
            
            # Overall summary
            if st.session_state.data_issues_detected:
                st.warning("⚠️ Data quality issues detected. Review the report below and consider applying fixes.")
            else:
                st.success("✅ No significant data quality issues detected.")
            
            # Missing Values
            st.subheader("Missing Values")
            if not report['missing_values'].empty:
                st.dataframe(report['missing_values'])
                
                # Offer fixes
                fix_option = st.selectbox(
                    "How would you like to handle missing values?",
                    ["No action", "Drop rows with missing values", "Drop columns with missing values", 
                     "Impute with mean (numeric)", "Impute with median (numeric)", "Impute with mode (all)"]
                )
                
                if fix_option != "No action" and st.button("Apply Missing Values Fix"):
                    strategy_map = {
                        "Drop rows with missing values": "drop_rows",
                        "Drop columns with missing values": "drop_columns",
                        "Impute with mean (numeric)": "impute_mean",
                        "Impute with median (numeric)": "impute_median",
                        "Impute with mode (all)": "impute_mode"
                    }
                    data, message = fix_missing_values(data, strategy_map[fix_option])
                    st.session_state.data = data
                    st.success(message)
                    st.experimental_rerun()
            else:
                st.success("No missing values found.")
            
            # Outliers
            st.subheader("Outliers")
            if report['outliers']:
                outlier_summary = []
                for col, details in report['outliers'].items():
                    outlier_summary.append({
                        'Column': col,
                        'Outliers Count': details['combined_count'],
                        'Percentage': f"{details['combined_percent']:.2f}%",
                        'Z-score Outliers': details['z_score_count'],
                        'IQR Outliers': details['iqr_count']
                    })
                
                st.write("Outliers detected using both Z-score (>3 std) and IQR (outside Q1-1.5*IQR and Q3+1.5*IQR) methods:")
                st.dataframe(pd.DataFrame(outlier_summary))
                
                # Offer fixes
                if outlier_summary:
                    col_to_fix = st.selectbox(
                        "Select column to fix outliers:",
                        [item['Column'] for item in outlier_summary]
                    )
                    
                    fix_strategy = st.selectbox(
                        "Choose outlier handling strategy:",
                        ["No action", "Remove outliers", "Cap outliers", "Log transform"]
                    )
                    
                    if fix_strategy != "No action" and st.button("Apply Outlier Fix"):
                        strategy_map = {
                            "Remove outliers": "remove",
                            "Cap outliers": "cap",
                            "Log transform": "log_transform"
                        }
                        data, message = fix_outliers(data, col_to_fix, strategy_map[fix_strategy])
                        st.session_state.data = data
                        st.success(message)
                        st.experimental_rerun()
                
                # Visualize outliers
                if outlier_summary:
                    st.subheader("Outlier Visualization")
                    col_to_visualize = st.selectbox(
                        "Select column to visualize:",
                        [item['Column'] for item in outlier_summary],
                        key="outlier_viz"
                    )
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    sns.boxplot(y=data[col_to_visualize], ax=ax1)
                    ax1.set_title(f'Boxplot of {col_to_visualize}')
                    
                    sns.histplot(data[col_to_visualize], ax=ax2, kde=True)
                    ax2.set_title(f'Distribution of {col_to_visualize}')
                    
                    st.pyplot(fig)
                    
                    # Highlight outliers in the dataset preview
                    if st.checkbox("Highlight outliers in dataset"):
                        selected_col = col_to_visualize
                        Q1 = data[selected_col].quantile(0.25)
                        Q3 = data[selected_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outlier_mask = (data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)
                        st.write(f"Showing rows with outliers in '{selected_col}':")
                        st.dataframe(data[outlier_mask].head(10))
            else:
                st.success("No significant outliers detected.")
            
            # Duplicates
            st.subheader("Duplicate Rows")
            if report['duplicates']:
                st.write(f"Found {report['duplicates']['count']} duplicate rows ({report['duplicates']['percent']:.2f}% of data).")
                
                if st.button("Show Sample Duplicates"):
                    st.write("Sample of duplicate rows:")
                    duplicate_indices = report['duplicates']['first_few']
                    if duplicate_indices:
                        st.dataframe(data.loc[duplicate_indices])
                
                if st.button("Remove Duplicates"):
                    data, message = fix_duplicates(data)
                    st.session_state.data = data
                    st.success(message)
                    st.experimental_rerun()
            else:
                st.success("No duplicate rows found.")
            
            # Inconsistent Formats
            st.subheader("Inconsistent Formats")
            if report['inconsistent_formats']:
                st.write("The following columns have inconsistent formats:")
                format_issues = []
                for col, details in report['inconsistent_formats'].items():
                    if details['type'] == 'date_format':
                        format_issues.append({
                            'Column': col,
                            'Issue Type': 'Date Format',
                            'Details': ', '.join(details['formats'])
                        })
                    elif details['type'] == 'text_case':
                        format_issues.append({
                            'Column': col,
                            'Issue Type': 'Text Case',
                            'Details': ', '.join(details['cases'].keys())
                        })
                
                st.dataframe(pd.DataFrame(format_issues))
                
                # Offer text case standardization
                text_cols = [col for col, details in report['inconsistent_formats'].items() 
                             if details['type'] == 'text_case']
                if text_cols:
                    col_to_standardize = st.selectbox(
                        "Select column to standardize text case:",
                        ["None"] + text_cols
                    )
                    
                    if col_to_standardize != "None":
                        case_option = st.selectbox(
                            "Select text case to apply:",
                            ["lowercase", "UPPERCASE", "Title Case"]
                        )
                        
                        if st.button("Standardize Text Case"):
                            case_map = {
                                "lowercase": "lower",
                                "UPPERCASE": "upper",
                                "Title Case": "title"
                            }
                            data, message = standardize_text_case(data, col_to_standardize, case_map[case_option])
                            st.session_state.data = data
                            st.success(message)
                            st.experimental_rerun()
            else:
                st.success("No inconsistent formats detected.")
            
            # Incorrect Data Types
            st.subheader("Potential Incorrect Data Types")
            if report['incorrect_data_types']:
                type_issues = []
                for col, details in report['incorrect_data_types'].items():
                    if details['suggested_type'] == 'numeric':
                        type_issues.append({
                            'Column': col,
                            'Current Type': details['current_type'],
                            'Suggested Type': details['suggested_type'],
                            'Convertible %': f"{details['convertible_percent']:.1f}%"
                        })
                    else:
                        type_issues.append({
                            'Column': col,
                            'Current Type': details['current_type'],
                            'Suggested Type': details['suggested_type'],
                            'Unique Values': details['unique_values']
                        })
                
                st.dataframe(pd.DataFrame(type_issues))
                
                # Offer type conversion
                if type_issues:
                    col_to_convert = st.selectbox(
                        "Select column to convert:",
                        [item['Column'] for item in type_issues]
                    )
                    
                    target_type = next((item['Suggested Type'] for item in type_issues if item['Column'] == col_to_convert), None)
                    
                    if st.button(f"Convert {col_to_convert} to {target_type}"):
                        data, message = fix_data_types(data, col_to_convert, target_type.lower())
                        st.session_state.data = data
                        st.success(message)
                        st.experimental_rerun()
            else:
                st.success("No incorrect data types detected.")
            
            # Invalid Values
            st.subheader("Invalid Values")
            if report['invalid_values']:
                invalid_summary = []
                for col, details in report['invalid_values'].items():
                    if 'issue' in details and details['issue'] == 'negative_values':
                        invalid_summary.append({
                            'Column': col,
                            'Issue': 'Negative Values',
                            'Count': details['count'],
                            'Percentage': f"{details['percent']:.2f}%",
                            'Examples': str(details.get('examples', []))[:30] + "..." if details.get('examples') else ""
                        })
                    
                    if 'zero_count' in details:
                        invalid_summary.append({
                            'Column': col,
                            'Issue': 'Zero Values',
                            'Count': details['zero_count'],
                            'Percentage': f"{details['zero_percent']:.2f}%",
                            'Examples': "0"
                        })
                
                st.dataframe(pd.DataFrame(invalid_summary))
                
                # Offer to view specific examples
                if invalid_summary:
                    col_to_inspect = st.selectbox(
                        "Select column to view invalid values:",
                        list(set([item['Column'] for item in invalid_summary]))
                    )
                    
                    issue_type = next((item['Issue'] for item in invalid_summary if item['Column'] == col_to_inspect), None)
                    
                    if issue_type == 'Negative Values':
                        st.write(f"Rows with negative values in {col_to_inspect}:")
                        st.dataframe(data[data[col_to_inspect] < 0].head(10))
                    elif issue_type == 'Zero Values':
                        st.write(f"Rows with zero values in {col_to_inspect}:")
                        st.dataframe(data[data[col_to_inspect] == 0].head(10))
            else:
                st.success("No invalid values detected.")
            
            # Column Mismatch
            st.subheader("Column Mismatch")
            missing_cols = report['column_mismatch']['missing']
            if missing_cols:
                st.warning(f"Missing commonly expected columns: {', '.join(missing_cols)}")
            else:
                st.success("No missing common columns detected.")
            
            # Overall Quality Score
            st.subheader("Data Quality Score")
            
            # Calculate a simple quality score based on issues found
            issue_weights = {
                'missing_values': 2.0 if not report['missing_values'].empty else 0,
                'outliers': 1.0 if report['outliers'] else 0,
                'duplicates': 1.5 if report['duplicates'] else 0,
                'inconsistent_formats': 1.0 if report['inconsistent_formats'] else 0,
                'incorrect_data_types': 1.5 if report['incorrect_data_types'] else 0,
                'invalid_values': 2.0 if report['invalid_values'] else 0,
                'column_mismatch': 0.5 if missing_cols else 0
            }
            
            total_issues = sum(issue_weights.values())
            max_issues = sum([weight for weight in issue_weights.values() if isinstance(weight, (int, float))])
            
            if max_issues > 0:
                quality_score = 100 * (1 - (total_issues / 9.5))  # 9.5 is the max possible issues weight
                
                # Display quality gauge
                st.write(f"Overall Data Quality Score: {quality_score:.1f}%")
                
                # Color based on score
                color = 'green' if quality_score >= 80 else 'orange' if quality_score >= 60 else 'red'
                st.markdown(f"""
                    <div style="background-color: {color}; width: {quality_score}%; height: 20px; border-radius: 10px;"></div>
                """, unsafe_allow_html=True)
                
                if quality_score < 60:
                    st.error("⚠️ Your data requires significant cleaning before modeling.")
                elif quality_score < 80:
                    st.warning("⚠️ Your data would benefit from some cleaning before modeling.")
                else:
                    st.success("✅ Your data is relatively clean and ready for modeling.")
        
        # Step 3: Data Cleaning (moved to data quality assessment)
        # Step now handled within the data quality assessment reports
        
        # Step 4: Exploratory Data Analysis (EDA)
        st.header("📊 Step 4: Exploratory Data Analysis")
        
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
        
        # Step 5: Feature Selection and Target Variable
        st.header("🎯 Step 5: Select Features and Target Variable")
        
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
        
        # Step 6: Model Selection
        st.header("🤖 Step 6: Select and Configure ML Model")
        
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
        
        # Step 7: Train/Test Split and Model Training
        st.header("🔄 Step 7: Train/Test Split and Model Training")
        
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
