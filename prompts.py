"""
Prompt generation functions for the Autonomous ML Agent.
This module contains all functions responsible for creating prompts for LLM interactions.
"""

import pandas as pd


def summarize_dataset(dataframe: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the dataset for LLM context.

    This function creates a detailed text summary that includes:
        - Column datatypes and schema information
        - Missing value counts and data completeness
        - Cardinality (unique value counts) for each categorical columns
        - Statistical summary for numerical columns
        - Sample data rows in csv format

    Args:
        dataframe (pd.DataFrame): The input dataframe to summarize

    Returns:
        str: A detailed summary of the dataset
    """
    import io

    try:
        # Create a string buffer to capture CSV output in memory
        buffer = io.StringIO()

        # Limit sample to first 15 rows to avoid overwhelming the LLM with too much data
        sample_rows = min(15, len(dataframe))

        # Convert sample rows to CSV format and write to buffer
        dataframe.head(sample_rows).to_csv(buffer, index=False)
        sample_csv = buffer.getvalue()

        # Extract datatype information for each column
        dtypes = dataframe.dtypes.astype(str).to_dict()

        # Count non-null values for each column (data completeness)
        non_null_counts = dataframe.notnull().sum().to_dict()

        # Count null/missing values for each column
        null_counts = dataframe.isnull().sum().to_dict()

        # Calculate unique value counts per column (cardinality
        # dropna = True excludes NaN values from the count
        nunique = dataframe.nunique(dropna=True).to_dict()

        # Identify numeric columns for statistical analysis
        numeric_cols = [c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]

        # Generate descriptive statistics for numeric columns only
        # Returns empty dicts if no numeric columns are found
        desc = dataframe[numeric_cols].describe().to_dict() if numeric_cols else {}

        # Build the summary report line by line
        lines = []

        # Section 1: Schema information (data types)
        lines.append("Schema (dtype):")
        for k, v in dtypes.items():
            lines.append(f"- {k}: {v}")
        lines.append("")  # Add an empty line after schema section
        
        # Section 2: Data completeness
        lines.append("Null/Non-Null Counts:")
        for c in dataframe.columns:
            lines.append(f"- {c}: non_nulls = {int(non_null_counts[c])}, \
                nulls = {int(null_counts[c])}")
        lines.append("")  # Add an empty line after data completeness section
        
        # Section 3: Cardinality (unique value counts)
        lines.append("Cardinality (nunique):")
        for k, v in nunique.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")  # Add an empty line after cardinality section
        
        # Section 4: Statistical summary for numeric columns
        if desc:
            lines.append("Numeric summary stats (describe):")
            for col, stats in desc.items():
                # Format each statistic with proper rounding and handle NaN values
                stat_line = ", ".join([f"{s}:{round(float(val), 4) if pd.notnull(val) else 'nan'}"
                                    for s, val in stats.items()])
                lines.append(f"- {col}: {stat_line}")
        lines.append("")  # Add an empty line after statistical summary section
        
        # Section 5: Sample data rows in csv format
        lines.append("Sample rows (CSV head):")
        lines.append(sample_csv)
        
        # Join all lines into a single string with line breaks
        summary = "\n".join(lines)
        return summary
    
    except Exception as e:
        return f"Error generating dataset summary: {str(e)}"


def build_cleaning_prompt(df: pd.DataFrame, target_column:str) -> str:
    """
    Build a prompt for data cleaning and preprocessing.
    
    Args:
        df (pd.DataFrame): The input dataframe
        target_column (str): The target column to predict
    Returns:
        str: The cleaning prompt
    """
    data_summary = summarize_dataset(df)

    prompt = f""" 

    You are an expert data scientist, specifically in the field
    of data cleaning and preprocessing. You are a given a dataframe summary and you are tasked with cleaning the dataset:

    {data_summary}

    The target column to predict is: {target_column}

    Make sure to handle the:
    - Missing values
    - Duplicates
    - Outliers
    - Drop columns that are not relevant to predicting the target column (i.e. IDs and Names)
    - Standardize the data accordingly (DO NOT STANDARDIZE THE TARGET COLUMN OR CATEGORICAL COLUMNS)
    - Use one-hot encoding for categorical columns
    - Select the categorical columns carefullly (i.e. cardinality is less than 10)

    Generate a standalone python script to clean the dataset, based on the data summary provided, and return a json property called "script."
    
    ## IMPORTANT REQUIREMENTS##
    - Make sure to load the data from the csv file called "/tmp/input.csv".
    - The script should be a python script that is standalone and can be executed to clean the data in a sandbox.
    - Make sure to save the cleaned data to a new csv file called "/tmp/cleaned.csv".
    - DO NOT drop the target column during cleaning - keep all columns for now.
    - DO NOT print to STDOUT or STDERR - only return the script.
    """
    return prompt


def build_model_training_prompt(df: pd.DataFrame, target_column: str) -> str:
    """
    Build a prompt for model training.
    
    Args:
        df (pd.DataFrame): The cleaned dataframe
        target_column (str): The target column to predict
        
    Returns:
        str: The model training prompt
    """
    data_summary = summarize_dataset(df)
    
    prompt = f"""
    You are an expert machine learning engineer. You are given a cleaned dataset and need to train multiple models to predict the target column.
    
    Dataset Summary:
    {data_summary}
    
    Target Column: {target_column}
    
    Your task is to:
    1. Load the cleaned dataset from "/tmp/cleaned.csv"
    2. Separate features (X) and target (y) - drop the target column from X
    3. Train the following models:
       - Logistic Regression (for classification) or Linear Regression (for regression)
       - Random Forest
       - XGBoost or Gradient Boosting
       - One additional model of your choice
    4. If classification, evaluate each model using the following metrics: Accuracy, Precision, Recall, F1 Score and ROC AUC Score
    5. If regression, evaluate each model using the following metrics: R2 Score, RMSE, MAE, and MAPE
    6. Save all model results in a dataframe and save it to "/tmp/model_results.csv"
    
    Important Requirements:
    - Use scikit-learn, pandas, numpy, and other standard ML libraries
    - Implement proper train/test split (80/20 or 70/30)
    - DO NOT perform cross-validation - just use the train/test split
    - Handle both classification and regression tasks automatically
    - Save model results to "/tmp/model_results.csv" (EXACT PATH REQUIRED))
    - DO NOT re-clean, re-scale, or re-encode the data - it's already cleaned
    - Focus only on model training, evaluation, and comparison
    - DO NOT print to STDOUT or STDERR - only return the script.
    
    Return a JSON object with a "script" property containing the complete Python code.
    """
    return prompt


def build_model_analysis_prompt(model_results_df: pd.DataFrame, target_column: str) -> str:
    """
    Build a prompt for analyzing model results and generating natural language summaries.
    
    Args:
        model_results_df (pd.DataFrame): The model results dataframe
        target_column (str): The target column that was predicted
        
    Returns:
        str: The model analysis prompt
    """
    prompt = f"""
    You are an expert data scientist and ML engineer. Analyze the following model results and provide comprehensive insights.
    
    Model Results:
    {model_results_df.to_string()}
    
    Target Column: {target_column}
    
    Your task is to:
    1. Analyze the performance of each model
    2. Identify the best performing model and explain why
    3. Provide insights on model behavior and potential issues
    4. Suggest improvements or next steps
    5. Generate a natural language summary of the results
    
    Please return a JSON object with the following structure:
    {{
        "best_model": "model_name",
        "best_score": score_value,
        "analysis": "detailed analysis of model performance",
        "insights": "key insights about the models and data",
        "recommendations": "suggestions for improvement",
        "summary": "natural language summary of the entire experiment"
    }}
    """
    return prompt


