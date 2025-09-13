from typing import Optional
from openai import OpenAI
import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
import io
import os
from e2b_code_interpreter import Sandbox

# Load environment variables from .env file
load_dotenv()

###################### FUNCTIONS #########################################################################################

# SUMMARIZES THE DATASET TO FEED INTO THE PROMPT
def summarize_dataset(dataframe: pd.DataFrame)->str:

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

    try:
        # Create a string buffer to capture CSV output in memory
        buffer = io.StringIO()

        # Limit sample to first 30 rows to avoid overwhelming the LLM with too much data
        sample_rows = min(30, len(dataframe))

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
        for k,v in dtypes.items():
            lines.append(f"- {k}: {v}")
        lines.append("") # Add an empty line after schema section
        
        # Section 2: Data completeness
        lines.append("Null/Non-Null Counts:")
        for c in dataframe.columns:
            lines.append(f"- {c}: non_nulls = {int(non_null_counts[c])}, \
                nulls = {int(null_counts[c])}")
        lines.append("") # Add an empty line after data completeness section
        
        # Section 3: Cardinality (unique value counts)
        lines.append("Cardinality (nunique):")
        for k,v in nunique.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("") # Add an empty line after cardinality section
        
        # Section 4: Statistical summary for numeric columns
        if desc:
            lines.append("Numeric summary stats (describe):")
            for col, stats in desc.items():
                # Format each statistic with proper rounding and handle NaN values
                stat_line = ", ".join([f"{s}:{round(float(val), 4) if pd.notnull(val) else 'nan'}"
                                    for s, val in stats.items()])
                lines.append(f"- {col}: {stat_line}")
        lines.append("") # Add an empty line after statistical summary section
        
        # Section 5: Sample data rows in csv format
        lines.append("Sample rows (CSV head):")
        lines.append(sample_csv)
        
        # Join all lines into a single string with line breaks
        summary = "\n".join(lines)
        return summary
    
    except Exception as e:
        st.error(f"Error generating dataset summary: {str(e)}")
        return "Error generating dataset summary"


# FUNCTION TO BUILD THE PROMPT FOR THE DATA CLEANING
# THIS PROMPT IS USED TO CLEAN THE DATAFRAME
# IT IS USED TO REMOVE ANY ROWS WITH MISSING VALUES
# AND TO ENSURE THAT THE DATA IS IN A FORMAT THAT IS SUITABLE FOR RUNNING MACHINE LEARNING MODELS
def build_cleaning_prompt(df):
    data_summary = summarize_dataset(df)

    prompt = f""" 

    You are an expert data scientist, specifically in the field
    of data cleaning and preprocessing. You are a given a dataframe summary and you are tasked with cleaning the dataset:

    {data_summary}

    Make sure to handle the:
    - Missing values
    - Duplicates
    - Outliers
    - Standardize the data accordingly
    - Use one-hot encoding for categorical columns

    Write a python script to clean the dataset, based on the data summary provided, and return a json property called "script."
    
    ## IMPORTANT ##
    - Make sure to load the data from the csv file called "/tmp/input.csv".
    - The script should be a python script that can be executed to clean the data.
    - Make sure to save the cleaned data to a new csv file called "/tmp/cleaned.csv".
    - The script should handle any errors gracefully and print status messages.
    """
    return prompt

# FUNCTION TO BUILD THE PROMPT FOR MODEL TRAINING
def build_model_training_prompt(df, target_column):
    """
    Build a prompt for training multiple ML models on the cleaned dataset.
    
    Args:
        df (pd.DataFrame): The cleaned dataframe
        target_column (str): The target column to predict
        
    Returns:
        str: The prompt for model training
    """
    data_summary = summarize_dataset(df)
    
    prompt = f"""
    You are an expert machine learning engineer. You are given a cleaned dataset and need to train multiple models to predict the target column.
    
    Dataset Summary:
    {data_summary}
    
    Target Column: {target_column}
    
    Your task is to:
    1. Load the cleaned dataset from "/tmp/cleaned.csv"
    2. Prepare the data for machine learning (feature engineering, train/test split)
    3. Train  the following models:
       - Logistic Regression (for classification) or Linear Regression (for regression)
       - Random Forest
       - XGBoost or Gradient Boosting
       - One additional model of your choice
    4. Evaluate each model using appropriate metrics
    5. Compare model performance
    6. Save the best model and results
    
    Important Requirements:
    - Use scikit-learn, pandas, numpy, and other standard ML libraries
    - Implement proper train/test split (80/20 or 70/30)
    - Use cross-validation for robust evaluation
    - Handle both classification and regression tasks automatically
    - Save model results to "/tmp/model_results.csv"
    - Save the best model using joblib to "/tmp/best_model.pkl"
    - Print detailed performance metrics for each model
    
    Return a JSON object with a "script" property containing the complete Python code.
    """
    return prompt

# SYSTEM PROMPT FOR THE LLM
SYSTEM_PROMPT = "You are a senior data engineer. Always return a strict JSON object matching the user's requested schema."

# USES THE PROMPT CREATED TO FEED INTO THE LLM
def get_llm_response(prompt:str) -> Optional[str]:
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
            )
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    SYSTEM_PROMPT
                )},
                {"role": "user", "content": prompt}
            ]
        )
        if not response or not getattr(response, "choices", None):
            return None
        text = response.choices[0].message.content or ""

        # EXPECT A JSON OBJECT WITH A SCRIPT PROPERTY
        try:
            # Handle OpenRouter's response format with special tokens
            if "<|message|>" in text:
                # Extract JSON content between <|message|> and the end
                json_start = text.find("<|message|>") + len("<|message|>")
                json_content = text[json_start:].strip()
            else:
                # If no special tokens, use the entire response
                json_content = text
            
            json_obj = json.loads(json_content)
            # GET THE SCRIPT VALUE FROM THE JSON OBJECT
            script_val = json_obj.get("script")
            # IF THE SCRIPT VALUE IS A STRING, AND IT IS NOT EMPTY, RETURN THE SCRIPT VALUE
            if isinstance(script_val, str) and script_val.strip():
                # USED TO REMOVE ANY EXTRA WHITESPACE OR NEWLINES
                return script_val.strip()
        except json.JSONDecodeError:
            st.error(f"Invalid JSON response: {text}")
            return None

        # IF NO SCRIPT FOUND, RETURN NONE
        return None

    except Exception as e:
        st.error(f"Error getting LLM response: {str(e)}")
        return None

def execute_in_e2b(script: str, csv_bytes: bytes):
    """
    Execute a Python script in an E2B sandbox with the provided CSV data.
    
    Args:
        script (str): The Python script to execute
        csv_bytes (bytes): The CSV file data as bytes
        
    Returns:
        tuple: (cleaned_csv_bytes, exec_info) or (None, exec_info) on error
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        st.error("E2B_API_KEY is not set")
        return None, {}
    
    # Create a dictionary to store execution results
    exec_info = {}
    
    try:
        # DOCUMENTATION: https://docs.e2b.dev/getting-started/introduction
        # Create E2B sandbox instance - E2B manages lifecycle automatically
        sandbox = Sandbox.create(api_key=api_key)
        
        # Upload the CSV file to the sandbox
        sandbox.files.write("/tmp/input.csv", csv_bytes)
        
        # Execute the script directly using run_code
        result = sandbox.run_code(script)
        
        # Store execution results
        exec_info["exit_code"] = getattr(result, "exit_code", 0)
        exec_info["stdout"] = getattr(result, "stdout", "")
        exec_info["stderr"] = getattr(result, "stderr", "")
        
        # Debug: Check what files exist in /tmp
        try:
            list_result = sandbox.run_code("import os; print('Files in /tmp:'); print(os.listdir('/tmp'))")
            exec_info["debug_files"] = getattr(list_result, "stdout", "")
            print("Debug - Files in /tmp:", exec_info["debug_files"])
        except Exception as e:
            exec_info["debug_files"] = f"Error listing files: {str(e)}"
        
        # Try to download the cleaned CSV file
        try:
            # The script should save the cleaned data to "/tmp/cleaned.csv"
            cleaned_bytes = sandbox.files.read("/tmp/cleaned.csv")
            return cleaned_bytes, exec_info
        except Exception as e:
            st.error(f"Error downloading cleaned file: {str(e)}")
            st.error(f"Debug info: {exec_info.get('debug_files', 'No debug info')}")
            return None, exec_info
                
    except Exception as e:
        st.error(f"Error executing script in E2B: {str(e)}")
        return None, exec_info
    
    finally:
        # Only kill after we've retrieved all results
        try:
            sandbox.kill()
        except:
            pass  # Ignore errors if sandbox is already dead

def execute_model_training_in_e2b(script: str, cleaned_csv_bytes: bytes):
    """
    Execute a model training script in an E2B sandbox with the cleaned CSV data.
    
    Args:
        script (str): The Python script to execute for model training
        cleaned_csv_bytes (bytes): The cleaned CSV file data as bytes
        
    Returns:
        tuple: (model_results_bytes, best_model_bytes, exec_info) or (None, None, exec_info) on error
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        st.error("E2B_API_KEY is not set")
        return None, None, {}
    
    # Create a dictionary to store execution results
    exec_info = {}
    
    try:
        # Create E2B sandbox instance with timeout
        sandbox = Sandbox.create(api_key=api_key, timeout=120)  # 2 minutes for model training
        
        # Upload the cleaned CSV file to the sandbox
        sandbox.files.write("/tmp/cleaned.csv", cleaned_csv_bytes)
        
        # Execute the model training script
        result = sandbox.run_code(script)
        
        # Store execution results
        exec_info["exit_code"] = getattr(result, "exit_code", 0)
        exec_info["stdout"] = getattr(result, "stdout", "")
        exec_info["stderr"] = getattr(result, "stderr", "")
        
        # Debug: Check what files exist in /tmp
        try:
            list_result = sandbox.run_code("import os; print('Files in /tmp:'); print(os.listdir('/tmp'))")
            exec_info["debug_files"] = getattr(list_result, "stdout", "")
        except Exception as e:
            exec_info["debug_files"] = f"Error listing files: {str(e)}"
        
        # Try to download the model results and best model
        model_results_bytes = None
        best_model_bytes = None
        
        try:
            # Download model results CSV
            model_results_bytes = sandbox.files.read("/tmp/model_results.csv")
        except Exception as e:
            st.warning(f"Could not download model results: {str(e)}")
        
        try:
            # Download best model pickle file
            best_model_bytes = sandbox.files.read("/tmp/best_model.pkl")
        except Exception as e:
            st.warning(f"Could not download best model: {str(e)}")
        
        return model_results_bytes, best_model_bytes, exec_info
                
    except Exception as e:
        st.error(f"Error executing model training in E2B: {str(e)}")
        return None, None, exec_info
    finally:
        # Clean up sandbox
        try:
            sandbox.kill()
        except:
            pass

###################### USER INTERFACE ##########################################################################################
# USES STREAMLIT TO BUILD THE UI AND ADD A TITLE
st.title("Autonomous ML Agent")
# DESCRIPTION TEXT TO GUIDE THE USER ON HOW TO START 
st.markdown("""
Welcome to the Autonomous ML Agent!

This autonomous machine learning agent ingests tabular datasets, automatically cleans and preprocesses the data, 
trains models, and optimizes them for target metrics such as accuracy, precision, or recall.

The entire pipeline is orchestrated by LLMs, where they generate and modify code, select appropriate algorithms, 
and iteratively refine the pipeline until the best-performing model is achieved.
""")

# USES STREAMLIT TO ALLOW USER TO UPLOAD A CSV FILE 
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# ONCE USER HAS UPLOADED A CSV FILE, IT IS READ INTO A PANDAS DATAFRAME
# THE DATAFRAME IS DISPLAYED TO THE USER ON THE APP
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    selected_column = st.selectbox("Select a column to predict", 
    df.columns.tolist(), 
    help = "The column to predict")

# Button to run the Autonomous ML Agent
button  = st.button("Run Autonomous ML Agent")

# ONCE THE BUTTON IS CLICKED, THE AUTONOMOUS ML AGENT STARTS RUNNING
if button:
    # SPINNER TO SHOW THAT THE AUTONOMOUS ML AGENT IS RUNNING
    with st.spinner("Running Autonomous ML Agent..."):
        # BUILD THE CLEANING PROMPT
        cleaning_prompt = build_cleaning_prompt(df)
        # DISPLAY THE CLEANING PROMPT IN AN ACCORDION
        with st.expander("Cleaning Prompt"):
            st.write(cleaning_prompt)
        script = get_llm_response(cleaning_prompt)
        # DISPLAY THE SCRIPT IN AN ACCORDION
        with st.expander("Script"):
            st.code(script)
        # SPINNER TO SHOW THAT THE SCRIPT IS BEING EXECUTED IN THE E2B SANDBOX
        with st.spinner("Executing cleaning script in E2B sandbox..."):
            # CONVERT THE DATAFRAME TO A CSV FILE AND ENCODE IT TO BYTES
            input_csv_bytes = df.to_csv(index=False).encode('utf-8')
            cleaned_bytes, exec_info = execute_in_e2b(script, input_csv_bytes)
            # DISPLAY THE EXECUTION INFO IN AN ACCORDION
            with st.expander("E2B Execution Info"):
                st.write(exec_info)
            # DISPLAY THE CLEANED DATA IN AN ACCORDION
            with st.expander("Cleaned Data"):
                # Handle both string and bytes from E2B files.read()
                if isinstance(cleaned_bytes, str):
                    # If it's a string, use StringIO
                    cleaned_df = pd.read_csv(io.StringIO(cleaned_bytes))
                else:
                    # If it's bytes, use BytesIO
                    cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                # DISPLAY THE CLEANED DATAFRAME
                st.write(cleaned_df)
                
                # STEP 2: MODEL TRAINING
                st.markdown("---")
                st.markdown("Model Training")
                
                if cleaned_bytes is not None:
                    # Build the model training prompt
                    model_training_prompt = build_model_training_prompt(cleaned_df, selected_column)
                    
                    # Display the model training prompt
                    with st.expander("Model Training Prompt"):
                        st.write(model_training_prompt)
                    
                    # Get the model training script from LLM
                    model_script = get_llm_response(model_training_prompt)
                    
                    # Display the model training script
                    with st.expander("Model Training Script"):
                        st.code(model_script)
                    
                    # Execute model training in E2B sandbox
                    with st.spinner("Training models in E2B sandbox..."):
                        model_results_bytes, best_model_bytes, model_exec_info = execute_model_training_in_e2b(
                            model_script, cleaned_bytes
                        )
                        
                        # Display model training execution info
                        with st.expander("Model Training Execution Info"):
                            st.write(model_exec_info)
                        
                        # Display model results if available
                        if model_results_bytes is not None:
                            with st.expander("Model Results"):
                                # Handle both string and bytes from E2B files.read()
                                if isinstance(model_results_bytes, str):
                                    model_results_df = pd.read_csv(io.StringIO(model_results_bytes))
                                else:
                                    model_results_df = pd.read_csv(io.BytesIO(model_results_bytes))
                                st.write(model_results_df)
                        
                        # Display best model info if available
                        if best_model_bytes is not None:
                            with st.expander("Best Model"):
                                st.success("Best model has been trained and saved!")
                                st.info("Model file is available for download (best_model.pkl)")
                        else:
                            st.warning("Best model file was not created. Check the execution info for details.")
                else:
                    st.error("Cannot proceed with model training - no cleaned data available.")