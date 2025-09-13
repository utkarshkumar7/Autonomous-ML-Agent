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

###################### USER INTERFACE ##########################################################################################
# USES STREAMLIT TO BUILD THE UI AND ADD A TITLE
st.title("Autonomous ML Agent")
# DESCRIPTION TEXT TO GUIDE THE USER ON HOW TO START 
st.markdown("Upload a CSV file to get started")

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

button  = st.button("Run Autonomous ML Agent")

if button:
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
        
        with st.spinner("Executing script in E2B sandbox..."):
            input_csv_bytes = df.to_csv(index=False).encode('utf-8')
            cleaned_bytes, exec_info = execute_in_e2b(script, input_csv_bytes)

            with st.expander("E2B Execution Info"):
                st.write(exec_info)
            
            with st.expander("Cleaned Data"):
                # Handle both string and bytes from E2B files.read()
                if isinstance(cleaned_bytes, str):
                    # If it's a string, use StringIO
                    cleaned_df = pd.read_csv(io.StringIO(cleaned_bytes))
                else:
                    # If it's bytes, use BytesIO
                    cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                st.write(cleaned_df)