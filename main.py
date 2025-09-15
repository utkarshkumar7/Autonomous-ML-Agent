"""
Autonomous ML Agent - Main Application
A Streamlit application that automates the entire ML pipeline from data cleaning to model deployment.
"""

import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv

# Import our custom modules
from prompts import (
    build_cleaning_prompt,
    build_model_training_prompt,
    build_model_analysis_prompt
)
from e2b_executor import (
    execute_in_e2b,
    execute_model_training_in_e2b
)
from ui_components import (
    create_leaderboard_ui,
    display_model_analysis,
    display_cleaned_data
)
from llm_client import (
    get_llm_response,
    get_llm_analysis_response
)

# Load environment variables from .env file
load_dotenv()

###################### USER INTERFACE ##########################################################################################

# USES STREAMLIT TO BUILD THE UI AND ADD A TITLE
st.title("Autonomous ML Agent 🕵🏻")

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
    button = st.button("Run Autonomous ML Agent")

    # ONCE THE BUTTON IS CLICKED, THE AUTONOMOUS ML AGENT STARTS RUNNING
    if button:
        # SPINNER TO SHOW THAT THE AUTONOMOUS ML AGENT IS RUNNING
        with st.spinner("Running Autonomous ML Agent..."):
            # STEP 1: DATA CLEANING
            st.markdown("### Step 1: Data Cleaning")
            
            # BUILD THE CLEANING PROMPT
            cleaning_prompt = build_cleaning_prompt(df, selected_column)
            
            # DISPLAY THE CLEANING PROMPT IN AN ACCORDION
            with st.expander("Cleaning Prompt"):
                st.write(cleaning_prompt)
            
            # GET THE CLEANING SCRIPT FROM LLM
            script = get_llm_response(cleaning_prompt)
            
            # DISPLAY THE SCRIPT IN AN ACCORDION
            with st.expander("Cleaning Script"):
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
                    if cleaned_bytes is not None:
                        cleaned_df = display_cleaned_data(cleaned_bytes)

                if cleaned_df is not None:         
                    # STEP 2: MODEL TRAINING
                    st.markdown("---")
                    st.markdown("### Step 2: Model Training")
                    
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
                    
                    # Initialize variables
                    model_results_df = None
                    model_exec_info = {}

                    # Execute model training in E2B sandbox
                    with st.spinner("Training models in E2B sandbox..."):
                        model_results_df, model_exec_info = execute_model_training_in_e2b(
                            model_script, cleaned_bytes
                        )
                        
                    # Display model training execution info
                    with st.expander("Model Training Execution Info"):
                        st.write(model_exec_info)
                        
                    # Display model results if available
                    if model_results_df is not None:
                        # Display the model results dataframe
                        st.write(model_results_df)
                        
                        # STEP 3: MODEL ANALYSIS AND LEADERBOARD
                        st.markdown("---")
                        st.markdown("### Step 3: Model Analysis & Leaderboard")
                        
                        # Generate model analysis using LLM
                        with st.spinner("Analyzing model results..."):
                            analysis_prompt = build_model_analysis_prompt(model_results_df, selected_column)
                            analysis_results = get_llm_analysis_response(analysis_prompt)
                        
                        # Display leaderboard and analysis
                        create_leaderboard_ui(model_results_df, analysis_results)
                        display_model_analysis(analysis_results)
                    else:
                        st.error("Model training failed. Check the execution info for details.")