"""
E2B sandbox execution functions for the Autonomous ML Agent.
This module contains all functions responsible for executing code in E2B sandboxes.
"""

import os
import streamlit as st
from typing import Optional, Tuple, Dict, Any
from e2b_code_interpreter import Sandbox
import pandas as pd
import io


def execute_in_e2b(script: str, csv_bytes: bytes) -> Tuple[Optional[bytes], Dict[str, Any]]:
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


def execute_model_training_in_e2b(script: str, cleaned_csv_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Execute a model training script in an E2B sandbox with the cleaned CSV data.
    
    Args:
        script (str): The Python script to execute for model training
        cleaned_csv_bytes (bytes): The cleaned CSV file data as bytes
        
    Returns:
        tuple: (model_results_df, exec_info) or (None, exec_info) on error
    """
    
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        st.error("E2B_API_KEY is not set")
        return None, {}
    
    # Create a dictionary to store execution results
    exec_info = {}
    
    try:
        # Create E2B sandbox instance with timeout
        sandbox = Sandbox.create(api_key=api_key, timeout=600)
        
        # Upload the cleaned CSV file to the sandbox
        sandbox.files.write("/tmp/cleaned.csv", cleaned_csv_bytes)
        
        # Execute the model training script
        result = sandbox.run_code(script)
        
        # Store execution results
        exec_info["exit_code"] = getattr(result, "exit_code", 0)
        exec_info["stdout"] = getattr(result, "stdout", "")
        exec_info["stderr"] = getattr(result, "stderr", "")
        
        # Try to download model results if script executed successfully
        if exec_info["exit_code"] == 0:
            try:
                model_results_bytes = sandbox.files.read("/tmp/model_results.csv")
                if isinstance(model_results_bytes, str):
                    model_results_df = pd.read_csv(io.StringIO(model_results_bytes))
                else:
                    model_results_df = pd.read_csv(io.BytesIO(model_results_bytes))
                return model_results_df, exec_info
            except Exception as e:
                st.error(f"Could not download model_results.csv: {str(e)}")
                return None, exec_info
        else:
            st.error(f"Model training script failed with exit code {exec_info['exit_code']}")
            return None, exec_info
                
    except Exception as e:
        st.error(f"Error executing model training in E2B: {str(e)}")
        return None, exec_info
    finally:
        # Clean up sandbox
        try:
            sandbox.kill()
        except:
            pass
