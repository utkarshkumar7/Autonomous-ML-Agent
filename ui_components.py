"""
UI component functions for the Autonomous ML Agent.
This module contains all functions responsible for creating and managing UI components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Dict, Any, Optional


def create_leaderboard_ui(model_results_df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None) -> None:
    """
    Create a leaderboard UI showing model performance metrics.
    
    Args:
        model_results_df (pd.DataFrame): The model results dataframe
        analysis_results (dict): LLM analysis results
    """
    st.markdown("### 🏆 Model Leaderboard")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display model results as a formatted table
        if not model_results_df.empty:
            # Sort by best metric (assuming first numeric column is the main metric)
            numeric_cols = model_results_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                main_metric = numeric_cols[0]
                model_results_df_sorted = model_results_df.sort_values(main_metric, ascending=False)
                
                # Highlight the best model
                st.dataframe(
                    model_results_df_sorted.style.highlight_max(axis=0, color='lightgreen'),
                    use_container_width=True
                )
            else:
                st.dataframe(model_results_df, use_container_width=True)
        else:
            st.warning("No model results available")
    
    with col2:
        # Display best model info
        if analysis_results and 'best_model' in analysis_results:
            st.markdown("#### 🥇 Best Model")
            st.success(f"**{analysis_results['best_model']}**")
            if 'best_score' in analysis_results:
                st.metric("Best Score", f"{analysis_results['best_score']:.4f}")
        
        # Display key metrics
        if not model_results_df.empty:
            st.markdown("#### 📊 Key Metrics")
            numeric_cols = model_results_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:  # Show top 3 metrics
                best_val = model_results_df[col].max()
                best_model = model_results_df.loc[model_results_df[col].idxmax(), 'Model']
                st.metric(f"Best {col}", f"{best_val:.4f}", f"({best_model})")


def display_model_analysis(analysis_results: Optional[Dict[str, Any]]) -> None:
    """
    Display LLM-generated model analysis and insights.
    
    Args:
        analysis_results (dict): LLM analysis results
    """
    if not analysis_results:
        return
    
    st.markdown("### 🔍 Model Analysis & Insights")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analysis", "💡 Insights", "🚀 Recommendations", "📝 Summary"])
    
    with tab1:
        if 'analysis' in analysis_results:
            st.markdown("#### Detailed Performance Analysis")
            st.write(analysis_results['analysis'])
    
    with tab2:
        if 'insights' in analysis_results:
            st.markdown("#### Key Insights")
            st.write(analysis_results['insights'])
    
    with tab3:
        if 'recommendations' in analysis_results:
            st.markdown("#### Improvement Recommendations")
            st.write(analysis_results['recommendations'])
    
    with tab4:
        if 'summary' in analysis_results:
            st.markdown("#### Executive Summary")
            st.info(analysis_results['summary'])


def display_cleaned_data(cleaned_bytes: bytes) -> pd.DataFrame:
    """
    Display cleaned data and return as DataFrame.
    
    Args:
        cleaned_bytes (bytes): The cleaned CSV data as bytes
        
    Returns:
        pd.DataFrame: The cleaned dataframe
    """
    # Handle both string and bytes from E2B files.read()
    if isinstance(cleaned_bytes, str):
        cleaned_df = pd.read_csv(io.StringIO(cleaned_bytes))
    else:
        cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
    
    # Display the cleaned dataframe
    st.write(cleaned_df)
    return cleaned_df


def display_model_results(model_results_bytes: bytes) -> pd.DataFrame:
    """
    Display model results and return as DataFrame.
    
    Args:
        model_results_bytes (bytes): The model results CSV data as bytes
        
    Returns:
        pd.DataFrame: The model results dataframe
    """
    # Handle both string and bytes from E2B files.read()
    if isinstance(model_results_bytes, str):
        model_results_df = pd.read_csv(io.StringIO(model_results_bytes))
    else:
        model_results_df = pd.read_csv(io.BytesIO(model_results_bytes))
    
    # Display the model results dataframe
    st.write(model_results_df)
    return model_results_df
