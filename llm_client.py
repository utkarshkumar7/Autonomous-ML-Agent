"""
LLM client functions for the Autonomous ML Agent.
This module contains all functions responsible for interacting with LLMs.
"""

import os
import json
import streamlit as st
from typing import Optional
from openai import OpenAI


# SYSTEM PROMPT FOR THE LLM
SYSTEM_PROMPT = "You are a senior data engineer. Always return a strict JSON object matching the user's requested schema."


def get_llm_response(prompt: str) -> Optional[str]:
    """
    Get a response from the LLM for the given prompt.
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        Optional[str]: The LLM response or None if error
    """
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        if not response or not getattr(response, "choices", None):
            return None
        text = response.choices[0].message.content or ""

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

        # IF NO SCRIPT FOUND, RETURN NONE
        return None

    except Exception as e:
        st.error(f"Error getting LLM response: {str(e)}")
        return None


def get_llm_analysis_response(prompt: str) -> Optional[dict]:
    """
    Get an analysis response from the LLM and parse it as JSON.
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        Optional[dict]: The parsed JSON response or None if error
    """
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        if not response or not getattr(response, "choices", None):
            return None
        text = response.choices[0].message.content or ""

        # Handle OpenRouter's response format with special tokens
        if "<|message|>" in text:
            # Extract JSON content between <|message|> and the end
            json_start = text.find("<|message|>") + len("<|message|>")
            json_content = text[json_start:].strip()
        else:
            # If no special tokens, use the entire response
            json_content = text
        
        return json.loads(json_content)

    except Exception as e:
        st.error(f"Error getting LLM analysis response: {str(e)}")
        return None
