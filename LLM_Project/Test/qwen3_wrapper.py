import requests
from typing import Dict, List, Optional, Union, Any

class Qwen3Client:
    """A wrapper for the Qwen3 API."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.qwen.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 100, 
                temperature: float = 0.7,
                top_p: float = 1.0,
                **kwargs) -> Dict[str, Any]:
        """Generate text based on the prompt."""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
            
        return response.json()
    
    def chat(self, 
            messages: List[Dict[str, str]], 
            max_tokens: int = 100,
            temperature: float = 0.7,
            **kwargs) -> Dict[str, Any]:
        """Chat completion with message history."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
            
        return response.json()