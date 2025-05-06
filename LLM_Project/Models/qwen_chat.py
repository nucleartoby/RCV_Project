import requests
from typing import Dict, List, Optional, Union, Any, Generator
import os
import sys
import json
import datetime

class Qwen3Client:
    
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
    
    def chat_stream(self, 
                  messages: List[Dict[str, str]], 
                  max_tokens: int = 100,
                  temperature: float = 0.7,
                  **kwargs) -> Generator[str, None, None]:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
            
        for line in response.iter_lines():
            if line:
                yield line.decode('utf-8')


def save_conversation(messages, filename=None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qwen3_conversation_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Qwen3 Conversation ===\n\n")
        for msg in messages:
            if msg["role"] != "system":
                f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
    
    print(f"Conversation saved to {filename}")


def display_help():
    print("\n=== Qwen3 Chat Commands ===")
    print("/help      - Display this help message")
    print("/exit      - Exit the chat")
    print("/save      - Save the conversation to a file")
    print("/save filename.txt - Save to specific file")
    print("/system    - Set a new system message")
    print("/temp 0.7  - Set temperature (0.0 to 1.0)")
    print("/tokens 500 - Set max tokens (1 to 4096)")
    print("/clear     - Clear conversation history")
    print("/stream    - Toggle streaming mode")
    print("========================\n")


def main():
    api_key = os.environ.get("QWEN3_API_KEY")
    if not api_key:
        api_key = input("Please enter your Qwen3 API key: ")
        if not api_key:
            print("Error: API key is required")
            sys.exit(1)

    client = Qwen3Client(api_key=api_key)

    temperature = 0.7
    max_tokens = 500
    use_streaming = False

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    print("\n===== Welcome to Qwen3 Chat =====")
    print("Type '/help' to see available commands")
    print("Type '/exit' or '/quit' to end the conversation.\n")

    while True:
        user_input = input("\nYou: ")

        if user_input.startswith('/'):
            parts = user_input[1:].split(' ', 1)
            command = parts[0].lower()
            
            if command in ["exit", "quit"]:
                print("\nThank you for chatting with Qwen3. Goodbye!")
                break
                
            elif command == "help":
                display_help()
                continue
                
            elif command == "save":
                if len(parts) > 1:
                    save_conversation(messages, parts[1])
                else:
                    save_conversation(messages)
                continue
                
            elif command == "system" and len(parts) > 1:
                system_message = parts[1]
                messages = [{"role": "system", "content": system_message}] + [
                    msg for msg in messages if msg["role"] != "system"
                ]
                print(f"System message updated to: {system_message}")
                continue
                
            elif command == "temp" and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    if 0.0 <= temperature <= 1.0:
                        print(f"Temperature set to {temperature}")
                    else:
                        print("Temperature must be between 0.0 and 1.0")
                    continue
                except ValueError:
                    print("Invalid temperature value")
                    continue
                    
            elif command == "tokens" and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    if max_tokens > 0:
                        print(f"Max tokens set to {max_tokens}")
                    else:
                        print("Max tokens must be positive")
                    continue
                except ValueError:
                    print("Invalid token value")
                    continue
                    
            elif command == "clear":
                system_message = messages[0]["content"] if messages and messages[0]["role"] == "system" else "You are a helpful assistant."
                messages = [{"role": "system", "content": system_message}]
                print("Conversation history cleared")
                continue
                
            elif command == "stream":
                use_streaming = not use_streaming
                print(f"Streaming mode: {'ON' if use_streaming else 'OFF'}")
                continue
                
            else:
                print(f"Unknown command: {command}")
                print("Type '/help' for a list of commands")
                continue

        messages.append({"role": "user", "content": user_input})
        
        try:
            if use_streaming:
                print("\nQwen3:", end=" ")
                assistant_message = ""

                for chunk in client.chat_stream(
                    messages=messages, 
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    try:
                        data = json.loads(chunk)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                assistant_message += content
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
                
                print()

                if assistant_message:
                    messages.append({"role": "assistant", "content": assistant_message})

                    if "usage" in data:
                        usage = data["usage"]
                        print(f"\n[Token usage: {usage.get('total_tokens', 'N/A')} tokens]")
            
            else:
                print("\nQwen3 is thinking...", end="\r")
                response = client.chat(
                    messages=messages, 
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                if "choices" in response and len(response["choices"]) > 0:
                    assistant_message = response["choices"][0]["message"]["content"]
                    print(f"\nQwen3: {assistant_message}")

                    messages.append({"role": "assistant", "content": assistant_message})

                    if "usage" in response:
                        usage = response["usage"]
                        print(f"\n[Token usage: {usage.get('total_tokens', 'N/A')} tokens]")
                else:
                    print("\nQwen3: [No response received]")
                    print("Unexpected API response format:", response)
        
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()