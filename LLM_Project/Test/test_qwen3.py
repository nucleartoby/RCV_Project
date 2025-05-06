from qwen3_client import Qwen3Client

API_KEY = "put api key here"

client = Qwen3Client(api_key=API_KEY)

def test_generate():
    result = client.generate(
        prompt="Write a short poem about artificial intelligence.",
        max_tokens=150,
        temperature=0.8
    )
    print("=== Text Generation Result ===")
    print(result)
    print("\n")

def test_chat():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are three benefits of AI?"}
    ]
    result = client.chat(
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )
    print("=== Chat Result ===")
    print(result)

if __name__ == "__main__":
    test_generate()
    test_chat()