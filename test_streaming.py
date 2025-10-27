"""
Test script for the streaming API
"""
import requests
import json
import time

def test_streaming():
    """Test the streaming endpoint"""
    url = "http://localhost:5000/stream"
    
    data = {
        "query": "do you know Gogel?"
    }
    
    print("Testing streaming endpoint...")
    print(f"Query: {data['query']}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=data, stream=True, timeout=60)
        response.raise_for_status()
        
        print("Streaming response:")
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    data_part = line[6:]  # Remove "data: " prefix
                    try:
                        parsed = json.loads(data_part)
                        
                        if "status" in parsed:
                            print(f"[STATUS] {parsed['status']}: {parsed.get('message', '')}")
                        elif "answer" in parsed:
                            print(parsed["answer"], end="", flush=True)
                        elif "error" in parsed:
                            print(f"\n[ERROR] {parsed['error']}")
                            
                    except json.JSONDecodeError:
                        print(f"[RAW] {data_part}")
        
        print(f"\n\nTotal time: {time.time() - start_time:.2f} seconds")
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Error: {e}")

def test_regular_endpoint():
    """Test the regular /ask endpoint with streaming=true"""
    url = "http://localhost:5000/ask"
    
    data = {
        "query": "do you know Gogel?",
        "streaming": True
    }
    
    print("\n\nTesting regular endpoint with streaming=True...")
    print(f"Query: {data['query']}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=data, stream=True, timeout=60)
        response.raise_for_status()
        
        print("Response:")
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    data_part = line[6:]  # Remove "data: " prefix
                    try:
                        parsed = json.loads(data_part)
                        
                        if "status" in parsed:
                            print(f"[STATUS] {parsed['status']}")
                        elif "answer" in parsed:
                            print(parsed["answer"], end="", flush=True)
                        elif "error" in parsed:
                            print(f"\n[ERROR] {parsed['error']}")
                            
                    except json.JSONDecodeError:
                        print(f"[RAW] {data_part}")
        
        print(f"\n\nTotal time: {time.time() - start_time:.2f} seconds")
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Flask RAG API Streaming Test ===")
    
    # Test both endpoints
    test_streaming()
    test_regular_endpoint()
    
    print("\n=== Test completed ===")