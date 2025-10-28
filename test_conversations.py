"""
Test script for conversation features
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_conversation_flow():
    """Test a complete conversation flow"""
    user_id = "test@example.com"
    
    print(f"=== Testing Conversation Flow for {user_id} ===\n")
    
    # Test 1: First question
    print("1. Asking first question...")
    response1 = requests.post(f"{BASE_URL}/ask", json={
        "query": "do you know Gogel?",
        "user_id": user_id,
        "streaming": False
    })
    
    if response1.status_code == 200:
        # Parse streaming response
        for line in response1.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = json.loads(line[6:])
                if "answer" in data:
                    print(f"Answer: {data['answer']}")
                    break
    else:
        print(f"Error: {response1.text}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 2: Follow-up question (should have context)
    print("2. Asking follow-up question...")
    response2 = requests.post(f"{BASE_URL}/ask", json={
        "query": "What services do they offer?",
        "user_id": user_id,
        "streaming": False
    })
    
    if response2.status_code == 200:
        for line in response2.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = json.loads(line[6:])
                if "answer" in data:
                    print(f"Answer: {data['answer']}")
                    break
    else:
        print(f"Error: {response2.text}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 3: Get conversation history
    print("3. Getting conversation history...")
    history_response = requests.get(f"{BASE_URL}/conversations/{user_id}/history")
    
    if history_response.status_code == 200:
        history_data = history_response.json()
        print(f"Conversation stats: {history_data['stats']}")
        print(f"Message count: {len(history_data['history'])}")
        
        for i, msg in enumerate(history_data['history']):
            print(f"  {i+1}. [{msg['role']}]: {msg['content'][:100]}...")
    else:
        print(f"Error getting history: {history_response.text}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 4: Test without user_id (should work but no history)
    print("4. Testing without user_id...")
    response3 = requests.post(f"{BASE_URL}/ask", json={
        "query": "What is Gogel again?",
        "streaming": False
    })
    
    if response3.status_code == 200:
        for line in response3.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = json.loads(line[6:])
                if "answer" in data:
                    print(f"Answer (no context): {data['answer']}")
                    break
    
    return user_id

def test_conversation_management():
    """Test conversation management endpoints"""
    print("\n=== Testing Conversation Management ===\n")
    
    # List all conversations
    print("1. Listing all conversations...")
    list_response = requests.get(f"{BASE_URL}/conversations")
    
    if list_response.status_code == 200:
        list_data = list_response.json()
        print(f"Active conversations: {list_data['active_conversations']}")
        print(f"Total count: {list_data['total_count']}")
        print(f"Service stats: {list_data['service_stats']}")
    else:
        print(f"Error: {list_response.text}")
    
    print("\n" + "-"*30 + "\n")
    
    return list_response.json().get('active_conversations', []) if list_response.status_code == 200 else []

def test_conversation_cleanup(user_id):
    """Test conversation cleanup"""
    print("2. Testing conversation cleanup...")
    
    # Clear specific user conversation
    clear_response = requests.post(f"{BASE_URL}/conversations/{user_id}/clear")
    
    if clear_response.status_code == 200:
        print(f"Successfully cleared conversation for {user_id}")
    else:
        print(f"Error clearing conversation: {clear_response.text}")
    
    # Verify conversation is cleared
    history_response = requests.get(f"{BASE_URL}/conversations/{user_id}/history")
    if history_response.status_code == 200:
        history_data = history_response.json()
        print(f"Messages after clear: {len(history_data['history'])}")

def test_streaming_with_conversation():
    """Test streaming with conversation context"""
    print("\n=== Testing Streaming with Conversations ===\n")
    
    user_id = "streaming@example.com"
    
    print("1. First streaming question...")
    response = requests.post(f"{BASE_URL}/stream", json={
        "query": "Tell me about Gogel's services"
    }, stream=True)
    
    if response.status_code == 200:
        print("Streaming response:")
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data_part = line[6:]
                try:
                    parsed = json.loads(data_part)
                    if "answer" in parsed:
                        print(parsed["answer"], end="", flush=True)
                    elif "status" in parsed:
                        print(f"\n[{parsed['status'].upper()}]", end="")
                        if "message" in parsed:
                            print(f" {parsed['message']}")
                except json.JSONDecodeError:
                    pass
        print("\n")

def test_health_and_stats():
    """Test health and stats endpoints"""
    print("\n=== Testing Health and Stats ===\n")
    
    # Health check
    health_response = requests.get(f"{BASE_URL}/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f"Health status: {health_data['status']}")
        print(f"Services: {health_data['services']}")
        print(f"Config: {health_data['config']}")
    
    print("\n" + "-"*30 + "\n")
    
    # Stats
    stats_response = requests.get(f"{BASE_URL}/stats")
    if stats_response.status_code == 200:
        stats_data = stats_response.json()
        if "conversations" in stats_data['stats']:
            print(f"Conversation stats: {stats_data['stats']['conversations']}")

if __name__ == "__main__":
    print("🤖 Flask RAG API Conversation Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        user_id = test_conversation_flow()
        active_conversations = test_conversation_management()
        test_conversation_cleanup(user_id)
        test_streaming_with_conversation()
        test_health_and_stats()
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")