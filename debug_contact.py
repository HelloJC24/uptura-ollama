"""
Debug script to test the BNGC contact information query
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_contact_query():
    """Test the BNGC contact information query"""
    print("=== Testing BNGC Contact Information Query ===\n")
    
    query = "what is the contact information of BNGC?"
    user_id = "debug@test.com"
    
    print(f"Query: {query}")
    print(f"User ID: {user_id}")
    print("-" * 50)
    
    # First, use the debug endpoint to see what's happening
    print("1. Running debug analysis...")
    debug_response = requests.post(f"{BASE_URL}/debug/query", json={
        "query": query
    })
    
    if debug_response.status_code == 200:
        debug_data = debug_response.json()
        
        print(f"Total chunks available: {debug_data['total_chunks']}")
        print(f"Relevant chunks found: {debug_data['relevant_chunks_found']}")
        print(f"Chunks above threshold: {debug_data['chunks_above_threshold']}")
        print(f"Max similarity: {debug_data['max_similarity']}")
        print(f"Min similarity threshold: {debug_data['min_similarity_threshold']}")
        print(f"Config - TOP_K: {debug_data['config']['top_k']}")
        
        print("\nTop similarities:")
        for i, sim in enumerate(debug_data['top_similarities'][:5]):
            print(f"  {i+1}. {sim['similarity']:.4f} - {sim['source_url']}")
            print(f"     Preview: {sim['text_preview'][:100]}...")
        
        if debug_data['relevant_chunks_found'] > 0:
            print(f"\nRelevant chunks that will be used:")
            for i, chunk in enumerate(debug_data['relevant_chunks']):
                print(f"  {i+1}. {chunk['source_url']}")
                print(f"     Preview: {chunk['text_preview'][:100]}...")
        else:
            print("\n❌ NO RELEVANT CHUNKS FOUND!")
            print("This means the similarity threshold is too high or the documents don't contain contact info.")
    else:
        print(f"Debug endpoint failed: {debug_response.text}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Now test the actual query
    print("2. Testing actual query...")
    response = requests.post(f"{BASE_URL}/ask", json={
        "query": query,
        "user_id": user_id,
        "streaming": False
    })
    
    if response.status_code == 200:
        print("Response:")
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = json.loads(line[6:])
                if "answer" in data:
                    print(f"Answer: {data['answer']}")
                elif "status" in data:
                    print(f"Status: {data['status']}")
    else:
        print(f"Query failed: {response.text}")

def test_different_contact_queries():
    """Test various contact-related queries"""
    print("\n=== Testing Different Contact Queries ===\n")
    
    queries = [
        "BNGC contact information",
        "Gogel contact details", 
        "how to contact BNGC",
        "BNGC phone number",
        "BNGC email address",
        "BNGC office address"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"{i}. Testing: {query}")
        
        debug_response = requests.post(f"{BASE_URL}/debug/query", json={
            "query": query
        })
        
        if debug_response.status_code == 200:
            debug_data = debug_response.json()
            print(f"   Max similarity: {debug_data['max_similarity']:.4f}")
            print(f"   Relevant chunks: {debug_data['relevant_chunks_found']}")
            
            if debug_data['relevant_chunks_found'] > 0:
                print(f"   ✅ Would find relevant content")
            else:
                print(f"   ❌ No relevant content found")
        else:
            print(f"   Error: {debug_response.status_code}")
        
        print()

def suggest_fixes():
    """Suggest potential fixes based on the debug results"""
    print("\n=== Suggested Fixes ===\n")
    
    print("If no relevant chunks are found:")
    print("1. Lower the MIN_SIMILARITY threshold (currently 0.25)")
    print("2. Increase TOP_K to retrieve more chunks")
    print("3. Check if the documents actually contain contact information")
    print("4. Reload documents to ensure they're properly processed")
    
    print("\nTo lower similarity threshold, set environment variable:")
    print("MIN_SIMILARITY=0.15")
    
    print("\nTo reload documents:")
    print("POST http://localhost:5000/reload-documents")

if __name__ == "__main__":
    print("🔍 BNGC Contact Information Debug Tool")
    print("=" * 60)
    
    try:
        test_contact_query()
        test_different_contact_queries()
        suggest_fixes()
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")