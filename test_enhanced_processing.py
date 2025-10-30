"""
Test script for enhanced document processing
Tests both static and dynamic content fetching capabilities
"""
import asyncio
import json
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_document_processor import enhanced_document_processor
from config import Config

async def test_enhanced_processing():
    """Test enhanced document processing capabilities"""
    
    print("🚀 Testing Enhanced Document Processing")
    print(f"Dynamic content enabled: {Config.ENABLE_DYNAMIC_CONTENT}")
    print(f"Browser type: {Config.BROWSER_TYPE}")
    print(f"API detection: {Config.DETECT_API_CALLS}")
    print("-" * 60)
    
    # Test URLs - mix of static and potentially dynamic content
    test_urls = [
        "https://google.com",  # Basic test
        "https://gogel.com.au",  # Target website
        "https://bngc.net.au",  # Target website
    ]
    
    for url in test_urls:
        print(f"\n📄 Testing: {url}")
        print("-" * 40)
        
        try:
            # Fetch enhanced content
            result = await enhanced_document_processor.fetch_document_content(url)
            
            # Display results
            print(f"Method used: {result.get('method', 'unknown')}")
            print(f"Error: {result.get('error', 'None')}")
            
            static_content = result.get('static_content', '')
            dynamic_content = result.get('dynamic_content', '')
            api_data = result.get('api_data', [])
            
            print(f"Static content length: {len(static_content)} characters")
            print(f"Dynamic content length: {len(dynamic_content)} characters")
            print(f"API endpoints found: {len(api_data)}")
            
            if api_data:
                print("API endpoints:")
                for api in api_data[:3]:  # Show first 3
                    print(f"  - {api.get('url', 'Unknown URL')}")
            
            # Get combined content
            all_content = enhanced_document_processor.get_all_content(result)
            print(f"Combined content length: {len(all_content)} characters")
            
            # Show a preview of the content
            preview = all_content[:300].replace('\n', ' ').strip()
            print(f"Content preview: {preview}...")
            
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
        
        print("-" * 40)
    
    print("\n✅ Enhanced document processing test completed!")

async def test_api_discovery():
    """Test API endpoint discovery"""
    print("\n🔍 Testing API Discovery")
    print("-" * 40)
    
    # Test with a website that might have APIs
    test_url = "https://gogel.com.au"
    
    try:
        result = await enhanced_document_processor.fetch_document_content(test_url)
        api_data = result.get('api_data', [])
        
        print(f"API discovery results for {test_url}:")
        print(f"Total API endpoints tested: {len(api_data)}")
        
        successful_apis = [api for api in api_data if api.get('data')]
        print(f"Successful API calls: {len(successful_apis)}")
        
        for api in successful_apis:
            print(f"  ✅ {api['url']}")
            text_content = api.get('text', '')
            print(f"     Data length: {len(text_content)} characters")
            if text_content:
                preview = text_content[:100].replace('\n', ' ').strip()
                print(f"     Preview: {preview}...")
        
    except Exception as e:
        print(f"❌ Error in API discovery test: {e}")

def test_static_processing():
    """Test static content processing (fallback)"""
    print("\n📄 Testing Static Content Processing")
    print("-" * 40)
    
    import requests
    from bs4 import BeautifulSoup
    
    test_url = "https://google.com"
    
    try:
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        static_text = ' '.join(chunk for chunk in chunks if chunk)
        
        print(f"Static processing successful for {test_url}")
        print(f"Content length: {len(static_text)} characters")
        preview = static_text[:200].replace('\n', ' ').strip()
        print(f"Preview: {preview}...")
        
    except Exception as e:
        print(f"❌ Error in static processing test: {e}")

async def main():
    """Run all tests"""
    print("🧪 Enhanced Document Processor Test Suite")
    print("=" * 60)
    
    # Test 1: Enhanced processing
    await test_enhanced_processing()
    
    # Test 2: API discovery
    await test_api_discovery()
    
    # Test 3: Static processing (baseline)
    test_static_processing()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    
    # Configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  ENABLE_DYNAMIC_CONTENT: {Config.ENABLE_DYNAMIC_CONTENT}")
    print(f"  BROWSER_TYPE: {Config.BROWSER_TYPE}")
    print(f"  USE_HEADLESS_BROWSER: {Config.USE_HEADLESS_BROWSER}")
    print(f"  BROWSER_WAIT_TIME: {Config.BROWSER_WAIT_TIME}")
    print(f"  DETECT_API_CALLS: {Config.DETECT_API_CALLS}")
    print(f"  API_WAIT_TIME: {Config.API_WAIT_TIME}")

if __name__ == "__main__":
    asyncio.run(main())