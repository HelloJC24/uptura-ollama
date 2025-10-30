"""
Enhanced Document Processor with Dynamic Content Support
Handles JavaScript-rendered content, API calls, and dynamic loading
"""
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Enhanced document processor with dynamic content support"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.discovered_apis: Set[str] = set()
        
    async def fetch_document_content(self, url: str) -> Dict[str, Any]:
        """Enhanced document fetching with dynamic content support"""
        logger.info(f"Fetching enhanced content from: {url}")
        
        result = {
            'url': url,
            'static_content': '',
            'dynamic_content': '',
            'api_data': [],
            'method': 'static',
            'error': None
        }
        
        try:
            # Always try static content first
            static_content = await self._fetch_static_content(url)
            result['static_content'] = static_content
            
            # Try dynamic content if enabled
            if Config.ENABLE_DYNAMIC_CONTENT:
                if Config.BROWSER_TYPE == "playwright" and PLAYWRIGHT_AVAILABLE:
                    dynamic_result = await self._fetch_with_playwright(url)
                elif Config.BROWSER_TYPE == "selenium" and SELENIUM_AVAILABLE:
                    dynamic_result = await self._fetch_with_selenium(url)
                else:
                    logger.warning(f"Dynamic content fetching not available. Browser type: {Config.BROWSER_TYPE}")
                    dynamic_result = None
                
                if dynamic_result:
                    result.update(dynamic_result)
                    result['method'] = 'dynamic'
            
            # Detect and fetch API endpoints
            if Config.DETECT_API_CALLS:
                api_data = await self._discover_and_fetch_apis(url, result.get('api_endpoints', []))
                result['api_data'] = api_data
                
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _fetch_static_content(self, url: str) -> str:
        """Fetch static HTML content"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error fetching static content from {url}: {e}")
            return ""
    
    async def _fetch_with_playwright(self, url: str) -> Dict[str, Any]:
        """Fetch content using Playwright for JavaScript rendering"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available")
            return {}
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=Config.USE_HEADLESS_BROWSER)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = await context.new_page()
                
                # Track network requests to discover API calls
                api_endpoints = []
                
                async def handle_response(response):
                    if response.url != url and (
                        '/api/' in response.url or 
                        response.headers.get('content-type', '').startswith('application/json')
                    ):
                        api_endpoints.append({
                            'url': response.url,
                            'method': response.request.method,
                            'status': response.status
                        })
                
                page.on("response", handle_response)
                
                # Navigate and wait for content
                await page.goto(url, wait_until='networkidle')
                
                # Wait for dynamic content to load
                await asyncio.sleep(Config.BROWSER_WAIT_TIME)
                
                # Try to wait for common dynamic content selectors
                try:
                    await page.wait_for_selector('.agent-card, .profile-card, .team-member, .dynamic-content', timeout=5000)
                except:
                    pass  # Timeout is expected if elements don't exist
                
                # Get the fully rendered content
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                dynamic_text = ' '.join(chunk for chunk in chunks if chunk)
                
                await browser.close()
                
                return {
                    'dynamic_content': dynamic_text,
                    'api_endpoints': api_endpoints
                }
                
        except Exception as e:
            logger.error(f"Error with Playwright for {url}: {e}")
            return {}
    
    async def _fetch_with_selenium(self, url: str) -> Dict[str, Any]:
        """Fetch content using Selenium for JavaScript rendering"""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available")
            return {}
        
        try:
            chrome_options = Options()
            if Config.USE_HEADLESS_BROWSER:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            
            driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            try:
                driver.get(url)
                
                # Wait for dynamic content
                time.sleep(Config.BROWSER_WAIT_TIME)
                
                # Try to wait for common dynamic selectors
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".agent-card, .profile-card, .team-member"))
                    )
                except:
                    pass  # Continue if elements not found
                
                # Get page source after JavaScript execution
                content = driver.page_source
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                dynamic_text = ' '.join(chunk for chunk in chunks if chunk)
                
                return {
                    'dynamic_content': dynamic_text,
                    'api_endpoints': []  # Selenium doesn't easily capture network requests
                }
                
            finally:
                driver.quit()
                
        except Exception as e:
            logger.error(f"Error with Selenium for {url}: {e}")
            return {}
    
    async def _discover_and_fetch_apis(self, base_url: str, api_endpoints: List[Dict]) -> List[Dict]:
        """Discover and fetch data from API endpoints"""
        api_data = []
        
        # Common API endpoint patterns to try
        base_domain = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
        common_apis = [
            f"{base_domain}/api/agents",
            f"{base_domain}/api/team",
            f"{base_domain}/api/profiles",
            f"{base_domain}/api/staff",
            f"{base_domain}/api/members",
            f"{base_domain}/wp-json/wp/v2/team",  # WordPress REST API
            f"{base_domain}/wp-json/wp/v2/staff",
        ]
        
        # Add discovered endpoints
        for endpoint in api_endpoints:
            if endpoint['url'] not in common_apis:
                common_apis.append(endpoint['url'])
        
        for api_url in common_apis:
            try:
                logger.info(f"Trying API endpoint: {api_url}")
                response = self.session.get(api_url, timeout=5)
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        try:
                            json_data = response.json()
                            api_data.append({
                                'url': api_url,
                                'data': json_data,
                                'text': self._extract_text_from_json(json_data)
                            })
                            logger.info(f"Successfully fetched JSON data from {api_url}")
                        except json.JSONDecodeError:
                            pass
                    else:
                        # Try to parse as text
                        text_content = response.text.strip()
                        if text_content:
                            api_data.append({
                                'url': api_url,
                                'data': text_content,
                                'text': text_content
                            })
                            
            except Exception as e:
                logger.debug(f"Failed to fetch from {api_url}: {e}")
                continue
        
        return api_data
    
    def _extract_text_from_json(self, data: Any) -> str:
        """Extract readable text from JSON data"""
        text_parts = []
        
        def extract_from_value(value):
            if isinstance(value, str):
                # Skip URLs and short strings
                if len(value) > 3 and not value.startswith(('http://', 'https://', 'www.')):
                    text_parts.append(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    # Include key names for context
                    if isinstance(v, str) and len(v) > 3:
                        text_parts.append(f"{k}: {v}")
                    else:
                        extract_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    extract_from_value(item)
        
        extract_from_value(data)
        return ' '.join(text_parts)
    
    def get_all_content(self, result: Dict[str, Any]) -> str:
        """Combine all content types into a single text"""
        all_content = []
        
        # Add static content
        if result.get('static_content'):
            all_content.append(result['static_content'])
        
        # Add dynamic content (might have additional info)
        if result.get('dynamic_content'):
            dynamic = result['dynamic_content']
            static = result.get('static_content', '')
            
            # Only add if significantly different from static
            if len(dynamic) > len(static) * 1.1:  # At least 10% more content
                all_content.append(dynamic)
        
        # Add API data
        for api_item in result.get('api_data', []):
            if api_item.get('text'):
                all_content.append(f"API Data from {api_item['url']}: {api_item['text']}")
        
        return ' '.join(all_content)

# Global instance
enhanced_document_processor = EnhancedDocumentProcessor()