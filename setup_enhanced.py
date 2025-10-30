"""
Installation script for enhanced document processing dependencies
Installs browser automation tools (Selenium, Playwright) and sets up drivers
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies for enhanced document processing")
    print("=" * 60)
    
    # Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements.txt"
    )
    
    if not success:
        print("⚠️ Requirements installation failed, trying individual packages...")
        
        packages = [
            "selenium>=4.0.0",
            "webdriver-manager>=3.8.0", 
            "playwright>=1.30.0",
            "beautifulsoup4",
            "requests"
        ]
        
        for package in packages:
            run_command(
                f"{sys.executable} -m pip install {package}",
                f"Installing {package}"
            )

def setup_playwright():
    """Setup Playwright browsers"""
    print("\n🎭 Setting up Playwright browsers")
    print("-" * 40)
    
    # Install Playwright browsers
    run_command(
        f"{sys.executable} -m playwright install chromium",
        "Installing Playwright Chromium browser"
    )
    
    # Install system dependencies (Linux/WSL)
    if os.name == 'posix':
        run_command(
            f"{sys.executable} -m playwright install-deps",
            "Installing Playwright system dependencies"
        )

def verify_installations():
    """Verify that installations work"""
    print("\n🔍 Verifying installations")
    print("-" * 40)
    
    # Test Selenium
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        print("✅ Selenium imports successful")
        
        # Test ChromeDriver download (doesn't start browser)
        try:
            ChromeDriverManager().install()
            print("✅ ChromeDriver installation works")
        except Exception as e:
            print(f"⚠️ ChromeDriver installation failed: {e}")
            
    except ImportError as e:
        print(f"❌ Selenium import failed: {e}")
    
    # Test Playwright
    try:
        import playwright
        print("✅ Playwright import successful")
        
        # Test async playwright import
        from playwright.async_api import async_playwright
        print("✅ Playwright async API import successful")
        
    except ImportError as e:
        print(f"❌ Playwright import failed: {e}")
    
    # Test enhanced processor
    try:
        from enhanced_document_processor import enhanced_document_processor, PLAYWRIGHT_AVAILABLE, SELENIUM_AVAILABLE
        print(f"✅ Enhanced document processor import successful")
        print(f"   Playwright available: {PLAYWRIGHT_AVAILABLE}")
        print(f"   Selenium available: {SELENIUM_AVAILABLE}")
        
    except ImportError as e:
        print(f"❌ Enhanced document processor import failed: {e}")

def main():
    """Main installation process"""
    print("🚀 Enhanced Document Processing Setup")
    print("=" * 60)
    print("This script will install browser automation tools for dynamic content fetching")
    print("Components:")
    print("  - Selenium WebDriver with Chrome")
    print("  - Playwright with Chromium")
    print("  - Required Python packages")
    print("=" * 60)
    
    # Install Python dependencies
    install_dependencies()
    
    # Setup Playwright
    setup_playwright()
    
    # Verify installations
    verify_installations()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Test the installation with: python test_enhanced_processing.py")
    print("2. Start the API with enhanced document processing enabled")
    print("3. Use /enhanced-reload endpoint to process documents with dynamic content")
    print("\nConfiguration:")
    print("  Set ENABLE_DYNAMIC_CONTENT=true in your environment")
    print("  Choose BROWSER_TYPE=playwright or selenium")
    print("  Set USE_HEADLESS_BROWSER=true for production")

if __name__ == "__main__":
    main()