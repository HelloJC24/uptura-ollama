# Enhanced Document Processing Setup Guide

This guide explains how to set up and use the enhanced document processing features that can fetch dynamic JavaScript-loaded content from websites.

## 🚀 Quick Start

### 1. Install Dependencies

Run the setup script to install browser automation tools:

```bash
python setup_enhanced.py
```

Or install manually:

```bash
pip install selenium>=4.0.0 webdriver-manager>=3.8.0 playwright>=1.30.0
python -m playwright install chromium
```

### 2. Configure Environment

Set these environment variables or update your `.env` file:

```bash
# Enable dynamic content fetching
ENABLE_DYNAMIC_CONTENT=true

# Choose browser (playwright or selenium)
BROWSER_TYPE=playwright

# Browser settings
USE_HEADLESS_BROWSER=true
BROWSER_WAIT_TIME=3

# API endpoint discovery
DETECT_API_CALLS=true
API_WAIT_TIME=2
```

### 3. Test the Setup

```bash
python test_enhanced_processing.py
```

### 4. Start the API

```bash
python ollama_api.py
```

## 🔧 Configuration Options

### Core Settings

- `ENABLE_DYNAMIC_CONTENT`: Enable/disable dynamic content fetching (default: false)
- `BROWSER_TYPE`: Choose between "playwright" or "selenium" (default: playwright)
- `USE_HEADLESS_BROWSER`: Run browser in headless mode (default: true)
- `BROWSER_WAIT_TIME`: Seconds to wait for dynamic content to load (default: 3)

### API Discovery

- `DETECT_API_CALLS`: Try to discover and fetch API endpoints (default: true)
- `API_WAIT_TIME`: Timeout for API endpoint requests (default: 2)

## 📡 New API Endpoints

### Enhanced Document Reload

```bash
POST /enhanced-reload
```

Reloads documents using enhanced processing with dynamic content fetching.

**Request:**
```json
{
  "urls": ["https://gogel.com.au", "https://bngc.net.au"]  // optional
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Enhanced document reload started in background",
  "urls": ["https://gogel.com.au"],
  "dynamic_content_enabled": true,
  "browser_type": "playwright",
  "api_detection_enabled": true
}
```

### Processing Status

```bash
GET /processing-status
```

Get detailed information about document processing including enhanced metadata.

**Response:**
```json
{
  "status": "success",
  "basic_stats": {
    "total_chunks": 45,
    "processed_urls": 2,
    "urls": ["https://gogel.com.au", "https://bngc.net.au"]
  },
  "enhanced_metadata": {
    "https://gogel.com.au": {
      "method": "dynamic",
      "has_dynamic_content": true,
      "api_endpoints": ["/api/team", "/api/properties"],
      "processed_timestamp": 1703123456,
      "chunk_count": 23
    }
  },
  "dynamic_content_config": {
    "enabled": true,
    "browser_type": "playwright",
    "headless": true,
    "wait_time": 3,
    "api_detection": true
  }
}
```

## 🌟 What This Solves

### Before (Static HTML Only)
- Only captured static HTML content
- Missed JavaScript-loaded team profiles
- No access to API-driven content
- Incomplete knowledge base for RAG

### After (Enhanced Processing)
- ✅ Captures JavaScript-rendered content
- ✅ Discovers and fetches API endpoints
- ✅ Gets dynamic team member cards
- ✅ Complete knowledge base with all content

## 🔍 How It Works

1. **Static Processing**: Always runs first as a baseline
2. **Dynamic Processing**: If enabled, uses browser automation to:
   - Load the full page with JavaScript
   - Wait for dynamic content to render
   - Capture network requests to find API endpoints
3. **API Discovery**: Attempts to fetch data from discovered APIs
4. **Content Combination**: Merges all content sources for comprehensive indexing

## 📊 Browser Comparison

### Playwright (Recommended)
- ✅ Better performance
- ✅ Network request interception
- ✅ Modern async API
- ✅ Better error handling

### Selenium
- ✅ More mature ecosystem
- ✅ Wider browser support
- ❌ No easy network monitoring
- ❌ Older API design

## 🐛 Troubleshooting

### Browser Installation Issues

**Windows:**
```bash
# If playwright install fails
python -m playwright install --force chromium
```

**Linux/WSL:**
```bash
# Install system dependencies
python -m playwright install-deps
sudo apt-get install -y chromium-browser
```

### Memory Issues

- Set `USE_HEADLESS_BROWSER=true` for production
- Reduce `BROWSER_WAIT_TIME` if pages load quickly
- Monitor memory usage during processing

### Network Issues

- Check firewall settings for browser automation
- Ensure target websites allow automated access
- Some sites may block automated requests

### Debugging

1. Check logs for browser automation errors
2. Test with `test_enhanced_processing.py`
3. Use `/processing-status` endpoint to see metadata
4. Try fallback to static processing if dynamic fails

## 📈 Performance Impact

- **Startup**: 2-3x longer due to browser setup
- **Processing**: 3-5x longer per document with dynamic content
- **Memory**: +100-200MB for browser processes
- **Quality**: Significantly better content capture

## 🚀 Production Deployment

1. Use headless mode: `USE_HEADLESS_BROWSER=true`
2. Optimize wait times based on target sites
3. Monitor memory usage and restart if needed
4. Consider running enhanced processing separately from API serving
5. Use caching aggressively to avoid re-processing

## 🔄 Migration Guide

### From Static to Enhanced

1. Install dependencies with `setup_enhanced.py`
2. Set `ENABLE_DYNAMIC_CONTENT=true`
3. Call `/enhanced-reload` to reprocess existing documents
4. Monitor `/processing-status` for completion
5. Test RAG quality with dynamic content

### Fallback Strategy

The system automatically falls back to static processing if:
- Browser automation fails
- Dynamic content is disabled
- Required packages are missing

This ensures the API remains functional even if enhanced processing isn't available.