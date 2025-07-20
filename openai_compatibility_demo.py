#!/usr/bin/env python3
"""
Demo script showing OpenAI API compatibility
"""

import os
import sys
from pathlib import Path

# Demo using requests (manual approach)
def demo_with_requests():
    print("=== Demo with requests library ===")
    
    import requests
    
    # Test file
    test_file_path = Path("tests/assets/test.opus")
    if not test_file_path.exists():
        print(f"Test file not found: {test_file_path}")
        return
    
    # OpenAI-compatible API call
    with open(test_file_path, "rb") as f:
        files = {"file": (test_file_path.name, f, "audio/opus")}
        data = {
            "model": "whisper-1",
            "response_format": "json",
            "language": "en",
            "temperature": 0.0
        }
        
        response = requests.post(
            "http://localhost:8000/v1/audio/transcriptions",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Transcription successful!")
            print(f"Text: {result.get('text', '')}")
            print(f"Language: {result.get('language', 'Unknown')}")
            if 'usage' in result:
                print(f"Duration: {result['usage'].get('seconds', 'Unknown')} seconds")
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"Error: {response.text}")

# Demo using OpenAI client library (if available)
def demo_with_openai_client():
    print("\n=== Demo with OpenAI client library ===")
    
    try:
        from openai import OpenAI
        
        # Create client pointing to our local server
        client = OpenAI(
            api_key="dummy-key",  # Not used but required
            base_url="http://localhost:8000/v1"
        )
        
        test_file_path = Path("tests/assets/test.opus")
        if not test_file_path.exists():
            print(f"Test file not found: {test_file_path}")
            return
        
        # OpenAI-compatible API call
        with open(test_file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json"
            )
            
            print("✓ Transcription successful with OpenAI client!")
            print(f"Text: {transcript.text}")
            
    except ImportError:
        print("OpenAI client library not installed. Install with: pip install openai")
    except Exception as e:
        print(f"✗ OpenAI client error: {e}")

# Demo different response formats
def demo_response_formats():
    print("\n=== Demo different response formats ===")
    
    import requests
    
    test_file_path = Path("tests/assets/test.opus")
    if not test_file_path.exists():
        print(f"Test file not found: {test_file_path}")
        return
    
    formats = ["json", "text", "srt", "verbose_json"]
    
    for fmt in formats:
        print(f"\n--- Format: {fmt} ---")
        
        with open(test_file_path, "rb") as f:
            files = {"file": (test_file_path.name, f, "audio/opus")}
            data = {
                "model": "whisper-1",
                "response_format": fmt,
                "temperature": 0.0
            }
            
            response = requests.post(
                "http://localhost:8000/v1/audio/transcriptions",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                if fmt in ["json", "verbose_json"]:
                    result = response.json()
                    print(f"✓ {fmt} format successful")
                    if fmt == "verbose_json":
                        print(f"  Segments: {len(result.get('segments', []))}")
                        print(f"  Language: {result.get('language', 'Unknown')}")
                        print(f"  Duration: {result.get('duration', 'Unknown')}")
                else:
                    print(f"✓ {fmt} format successful")
                    # Show first few lines for text formats
                    lines = response.text.split('\n')[:3]
                    for line in lines:
                        if line.strip():
                            print(f"  {line}")
            else:
                print(f"✗ {fmt} format failed: {response.status_code}")

def main():
    print("Whisper Diarization Server - OpenAI Compatibility Demo")
    print("=" * 60)
    
    # Check if server is running
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not running or not healthy")
            print("Please start the server with: python start_server.py")
            return
    except:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("Please start the server with: python start_server.py")
        return
    
    print("✅ Server is running and healthy\n")
    
    # Run demos
    demo_with_requests()
    demo_with_openai_client()
    demo_response_formats()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\nFor more examples, check the SERVER_README.md file")

if __name__ == "__main__":
    main()