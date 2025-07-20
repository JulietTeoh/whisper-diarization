#!/usr/bin/env python3
"""
Test script for the Whisper Diarization Server
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_models_endpoint():
    """Test the models endpoint"""
    try:
        response = requests.get("http://localhost:8000/v1/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Models endpoint passed: {len(data['data'])} models available")
            return True
        else:
            print(f"✗ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Models endpoint error: {e}")
        return False

def test_transcription_endpoint():
    """Test the transcription endpoint with the existing test file"""
    test_file_path = Path("tests/assets/test.opus")
    
    if not test_file_path.exists():
        print(f"✗ Test file not found: {test_file_path}")
        return False
    
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": (test_file_path.name, f, "audio/opus")}
            data = {
                "model": "whisper-1",
                "response_format": "json",
                "temperature": 0.0
            }
            
            print(f"Testing transcription with {test_file_path.name}...")
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8000/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Transcription completed in {end_time - start_time:.2f} seconds")
                print(f"  Text: {result.get('text', '')[:100]}...")
                print(f"  Language: {result.get('language', 'Unknown')}")
                return True
            else:
                print(f"✗ Transcription failed: {response.status_code}")
                print(f"  Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ Transcription error: {e}")
        return False

def test_transcription_formats():
    """Test different response formats"""
    test_file_path = Path("tests/assets/test.opus")
    
    if not test_file_path.exists():
        print(f"✗ Test file not found: {test_file_path}")
        return False
    
    formats = ["json", "text", "srt", "verbose_json"]
    
    for fmt in formats:
        try:
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
                    data=data,
                    timeout=300
                )
                
                if response.status_code == 200:
                    print(f"✓ Format {fmt} test passed")
                else:
                    print(f"✗ Format {fmt} test failed: {response.status_code}")
                    
        except Exception as e:
            print(f"✗ Format {fmt} test error: {e}")

def main():
    print("Testing Whisper Diarization Server...")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_endpoint()
    models_ok = test_models_endpoint()
    
    if not health_ok or not models_ok:
        print("\n❌ Basic endpoint tests failed. Is the server running?")
        return False
    
    print("\n📋 Testing transcription functionality...")
    transcription_ok = test_transcription_endpoint()
    
    if transcription_ok:
        print("\n📋 Testing response formats...")
        test_transcription_formats()
    
    print("\n" + "=" * 50)
    if health_ok and models_ok and transcription_ok:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)