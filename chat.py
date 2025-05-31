#!/usr/bin/env python3
"""
Simple test script to verify OpenAI API integration
Run this to test if your API key works before integrating with Flask
"""

import os
import openai
from dotenv import load_dotenv

def test_openai_setup():
    """Test OpenAI API setup and connection"""
    
    print("=== OpenAI Setup Test ===")
    
    # Load environment variables
    print("1. Loading environment variables...")
    load_dotenv()
    
    # Check current directory
    print(f"   Current directory: {os.getcwd()}")
    
    # Check if .env file exists
    env_exists = os.path.exists('.env')
    print(f"   .env file exists: {env_exists}")
    
    if env_exists:
        print("   .env file content:")
        try:
            with open('.env', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('OPENAI_API_KEY'):
                        key_line = line.strip()
                        # Hide most of the key for security
                        if '=' in key_line:
                            key_part = key_line.split('=', 1)[1]
                            if len(key_part) > 10:
                                masked_key = key_part[:10] + '*' * (len(key_part) - 10)
                                print(f"   {key_line.split('=')[0]}={masked_key}")
                            else:
                                print(f"   {key_line} (KEY TOO SHORT!)")
                        break
                else:
                    print("   No OPENAI_API_KEY found in .env file!")
        except Exception as e:
            print(f"   Error reading .env: {e}")
    
    # Get API key from environment
    print("\n2. Checking environment variables...")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        print(f"   API key loaded: Yes")
        print(f"   Key length: {len(api_key)} characters")
        print(f"   Starts with: {api_key[:10]}...")
        print(f"   Ends with: ...{api_key[-10:]}")
        
        # Check key format
        if api_key.startswith('sk-'):
            print("   ✅ Key format looks correct")
        else:
            print("   ❌ Key should start with 'sk-'")
            
        if len(api_key) >= 40:
            print("   ✅ Key length looks reasonable")
        else:
            print("   ❌ Key seems too short")
    else:
        print("   ❌ No API key found in environment!")
        return False
    
    # Set OpenAI API key
    print("\n3. Setting up OpenAI client...")
    openai.api_key = api_key
    
    # Test API connection
    print("\n4. Testing OpenAI API connection...")
    try:
        # Simple test request
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Say "Hello from OpenAI!" and nothing else.'}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        message = response.choices[0].message.content.strip()
        print(f"   ✅ API Response: {message}")
        print("   ✅ OpenAI API is working correctly!")
        return True
        
    except openai.error.AuthenticationError as e:
        print(f"   ❌ Authentication Error: {e}")
        print("   Check your API key at: https://platform.openai.com/api-keys")
        return False
        
    except openai.error.RateLimitError as e:
        print(f"   ❌ Rate Limit Error: {e}")
        print("   You may have exceeded your API quota")
        return False
        
    except openai.error.InvalidRequestError as e:
        print(f"   ❌ Invalid Request: {e}")
        return False
        
    except Exception as e:
        print(f"   ❌ Unexpected Error: {e}")
        return False

def interactive_chat():
    """Simple interactive chat for testing"""
    
    print("\n=== Interactive Chat Test ===")
    print("Type 'quit' to exit\n")
    
    conversation = [
        {'role': 'system', 'content': 'You are Skeeze, a helpful AI assistant specializing in skin health.'}
    ]
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Add user message
            conversation.append({'role': 'user', 'content': user_input})
            
            # Get AI response
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=conversation[-10:],  # Keep last 10 messages
                max_tokens=200,
                temperature=0.7
            )
            
            ai_message = response.choices[0].message.content.strip()
            conversation.append({'role': 'assistant', 'content': ai_message})
            
            print(f"Skeeze: {ai_message}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    """Main function to run tests"""
    
    print("🧪 OpenAI Integration Test Script")
    print("=" * 40)
    
    # Test setup
    if test_openai_setup():
        print("\n🎉 Setup test passed!")
        
        # Ask if user wants to try interactive chat
        try:
            test_chat = input("\nWould you like to test interactive chat? (y/n): ").lower().strip()
            if test_chat in ['y', 'yes']:
                interactive_chat()
        except KeyboardInterrupt:
            print("\nSkipping interactive test.")
    else:
        print("\n❌ Setup test failed!")
        print("\nTroubleshooting steps:")
        print("1. Make sure .env file exists in the same directory as this script")
        print("2. Check your OPENAI_API_KEY in the .env file")
        print("3. Get a new API key from: https://platform.openai.com/api-keys")
        print("4. Make sure your OpenAI account has billing/credits set up")

if __name__ == "__main__":
    main()