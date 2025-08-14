#!/usr/bin/env python3
"""
Simple Interactive RML Chat Interface
Ask questions directly to the RML system without curl commands
"""

import requests
import json
import sys

def ask_rml(question):
    """Ask a question to the RML system"""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"message": question},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('answer', 'No answer received')
        else:
            return f"Error: Server returned status {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to RML server. Make sure it's running on http://127.0.0.1:8000"
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🤖 RML Interactive Chat Interface")
    print("=" * 50)
    print("Ask any question to the RML system!")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for example questions")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if question.lower() == 'help':
                print("\n📚 Example Questions:")
                print("• What is machine learning?")
                print("• What latency does RML claim?")
                print("• What is artificial intelligence?")
                print("• How much hallucination reduction does RML claim?")
                print("• What industry use cases does RML serve?")
                print("• Hello, how are you today?")
                continue
                
            if not question:
                continue
                
            print("\n🔄 Thinking...")
            answer = ask_rml(question)
            
            print("\n🤖 RML Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 