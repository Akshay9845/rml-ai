"""
Command Line Interface for RML System
"""

import argparse
import sys
from typing import Optional

from .core import RMLSystem
from .config import RMLConfig


def interactive_chat(config: RMLConfig):
    """Interactive chat interface"""
    print("🤖 RML Interactive Chat Interface")
    print("=" * 50)
    print("Ask any question to the RML system!")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for example questions")
    print("Type 'config' to see current configuration")
    print("=" * 50)
    
    try:
        rml = RMLSystem(config)
        print(f"✅ RML system loaded with {rml.memory.get_stats()['total_entries']} entries")
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
                
                if question.lower() == 'config':
                    print(f"\n⚙️ Current Configuration:")
                    print(config)
                    continue
                    
                if not question:
                    continue
                    
                print("\n🔄 Thinking...")
                response = rml.query(question)
                
                print("\n🤖 RML Answer:")
                print("-" * 40)
                print(response.answer)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                
    except Exception as e:
        print(f"❌ Failed to initialize RML system: {e}")
        sys.exit(1)


def single_query(config: RMLConfig, question: str):
    """Single query mode"""
    try:
        rml = RMLSystem(config)
        response = rml.query(question)
        
        print(f"\n🤖 Question: {question}")
        print("=" * 50)
        print(response.answer)
        print("=" * 50)
        print(f"⏱️ Response time: {response.response_ms:.2f}ms")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RML-AI: Resonant Memory Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rml-ai                          # Interactive chat mode
  rml-ai -q "What is RML?"       # Single query mode
  rml-ai --config                # Show configuration
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        help="Single query to process (non-interactive mode)"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration and exit"
    )
    
    parser.add_argument(
        "--encoder-model",
        default="intfloat/e5-base-v2",
        help="Encoder model to use"
    )
    
    parser.add_argument(
        "--decoder-model",
        default="microsoft/phi-1_5",
        help="Decoder model to use"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--dataset-path",
        default="data/rml_data.jsonl",
        help="Path to dataset file"
    )
    
    parser.add_argument(
        "--max-entries",
        type=int,
        default=1000,
        help="Maximum number of dataset entries to load"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = RMLConfig(
        encoder_model=args.encoder_model,
        decoder_model=args.decoder_model,
        device=args.device,
        dataset_path=args.dataset_path,
        max_entries=args.max_entries
    )
    
    # Handle different modes
    if args.config:
        print("⚙️ RML Configuration:")
        print(config)
        return
    
    if args.query:
        single_query(config, args.query)
    else:
        interactive_chat(config)


if __name__ == "__main__":
    main() 