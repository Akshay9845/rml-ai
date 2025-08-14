#!/usr/bin/env python3
"""
Enhanced RML Pipeline with Improved Response Generation

Features:
- Semantic search with BGE-Large encoder
- Enhanced response generation with DialoGPT
- Improved context handling and prompt engineering
- Better error handling and validation
"""

import os
import sys
import re
import logging
import time
import psutil
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM
from rml_knowledge_base_optimized import RMLKnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rml_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class RMLPipeline:
    def __init__(self, data_dir: str = "data", device: str = None, min_confidence: float = 0.7):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        
        # Initialize knowledge base
        self.knowledge_base = RMLKnowledgeBase()
        
        # Load a more capable decoder model
        self.decoder_name = "microsoft/DialoGPT-medium"  # Upgraded from small to medium
        try:
            logger.info(f"Loading tokenizer and model: {self.decoder_name}")
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                self.decoder_name,
                padding_side='left',
                truncation_side='left'
            )
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            
            # Load model with better configuration
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.decoder_name,
                device_map="auto" if self.device == 'cuda' else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.decoder.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to CPU if CUDA is not available
            if 'cuda' in str(e).lower():
                logger.info("Falling back to CPU")
                self.device = 'cpu'
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    self.decoder_name,
                    device_map=None,
                    torch_dtype=torch.float32
                ).to(self.device)
                self.decoder.eval()
            else:
                raise
        
        # Load data
        if os.path.exists(data_dir):
            self._load_training_data(data_dir)
        else:
            self._load_sample_data()
    
    def _load_training_data(self, data_dir: str, max_files: int = 1000, batch_size: int = 50):
        """Load training data from directory with progress tracking and memory management
        
        Args:
            data_dir: Directory containing training data files
            max_files: Maximum number of files to load (0 for no limit)
            batch_size: Number of documents to process in a single batch
        """
        import json
        import time
        from pathlib import Path
        from tqdm import tqdm
        
        def process_jsonl_file(file_path: Path):
            """Process a single JSONL file and yield documents"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if isinstance(data, dict):
                                if 'text' in data:
                                    yield data['text']
                                elif 'content' in data:
                                    yield data['content']
                                elif 'document' in data:
                                    yield data['document']
                            elif isinstance(data, str):
                                yield data
                        except json.JSONDecodeError as je:
                            logger.warning(f"Invalid JSON in {file_path}: {je}")
                            continue
                        except Exception as e:
                            logger.error(f"Error parsing line in {file_path}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        try:
            logger.info(f"🔍 Scanning for training data in {data_dir}")
            
            # Define priority order for data files
            priority_patterns = [
                '**/all_rml_training_data.jsonl',
                '**/rml_training_data.jsonl',
                '**/real_training_data.jsonl',
                '**/proper_training_data.jsonl',
                '**/language_training_data.jsonl',
                '**/pile_rml_final/*.jsonl',
                '**/real_redpajama/*.jsonl',
                '**/*.jsonl'  # Catch-all for other JSONL files
            ]
            
            # Find files in priority order
            files = []
            for pattern in priority_patterns:
                matched_files = list(Path(data_dir).glob(pattern))
                # Only add files that haven't been added yet
                files.extend(f for f in matched_files if f not in files)
                if max_files and len(files) >= max_files:
                    files = files[:max_files]
                    break
            
            if not files:
                logger.warning(f"No training files found in {data_dir}")
                return
            
            total_files = len(files)
            logger.info(f"✅ Found {total_files} data files to process")
            
            # Process files in batches to manage memory
            for batch_idx in range(0, total_files, batch_size):
                batch_files = files[batch_idx:batch_idx + batch_size]
                batch_docs = []
                batch_size_mb = 0
                
                # Process each file in the current batch
                for file_path in tqdm(batch_files, desc=f"Processing batch {batch_idx//batch_size + 1}/{(total_files-1)//batch_size + 1}"):
                    try:
                        # Count lines for progress tracking
                        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
                        with tqdm(process_jsonl_file(file_path), total=total_lines, 
                                desc=f"  {file_path.name}", leave=False) as pbar:
                            for doc in pbar:
                                if doc and isinstance(doc, str) and doc.strip():
                                    batch_docs.append(doc.strip())
                                    batch_size_mb += len(doc.encode('utf-8')) / (1024 * 1024)  # Track MB
                                    
                                    # Process batch if it reaches the size limit
                                    if len(batch_docs) >= 1000 or batch_size_mb > 100:  # 100MB or 1000 docs
                                        self._process_batch(batch_docs, batch_idx, total_files)
                                        batch_docs = []
                                        batch_size_mb = 0
                        
                        logger.info(f"✅ Processed {file_path.name} ({len(batch_docs)} documents)")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {file_path}: {str(e)}")
                        continue
                
                # Process any remaining documents in the batch
                if batch_docs:
                    self._process_batch(batch_docs, batch_idx, total_files)
        
        except Exception as e:
            logger.error(f"❌ Error in _load_training_data: {str(e)}")
            raise
    
    def _process_batch(self, batch_docs, batch_idx, total_batches):
        """Process a batch of documents and add to knowledge base"""
        if not batch_docs:
            return
            
        try:
            start_time = time.time()
            self.knowledge_base.add_documents(batch_docs)
            elapsed = time.time() - start_time
            docs_per_sec = len(batch_docs) / elapsed if elapsed > 0 else 0
            
            logger.info(
                f"📦 Added batch {batch_idx + 1}/{total_batches} "
                f"({len(batch_docs)} docs, {docs_per_sec:.1f} docs/sec)"
            )
            
        except Exception as e:
            logger.error(f"❌ Error adding batch to knowledge base: {str(e)}")
            # Try adding documents one by one to identify problematic ones
            success_count = 0
            for doc in batch_docs:
                try:
                    self.knowledge_base.add_documents([doc])
                    success_count += 1
                except Exception as doc_error:
                    logger.warning(f"Skipped problematic document: {str(doc_error)}")
            
            if success_count > 0:
                logger.info(f"✅ Recovered by adding {success_count}/{len(batch_docs)} documents individually")
            else:
                logger.warning("Failed to add any documents from this batch")
            
            # Clean up to free memory
            del batch_docs
            
            # Small delay to prevent resource exhaustion
            time.sleep(0.5)
            
            # Re-raise the exception to be handled by the caller
            raise
            
        logger.info(f"✅ Successfully processed batch {batch_idx + 1}/{total_batches}")
    
    def _load_sample_data(self) -> None:
        """Load sample data when no training data is available"""
        samples = [
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "Python is an interpreted, high-level, general-purpose programming language.",
            "Machine learning is the study of computer algorithms that improve automatically through experience.",
            "The Great Wall of China is a series of fortifications built across the historical northern borders of China."
        ]
        logger.info("Loading sample data...")
        for doc in samples:
            self.knowledge_base.add_document(doc)
    
    def _extract_knowledge(self, text: str, max_length: int = 100000) -> List[Dict[str, Any]]:
        """Extract knowledge triples from text using spaCy
        
        Args:
            text: Input text to process
            max_length: Maximum length of text to process (in characters)
            
        Returns:
            List of extracted knowledge triples
        """
        if not text or not text.strip():
            return []
            
        # Truncate text if too long
        if len(text) > max_length:
            logger.warning(f"Truncating text from {len(text)} to {max_length} characters")
            text = text[:max_length]
            
        try:
            doc = self.knowledge_base.nlp(text)
            triples = []
            
            # Extract entities and their relations
            entities = [ent for sent in doc.sents for ent in sent.ents]
            
            # Limit number of entities to process to avoid combinatorial explosion
            max_entities = 100
            if len(entities) > max_entities:
                entities = entities[:max_entities]
                logger.debug(f"Limiting to first {max_entities} entities")
            
            # Create entity pairs and relations
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i+1:i+3]:  # Look at next 2 entities
                    if ent1.sent == ent2.sent:  # Only relate entities in same sentence
                        # Simple relation based on dependency path
                        relation = self._find_relation(ent1, ent2, ent1.sent)
                        if relation:
                            triples.append({
                                'subject': ent1.text,
                                'relation': relation,
                                'object': ent2.text,
                                'confidence': 0.9  # High confidence for same-sentence relations
                            })
            
            return triples
            
        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}")
            return []
        
        return triples
    
    def _find_relation(self, ent1, ent2, sent) -> Optional[str]:
        """Find relation between two entities in a sentence"""
        # Get dependency path between entities
        path = []
        e1 = ent1.root
        e2 = ent2.root
        
        # Get path from ent1 to root
        path1 = []
        while e1 != e1.head:
            path1.append(e1.dep_)
            e1 = e1.head
        
        # Get path from ent2 to root
        path2 = []
        while e2 != e2.head:
            path2.append(e2.dep_)
            e2 = e2.head
        
        # Find common ancestor
        while path1 and path2 and path1[-1] == path2[-1]:
            path1.pop()
            path2.pop()
        
        # Build relation path
        relation = ' '.join(path1[::-1] + path2)
        return relation if relation else "related_to"
    
    def _build_prompt(self, query: str, context: str, knowledge: List[Dict]) -> str:
        """Build the prompt for the decoder
        
        Args:
            query: User query
            context: Retrieved context
            knowledge: Extracted knowledge triples
            
        Returns:
            Formatted prompt string
        """
        # Format knowledge triples (limit to top 3 most relevant)
        knowledge_str = "\n".join([f"- {k['subject']} | {k['relation']} | {k['object']}" 
                                 for k in knowledge[:3]])
        
        # Truncate context to prevent overflow
        max_context_length = 500
        if len(context) > max_context_length:
            context = context[:max_context_length] + "... [truncated]"
        
        prompt = f"""Answer the question based on the context below. If you don't know the answer, say so.

RELEVANT CONTEXT:
{context}

KNOWLEDGE GRAPH:
{knowledge_str}

QUESTION: {query}

INSTRUCTIONS:
1. Carefully analyze the context and knowledge graph above
2. Provide a clear, concise, and factual answer
3. If the information is insufficient, say "I don't have enough information to answer that question."
4. Do not make up information or include anything not in the context
5. Format your response in clear, well-structured sentences

ANSWER:"""
        return prompt

    def _validate_response(self, response: str, query: str = None) -> bool:
        """More lenient validation of the generated response
        
        Args:
            response: Generated response to validate
            query: Original user query for context (optional)
            
        Returns:
            bool: True if response is valid, False otherwise
        """
        if not response or not response.strip():
            return False
            
        # More lenient length check
        response = response.strip()
        if len(response) < 5 or len(response) > 1000:
            return False
            
        # Check for obvious error patterns
        error_phrases = [
            "i don't know",
            "no information",
            "error",
            "as an ai",
            "i cannot",
            "unable to"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in error_phrases):
            return False
            
        # Optional: Check relevance to query if query is provided
        if query:
            query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
            if query_words:
                matching_words = sum(1 for w in query_words if w in response_lower)
                if matching_words / len(query_words) < 0.2:  # Reduced threshold
                    return False
        
        return True
        
    def process_text(self, text: str, max_retries: int = 3) -> str:
        """Process text through the pipeline with retries and improved response generation
        
        Args:
            text: Input text to process
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response or error message
        """
        logger.info(f"Processing query: {text}")
        
        for attempt in range(max_retries):
            try:
                # 1. Semantic search with more context
                logger.debug("Performing semantic search...")
                search_results = self.knowledge_base.semantic_search(text, k=5)
                
                if not search_results:
                    logger.warning("No relevant search results found")
                    return "I couldn't find any relevant information for your query. Could you provide more details?"
                
                # 2. Extract and process context
                context_parts = []
                all_knowledge = []
                
                for doc, score in search_results:
                    if score < 0.5:  # Slightly lower threshold for more context
                        continue
                        
                    context_parts.append(doc)
                    # Extract knowledge from each relevant document
                    all_knowledge.extend(self._extract_knowledge(doc))
                
                if not context_parts:
                    return "I found some information, but it's not very relevant to your question. Could you rephrase?"
                
                # Limit context length to prevent token overflow
                context = "\n---\n".join(context_parts)[:8000]  # Limit context length
                
                # 3. Generate response with better parameters
                prompt = self._build_prompt(text, context, all_knowledge)
                logger.debug(f"Generated prompt (first 200 chars): {prompt[:200]}...")
                
                # Tokenize with better parameters
                try:
                    inputs = self.decoder_tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=1024,  # Reasonable context window
                        truncation=True,
                        padding='max_length',
                        add_special_tokens=True
                    ).to(self.device)
                except Exception as e:
                    logger.error(f"Tokenization error: {e}")
                    return "I encountered an error processing your request. Please try again with a different question."
                
                # Simplified generation parameters
                generation_params = {
                    'max_new_tokens': 50,     # Very short responses
                    'temperature': 0.7,       # Less random
                    'top_p': 0.9,            # Narrower sampling
                    'top_k': 50,             # Consider fewer tokens
                    'do_sample': True,        # Enable sampling
                    'num_beams': 1,           # No beam search
                    'no_repeat_ngram_size': 2, # Less strict on repetition
                    'pad_token_id': self.decoder_tokenizer.eos_token_id,
                }
                
                logger.debug(f"Using generation params: {generation_params}")
                logger.debug(f"Input length: {inputs['input_ids'].shape[1]}")
                
                with torch.no_grad():
                    try:
                        outputs = self.decoder.generate(
                            **inputs,
                            **generation_params
                        )
                        
                        # Decode the generated text
                        response = self.decoder_tokenizer.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )
                        
                        # Extract just the answer part (after the last 'ANSWER:')
                        if 'ANSWER:' in response:
                            response = response.split('ANSWER:')[-1].strip()
                        
                        # Validate the response
                        if self._validate_response(response, text):
                            logger.info("Successfully generated response")
                            return response
                        else:
                            logger.warning(f"Low quality response on attempt {attempt + 1}")
                            
                    except Exception as e:
                        logger.error(f"Generation error on attempt {attempt + 1}: {e}")
                        
                # Small delay before retry
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in process_text (attempt {attempt + 1}): {e}")
                time.sleep(1)  # Longer delay on error
        
        # If we get here, all retries failed
        return "I'm having trouble generating a good response. Could you rephrase your question or provide more details?"

def main():
    """Test the RML pipeline with interactive or batch mode"""
    try:
        # Initialize the pipeline
        print("🚀 Initializing RML Pipeline...")
        pipeline = RMLPipeline()
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            # Batch mode: process each argument as a query
            for query in sys.argv[1:]:
                print(f"\n🔍 Query: {query}")
                response = pipeline.process_text(query)
                print(f"💬 Response: {response}")
        else:
            # Interactive mode
            print("\n✨ RML Pipeline Ready! Enter your queries (or 'quit' to exit):")
            while True:
                try:
                    query = input("\nYou: ").strip()
                    if query.lower() in ('quit', 'exit', 'q'):
                        break
                        
                    if not query:
                        continue
                        
                    print("\n🤖 Thinking...")
                    response = pipeline.process_text(query)
                    print(f"\n💬 Response: {response}")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    continue
                    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()
