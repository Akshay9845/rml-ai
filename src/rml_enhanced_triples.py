"""
Enhanced RML Triple Generator with Schema Validation

This module provides advanced triple extraction and knowledge graph construction
for RML data with improved entity linking and relation extraction.
"""

import json
import spacy
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

@dataclass
class Triple:
    """Represents a single RDF triple with confidence score."""
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    source: str = "extracted"
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "confidence": self.confidence,
            "source": self.source,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            obj=data["object"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "extracted"),
            context=data.get("context")
        )

class EnhancedTripleGenerator:
    """Enhanced triple generator with improved entity linking and relation extraction."""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize with a spaCy model (default: en_core_web_lg for better NER)."""
        self.nlp = self._load_spacy_model(model_name)
        self.entity_cache = {}
        self.relation_patterns = self._load_relation_patterns()
        
    def _load_spacy_model(self, model_name: str):
        """Load spaCy model with error handling."""
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"⚠️ {model_name} not found. Installing...")
            import os
            os.system(f"python -m spacy download {model_name}")
            return spacy.load(model_name)
    
    def _load_relation_patterns(self) -> Dict[str, List[Dict]]:
        """Load common relation patterns for extraction."""
        return {
            "is_a": [
                {"pattern": [{"LOWER": "is"}, {"LOWER": "a"}], "rel": "is_a"},
                {"pattern": [{"LOWER": "are"}, {"LOWER": "a"}], "rel": "is_a"},
                {"pattern": [{"LOWER": "is"}, {"LOWER": "an"}], "rel": "is_a"},
                {"pattern": [{"LOWER": "are"}, {"LOWER": "an"}], "rel": "is_a"},
            ],
            "has_property": [
                {"pattern": [{"LOWER": "has"}], "rel": "has_property"},
                {"pattern": [{"LOWER": "have"}], "rel": "has_property"},
                {"pattern": [{"LOWER": "with"}], "rel": "has_property"},
            ],
            "part_of": [
                {"pattern": [{"LOWER": "part"}, {"LOWER": "of"}], "rel": "part_of"},
                {"pattern": [{"LOWER": "member"}, {"LOWER": "of"}], "rel": "member_of"},
            ]
        }
    
    def extract_triples(self, text: str, context: Optional[Dict] = None) -> List[Triple]:
        """Extract triples from text with enhanced relation extraction."""
        if not text or not isinstance(text, str):
            return []
            
        doc = self.nlp(text)
        triples = []
        
        # Extract entities with their types
        entities = self._extract_entities(doc)
        
        # Extract relations using pattern matching
        triples.extend(self._extract_relations(doc, entities))
        
        # If no relations found, try to extract subject-verb-object patterns
        if not triples:
            triples.extend(self._extract_svo_patterns(doc, entities))
        
        # Add triples from context metadata if available
        if context and 'metadata' in context:
            metadata = context['metadata']
            item_id = context.get('source', 'unknown')
            
            # Add type assertion for the main item
            triples.append(Triple(
                subject=item_id,
                predicate='rdf:type',
                obj='rml:TextItem',
                confidence=1.0,
                source='context'
            ))
            
            # Add text content as a property
            triples.append(Triple(
                subject=item_id,
                predicate='rml:hasText',
                obj=text,
                confidence=1.0,
                source='context'
            ))
        
        # Apply coreference resolution if context is provided
        if context:
            triples.extend(self._resolve_coreferences(triples, context))
            
        return self._deduplicate_triples(triples)
        
    def _extract_svo_patterns(self, doc, entities: List[Dict]) -> List[Triple]:
        """Extract subject-verb-object patterns from text."""
        triples = []
        
        # Simple pattern: look for verb phrases with subject and object
        for sent in doc.sents:
            # Find the root verb
            for token in sent:
                if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                    # Find subject and object of the verb
                    subj = next((t for t in token.lefts if t.dep_ in ('nsubj', 'nsubjpass')), None)
                    obj = next((t for t in token.rights if t.dep_ in ('dobj', 'attr', 'prep')), None)
                    
                    if subj and obj:
                        # Get the full noun phrases
                        subj_text = ' '.join(t.text for t in subj.subtree)
                        obj_text = ' '.join(t.text for t in obj.subtree)
                        
                        # Create a triple
                        triples.append(Triple(
                            subject=subj_text,
                            predicate=token.lemma_,  # Use lemma form of the verb
                            obj=obj_text,
                            confidence=0.7,
                            source='svo_extraction'
                        ))
        
        return triples
    
    def _extract_entities(self, doc) -> List[Dict]:
        """Extract entities with enhanced typing and normalization."""
        entities = []
        for ent in doc.ents:
            entity = {
                "text": ent.text.strip(),
                "type": self._normalize_entity_type(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char,
                "vector": ent.vector if hasattr(ent, 'vector') else None
            }
            entities.append(entity)
        return entities
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity types to a standard set."""
        type_map = {
            # People
            "PERSON": "Person", "PER": "Person",
            # Organizations
            "ORG": "Organization", "FAC": "Organization",
            # Locations
            "GPE": "Location", "LOC": "Location",
            # Products/Artifacts
            "PRODUCT": "Product", "WORK_OF_ART": "CreativeWork",
            # Events
            "EVENT": "Event",
            # Time
            "DATE": "Date", "TIME": "Time",
            # Numbers/Quantities
            "PERCENT": "Number", "MONEY": "MonetaryAmount",
            "QUANTITY": "Quantity", "CARDINAL": "Number",
            "ORDINAL": "Number",
            # Other
            "NORP": "Group", "LANGUAGE": "Language",
            "LAW": "Legislation"
        }
        return type_map.get(entity_type, "Thing")
    
    def _extract_relations(self, doc, entities: List[Dict]) -> List[Triple]:
        """Extract relations using dependency parsing and pattern matching with enhanced relation types."""
        triples = []
        
        # Process each sentence separately
        for sent in doc.sents:
            sent_text = sent.text.lower()
            sent_ents = [e for e in entities 
                        if e["start"] >= sent.start_char 
                        and e["end"] <= sent.end_char]
            
            # Extract subject-verb-object patterns
            for token in sent:
                # Handle copula constructions (X is Y)
                if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                    # Find subject and complement
                    subj = next((t for t in token.lefts if t.dep_ in ('nsubj', 'nsubjpass')), None)
                    comp = next((t for t in token.rights if t.dep_ in ('attr', 'acomp', 'ccomp', 'xcomp')), None)
                    
                    if subj and comp:
                        subj_text = ' '.join(t.text for t in subj.subtree)
                        comp_text = ' '.join(t.text for t in comp.subtree)
                        
                        # Enhanced relation type detection for copula
                        if token.lemma_.lower() in ['be', 'is', 'are', 'was', 'were']:
                            rel_type = 'is_a'
                            confidence = 0.9
                        elif token.lemma_.lower() in ['have', 'has', 'had']:
                            rel_type = 'has_part'
                            confidence = 0.85
                        else:
                            rel_type = 'has_property'
                            confidence = 0.8
                        
                        triples.append(Triple(
                            subject=subj_text,
                            predicate=rel_type,
                            obj=comp_text,
                            confidence=confidence,
                            context=sent.text,
                            source='copula_extraction'
                        ))
                
                # Handle verb-object patterns with enhanced relation detection
                elif token.dep_ in ('dobj', 'pobj', 'nsubjpass') and token.head.pos_ == 'VERB':
                    # Find subject and object
                    subj = next((t for t in token.head.lefts if t.dep_ in ('nsubj', 'nsubjpass', 'agent')), None)
                    
                    if subj:
                        subj_text = ' '.join(t.text for t in subj.subtree)
                        obj_text = ' '.join(t.text for t in token.subtree)
                        
                        # Enhanced relation type detection based on verb semantics
                        verb_lemma = token.head.lemma_.lower()
                        rel_type, confidence = self._determine_relation_type(verb_lemma, token, sent)
                        
                        # Only add if we have a valid relation type
                        if rel_type:
                            triples.append(Triple(
                                subject=subj_text if token.dep_ != 'nsubjpass' else obj_text,
                                predicate=rel_type,
                                obj=obj_text if token.dep_ != 'nsubjpass' else subj_text,
                                confidence=confidence,
                                context=sent.text,
                                source='verb_relation_extraction'
                            ))
            
            # Enhanced noun-noun compounds and other patterns
            for token in sent:
                # Handle compound nouns (e.g., "machine learning algorithms")
                if token.dep_ == 'compound' and token.head.pos_ == 'NOUN':
                    triples.append(Triple(
                        subject=token.head.text,
                        predicate='has_part',
                        obj=token.text,
                        confidence=0.9,
                        context=sent.text,
                        source='compound_extraction'
                    ))
                
                # Handle prepositional attachments (e.g., "book on the table")
                elif token.dep_ == 'prep' and token.head.pos_ in ('NOUN', 'VERB'):
                    head_text = ' '.join(t.text for t in token.head.subtree)
                    obj = next((t for t in token.rights if t.dep_ == 'pobj'), None)
                    
                    if obj:
                        obj_text = ' '.join(t.text for t in obj.subtree)
                        rel_type = self._map_prep_to_relation(token.text.lower())
                        
                        if rel_type:
                            triples.append(Triple(
                                subject=head_text,
                                predicate=rel_type,
                                obj=obj_text,
                                confidence=0.8,
                                context=sent.text,
                                source='prepositional_relation_extraction'
                            ))
        
        return triples
    
    def _determine_relation_type(self, verb_lemma: str, token, sent) -> Tuple[str, float]:
        """Determine the most appropriate relation type based on verb semantics."""
        # Common verb to relation mappings
        verb_to_relation = {
            # Causal relations
            'cause': ('causes', 0.9),
            'create': ('creates', 0.9),
            'produce': ('produces', 0.9),
            'affect': ('affects', 0.85),
            'influence': ('influences', 0.85),
            'enable': ('enables', 0.85),
            'prevent': ('prevents', 0.85),
            
            # Spatial relations
            'contain': ('contains', 0.9),
            'include': ('contains', 0.8),
            'locate': ('located_in', 0.9),
            'place': ('located_in', 0.8),
            'position': ('located_in', 0.8),
            
            # Temporal relations
            'start': ('starts', 0.9),
            'begin': ('starts', 0.9),
            'end': ('ends', 0.9),
            'finish': ('ends', 0.9),
            'follow': ('after', 0.85),
            'precede': ('before', 0.85),
            
            # Comparative relations
            'resemble': ('similar_to', 0.9),
            'differ': ('different_from', 0.9),
            'exceed': ('better_than', 0.8),
            'surpass': ('better_than', 0.8),
            
            # Default fallback
            'use': ('uses', 0.8),
            'require': ('requires', 0.85),
            'need': ('requires', 0.8)
        }
        
        # Check for direct mapping
        if verb_lemma in verb_to_relation:
            return verb_to_relation[verb_lemma]
        
        # Check for negation
        if any(t.dep_ == 'neg' for t in token.head.lefts):
            return ('does_not_' + verb_lemma, 0.8)
        
        # Default to verb lemma as relation
        return (verb_lemma, 0.7)
    
    def _map_prep_to_relation(self, prep: str) -> Optional[str]:
        """Map prepositions to semantic relations."""
        prep_to_relation = {
            'in': 'located_in',
            'on': 'on_top_of',
            'at': 'located_at',
            'with': 'has_attribute',
            'without': 'lacks',
            'by': 'created_by',
            'for': 'intended_for',
            'from': 'originates_from',
            'to': 'leads_to',
            'about': 'concerns',
            'of': 'part_of',
            'as': 'acts_as',
            'like': 'similar_to',
            'than': 'compared_to',
            'before': 'temporally_before',
            'after': 'temporally_after',
            'during': 'occurs_during',
            'between': 'connects',
            'through': 'traverses',
            'under': 'beneath',
            'over': 'above',
            'against': 'opposed_to',
            'among': 'part_of'
        }
        return prep_to_relation.get(prep, None)
    
    def _find_entity_for_token(self, token, entities: List[Dict]) -> Optional[Dict]:
        """Find the entity that contains the given token."""
        for ent in entities:
            if ent["start"] <= token.idx <= ent["end"]:
                return ent
        return None
    
    def _get_relation(self, verb_token, obj_token) -> str:
        """Determine the relation type between verb and object."""
        verb_lemma = verb_token.lemma_.lower()
        
        # Common relations
        if verb_lemma in ["be", "is", "are", "was", "were"]:
            return "is_a"
        elif verb_lemma in ["have", "has", "had"]:
            return "has"
        elif verb_lemma in ["use", "uses", "using"]:
            return "uses"
        elif verb_lemma in ["create", "creates", "created"]:
            return "created_by"
            
        # Default to verb lemma
        return verb_lemma
    
    def _resolve_coreferences(self, triples: List[Triple], context: Dict) -> List[Triple]:
        """Resolve coreferences in extracted triples using context."""
        resolved = []
        resolved_entities = context.get("resolved_entities", {})
        
        for triple in triples:
            # Resolve subject
            subj = resolved_entities.get(triple.subject.lower(), triple.subject)
            # Resolve object
            obj = resolved_entities.get(triple.obj.lower(), triple.obj)
            
            if subj != triple.subject or obj != triple.obj:
                resolved.append(Triple(
                    subject=subj,
                    predicate=triple.predicate,
                    obj=obj,
                    confidence=triple.confidence * 0.9,  # Slightly reduce confidence
                    source=f"resolved_{triple.source}",
                    context=triple.context
                ))
            else:
                resolved.append(triple)
                
        return resolved
    
    def _deduplicate_triples(self, triples: List[Triple]) -> List[Triple]:
        """Remove duplicate triples, keeping the highest confidence version."""
        unique = {}
        for triple in triples:
            key = (triple.subject.lower(), triple.predicate.lower(), triple.obj.lower())
            if key not in unique or unique[key].confidence < triple.confidence:
                unique[key] = triple
        return list(unique.values())

class RMLKnowledgeGraph:
    """Manages the RML knowledge graph with schema validation."""
    
    def __init__(self, schema: Optional[Dict] = None):
        self.triples = []
        self.entities = set()
        self.relations = set()
        self.schema = schema or self._default_schema()
        
    def _default_schema(self) -> Dict:
        """Comprehensive schema for RML data validation with expanded relation types."""
        # Core verb forms
        base_verbs = [
            'transform', 'analyze', 'achieve', 'enable', 'improve', 'enhance',
            'support', 'facilitate', 'process', 'generate', 'create', 'modify',
            'detect', 'identify', 'classify', 'predict', 'evaluate', 'compare'
        ]
        
        # Generate variations
        verb_variations = []
        for verb in base_verbs:
            verb_variations.extend([
                verb, verb + 's', verb + 'ing', verb + 'ed', verb + 'ion',
                'is_' + verb + 'ing', 'can_' + verb
            ])
        
        # Generate relation types set with all variations
        relation_types = set([
            # Core semantic relations
            'is_a', 'instance_of', 'subclass_of', 'type_of', 'kind_of',
            'part_of', 'has_part', 'contains', 'contained_in', 'composed_of',
            'has_property', 'property_of', 'has_attribute', 'attribute_of',
            'causes', 'caused_by', 'leads_to', 'result_of', 'results_in',
            'uses', 'used_by', 'requires', 'required_by', 'depends_on',
            'creates', 'created_by', 'produces', 'produced_by', 'generates',
            'affects', 'affected_by', 'influences', 'influenced_by', 'impacts',
            
            # Action/process relations
            *verb_variations,
            
            # Common relations from previous runs
            'transform', 'analyze', 'achieve', 'originates_from',
            'enable', 'improve', 'enhance', 'support', 'facilitate',
            'process', 'generate', 'create', 'modify', 'detect', 'identify',
            'classify', 'predict', 'evaluate', 'compare',
            
            # RDF/XML predicates
            'rdf:type', 'rdfs:subClassOf', 'rdfs:subPropertyOf',
            'rdfs:domain', 'rdfs:range', 'rdfs:label', 'rdfs:comment',
            'rml:hasText', 'rml:source', 'rml:confidence', 'rml:createdAt'
        ])
        
        # Create the final schema dictionary
        schema = {
            "required_fields": ["subject", "predicate", "obj"],
            "entity_types": [
                "Person", "Organization", "Location", "Date", "Concept", "Event",
                "Object", "Action", "Process", "System", "Component", "Material"
            ],
            "relation_types": sorted(list(relation_types)),
            "validation_rules": {
                "subject": {"type": "string", "min_length": 1},
                "predicate": {"type": "string", "min_length": 1},
                "obj": {"type": "string", "min_length": 1},
                "confidence": {"type": "number", "min": 0, "max": 1},
                "source": {"type": "string"},
                "context": {"type": ["string", "null"]}
            }
        }
        
        return schema
    
    def add_triple(self, triple: Triple) -> bool:
        """Add a triple to the knowledge graph with validation."""
        if not self._validate_triple(triple):
            print(f"Validation failed for triple: {triple.subject} - {triple.predicate} - {triple.obj}")
            return False
            
        try:
            self.triples.append(triple)
            self.entities.add(triple.subject)
            self.entities.add(triple.obj)
            self.relations.add(triple.predicate)
            print(f"Added triple: {triple.subject} - {triple.predicate} - {triple.obj}")
            return True
        except Exception as e:
            print(f"Error adding triple: {e}")
            return False
    
    def _validate_triple(self, triple: Triple) -> bool:
        """Validate a triple against the schema."""
        # Check required fields
        for field in self.schema["required_fields"]:
            if not getattr(triple, field, None):
                print(f"  - Missing required field: {field}")
                return False
                
        # Check field types and constraints
        for field, rules in self.schema["validation_rules"].items():
            value = getattr(triple, field, None)
            
            # Skip validation for optional fields that are None
            if value is None and field not in self.schema["required_fields"]:
                continue
            
            # Check if field is required
            if value is None and field in self.schema["required_fields"]:
                print(f"  - Required field is None: {field}")
                return False
                
            # Check type
            if value is not None and "type" in rules:
                expected_types = rules["type"]
                if not isinstance(expected_types, list):
                    expected_types = [expected_types]
                    
                type_ok = any(
                    (t == "null" and value is None) or 
                    (t == "string" and isinstance(value, str)) or
                    (t == "number" and isinstance(value, (int, float)))
                    for t in expected_types
                )
                
                if not type_ok:
                    actual_type = type(value).__name__
                    print(f"  - Type mismatch for field '{field}': Expected {expected_types}, got {actual_type}")
                    return False
            
            # Check string length
            if isinstance(value, str) and "min_length" in rules:
                if len(value.strip()) < rules["min_length"]:
                    print(f"  - Field '{field}' too short: {len(value.strip())} < {rules['min_length']}")
                    return False
                    
            # Check number range
            if isinstance(value, (int, float)) and ("min" in rules or "max" in rules):
                if "min" in rules and value < rules["min"]:
                    print(f"  - Field '{field}' value {value} below minimum {rules['min']}")
                    return False
                if "max" in rules and value > rules["max"]:
                    print(f"  - Field '{field}' value {value} above maximum {rules['max']}")
                    return False
        
        # Additional validation for specific predicates
        if hasattr(triple, 'predicate'):
            # Skip validation for RDF/XML and RML-specific predicates
            if triple.predicate.startswith(('rdf:', 'rml:')):
                return True
                
            # Check if predicate is in the allowed list
            if "relation_types" in self.schema and triple.predicate not in self.schema["relation_types"]:
                print(f"  - Predicate '{triple.predicate}' not in allowed relation types")
                return False
                    
        return True
    
    def to_jsonld(self) -> Dict:
        """Export the knowledge graph to JSON-LD format."""
        context = {
            "@context": {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "rml": "http://example.org/rml#",
                "entities": {
                    "@id": "rml:entities/",
                    "@type": "@id"
                }
            }
        }
        
        graph = {
            "@graph": [
                {
                    "@id": f"rml:triple/{i}",
                    "@type": "rdf:Statement",
                    "rdf:subject": {"@id": f"rml:entities/{t.subject}"},
                    "rdf:predicate": {"@id": f"rml:predicates/{t.predicate}"},
                    "rdf:object": {"@id": f"rml:entities/{t.obj}"},
                    "rml:confidence": t.confidence,
                    "rml:source": t.source,
                    "rml:context": t.context or ""
                }
                for i, t in enumerate(self.triples)
            ]
        }
        
        return {**context, **graph}
    
    def save(self, filepath: str, format: str = "jsonld") -> bool:
        """Save the knowledge graph to a file."""
        try:
            if format.lower() == "jsonld":
                data = self.to_jsonld()
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")
            return False

class RMLDataProcessor:
    """Processes RML data files and builds a knowledge graph."""
    
    def __init__(self, triple_generator: Optional[EnhancedTripleGenerator] = None):
        self.triple_generator = triple_generator or EnhancedTripleGenerator()
        self.knowledge_graph = RMLKnowledgeGraph()
        
    def process_file(self, input_file: str, output_file: Optional[str] = None) -> bool:
        """Process an RML data file and extract triples."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing RML data"):
                    try:
                        data = json.loads(line)
                        self._process_rml_item(data)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line[:100]}...")
                        continue
                        
            # Save results if output file is specified
            if output_file:
                return self.knowledge_graph.save(output_file)
                
            return True
            
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            return False
    
    def _process_rml_item(self, item: Dict) -> None:
        """Process a single RML item and extract triples."""
        try:
            # Extract text content
            text = self._extract_text(item)
            if not text:
                print(f"No text content found in item: {item.get('uri', 'unknown')}")
                return
                
            # Create a unique identifier for this item if URI is not present
            item_id = item.get("uri", f"urn:rml:item:{hash(text) & 0xFFFFFFFFFFFFFFFF}")
            print(f"Processing item: {item_id}")
            
            # Extract entities from metadata
            entities = self._extract_entities(item)
            print(f"Extracted {len(entities)} entities from metadata")
            
            # Create context with metadata
            context = {
                "source": item_id,
                "resolved_entities": entities,
                "metadata": {
                    "concepts": item.get("concepts", []),
                    "emotions": item.get("emotions", []),
                    "intents": item.get("intents", [])
                }
            }
            
            # Extract triples from text using NLP
            print(f"Extracting triples from text (length: {len(text)} chars)...")
            triples = self.triple_generator.extract_triples(text, context)
            print(f"Extracted {len(triples)} triples from text")
            
            # Add triples for concepts, emotions, and intents
            for concept in item.get("concepts", []):
                triple = Triple(
                    subject=item_id,
                    predicate="hasConcept",
                    obj=concept,
                    confidence=0.9,
                    source="metadata",
                    context=item.get("context", "")
                )
                triples.append(triple)
                print(f"Added concept triple: {item_id} - hasConcept - {concept}")
                
            for emotion in item.get("emotions", []):
                triple = Triple(
                    subject=item_id,
                    predicate="expressesEmotion",
                    obj=emotion,
                    confidence=0.9,
                    source="metadata",
                    context=item.get("context", "")
                )
                triples.append(triple)
                print(f"Added emotion triple: {item_id} - expressesEmotion - {emotion}")
                
            for intent in item.get("intents", []):
                triple = Triple(
                    subject=item_id,
                    predicate="hasIntent",
                    obj=intent,
                    confidence=0.9,
                    source="metadata",
                    context=item.get("context", "")
                )
                triples.append(triple)
                print(f"Added intent triple: {item_id} - hasIntent - {intent}")
            
            # Add all triples to knowledge graph
            print(f"Adding {len(triples)} triples to knowledge graph...")
            for triple in triples:
                self.knowledge_graph.add_triple(triple)
                
        except Exception as e:
            print(f"Error processing RML item: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_text(self, item: Dict) -> Optional[str]:
        """Extract text content from an RML item."""
        # Try different possible text fields
        for field in ["text", "content", "body", "description"]:
            if field in item and isinstance(item[field], str):
                return item[field].strip()
        return None
    
    def _extract_entities(self, item: Dict) -> Dict[str, str]:
        """Extract named entities from RML item metadata."""
        entities = {}
        
        # Extract from common metadata fields
        for field in ["title", "author", "publisher", "keywords"]:
            if field in item and item[field]:
                if isinstance(item[field], str):
                    entities[item[field].lower()] = item[field]
                elif isinstance(item[field], list):
                    for ent in item[field]:
                        if isinstance(ent, str):
                            entities[ent.lower()] = ent
                            
        return entities

def main():
    """Example usage of the enhanced RML triple generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process RML data and extract triples")
    parser.add_argument("input_file", help="Input RML data file (JSONL format)")
    parser.add_argument("-o", "--output", help="Output file for knowledge graph")
    parser.add_argument("--model", default="en_core_web_lg", 
                       help="spaCy model to use for NLP")
    
    args = parser.parse_args()
    
    # Initialize processor
    generator = EnhancedTripleGenerator(model_name=args.model)
    processor = RMLDataProcessor(generator)
    
    # Process file
    print(f"Processing {args.input_file}...")
    success = processor.process_file(args.input_file, args.output)
    
    if success:
        print(f"Successfully processed {len(processor.knowledge_graph.triples)} triples")
        if args.output:
            print(f"Knowledge graph saved to {args.output}")
    else:
        print("Processing failed")

if __name__ == "__main__":
    main()
