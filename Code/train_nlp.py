"""
NLP Next-Word Predictor Training Script
Trains a statistical N-gram Language Model from external text files / corpora
and packages it into 'nlp_model.pkl' for real-time sub-millisecond inference.

Usage:
    python Code/train_nlp.py
    python Code/train_nlp.py --corpus my_custom_data.txt
"""

import os
import sys
import argparse
import pickle
import re
from collections import defaultdict, Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS_PATH = os.path.join(SCRIPT_DIR, 'training_corpus.txt')
DEFAULT_VISION_MODEL_PATH = os.path.join(SCRIPT_DIR, 'mp_mlp_model.pkl')
OUTPUT_MODEL_PATH = os.path.join(SCRIPT_DIR, 'nlp_model.pkl')

def get_current_vocab(vision_model_path):
    """Extract known sign vocabulary from the trained computer vision model."""
    if os.path.exists(vision_model_path):
        try:
            with open(vision_model_path, 'rb') as f:
                bundle = pickle.load(f)
            le = bundle.get('label_encoder')
            if le is not None and hasattr(le, 'classes_'):
                return list(le.classes_)
        except Exception as e:
            print(f"Warning: Could not read classes from {vision_model_path}: {e}")
    
    # Fallback to standard 31 classes
    return [
        'Born', 'Brain', 'Hello', 'I', 'Leave', 'Love', 'MY', 'Monk',
        'NO', 'Now', 'OK', 'Play', 'SIGN', 'School', 'Small', 'SomeTimes',
        'Thanks you', 'Where', 'Yes', 'family', 'friend', 'go', 'good',
        'help', 'how', 'please', 'see', 'want', 'what', 'when', 'you'
    ]

def build_canonical_map(vocab):
    """Map lowercased versions of words to their exact model class names."""
    c_map = {}
    for v in vocab:
        c_map[v.lower()] = v
    # Handle common idioms and aliases
    c_map['thanks'] = 'Thanks you'
    c_map['thank you'] = 'Thanks you'
    c_map['sometimes'] = 'SomeTimes'
    c_map['my'] = 'MY' if 'MY' in vocab else 'my'
    c_map['no'] = 'NO' if 'NO' in vocab else 'No'
    return c_map

def tokenize_sentence(sentence, canonical_map):
    """
    Clean and tokenize a sentence, mapping known words to their canonical casing.
    Handles multi-word tokens like 'Thanks you'.
    """
    cleaned = sentence.strip()
    if not cleaned or cleaned.startswith('#'):
        return []

    # Replace punctuation with spaces
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    raw_tokens = [t.strip() for t in cleaned.split() if t.strip()]
    if not raw_tokens:
        return []

    tokens = []
    i = 0
    while i < len(raw_tokens):
        # Check for 2-word idioms like "thanks you" or "thank you"
        if i + 1 < len(raw_tokens):
            two_word = f"{raw_tokens[i].lower()} {raw_tokens[i+1].lower()}"
            if two_word in canonical_map:
                tokens.append(canonical_map[two_word])
                i += 2
                continue

        w_lower = raw_tokens[i].lower()
        if w_lower in canonical_map:
            tokens.append(canonical_map[w_lower])
        else:
            tokens.append(raw_tokens[i].capitalize())
        i += 1

    return tokens

def train_ngram_model(corpus_paths, vocab):
    canonical_map = build_canonical_map(vocab)
    
    unigrams = Counter()
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)
    quadgrams = defaultdict(Counter)
    
    total_sentences = 0
    total_tokens = 0

    for c_path in corpus_paths:
        if not os.path.exists(c_path):
            print(f"Warning: Corpus file '{c_path}' not found, skipping.")
            continue
            
        print(f"Reading training corpus: {c_path}")
        with open(c_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                tokens = tokenize_sentence(line, canonical_map)
                if not tokens or len(tokens) < 1:
                    continue
                
                total_sentences += 1
                total_tokens += len(tokens)

                for i, word in enumerate(tokens):
                    unigrams[word] += 1
                    
                    if i >= 1:
                        prev1 = tokens[i - 1]
                        bigrams[prev1][word] += 1
                        
                    if i >= 2:
                        prev2 = (tokens[i - 2], tokens[i - 1])
                        trigrams[prev2][word] += 1
                        
                    if i >= 3:
                        prev3 = (tokens[i - 3], tokens[i - 2], tokens[i - 1])
                        quadgrams[prev3][word] += 1

    # Ensure all signs in the vocabulary have at least a baseline frequency
    for v in vocab:
        if unigrams[v] == 0:
            unigrams[v] = 1

    return {
        'vocab': vocab,
        'canonical_map': canonical_map,
        'unigrams': dict(unigrams),
        'bigrams': {k: dict(v) for k, v in bigrams.items()},
        'trigrams': {k: dict(v) for k, v in trigrams.items()},
        'quadgrams': {k: dict(v) for k, v in quadgrams.items()},
        'total_sentences': total_sentences,
        'total_tokens': total_tokens,
    }

def print_model_stats(model_bundle):
    print("\n" + "=" * 60)
    print("  NLP MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Sentences Processed : {model_bundle['total_sentences']}")
    print(f"Total Tokens Processed    : {model_bundle['total_tokens']}")
    print(f"Vocabulary Size           : {len(model_bundle['vocab'])} signs")
    print(f"Bigram Transitions Learned: {len(model_bundle['bigrams'])}")
    print(f"Trigram Contexts Learned  : {len(model_bundle['trigrams'])}")
    print(f"4-gram Contexts Learned   : {len(model_bundle['quadgrams'])}")
    print("-" * 60)
    print("Sample Top Predictions for Common Starters:")

    sample_starters = ['Hello', 'I', 'how', 'what', 'Where', 'please', 'MY', 'when']
    for s in sample_starters:
        # Check canonical match
        c_word = model_bundle['canonical_map'].get(s.lower(), s)
        next_words = model_bundle['bigrams'].get(c_word, {})
        top_3 = sorted(next_words.items(), key=lambda x: x[1], reverse=True)[:3]
        sugg_str = ", ".join([f"{w} ({cnt})" for w, cnt in top_3]) if top_3 else "None yet"
        print(f"  [{c_word:8}] -> {sugg_str}")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Train NLP next-word model on text data")
    parser.add_argument('--corpus', nargs='+', default=[DEFAULT_CORPUS_PATH],
                        help="Path(s) to training corpus text file(s)")
    parser.add_argument('--vision_model', default=DEFAULT_VISION_MODEL_PATH,
                        help="Path to mp_mlp_model.pkl to read active sign vocabulary")
    parser.add_argument('--output', default=OUTPUT_MODEL_PATH,
                        help="Path to save trained nlp_model.pkl")
    args = parser.parse_args()

    print("=" * 60)
    print("  TRAINING SIGN LANGUAGE NLP PREDICTOR")
    print("=" * 60)
    
    vocab = get_current_vocab(args.vision_model)
    print(f"Loaded {len(vocab)} signs from vision model: {vocab[:8]}...")
    
    model_bundle = train_ngram_model(args.corpus, vocab)
    
    if model_bundle['total_sentences'] == 0:
        print(f"\n[Warning] No sentences found in {args.corpus}!")
        print("Model file will NOT be overwritten with an empty model.")
        return

    # Save trained model bundle
    with open(args.output, 'wb') as f:
        pickle.dump(model_bundle, f)
    
    print(f"[Success] Saved trained NLP model to: {args.output}")
    print_model_stats(model_bundle)

if __name__ == "__main__":
    main()
