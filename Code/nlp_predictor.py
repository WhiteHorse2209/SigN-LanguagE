"""
General AI Next-Word Predictor for SignBridge / SigN-LanguagE.
Combines:
1. Local Generative AI (Ollama with auto-detected models: llama3.2:3b, qwen3, etc.)
   for true natural English predictive text autocompletion.
2. Zero-latency General English N-Gram & Markov model (sub-millisecond local response
   so the camera stream never stutters).
3. Automatic loading of custom external data models (nlp_model.pkl).
4. Dynamic online learning during active sessions.
"""

import os
import re
import pickle
import threading
from collections import defaultdict, Counter
from typing import List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, 'nlp_model.pkl')

def get_default_ollama_model() -> str:
    """Auto-detect which model is installed in the local Ollama instance."""
    try:
        import ollama
        resp = ollama.list()
        names = [m.model for m in resp.models]
        # Preference order based on conversational speed & intelligence
        for pref in ['llama3.2:3b', 'llama3.2:latest', 'llama3.2', 'qwen3:1.7b', 'qwen2:0.5b']:
            if pref in names:
                return pref
        if names:
            return names[0]
    except Exception:
        pass
    return 'llama3.2:3b'

# Comprehensive General English Language Model Prior (Top transitions for natural English conversation)
GENERAL_ENGLISH_TRANSITIONS = {
    # Question words
    'how': Counter({'are': 35, 'you': 25, 'do': 20, 'is': 15, 'can': 10}),
    'how are': Counter({'you': 40, 'things': 15, 'we': 10, 'they': 8}),
    'how do': Counter({'you': 35, 'we': 15, 'i': 12}),
    'where': Counter({'are': 30, 'is': 25, 'do': 20, 'you': 15, 'can': 10}),
    'where are': Counter({'you': 40, 'we': 15, 'the': 12}),
    'where is': Counter({'the': 30, 'my': 25, 'your': 15}),
    'what': Counter({'is': 30, 'do': 25, 'are': 20, 'you': 15, 'can': 10}),
    'what do': Counter({'you': 45, 'we': 15, 'they': 10}),
    'what is': Counter({'that': 25, 'this': 20, 'your': 18, 'the': 15}),
    'when': Counter({'will': 25, 'are': 20, 'do': 20, 'can': 15, 'you': 12}),
    'when will': Counter({'you': 30, 'we': 20, 'it': 15}),
    'why': Counter({'do': 30, 'are': 25, 'is': 15, 'did': 15}),
    'who': Counter({'is': 35, 'are': 25, 'will': 15, 'can': 12}),

    # Pronouns & Subject starters
    'i': Counter({'want': 35, 'am': 28, 'love': 25, 'have': 20, 'need': 18, 'can': 15, 'see': 12, 'go': 10}),
    'i want': Counter({'to': 45, 'you': 25, 'help': 20, 'see': 18, 'go': 15, 'more': 10}),
    'i want to': Counter({'see': 30, 'help': 25, 'go': 20, 'know': 18, 'play': 15}),
    'i love': Counter({'you': 50, 'my': 30, 'this': 18, 'to': 15, 'school': 12}),
    'i love you': Counter({'so': 25, 'very': 20, 'too': 18}),
    'i am': Counter({'good': 30, 'happy': 25, 'here': 20, 'ready': 15, 'fine': 12}),
    'i have': Counter({'a': 30, 'to': 25, 'no': 15, 'some': 12, 'been': 10}),
    'i need': Counter({'help': 35, 'you': 25, 'to': 20, 'time': 12}),
    'i can': Counter({'help': 35, 'see': 25, 'do': 20, 'go': 15}),
    'i see': Counter({'you': 35, 'my': 25, 'the': 20, 'what': 15}),

    'you': Counter({'are': 35, 'can': 25, 'have': 20, 'want': 18, 'know': 15, 'look': 12}),
    'you are': Counter({'my': 30, 'good': 25, 'welcome': 20, 'so': 15}),
    'you can': Counter({'help': 30, 'do': 25, 'see': 20, 'go': 15}),
    'you have': Counter({'to': 25, 'a': 20, 'been': 15}),
    'you want': Counter({'to': 35, 'help': 20, 'me': 18}),

    'my': Counter({'friend': 35, 'family': 30, 'name': 25, 'school': 20, 'life': 12}),
    'my friend': Counter({'is': 35, 'and': 20, 'good': 18, 'help': 15}),
    'my family': Counter({'is': 35, 'loves': 20, 'and': 18, 'good': 15}),
    'my school': Counter({'is': 35, 'good': 20, 'small': 15}),

    # Common greetings, requests, responses
    'hello': Counter({'how': 35, 'my': 25, 'friend': 20, 'there': 15, 'everyone': 10}),
    'hello my': Counter({'friend': 40, 'family': 25, 'name': 15}),
    'please': Counter({'help': 40, 'tell': 20, 'let': 18, 'give': 15, 'come': 12, 'see': 10}),
    'please help': Counter({'me': 45, 'my': 25, 'you': 15}),
    'thanks': Counter({'you': 50, 'very': 25, 'a': 15}),
    'thanks you': Counter({'very': 35, 'so': 25, 'my': 20, 'friend': 15}),
    'thank you': Counter({'very': 35, 'so': 25, 'friend': 15}),
    'good': Counter({'morning': 30, 'to': 25, 'see': 20, 'friend': 18, 'job': 15}),
    'good to': Counter({'see': 45, 'meet': 30, 'know': 15}),
    'yes': Counter({'i': 35, 'please': 25, 'sure': 20, 'of': 15}),
    'yes i': Counter({'do': 30, 'can': 25, 'want': 20, 'am': 18}),
    'no': Counter({'problem': 35, 'i': 25, 'way': 20, 'thanks': 15}),
    'ok': Counter({'i': 30, 'see': 25, 'thank': 20, 'let': 15}),
    'help': Counter({'me': 40, 'you': 25, 'with': 20, 'now': 15}),
    'see': Counter({'you': 45, 'this': 20, 'the': 15}),
    'go': Counter({'to': 40, 'now': 25, 'home': 20, 'school': 15}),
    'play': Counter({'with': 35, 'now': 25, 'a': 15}),
    'leave': Counter({'now': 40, 'school': 20, 'me': 15}),
}

class NextWordPredictor:
    """
    True AI Next-Word Prediction Engine.
    Blends sub-millisecond General English N-Gram priors with
    asynchronous Ollama local LLM contextual intelligence.
    """
    def __init__(self, vocab: Optional[List[str]] = None, use_ollama: bool = True,
                 ollama_model: Optional[str] = None, model_path: Optional[str] = None):
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model or get_default_ollama_model()
        self.model_path = model_path or DEFAULT_MODEL_PATH
        
        # General English N-gram prior
        self.unigrams = Counter({'i': 50, 'you': 45, 'the': 40, 'to': 35, 'a': 30,
                                 'how': 25, 'what': 25, 'where': 20, 'want': 20, 'is': 20,
                                 'help': 18, 'good': 18, 'love': 18, 'my': 18, 'please': 15})
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.quadgrams = defaultdict(Counter)
        
        # Known sign vocabulary (for prioritizing or formatting matching signs)
        self.vocab = list(vocab) if vocab else []
        self.canonical_map = {v.lower(): v for v in self.vocab} if self.vocab else {}
        
        # LLM cache & async locks
        self.llm_cache = {}
        self.pending_llm_queries = set()
        self.cache_lock = threading.Lock()

        # Load general English baseline transitions
        self._load_general_english_transitions()

        # Load custom trained model (from train_nlp.py) if available
        self._load_custom_trained_model(self.model_path)

        # Update vocabulary if provided
        if vocab:
            for v in vocab:
                if v not in self.vocab:
                    self.vocab.append(v)
                self.canonical_map[v.lower()] = v

    def _load_general_english_transitions(self):
        """Seed the engine with rich general conversational English transitions."""
        for phrase, counter in GENERAL_ENGLISH_TRANSITIONS.items():
            tokens = phrase.split()
            if len(tokens) == 1:
                self.bigrams[tokens[0]].update(counter)
            elif len(tokens) == 2:
                self.trigrams[(tokens[0], tokens[1])].update(counter)
            for w, c in counter.items():
                self.unigrams[w] += c

    def _load_custom_trained_model(self, path: str):
        """Merge trained N-grams from train_nlp.py if available."""
        if not os.path.exists(path):
            return
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # Only merge if data has actual sentences trained
            if data.get('total_sentences', 0) > 0:
                for k, v in data.get('unigrams', {}).items():
                    self.unigrams[k.lower()] += v
                for k, v in data.get('bigrams', {}).items():
                    k_lower = k.lower() if isinstance(k, str) else k
                    for w, c in v.items():
                        self.bigrams[k_lower][w.lower()] += c
                for k, v in data.get('trigrams', {}).items():
                    k_lower = tuple(x.lower() for x in k) if isinstance(k, tuple) else k
                    for w, c in v.items():
                        self.trigrams[k_lower][w.lower()] += c
                for k, v in data.get('quadgrams', {}).items():
                    k_lower = tuple(x.lower() for x in k) if isinstance(k, tuple) else k
                    for w, c in v.items():
                        self.quadgrams[k_lower][w.lower()] += c
                
                # Merge vocabulary
                for v in data.get('vocab', []):
                    if v not in self.vocab:
                        self.vocab.append(v)
                    self.canonical_map[v.lower()] = v
        except Exception:
            pass

    def _clean_token(self, token: str) -> str:
        """Strip punctuation and normalize word format."""
        cleaned = re.sub(r'[^\w\s]', '', token).strip()
        return cleaned

    def _format_token_casing(self, word: str) -> str:
        """Return the word in sign vocabulary casing if present, else natural lowercase."""
        w_lower = word.lower()
        if w_lower in self.canonical_map:
            return self.canonical_map[w_lower]
        return word

    def _fetch_ollama_async(self, context_str: str):
        """
        Query the local Ollama LLM in a background thread for true generative
        AI natural next-word autocomplete predictions.
        """
        def _worker():
            try:
                import ollama
                prompt = (
                    f"You are a smartphone predictive text autocomplete engine. "
                    f"Given the prefix text: \"{context_str}\", predict the 3 most likely single words that immediately follow in natural conversational English. "
                    f"Reply ONLY with a comma-separated list of the 3 single words (e.g. are, do, can). No explanations, no numbering, no punctuation."
                )
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.3}
                )
                content = response['message']['content'].strip()
                raw_tokens = [t.strip(' "\'.,') for t in content.replace('\n', ',').split(',')]
                
                valid_preds = []
                last_word = context_str.split()[-1].lower() if context_str.split() else ''
                for tok in raw_tokens:
                    cleaned = self._clean_token(tok)
                    # Take only first word if multiple words returned
                    first_word = cleaned.split()[0] if cleaned.split() else ''
                    if not first_word or len(first_word) < 1:
                        continue
                    if first_word.lower() != last_word and first_word.lower() not in [x.lower() for x in valid_preds]:
                        valid_preds.append(self._format_token_casing(first_word))

                with self.cache_lock:
                    if valid_preds:
                        self.llm_cache[context_str] = valid_preds[:3]
            except Exception:
                pass
            finally:
                with self.cache_lock:
                    self.pending_llm_queries.discard(context_str)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def predict(self, words: List[str], top_k: int = 3) -> List[str]:
        """
        Main AI next-word prediction method.
        Returns top_k general English word predictions given the preceding context.
        Operates in 0 milliseconds and seamlessly incorporates Ollama AI suggestions.
        """
        clean_words = [self._clean_token(w) for w in words if w and self._clean_token(w)]
        
        # When no words signed yet: recommend top conversational starters
        if not clean_words:
            starters = ['Hello', 'How', 'I', 'Where', 'What', 'Please']
            formatted = [self._format_token_casing(s) for s in starters]
            return formatted[:top_k]

        context_str = " ".join(clean_words)
        lower_tokens = [w.lower() for w in clean_words]
        last_tok = lower_tokens[-1]

        # 1. Check if Ollama LLM predictions are available in cache
        llm_preds = []
        with self.cache_lock:
            if context_str in self.llm_cache:
                llm_preds = self.llm_cache[context_str]
            elif self.use_ollama and context_str not in self.pending_llm_queries:
                self.pending_llm_queries.add(context_str)
                self._fetch_ollama_async(context_str)

        # 2. Compute General English N-Gram predictions
        scores = defaultdict(float)

        # 4-gram context
        if len(lower_tokens) >= 3:
            prev3 = (lower_tokens[-3], lower_tokens[-2], lower_tokens[-1])
            for cand, count in self.quadgrams.get(prev3, {}).items():
                scores[cand] += count * 4.0

        # Trigram context
        if len(lower_tokens) >= 2:
            prev2 = (lower_tokens[-2], lower_tokens[-1])
            phrase2 = f"{lower_tokens[-2]} {lower_tokens[-1]}"
            for cand, count in self.trigrams.get(prev2, {}).items():
                scores[cand] += count * 3.0
            for cand, count in GENERAL_ENGLISH_TRANSITIONS.get(phrase2, {}).items():
                scores[cand] += count * 3.0

        # Bigram context
        if len(lower_tokens) >= 1:
            prev1 = lower_tokens[-1]
            for cand, count in self.bigrams.get(prev1, {}).items():
                scores[cand] += count * 2.0
            for cand, count in GENERAL_ENGLISH_TRANSITIONS.get(prev1, {}).items():
                scores[cand] += count * 2.0

        # Unigram background prior
        for cand, count in self.unigrams.most_common(15):
            scores[cand] += count * 0.1

        # Avoid predicting the exact same word consecutively
        if last_tok in scores:
            scores[last_tok] *= 0.05

        sorted_ngram = sorted(scores.keys(), key=lambda w: scores[w], reverse=True)

        # 3. Blend LLM predictions + General English N-Gram predictions
        combined = []
        for p in llm_preds:
            fmt = self._format_token_casing(p)
            if fmt.lower() != last_tok and fmt.lower() not in [x.lower() for x in combined]:
                combined.append(fmt)

        for p in sorted_ngram:
            fmt = self._format_token_casing(p)
            if fmt.lower() != last_tok and fmt.lower() not in [x.lower() for x in combined]:
                combined.append(fmt)
            if len(combined) >= top_k:
                break

        # Fallback guarantee
        if len(combined) < top_k:
            fallbacks = ['you', 'to', 'is', 'good', 'my', 'help', 'now']
            for fb in fallbacks:
                fmt = self._format_token_casing(fb)
                if fmt.lower() != last_tok and fmt.lower() not in [x.lower() for x in combined]:
                    combined.append(fmt)
                if len(combined) >= top_k:
                    break

        return combined[:top_k]

    def record_selection(self, current_words: List[str], chosen_word: str):
        """Reinforce the language model when a user picks a prediction."""
        full_seq = [w.lower() for w in current_words if w] + [chosen_word.lower()]
        for i, word in enumerate(full_seq):
            self.unigrams[word] += 1
            if i >= 1:
                self.bigrams[full_seq[i - 1]][word] += 1
            if i >= 2:
                self.trigrams[(full_seq[i - 2], full_seq[i - 1])][word] += 1
