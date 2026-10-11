"""
Interactive Test Script for Next-Word Prediction & Suggestions
Allows instant testing of the NLP suggestion engine in the terminal
without needing a webcam.
"""

import sys
import os

# Add local path to import nlp_predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nlp_predictor import NextWordPredictor

def run_automated_tests(predictor):
    print("=" * 60)
    print("  AUTOMATED PREDICTION TESTS")
    print("=" * 60)
    test_cases = [
        [],
        ["Hello"],
        ["how"],
        ["what"],
        ["Where"],
        ["I"],
        ["I", "want"],
        ["please"],
        ["MY"],
        ["MY", "friend"],
        ["Thanks you"],
    ]

    for tc in test_cases:
        context_disp = " ".join(tc) if tc else "(Start of sentence)"
        suggestions = predictor.predict(tc, top_k=3)
        formatted_suggs = "   ".join([f"[{i+1}] {w}" for i, w in enumerate(suggestions)])
        print(f"Context: {context_disp:25} -> Next Suggestions: {formatted_suggs}")
    print("=" * 60 + "\n")

def run_interactive_mode(predictor):
    print("=" * 60)
    print("  INTERACTIVE SUGGESTION SIMULATOR")
    print("=" * 60)
    print("Type a word or press [1, 2, 3] to pick a suggestion.")
    print("Commands: 'clear' to reset, 'exit' to quit.\n")

    sentence = []
    while True:
        suggestions = predictor.predict(sentence, top_k=3)
        curr_str = " ".join(sentence) if sentence else "(Empty)"
        print(f"\n📝 Current sentence: [ {curr_str} ]")
        print("💡 Next Suggestions:")
        for i, s in enumerate(suggestions):
            print(f"    [{i+1}] {s}")

        choice = input("\n👉 Enter 1/2/3 to select suggestion, type custom word, or 'exit': ").strip()
        if choice.lower() == 'exit':
            print("Exiting simulator.")
            break
        elif choice.lower() == 'clear':
            sentence = []
            print("Cleared sentence.")
            continue
        elif choice in ['1', '2', '3']:
            idx = int(choice) - 1
            if idx < len(suggestions):
                chosen = suggestions[idx]
                predictor.record_selection(sentence, chosen)
                sentence.append(chosen)
                print(f"Selected: '{chosen}'")
        elif choice:
            predictor.record_selection(sentence, choice)
            sentence.append(choice)
            print(f"Added: '{choice}'")

if __name__ == "__main__":
    predictor = NextWordPredictor()
    run_automated_tests(predictor)
    
    # If run in interactive terminal, prompt user
    if sys.stdin.isatty():
        try:
            run_interactive_mode(predictor)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
