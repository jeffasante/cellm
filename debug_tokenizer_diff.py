#!/usr/bin/env python3
"""
Debug the discrepancy between Python and Rust tokenizer results
"""

import json
from pathlib import Path

def analyze_tokenizer_encoding():
    """Analyze how the tokenizer encodes text to understand the Rust vs Python difference"""
    
    tokenizer_path = Path("models/hf/qwen3.5-0.8b/tokenizer.json")
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check the vocabulary structure
    vocab = data.get('model', {}).get('vocab', {})
    print(f"Vocabulary size: {len(vocab)}")
    
    # Find some key tokens
    key_tokens = {}
    
    # Look for "Hello" in the vocab
    for token, token_id in vocab.items():
        if token == "Hello":
            key_tokens["Hello"] = token_id
        elif token == "Ġworld":
            key_tokens["Ġworld"] = token_id
        elif token == "ä½łå¥½":  # This is how "你好" appears in the vocab
            key_tokens["ä½łå¥½"] = token_id
    
    print("\nKey vocabulary entries:")
    for token, token_id in key_tokens.items():
        print(f"  '{token}' -> ID {token_id}")
    
    # Check what tokens 96304 and 1562 actually represent
    print(f"\nLooking up problematic token IDs from Rust:")
    for token_id in [96304, 1562]:
        if token_id in vocab:
            token_text = list(vocab.keys())[list(vocab.values()).index(token_id)]
            print(f"  ID {token_id}: '{token_text}'")
        else:
            print(f"  ID {token_id}: Not in main vocab")
            
            # Check if it's in added_tokens
            added_tokens = data.get('added_tokens', [])
            for token in added_tokens:
                if token.get('id') == token_id:
                    print(f"  ID {token_id}: '{token.get('content')}' (added token)")
                    break
    
    # Check BPE merges to understand encoding
    merges = data.get('model', {}).get('merges', [])
    print(f"\nBPE merges count: {len(merges)}")
    
    # Look for merges that might affect "Hello"
    hello_merges = [merge for merge in merges if 'Hello' in merge or 'Ġ' in merge]
    print(f"Merges involving 'Hello' or space: {len(hello_merges)}")
    for merge in hello_merges[:5]:  # Show first 5
        print(f"  {merge}")

def test_encoding_steps():
    """Show step-by-step encoding process"""
    
    try:
        from tokenizers import Tokenizer
        
        tokenizer = Tokenizer.from_file("models/hf/qwen3.5-0.8b/tokenizer.json")
        
        print("\n=== Step-by-step encoding of 'Hello' ===")
        
        # Get the encoding
        encoding = tokenizer.encode("Hello")
        
        print(f"Input: 'Hello'")
        print(f"Tokens: {encoding.tokens}")
        print(f"IDs: {encoding.ids}")
        print(f"Offsets: {encoding.offsets}")
        
        # Check what the tokenizer sees
        print(f"\nNormalized: '{encoding.normalized}'")
        print(f"Original: '{encoding.original}'")
        
        # Test character by character
        print(f"\n=== Character analysis ===")
        for i, char in enumerate("Hello"):
            char_encoding = tokenizer.encode(char)
            print(f"  '{char}' -> {char_encoding.ids} -> '{tokenizer.decode(char_encoding.ids)}'")
            
    except ImportError:
        print("tokenizers library not available")
    except Exception as e:
        print(f"Error: {e}")

def check_tokenizer_config():
    """Check tokenizer configuration that might affect encoding"""
    
    config_path = Path("models/hf/qwen3.5-0.8b/tokenizer_config.json")
    
    if not config_path.exists():
        print("tokenizer_config.json not found")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n=== Tokenizer configuration ===")
    
    key_config = [
        "add_prefix_space",
        "add_bos_token", 
        "add_eos_token",
        "clean_up_tokenization_spaces",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "use_fast"
    ]
    
    for key in key_config:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # Check chat template
    if "chat_template" in config:
        chat_template = config["chat_template"]
        print(f"\nChat template (first 200 chars): {chat_template[:200]}...")

def main():
    print("Debugging tokenizer Rust vs Python discrepancy...")
    
    analyze_tokenizer_encoding()
    test_encoding_steps()
    check_tokenizer_config()
    
    print(f"\n=== Summary ===")
    print(f"Python tokenizes 'Hello' as ID 9419")
    print(f"Rust tokenizes 'Hello' as ID 96304 (incorrect)")
    print(f"This suggests a bug in the Rust tokenizer implementation")
    print(f"Possible causes:")
    print(f"  1. Different normalization settings")
    print(f"  2. Different BPE merge application")
    print(f"  3. Character encoding issues")
    print(f"  4. Version mismatch in tokenizers crate")

if __name__ == "__main__":
    main()
