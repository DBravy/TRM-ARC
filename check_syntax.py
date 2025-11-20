"""
Quick syntax check for slot attention implementation.
Verifies the model file can be parsed without running it.
"""

import ast
import sys

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "✓ Syntax valid"
    except SyntaxError as e:
        return False, f"✗ Syntax error: {e}"
    except Exception as e:
        return False, f"✗ Error: {e}"

if __name__ == "__main__":
    print("="*60)
    print("SYNTAX CHECK FOR SLOT ATTENTION IMPLEMENTATION")
    print("="*60)

    files_to_check = [
        "models/recursive_reasoning/trm.py",
        "test_slot_attention.py",
    ]

    all_valid = True
    for filepath in files_to_check:
        valid, message = check_file_syntax(filepath)
        print(f"\n{filepath}")
        print(f"  {message}")
        if not valid:
            all_valid = False

    print("\n" + "="*60)
    if all_valid:
        print("✓ ALL FILES HAVE VALID SYNTAX")
        print("="*60)
        print("\nThe implementation is syntactically correct!")
        print("\nTo run full tests with PyTorch, use:")
        print("  python test_slot_attention.py")
        print("\nMake sure you have the required packages installed:")
        print("  pip install -r requirements.txt")
    else:
        print("✗ SYNTAX ERRORS FOUND")
        print("="*60)
        sys.exit(1)
