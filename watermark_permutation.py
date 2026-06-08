#!/usr/bin/env python3
"""
Bit Permutation for Watermark Security
Scrambles bit order using a secret key
"""

import numpy as np
import os

PERMUTATION_KEY_PATH = 'permutation_key.bin'


def generate_permutation_key(seed: int = None) -> np.ndarray:
    """
    Generate random 32-bit permutation key

    Args:
        seed: Optional random seed for reproducibility

    Returns:
        32-element array with permutation [0-31]
    """
    if seed is not None:
        np.random.seed(seed)

    key = np.random.permutation(32)
    return key


def save_permutation_key(key: np.ndarray, path: str = PERMUTATION_KEY_PATH):
    """Save permutation key to binary file"""
    if len(key) != 32:
        raise ValueError(f"Key must be 32 elements, got {len(key)}")

    if not is_valid_permutation(key):
        raise ValueError("Key is not a valid permutation")

    # Save as uint8 array (values 0-31)
    key_uint8 = key.astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(key_uint8.tobytes())

    print(f"[INFO] Permutation key saved to: {path}")
    print(f"[WARNING] Keep {path} SECRET and backup safely!")


def load_permutation_key(path: str = PERMUTATION_KEY_PATH) -> np.ndarray:
    """Load permutation key from binary file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Permutation key not found: {path}")

    with open(path, 'rb') as f:
        key_bytes = f.read()

    if len(key_bytes) != 32:
        raise ValueError(f"Invalid key file size: {len(key_bytes)} (expected 32)")

    key = np.frombuffer(key_bytes, dtype=np.uint8)

    # Validate key is a valid permutation
    if not is_valid_permutation(key):
        raise ValueError("Invalid permutation key in file")

    return key


def is_valid_permutation(key: np.ndarray) -> bool:
    """Check if key is a valid permutation of 0-31"""
    if len(key) != 32:
        return False
    return set(key) == set(range(32))


def permute_bits(data_32bit: int, key: np.ndarray) -> int:
    """
    Apply permutation to 32-bit data

    Args:
        data_32bit: Input 32-bit value
        key: Permutation key

    Returns:
        Permuted 32-bit value
    """
    if data_32bit >= (1 << 32):
        raise ValueError(f"Data must be 32 bits")

    permuted = 0
    for i in range(32):
        # Get bit i from input
        bit = (data_32bit >> i) & 1
        # Place it at position key[i] in output
        permuted |= bit << int(key[i])

    return int(permuted)


def unpermute_bits(permuted_32bit: int, key: np.ndarray) -> int:
    """
    Reverse permutation (decode)

    Args:
        permuted_32bit: Permuted 32-bit value
        key: Same permutation key used for encoding

    Returns:
        Original 32-bit value
    """
    if permuted_32bit >= (1 << 32):
        raise ValueError(f"Data must be 32 bits")

    original = 0
    for i in range(32):
        # Get bit from position key[i] in input
        bit = (permuted_32bit >> int(key[i])) & 1
        # Place it at position i in output
        original |= bit << i

    return int(original)


def initialize_key_if_needed(path: str = PERMUTATION_KEY_PATH) -> np.ndarray:
    """
    Generate and save key if it doesn't exist
    Called during encode.py initialization
    """
    if not os.path.exists(path):
        print(f"[INFO] Generating new permutation key...")
        key = generate_permutation_key()
        save_permutation_key(key, path)
        return key
    else:
        return load_permutation_key(path)


# Self-test
if __name__ == '__main__':
    print("Testing watermark_permutation.py")
    print("=" * 60)

    # Generate test key
    test_key = generate_permutation_key(seed=42)
    print(f"Test key: {test_key[:8]}... (showing first 8)")

    # Test permute/unpermute
    test_data = 0xDEADBEEF
    permuted = permute_bits(test_data, test_key)
    unpermuted = unpermute_bits(permuted, test_key)

    print(f"Original:   0x{test_data:08X}")
    print(f"Permuted:   0x{permuted:08X}")
    print(f"Unpermuted: 0x{unpermuted:08X}")

    assert unpermuted == test_data, f"Permutation failed: {test_data:08X} != {unpermuted:08X}"

    print("\n✅ All permutation tests passed!")
