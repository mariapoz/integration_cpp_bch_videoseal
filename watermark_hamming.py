#!/usr/bin/env python3
"""
Extended Hamming (32, 24, 2) Error Correction Code
Corrects 1-bit errors, detects 2-bit errors (SECDED)
"""

def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0


def hamming_encode(data_24bit: int) -> int:
    """
    Extended Hamming(32,24) encoding with SECDED
    
    Args:
        data_24bit: 24-bit watermark data
    
    Returns:
        32-bit encoded word with error correction
    """
    if data_24bit >= (1 << 24):
        raise ValueError(f"Data must be 24 bits, got: {data_24bit}")
    
    encoded = 0
    data_idx = 0
    
    # Step 1: Place data bits at non-power-of-2 positions (except 32)
    for pos in range(1, 33):
        if not is_power_of_2(pos) and pos != 32:
            # Extract next data bit
            bit = (data_24bit >> data_idx) & 1
            encoded |= bit << (pos - 1)
            data_idx += 1
    
    # Step 2: Calculate parity bits (positions 1, 2, 4, 8, 16)
    for p in [1, 2, 4, 8, 16]:
        parity = 0
        for pos in range(1, 32):
            if pos & p:  # Check if bit position is covered by this parity
                parity ^= (encoded >> (pos - 1)) & 1
        encoded |= parity << (p - 1)
    
    # Step 3: Overall parity (SECDED at position 32)
    overall = 0
    for pos in range(31):
        overall ^= (encoded >> pos) & 1
    encoded |= overall << 31
    
    return encoded


def hamming_decode(encoded_32bit: int) -> tuple[int, bool, int]:
    """
    Extended Hamming(32,24) decoding with error correction
    
    Returns:
        (decoded_data_24bit, is_valid, error_position)
        - decoded_data_24bit: Corrected 24-bit data
        - is_valid: True if no errors or corrected successfully
        - error_position: 0=no error, 1-31=corrected position, -1=uncorrectable
    """
    if encoded_32bit >= (1 << 32):
        raise ValueError(f"Encoded data must be 32 bits, got: {encoded_32bit}")
    
    # Step 1: Calculate syndrome
    syndrome = 0
    for p in [1, 2, 4, 8, 16]:
        parity = 0
        for pos in range(1, 32):
            if pos & p:
                parity ^= (encoded_32bit >> (pos - 1)) & 1
        if parity != 0:
            syndrome |= p
    
    # Step 2: Check overall parity
    overall_parity = 0
    for pos in range(32):
        overall_parity ^= (encoded_32bit >> pos) & 1
    
    # Step 3: Error detection and correction
    if syndrome == 0 and overall_parity == 0:
        # No errors
        error_pos = 0
        is_valid = True
    elif syndrome != 0 and overall_parity != 0:
        # Single-bit error - CORRECT IT
        error_pos = syndrome
        if 1 <= error_pos <= 31:
            encoded_32bit ^= (1 << (error_pos - 1))  # Flip error bit
        is_valid = True
    elif syndrome == 0 and overall_parity != 0:
        # Error in overall parity bit (position 32)
        error_pos = 32
        is_valid = True
    else:
        # Double-bit error - CANNOT CORRECT
        error_pos = -1
        is_valid = False
    
    # Step 4: Extract 24 data bits
    decoded_24bit = 0
    data_idx = 0
    for pos in range(1, 33):
        if not is_power_of_2(pos) and pos != 32:
            bit = (encoded_32bit >> (pos - 1)) & 1
            decoded_24bit |= bit << data_idx
            data_idx += 1
    
    return decoded_24bit, is_valid, error_pos


# Self-test
if __name__ == '__main__':
    print("Testing watermark_hamming.py")
    print("=" * 60)
    
    # Test encoding/decoding
    test_data = 0xABCDEF  # 24-bit test pattern
    
    # Encode
    encoded = hamming_encode(test_data)
    print(f"Original: 0x{test_data:06X} (24 bits)")
    print(f"Encoded:  0x{encoded:08X} (32 bits)")
    
    # Decode (no errors)
    decoded, valid, err_pos = hamming_decode(encoded)
    print(f"Decoded:  0x{decoded:06X}, Valid: {valid}, Error: {err_pos}")
    assert decoded == test_data and valid and err_pos == 0
    
    # Test 1-bit error correction
    corrupted = encoded ^ (1 << 10)  # Flip bit 10
    decoded, valid, err_pos = hamming_decode(corrupted)
    print(f"Corrupted (1 bit): Decoded: 0x{decoded:06X}, Valid: {valid}, Error pos: {err_pos}")
    assert decoded == test_data and valid and err_pos == 11  # Position 11 (0-indexed bit 10)
    
    # Test 2-bit error detection
    corrupted = encoded ^ (1 << 10) ^ (1 << 15)  # Flip bits 10 and 15
    decoded, valid, err_pos = hamming_decode(corrupted)
    print(f"Corrupted (2 bits): Valid: {valid}, Error pos: {err_pos}")
    assert not valid and err_pos == -1
    
    print("\n✅ All Hamming tests passed!")
