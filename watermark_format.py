#!/usr/bin/env python3
"""
Watermark Format - Epoch-based ID packing/unpacking

Adapted from spread-spectrum-master/psychoacoustic_watermarker/util.py
Format: "DDMMYYXXX" (9 digits) → 24 bits (10-bit ID + 14-bit epoch days)
"""

from datetime import datetime


def isInt(s):
    """Check if string can be converted to int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def make_watermark_string(id=0, current_dt=datetime.now()) -> str:
    """
    Create watermark string from ID and date.

    Args:
        id: Sequence number (0-999)
        current_dt: datetime object

    Returns:
        string in "DDMMYYXXX" format (9 digits)
    """
    if not (0 <= id <= 999):
        raise ValueError(f'ID must be 0-999, got: {id}')

    return '{:02d}{:02d}{:02d}{:03d}'.format(
        current_dt.day,
        current_dt.month,
        current_dt.year % 100,
        id
    )


def pack_watermark(watermark: str) -> int:
    """
    Pack 9-digit watermark ("DDMMYYXXX") into 24 bits.

    Bit layout (24 bits):
    - Bits 0-9:   id (10 bits, 0-999)
    - Bits 10-23: epoch day from 01.01.2025 (14 bits, 01.01.2025-09.11.2069)

    Args:
        watermark: string in "DDMMYYXXX" format (9 digits)

    Returns:
        packed watermark as 24-bit integer
    """
    if len(watermark) != 9 or not isInt(watermark):
        raise ValueError(f'Watermark string must be 9 digits, got: {watermark}')

    day = int(watermark[0:2])
    month = int(watermark[2:4])
    year = int(watermark[4:6])
    id = int(watermark[6:9])

    # Validate ranges
    if not (1 <= day <= 31):
        raise ValueError(f'Invalid day: {day}')
    if not (1 <= month <= 12):
        raise ValueError(f'Invalid month: {month}')
    if not (0 <= year <= 99):
        raise ValueError(f'Invalid year: {year}')
    if not (0 <= id <= 999):
        raise ValueError(f'Invalid id: {id}')

    # Calculate epoch days from 01.01.2025
    epoch_start_dt = datetime(2025, 1, 1)
    current_dt = datetime(year + 2000, month, day)
    delta_day = (current_dt.date() - epoch_start_dt.date()).days

    if delta_day < 0:
        raise ValueError(f'Date before epoch start (01.01.2025): {day:02d}.{month:02d}.{year + 2000:04d}')
    if delta_day > 2 ** 14 - 1:
        raise ValueError(f'Maximum available date is 09.11.2069, got: {day:02d}.{month:02d}.{year + 2000:04d}')

    # Pack into 24 bits: [id (10 bits)][epoch_days (14 bits)]
    packed = id | (delta_day << 10)

    return packed


def unpack_watermark(label: int) -> str:
    """
    Unpack 24-bit label to 9-digit watermark string in "DDMMYYXXX" format.

    Args:
        label: detected watermark in 24-bits format

    Returns:
        string in "DDMMYYXXX" format (9 digits)
    """
    if label >= (1 << 24):
        raise ValueError(f'Label must be 24 bits, got: {label}')

    # Extract components
    id = label & (2 ** 10 - 1)  # 10 bits
    delta_day = (label >> 10) & (2 ** 14 - 1)  # 14 bits

    # Convert epoch days back to date
    from datetime import timedelta
    epoch_start_dt = datetime(2025, 1, 1)
    dt = epoch_start_dt + timedelta(days=delta_day)

    return '{:02d}{:02d}{:02d}{:03d}'.format(dt.day, dt.month, dt.year % 100, id)


def check_watermark(label: int, current_dt=datetime.now()) -> bool:
    """
    Check watermark to contain valid date and id.

    Args:
        label: detected watermark in 24-bits format
        current_dt: datetime object for validation

    Returns:
        True if id is 0-999 and date is between 01.01.2025 and current date, False otherwise
    """
    if label >= (1 << 24):
        return False

    # Extract components
    id = label & (2 ** 10 - 1)
    if id > 999:
        print(f'[WARNING] Invalid id: {id} (expected 0-999)')
        return False

    delta_day = (label >> 10) & (2 ** 14 - 1)

    # Check date range
    from datetime import timedelta
    epoch_start_dt = datetime(2025, 1, 1)
    delta_day_max = (current_dt.date() - epoch_start_dt.date()).days

    if delta_day > delta_day_max:
        dt = epoch_start_dt + timedelta(days=delta_day)
        print(f'[WARNING] Date in future: {dt.day:02d}.{dt.month:02d}.{dt.year:04d}')
        return False

    return True


def format_watermark_info(watermark_string: str) -> dict:
    """
    Parse watermark string into human-readable components.

    Args:
        watermark_string: "DDMMYYXXX" format

    Returns:
        dict with parsed components
    """
    if len(watermark_string) != 9:
        return None

    day = int(watermark_string[0:2])
    month = int(watermark_string[2:4])
    year = int(watermark_string[4:6])
    id = int(watermark_string[6:9])

    return {
        'id': watermark_string,
        'date': f"{day:02d}.{month:02d}.20{year:02d}",
        'day': day,
        'month': month,
        'year': year,
        'sequence': id,
        'valid': True  # Can add more validation here
    }


# Self-test
if __name__ == '__main__':
    print("Testing watermark_format.py")
    print("=" * 60)

    # Test 1: Make watermark string
    test_date = datetime(2025, 11, 8)
    wm_str = make_watermark_string(id=42, current_dt=test_date)
    print(f"Generated watermark: {wm_str}")
    assert wm_str == "081125042", f"Expected '081125042', got '{wm_str}'"

    # Test 2: Pack
    packed = pack_watermark(wm_str)
    print(f"Packed to 24 bits: 0x{packed:06X} ({packed})")

    # Test 3: Unpack
    unpacked = unpack_watermark(packed)
    print(f"Unpacked: {unpacked}")
    assert unpacked == wm_str, f"Pack/unpack mismatch: {wm_str} != {unpacked}"

    # Test 4: Check watermark
    is_valid = check_watermark(packed, current_dt=datetime.now())
    print(f"Valid: {is_valid}")
    assert is_valid, "Watermark should be valid"

    # Test 5: Format info
    info = format_watermark_info(unpacked)
    print(f"Info: {info}")

    print("\n✅ All watermark_format tests passed!")
