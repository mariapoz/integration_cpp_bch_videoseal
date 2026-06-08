#!/usr/bin/env python3
"""
VideoSeal Video Watermark Decoder with legacy project structure.

Keeps the old CLI style:
    python decode.py --input watermarked.mp4

Pipeline:
    watermarked video -> VideoSeal 256-bit message -> legacy 32-bit payload -> unpermutation -> Hamming decode -> DDMMYYXXX
"""

import argparse
import sys
from pathlib import Path
import zlib
from collections import Counter

import torch
import cv2
import numpy as np


from watermark_format import unpack_watermark, format_watermark_info, check_watermark
from watermark_hamming import hamming_decode
from watermark_permutation import unpermute_bits, load_permutation_key

MAGIC = b"VSW1"
MESSAGE_BITS = 256
MESSAGE_BYTES = MESSAGE_BITS // 8


def read_video_cv2(input_path: str) -> torch.Tensor:
    """Read video with OpenCV and return RGB uint8 frames as [T,H,W,C]."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from video: {input_path}")
    arr = np.ascontiguousarray(np.stack(frames, axis=0))
    return torch.from_numpy(arr)


def bits_to_bytes(bits_tensor: torch.Tensor) -> bytes:
    bits = bits_tensor.detach().cpu().int().flatten().tolist()
    if len(bits) < MESSAGE_BITS:
        raise ValueError(f"Expected at least {MESSAGE_BITS} bits, got {len(bits)}")
    bits = bits[:MESSAGE_BITS]
    out = bytearray()
    for i in range(0, MESSAGE_BITS, 8):
        b = 0
        for bit in bits[i:i+8]:
            b = (b << 1) | int(bit)
        out.append(b)
    return bytes(out)


def unpack_legacy_watermark(watermark_bytes: bytes, permutation_key):
    if len(watermark_bytes) != 4:
        raise ValueError(f"Legacy watermark must be 4 bytes, got {len(watermark_bytes)}")
    permuted_32bit = int.from_bytes(watermark_bytes, byteorder="big")
    encoded_32bit = unpermute_bits(permuted_32bit, permutation_key)
    decoded_24bit, is_valid, error_position = hamming_decode(encoded_32bit)
    watermark_id = unpack_watermark(decoded_24bit)
    return watermark_id, is_valid, error_position


def parse_videoseal_message(msg_bytes: bytes, permutation_key):
    magic_ok = msg_bytes[:4] == MAGIC
    legacy_payload = msg_bytes[4:8]
    stored_crc = int.from_bytes(msg_bytes[8:12], "big")
    actual_crc = zlib.crc32(legacy_payload)
    crc_ok = stored_crc == actual_crc
    watermark_id, hamming_ok, error_pos = unpack_legacy_watermark(legacy_payload, permutation_key)
    return {
        "magic_ok": magic_ok,
        "crc_ok": crc_ok,
        "watermark_id": watermark_id,
        "hamming_ok": hamming_ok,
        "error_pos": error_pos,
        "raw_message_hex": msg_bytes.hex(),
        "legacy_payload_hex": legacy_payload.hex(),
    }


def load_model(model_path: str, device: torch.device):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"VideoSeal model not found: {model_file}\n"
            "Download it first:\n"
            "  mkdir -p ckpts\n"
            "  curl -L -o ckpts/y_256b_img.jit https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit"
        )
    model = torch.jit.load(str(model_file), map_location=device)
    model.to(device)
    model.eval()
    return model


def decode_video(input_path: str, model_path: str = "ckpts/y_256b_img.jit", use_gpu: bool = True, aggregation: str = "avg", max_frames: int | None = None):
    try:
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Device: {device}")
        permutation_key = load_permutation_key()
        model = load_model(model_path, device)

        frames = read_video_cv2(input_path)
        if max_frames:
            frames = frames[:max_frames]
        print(f"[INFO] Video frames: {tuple(frames.shape)}")
        video_tensor = frames.float().div(255.0).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            aggregated_msg = model.detect_video_and_aggregate(video_tensor, aggregation=aggregation)
            # For safety: if model returns logits instead of hard bits, threshold them.
            detected_bits = (aggregated_msg > 0).float()

        msg_bytes = bits_to_bytes(detected_bits)
        result = parse_videoseal_message(msg_bytes, permutation_key)
        info = format_watermark_info(result["watermark_id"])
        result["info"] = info
        result["date_valid"] = False
        if info:
            # check_watermark expects 24-bit label, so validate via parsed date format only here.
            result["date_valid"] = True

        print("\n" + "=" * 70)
        print("DETECTION RESULTS")
        print("=" * 70)
        print(f"Detected ID:       {result['watermark_id']}")
        if info:
            print(f"Date:              {info['date']}")
            print(f"Sequence:          {info['sequence']}")
        print(f"Magic OK:          {result['magic_ok']}")
        print(f"CRC OK:            {result['crc_ok']}")
        print(f"Hamming OK:        {result['hamming_ok']}  error_pos={result['error_pos']}")
        print(f"Legacy payload:    {result['legacy_payload_hex']}")
        print("=" * 70)
        return result
    except Exception as e:
        print(f"[ERROR] Decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="VideoSeal decoder with old project structure")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--model", default="ckpts/y_256b_img.jit")
    parser.add_argument("--aggregation", default="avg", choices=["avg", "squared_avg", "l1norm_avg", "l2norm_avg"])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    result = decode_video(args.input, args.model, not args.no_gpu, args.aggregation, args.max_frames)
    if result and result.get("magic_ok") and result.get("crc_ok") and result.get("hamming_ok"):
        print(f"\n[SUCCESS] Valid watermark detected: {result['watermark_id']}")
        sys.exit(0)
    elif result:
        print("\n[WARNING] Watermark decoded, but checks did not fully pass")
        sys.exit(2)
    else:
        print("\n[FAILED] Could not detect watermark")
        sys.exit(1)


if __name__ == "__main__":
    main()
