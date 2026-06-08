#!/usr/bin/env python3
"""
Streaming VideoSeal encoder.

Does NOT load the whole video into RAM. It reads a small chunk of frames,
embeds the watermark, writes the chunk through cv2.VideoWriter, and then frees memory.

Usage:
    python3 encode.py --input input.mp4 --output out.mp4 --id 081225042
    python3 encode.py --input input.mp4 --output out.mp4 --id 081225042 --chunk-size 1
"""

import argparse
import sys
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from watermark_format import pack_watermark, make_watermark_string
from watermark_hamming import hamming_encode
from watermark_permutation import permute_bits, initialize_key_if_needed

MAGIC = b"VSW1"
MESSAGE_BITS = 256
MESSAGE_BYTES = MESSAGE_BITS // 8


def bytes_to_bits(data: bytes, device=None) -> torch.Tensor:
    """Convert bytes to a [1, 256] float tensor with 0/1 bits."""
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)


def pack_watermark_id(id_string: str, permutation_key) -> bytes:
    """DDMMYYXXX -> 4 bytes using old pipeline: format -> Hamming -> permutation."""
    if len(id_string) != 9 or not id_string.isdigit():
        raise ValueError(f"ID must be 9 digits in DDMMYYXXX format, got: {id_string}")

    packed_24bit = pack_watermark(id_string)
    encoded_32bit = hamming_encode(packed_24bit)
    permuted_32bit = permute_bits(encoded_32bit, permutation_key)
    return permuted_32bit.to_bytes(4, byteorder="big")


def make_videoseal_message(id_string: str, permutation_key) -> bytes:
    """
    Build 32-byte / 256-bit VideoSeal message.

    Layout:
        0..3   MAGIC = VSW1
        4..7   old 32-bit payload: DDMMYYXXX -> Hamming -> permutation
        8..11  CRC32 of old 32-bit payload
        12..31 deterministic filler
    """
    legacy_payload = pack_watermark_id(id_string, permutation_key)
    crc = zlib.crc32(legacy_payload).to_bytes(4, "big")

    filler_seed = MAGIC + legacy_payload + crc
    filler = bytearray()
    counter = 0
    while len(filler) < 20:
        filler.extend(zlib.crc32(filler_seed + counter.to_bytes(4, "big")).to_bytes(4, "big"))
        counter += 1

    msg = MAGIC + legacy_payload + crc + bytes(filler[:20])
    assert len(msg) == MESSAGE_BYTES
    return msg


def generate_watermark_id(sequence_number: int = 1) -> str:
    if not (0 <= sequence_number <= 999):
        raise ValueError(f"Sequence must be 0-999, got {sequence_number}")
    return make_watermark_string(id=sequence_number, current_dt=datetime.now())


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


def open_cv2_writer(output_path: str, width: int, height: int, fps: float):
    """Open an OpenCV VideoWriter for MP4 output without external ffmpeg binary.

    Tries several common codecs because availability depends on the OpenCV build.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # For .mp4 on macOS, mp4v is usually available. avc1/H264 may or may not be.
    candidates = ["mp4v", "avc1", "H264", "XVID", "MJPG"]

    last_writer = None
    for fourcc_name in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            print(f"[INFO] OpenCV VideoWriter codec: {fourcc_name}")
            return writer
        last_writer = writer
        try:
            writer.release()
        except Exception:
            pass

    raise RuntimeError(
        "Could not open cv2.VideoWriter. "
        "Try output path with .mp4 extension, or install ffmpeg/use the ffmpeg-based encoder."
    )


def embed_chunk(model, frames_rgb: list[np.ndarray], message: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Embed watermark into a small chunk of RGB uint8 frames.

    Input:
        frames_rgb: list of HxWx3 uint8 RGB frames
    Output:
        HxWx3 uint8 RGB frames as numpy array [T,H,W,C]
    """
    arr = np.ascontiguousarray(np.stack(frames_rgb, axis=0))  # [T,H,W,C], uint8
    x = torch.from_numpy(arr).to(device=device, dtype=torch.float32).div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]

    with torch.no_grad():
        # Same VideoSeal call style as in the previous non-streaming encoder,
        # but now only for a small chunk instead of the whole video.
        y = model.embed(x, message, is_video=True)
        y = y.clamp(0.0, 1.0)

    out = y.detach().cpu().mul(255.0).round().to(torch.uint8)
    out = out.permute(0, 2, 3, 1).contiguous().numpy()  # [T,H,W,C], RGB uint8

    del x, y
    return out


def encode_video_streaming(
    input_path: str,
    output_path: str,
    watermark_id: str,
    model_path: str = "ckpts/y_256b_img.jit",
    use_gpu: bool = True,
    chunk_size: int = 2,
    first_minute_only: bool = False,
    crf: int = 23,
    scaling_w: float | None = None,
    step_size: int | None = None,
) -> bool:
    cap = None
    writer = None

    try:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Watermark ID: {watermark_id}")
        print(f"[INFO] Streaming mode, chunk_size={chunk_size}")

        permutation_key = initialize_key_if_needed()
        msg_bytes = make_videoseal_message(watermark_id, permutation_key)
        message = bytes_to_bits(msg_bytes, device=device)
        print(f"[INFO] VideoSeal message: {len(msg_bytes)} bytes / {message.shape[1]} bits")

        model = load_model(model_path, device)

        if scaling_w is not None and hasattr(model, "blender"):
            model.blender.scaling_w = float(scaling_w)
            print(f"[INFO] scaling_w set to {scaling_w}")
        if step_size is not None and hasattr(model, "step_size"):
            model.step_size = int(step_size)
            print(f"[INFO] step_size set to {step_size}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if first_minute_only:
            max_frames = int(fps * 60)
            if total_frames > 0:
                max_frames = min(max_frames, total_frames)
        else:
            max_frames = total_frames if total_frames > 0 else None

        print(f"[INFO] Video: {width}x{height}, fps={fps:.3f}, total_frames={total_frames}")
        if max_frames is not None:
            print(f"[INFO] Frames to process: {max_frames}")

        writer = open_cv2_writer(output_path, width, height, fps)

        chunk = []
        processed = 0
        written = 0

        while True:
            if max_frames is not None and processed >= max_frames:
                break

            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            chunk.append(frame_rgb)
            processed += 1

            if len(chunk) >= chunk_size:
                out_chunk = embed_chunk(model, chunk, message, device)
                for frame_rgb_out in out_chunk:
                    frame_bgr_out = cv2.cvtColor(frame_rgb_out, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr_out)
                written += len(out_chunk)
                chunk.clear()

                if written % max(1, chunk_size * 10) == 0:
                    if max_frames:
                        pct = written / max_frames * 100
                        print(f"[PROGRESS] {written}/{max_frames} frames ({pct:.1f}%)", flush=True)
                    else:
                        print(f"[PROGRESS] {written} frames", flush=True)

        if chunk:
            out_chunk = embed_chunk(model, chunk, message, device)
            for frame_rgb_out in out_chunk:
                frame_bgr_out = cv2.cvtColor(frame_rgb_out, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr_out)
            written += len(out_chunk)
            chunk.clear()

        cap.release()
        cap = None

        writer.release()
        writer = None

        print(f"[SUCCESS] Saved: {output_path}")
        print(f"[SUCCESS] Processed frames: {written}")
        return True

    except Exception as e:
        print(f"[ERROR] Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Streaming VideoSeal encoder with old project structure")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--id", help="Watermark ID, format DDMMYYXXX")
    parser.add_argument("--sequence", type=int, help="Auto-generate ID with sequence number 0-999")
    parser.add_argument("--model", default="ckpts/y_256b_img.jit", help="Path to VideoSeal TorchScript model")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--chunk-size", type=int, default=2, help="How many frames to process at once. Use 1 if RAM is low.")
    parser.add_argument("--first-minute", action="store_true", help="Process only the first 60 seconds")
    parser.add_argument("--crf", type=int, default=23, help="Kept for compatibility; ignored by cv2.VideoWriter version")
    parser.add_argument("--scaling-w", type=float, default=None, help="Optional VideoSeal watermark strength")
    parser.add_argument("--step-size", type=int, default=None, help="Optional VideoSeal video step size")
    args = parser.parse_args()

    if args.id:
        watermark_id = args.id
    elif args.sequence is not None:
        watermark_id = generate_watermark_id(args.sequence)
        print(f"[INFO] Auto-generated ID: {watermark_id}")
    else:
        print("[ERROR] Must specify --id or --sequence")
        sys.exit(1)

    ok = encode_video_streaming(
        input_path=args.input,
        output_path=args.output,
        watermark_id=watermark_id,
        model_path=args.model,
        use_gpu=not args.no_gpu,
        chunk_size=args.chunk_size,
        first_minute_only=args.first_minute,
        crf=args.crf,
        scaling_w=args.scaling_w,
        step_size=args.step_size,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
