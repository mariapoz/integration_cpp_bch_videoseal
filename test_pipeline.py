#!/usr/bin/env python3
"""Unit tests for reusable Format → Hamming → Permutation pipeline and 256-bit VideoSeal wrapper."""
from datetime import datetime
import zlib
from watermark_format import pack_watermark, unpack_watermark, make_watermark_string
from watermark_hamming import hamming_encode, hamming_decode
from watermark_permutation import generate_permutation_key, permute_bits, unpermute_bits

MAGIC=b'VSW1'

def make_message(wm, key):
    p=pack_watermark(wm)
    h=hamming_encode(p)
    q=permute_bits(h,key).to_bytes(4,'big')
    crc=zlib.crc32(q).to_bytes(4,'big')
    return MAGIC+q+crc+bytes(20)

def parse_message(msg,key):
    assert len(msg)==32
    assert msg[:4]==MAGIC
    q=msg[4:8]
    assert zlib.crc32(q)==int.from_bytes(msg[8:12],'big')
    h=unpermute_bits(int.from_bytes(q,'big'),key)
    d,valid,err=hamming_decode(h)
    return unpack_watermark(d), valid, err

def main():
    key=generate_permutation_key(seed=12345)
    wm=make_watermark_string(id=42,current_dt=datetime(2025,11,8))
    msg=make_message(wm,key)
    got,valid,err=parse_message(msg,key)
    print('original:',wm)
    print('decoded: ',got)
    print('valid:',valid,'err:',err)
    assert got==wm and valid and err==0
    # 1-bit error in legacy 32-bit section should be corrected after unpermutation/Hamming.
    corrupted=bytearray(msg)
    corrupted[5]^=0b00000001
    got,valid,err=parse_message(bytes(corrupted),key)
    print('after 1-bit legacy error:',got,valid,err)
    assert got==wm and valid and err>0
    print('\n✅ All unit tests passed')

if __name__=='__main__': main()
