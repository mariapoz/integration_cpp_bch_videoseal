# DESIGN: миграция RivaGAN → VideoSeal при сохранении старой структуры

## Цель

Сохранить старую структуру проекта `INTEGRATION_CPP_BCH`, но заменить слой нейросетевого встраивания с RivaGAN на VideoSeal.

## Что переиспользовано

Из старого проекта оставлены без принципиальных изменений:

1. `watermark_format.py` — упаковка ID `DDMMYYXXX` в 24 бита.
2. `watermark_hamming.py` — Extended Hamming `(32,24)`.
3. `watermark_permutation.py` — перестановка 32 битов по секретному ключу.
4. `permutation_key.bin` — существующий ключ.
5. CLI-идея `encode.py`, `decode.py`, `batch_encode.py`.

## Новый encode pipeline

```text
Input ID: DDMMYYXXX
  ↓
Format pack: 24 bits
  ↓
Hamming ECC: 32 bits
  ↓
Permutation: 32 bits
  ↓
VideoSeal wrapper: 256 bits
  ↓
VideoSeal embed(video, message)
  ↓
watermarked.mp4
```

## 256-bit wrapper

VideoSeal TorchScript-модель `y_256b_img.jit` принимает сообщение длиной 256 бит. Старый payload занимает только 32 бита, поэтому он помещается внутрь 256-битного сообщения.

Layout:

```text
bytes 0..3   MAGIC = VSW1
bytes 4..7   old 32-bit permuted watermark
bytes 8..11  CRC32(old 32-bit payload)
bytes 12..31 deterministic filler
```

## Новый decode pipeline

```text
watermarked video
  ↓
VideoSeal detect_video_and_aggregate(video)
  ↓
256-bit message
  ↓
MAGIC + CRC check
  ↓
extract old 32-bit payload
  ↓
unpermutation
  ↓
Hamming decode
  ↓
Format unpack
  ↓
ID DDMMYYXXX
```

## Почему так удобно

- Не нужно переписывать формат ID.
- Не нужно менять batch-сценарий.
- Старые модули ECC/permutation остаются полезными.
- Можно честно сравнить старый RivaGAN-проект и новый VideoSeal-проект по одинаковым ID.

## Что измерять

1. `Magic OK` — VideoSeal смог восстановить структуру сообщения.
2. `CRC OK` — 32-битный payload не повреждён.
3. `Hamming OK` — ECC восстановила старый payload.
4. `Detected ID` — итоговый ID.
5. Для сравнения с RivaGAN: PSNR/SSIM/VMAF и success rate после атак.
