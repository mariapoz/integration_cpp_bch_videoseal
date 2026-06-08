# VideoSeal Watermarking with Hamming ECC + Permutation

Это версия проекта в той же структуре, что старый `INTEGRATION_CPP_BCH`, но вместо RivaGAN используется **VideoSeal TorchScript**.

## Структура

```text
INTEGRATION_CPP_BCH_VIDEOSEAL/
├── .gitignore
├── batch_encode.py
├── decode.py
├── DESIGN.md
├── encode.py
├── permutation_key.bin
├── README.md
├── requirements.txt
├── test_pipeline.py
├── watermark_format.py
├── watermark_hamming.py
└── watermark_permutation.py
```

Переиспользованы старые модули:

- `watermark_format.py` — формат ID `DDMMYYXXX`;
- `watermark_hamming.py` — Hamming ECC;
- `watermark_permutation.py` — перестановка битов;
- `permutation_key.bin` — тот же ключ перестановки.

Новая часть — `encode.py` и `decode.py`: они используют VideoSeal вместо RivaGAN.

## Архитектура

```text
DDMMYYXXX
→ watermark_format, 24 bits
→ Hamming(32,24), 32 bits
→ permutation, 32 bits
→ wrapper до 256 bits
→ VideoSeal embedder
→ watermarked video
```

Обратно:

```text
watermarked video
→ VideoSeal detector + temporal aggregation
→ 256-bit message
→ legacy 32-bit payload
→ unpermutation
→ Hamming decode
→ DDMMYYXXX
```

## Установка

```bash
cd INTEGRATION_CPP_BCH_VIDEOSEAL
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Скачать модель VideoSeal

```bash
mkdir -p ckpts
curl -L -o ckpts/y_256b_img.jit https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit
```

## Проверить внутренний pipeline без нейросети

```bash
python3 test_pipeline.py
```

## Встроить watermark

```bash
python3 encode.py --input video.mp4 --output watermarked.mp4 --id 081225042
```

или с автогенерацией ID по текущей дате:

```bash
python3 encode.py --input video.mp4 --output watermarked.mp4 --sequence 1
```

Для быстрого теста только первая минута:

```bash
python3 encode.py --input video.mp4 --output watermarked.mp4 --sequence 1 --first-minute
```

## Декодировать watermark

```bash
python3 decode.py --input watermarked.mp4
```

Успешный результат должен показать:

```text
Magic OK: True
CRC OK: True
Hamming OK: True
Detected ID: ...
```

## Batch encoding

```bash
python3 batch_encode.py --input-dir videos --output-dir outputs --start-sequence 1
```

## Важное

`permutation_key.bin` нужен и для encode, и для decode. Если заменить или потерять ключ, старые видео не декодируются.

Модель VideoSeal в этом проекте не хранится внутри архива, её нужно скачать отдельно командой выше.
