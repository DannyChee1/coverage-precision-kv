# Compressed Before the Question

When shrinking a document's KV cache before you know what will be asked (question-hidden), keep more tokens at fewer bits rather than fewer tokens at more bits.

## Results

Accuracy (%) with the cache at 1/64 of its full size, all quantization overhead counted. Both compressed caches use KVzip+ to choose which tokens to keep.

| Model | Benchmark | Full cache | 4-bit keys and values | 3-bit keys, 2-bit values |
|---|---|---|---|---|
| Llama-3.1-8B | RULER-4K | 95.4 | 68.1 | 80.9 |
| Llama-3.1-8B | QuALITY | 71.3 | 57.1 | 62.3 |
| Qwen3-8B | RULER-4K | 95.8 | 54.8 | 71.2 |
| Qwen3-8B | QuALITY | 77.0 | 55.9 | 61.8 |
| Qwen3-14B | RULER-4K | 95.3 | 62.0 | 77.9 |
| Qwen3-14B | QuALITY | 80.9 | 62.1 | 66.9 |

- Fewer bits per token wins by 13 to 16 points on RULER-4K, and also at 16K tokens and on summarization. At 1/32 the gap is small.
- Most of the 4-bit cache's mistakes go away when it keeps twice as many tokens; few go away when it keeps 16 bits.
- KVzip+ keeps answer tokens only slightly more often than chance (1.03 to 1.16x), so the cache holds the answer only if it holds enough of the document.
- Keys need about 3 bits and values about 2. A smarter bit allocator (RDKV-style) does worse because it keeps fewer tokens.

## How it works

- The model reads the document once, and KVzip+ scores each cached token by how much it helps rebuild the document.
- The highest-scoring tokens are kept until the byte budget runs out, counting every scale and offset.
- Kept keys are stored at 3 bits per channel and values at 2 bits per token; the first four tokens stay at 16 bits.
- Questions are answered from this compressed cache.

## Limitations

- Re-reading the document still answers better: at 1/64 size the cache trails the full one by 15 to 25 points on RULER-4K.
- Memory savings are measured from stored bytes; there is no fast attention kernel.
- Tested on 8B and 14B models, with documents up to 16K tokens.
- The answer-blind result holds for question answering; on needle-in-a-haystack tasks KVzip+ keeps the needles at 2 to 3x chance.
