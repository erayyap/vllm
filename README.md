# Fork: vLLM 0.33.0 + Gemma 4 voice-session patches

Branch `gemma4-voice` = tag v0.33.0 + patches to run Gemma 4 31B as 20 resident voice-call sessions (up to 64k
context each, FP8 KV) on one H100 80GB. All patches are Python/Triton/Gluon only; no CUDA extension changes.
The full write-up and every measurement are in [`gemma4_voice/README.md`](gemma4_voice/README.md).

| commit | what | tested |
|---|---|---|
| Gemma 4 voice sessions: engine patches | pause-SWA bug fix, stream-delta-resume, K-only global KV + attn512 kernel hooks, JIT lookahead, coalescing, pace, turn-first | yes, H100 |
| Scheduler: load-split (`VLLM_LOAD_SPLIT=1`) | context-load chunks never share a step with live calls' work | **no**: written after the GPU was returned |
| attn512 kernel in the package + voice server | `vllm/attn512/` (kernel, tuned config, tests), `gemma4_voice/` (server, load generator, README) | kernel yes; packaging no |

Measured with the measured setup (`VLLM_LOAD_SPLIT=0`): TTFT < 100 ms holds for resident sessions in steady
state at ~28k context (p50 63–67 ms, p95 83–99 ms through the WebSocket server). It is missed while calls join
(p95 110–128 ms at 28k, 210 ms at ~59k) and at ~59k steady state (p95 105 ms). Load-split is the candidate fix
for the join case.

## Install

Either install this fork with vLLM 0.33.0's prebuilt binaries:

```bash
git clone -b gemma4-voice <this repo> vllm-gemma4-voice && cd vllm-gemma4-voice
pip download vllm==0.33.0 --no-deps --only-binary=:all: -d /tmp/vllm-whl
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$(ls /tmp/vllm-whl/vllm-0.33.0-*.whl)" pip install -e .
pip install xxhash pyarrow fastapi uvicorn websockets
```

or patch a stock install. This is how the measurement box was set up:

```bash
pip install vllm==0.33.0 xxhash pyarrow fastapi uvicorn websockets
git diff 4ba787e HEAD -- vllm/ ':!vllm/attn512/__init__.py' > gemma4_voice_vllm0.33.0.patch   # or use the shipped one
cd "$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" \
  && patch -p1 < /path/to/gemma4_voice_vllm0.33.0.patch
```

## Run the fastest configuration

`gemma4_voice/server.py` sets every engine knob to the fastest measured value (env defaults, each overridable):
K-only global KV with the attn512 Gluon decode kernel (`VLLM_KONLY=2`), MTP with 3 draft tokens, FP8 KV, JIT
lookahead, coalescing, pace, turn-first, context-load chunk cap 256, and now load-split.

```bash
cd gemma4_voice
# default: FP8-block weights + MTP drafter (KL to BF16 0.015, top-1 97.2%); fits 20 x ~48k, the requirement is 20 x 30k
ATTN512_IMPL=gluon python3 server.py --warmup
# optional: Google QAT int4 weights, a bit faster (server p95 83 vs 93-99 ms) but lower fidelity (KL 0.09, top-1 92.4%)
ATTN512_IMPL=gluon python3 server.py --model google/gemma-4-31B-it-qat-w4a16-ct \
    --spec-model google/gemma-4-31B-it-qat-q4_0-unquantized-assistant --warmup
# reproduce the measured numbers (load-split off)
VLLM_LOAD_SPLIT=0 ATTN512_IMPL=gluon python3 server.py --warmup
```
`--model` / `--spec-model` take HF ids or local dirs; `--spec-model none` turns speculative decoding off.

Make sure the fast kernel is the one running. The decode kernel exists only in the Gluon implementation, and
the default `ATTN512_IMPL=auto` silently falls back to the slower Triton kernel if Gluon does not import or
fails its self-test. Run the kernel tests once on the target GPU (`python3 vllm/attn512/test_konly.py`), then
start the server with `ATTN512_IMPL=gluon`: that skips the self-test and fails loudly instead of falling back.

Kernel knobs (attn512 decode kernel; microbenchmark only, B=20, 30k keys, q=4 as with MTP 3):

| env | default | best in the sweep | kernel time |
|---|---|---|---|
| `ATTN512_DEC_RING` | 8 | 16 | 1.062 → 1.028 ms together with 8 splits |
| `ATTN512_DEC_LAG` | 1 | 1 (q=4), 2 with `ATTN512_DEC_PAIR=1` (q=1) | q=1: 0.978 → 0.903 ms |
| `ATTN512_DEC_OVH` | 4 | split heuristic overhead | under CUDA graphs vLLM sizes splits for `VLLM_KONLY_GRAPH_KV`=65536 keys |

The global layers are ~9% of a decode step, so these give well under 1% end to end; they were not run through
vLLM. `ATTN512_DEC_RING=16` is the only one worth trying.

---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
vllm download-kernels  # precompiled kernels for faster startup
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
