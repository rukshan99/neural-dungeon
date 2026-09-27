# The Surface

> *You climb the last stair. The light is very bright. Somebody hands you a coffee.*

If `dungeon status` says **Dungeon Master**, you have written, from scratch and with tests watching: broadcasting rules, gradient descent, an autograd engine, a training loop with regularization, PyTorch pipelines that reproduce bitwise and train in mixed precision across workers, two tokenizers, multi-head attention with rotary positions and grouped queries, a GPT you trained on the dungeon's own chronicles, LoRA, instruction tuning and DPO, a retrieval pipeline that refuses to hallucinate, a tool-using agent that an Imp could not hijack (and that redacts what it should not repeat), an evaluation harness that a Gorgon could not game, an inference server with a KV cache, and a watchtower that notices the world drifting before the accuracy does. That is a real skill set. It is roughly the syllabus of an applied AI engineering role, and you did the parts most people skip.

## What you now know that you might not have noticed

- **Shapes are the type system of ML.** Most bugs you will meet in the wild are Floor 0 bugs wearing a bigger model.
- **Everything is a loss going down.** Optimizers, schedules, regularization, fine-tuning, even RAG thresholds are all ways of steering the same descent.
- **Libraries are earned.** You wrote backprop before calling `.backward()` and attention before calling `nn.MultiheadAttention`. When they misbehave now, you will know where to look.
- **Determinism is a feature you build, not a default you get.**
- **The model is the smallest part of a production system.** Retries, budgets, injection defenses, evals and latency percentiles are where systems succeed or fail.
- **Metrics lie when you stare at them.** Measure two things, hold something out, and check whether the judge has a favourite corner.

## Where to go next

Foundations
- *Deep Learning* (Goodfellow, Bengio, Courville): the textbook behind Floors 1 to 3.
- Andrej Karpathy's *micrograd* and *nanoGPT* repositories: the spiritual ancestors of Floors 2 and 7.
- *Dive into Deep Learning* (d2l.ai): free, code-first, broad.

Transformers and training
- Vaswani et al., *Attention Is All You Need* (2017).
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019).
- Dao et al., *FlashAttention* (2022): what the Tiled Gaze secret room was approximating.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).

Systems
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020).
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023).
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023).
- Chip Huyen, *Designing Machine Learning Systems* and *AI Engineering*.

Safety and evaluation
- Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* (2023).
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023): position bias, verbosity bias, and how to measure a judge.

## Keep digging

- **Replay a floor with constraints.** Clear Floor 7 with half the parameters. Clear Floor 12's boss with a 5x speed bar instead of 3x.
- **Point your agent at a real model.** `dungeon/artifacts/providers.py` implements the same interface the mocks do.
- **Add a floor.** The dungeon has room for convolutions, reinforcement learning, diffusion, distributed training, and whatever you are curious about. [CONTRIBUTING.md](CONTRIBUTING.md) is the blueprint.
- **Teach it.** The fastest way to find out whether you understand a boss's weakness is to explain it to someone standing at the Threshold.

Thank you for playing. Mind the Basilisk on your way out.
