# Localist-LLM — Minimalist Reference Implementation

This repository provides a **minimal, conceptual realisation** of key ideas behind
*Localist LLMs*: transformer-based language models with a controllable **locality
dial** that interpolates between global and locality-biased attention.

The goal of this repository is to demonstrate the **high-level concept** of
tunable locality in attention mechanisms using a small amount of educational
PyTorch code.  
It is **not** a production system and **does not** implement the full Localist LLM
architecture.

> ⚠️ **Important — No Open-Source License**  
> This repository is released **without an open-source license**.  
> All rights are reserved by the author. Viewing the code is permitted; reuse,
> modification, or redistribution requires **written permission**.

> ⚠️ **Patent Notice**  
> This repository contains a deliberately simplified implementation.  
> Full Localist LLM methods, architectures, safety mechanisms, and training
> procedures are covered by **pending patent applications**.  
> Proprietary components are intentionally omitted.

---

## Features (Conceptual Only)

- A simple **locality dial** in the range `[0, 1]`, controlling the degree of
  locality in attention.
- A **generic distance-based locality bias**, used purely for illustration.
- A tiny transformer encoder block demonstrating how a locality dial can
  influence attention patterns.
- An example script printing attention matrices for different dial settings.

These components are sufficient to illustrate the *effect* of controllable
locality, but they do **not** represent the full invention.

---

## Installation

```bash
pip install torch
git clone https://github.com/JoDiederich/Localist-LLM.git
cd Localist-LLM
