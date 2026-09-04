# DEF-FSM

Code for **"Decoupling Turn-Taking from Semantics: A Decoupled Data Approach for Finite-State-Machine-Based Full-Duplex Dialogue"** (EMNLP 2026).

Paper: [arXiv:2609.03321](https://arxiv.org/abs/2609.03321)

(D: **D**ecoupled, E: **E**vent-guided transformation, F: **F**ull-duplex)

The Neural Finite State Machine (NFSM) framework serializes turn-taking control and response generation onto a single causal "tape" under plain next-token prediction. Its weakness is the data: LLM-generated transcripts cannot reproduce the fine-grained acoustic timing of real conversation. 

This repository implements the decoupled alternative — turn-taking is learned from real Human-Human (HH) spoken dialogue, semantic behaviour from configurable Human-Agent (HA) text — together with the rule-based event-guided transformation that turns HH audio into FSM tapes, the Source-Aware Calibrated (SAC) loss, and the real-time FSM inference system.

## What is in this release

- `data_transformation/` — the HH event-guided transformation pipeline, the HA rewriting pipeline, and the reproduced NFSM synthetic data generation pipeline.
- `training/` — SAC-loss fine-tuning.
- `def_fsm/` — the real-time inference system of FSM: perception ([Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) or [SimulStreaming](https://github.com/ufal/SimulStreaming)), cognition ([llama.cpp](https://github.com/ggml-org/llama.cpp)), and motor ([Kokoro](https://github.com/hexgrad/kokoro)).
- `configs/` — one config of experiment reported in the paper.
- `evaluation/` — interruption handling, plus the Full-Duplex-Bench and VoiceBench integrations.

## Install

```bash
git clone Liyht/def-fsm && cd def-fsm
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                       # data transformation + training
uv pip install -e ".[asr,inference]"      # + the real-time system
```

Extras: `asr` (perception and the ASR stage of the data pipeline), `inference` (llama.cpp and
Kokoro), `rewrite` (vLLM and the OpenAI client, for the LLM stages of the data pipeline and the
vLLM-backed evaluations).

### The word-level perception module (optional)

The paper's main configuration uses Faster-Whisper and needs nothing extra. The word-level
SimulStreaming variant is an out-of-tree fork:

```bash
git clone https://github.com/ufal/SimulStreaming && cd SimulStreaming
git checkout 240be1f
git apply /path/to/def-fsm/third_party/patches/simulstreaming.patch
```

Then set `SIMULSTREAMING_PATH` in `configs/paths.yaml`. Both perception modules expect a Whisper
`large-v3-turbo` checkpoint at `WHISPER_MODEL_PATH`.

## Data

| Corpus | Role | Access |
| --- | --- | --- |
| Switchboard | HH | LDC licence required |
| Fisher | HH | LDC licence required |
| [ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) | HA | public on the Hugging Face Hub |

**The Switchboard and Fisher transcripts, and every tape derived from them, cannot be
redistributed.** They must be obtained from the LDC and transformed locally.

### Building the corpora

```bash
bash scripts/prepare_data.sh              # every stage, in order
bash scripts/prepare_data.sh hh-tape      # or one stage at a time
```

| Stage | What it does |
| --- | --- |
| `hh-asr` | transcribes the user channel with the perception module the FSM deploys |
| `hh-refine` | splits the long Fisher reference utterances into clauses, interpolating a timestamp for each |
| `hh-tape` | restores punctuation on the agent channel, classifies the timeline into the 7 turn-taking events, then serializes each segment through 14 mapping rules |
| `ha-clean` | rule-based filtering of ShareGPT |
| `ha-rewrite` | LLM rewriting into a spoken register (Qwen3-32B, loaded with vLLM in-process) |
| `ha-tape` | serializes the rewritten dialogues into schema-compliant tapes |
| `nfsm-generate` | reproduces the NFSM synthetic training and test (used in `evaluation/fsm_bench`) data via the OpenAI API |
| `nfsm-tape` | serializes NFSM synthetic data to tape|
| `statistics` | token-type ratios and split sizes |


## Training

```bash
bash scripts/train.sh configs/base.yaml                                   # the main model
bash scripts/train.sh configs/base.yaml configs/experiments/wo_sac.yaml   # an ablation
bash scripts/train_all.sh                                          # everything below
```

An experiment file only carries the keys that differ from `configs/base.yaml`. GPU count and the starting device come from the environment: `NUM_GPU=4 START_GPU=0 bash scripts/train.sh ...`. 

Two points of notation, because the config keys do not name them the way the paper does:

- **α** is expressed through `dataset_loss_weights`, which is `[response_weight,   transition_weight]` per source. α = 0.6 means `[0.4, 0.6]` on the HH sources and `[0.6, 0.4]` on   the HA source.
- **"w/o SAC"** means `loss_function: loss_modification` with `normalize_by_count: false` and   uniform `[0.5, 0.5]` weights, i.e. plain cross-entropy. The SAC loss is   `loss_function: logit_adjustment` and   `prob_denominator: transition`, so the prior is computed over the restricted transition   vocabulary.

## Pretrained models

Both fine-tuned checkpoints live in one repository on the Hugging Face Hub:
[Liyht/def-fsm-qwen3-4b](https://huggingface.co/Liyht/def-fsm-qwen3-4b).

| Path in that repository | Checkpoint | Use it for |
| --- | --- | --- |
| root | `improved-data` | Trained with improved data pipeline. |
| `paper/` | `paper` | Reproducing the numbers reported in the paper. |

```bash
hf download Liyht/def-fsm-qwen3-4b --exclude "paper/*" --local-dir def-fsm-qwen3-4b
hf download Liyht/def-fsm-qwen3-4b --include "paper/*"  --local-dir def-fsm-qwen3-4b
```

The two differ only in training data and one hyperparameter. `improved-data` was trained after
the paper, on tapes rebuilt with two fixes: sub-50ms user/agent overlaps are now treated as ASR
timestamp jitter rather than real overlapping speech, and the logit adjustment was softened to
`tau = 0.75` so that it reduces over-suppressing `[C.SPEAK]`. Every score reported in the paper
comes from `paper`, which was trained and evaluated on the earlier tapes.

Throughout this README, `<checkpoint-dir>` is either a training output directory or a downloaded
copy of these weights.

## Inference

Export a checkpoint to GGUF, then talk to it:

```bash
hf download Liyht/def-fsm-qwen3-4b --exclude "paper/*" --local-dir def-fsm-qwen3-4b

bash scripts/convert_to_gguf.sh def-fsm-qwen3-4b     # writes def-fsm-qwen3-4b_gguf_f16/
bash scripts/run_demo.sh def-fsm-qwen3-4b_gguf_f16/<model>.gguf
```

A training output directory works the same way; point the converter at its
`best_loss_..._step_...` subdirectory instead.

`scripts/run_demo.sh` passes its extra arguments to [demo/app_local.py](demo/app_local.py), so
`--asr-worker simul` and `--devices cuda:0 cuda:1 cuda:2` both work.

## Evaluation

Each directory's README covers the setup.

1. [evaluation/state_transition_f1/](evaluation/state_transition_f1/) is the evaluation of the
state transition tokens on the validation and test splits about turn-taking proficiency:

    ```bash
    bash evaluation/state_transition_f1/state_transition_f1.sh <training-output-dir>
    ```

    It reads a training output directory directly, reusing the `config.yaml` the run saved, so
    evaluation sees exactly the tokenizer, prompts and data paths it trained with. It reports the
    per-type F1 and writes the confusion matrices of Figure 3.

2. [evaluation/full_duplex_bench/](evaluation/full_duplex_bench/) is the evaluation of
[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1.0 and v1.5 about turn-taking capability:

    ```bash
    bash evaluation/full_duplex_bench/run_v1.sh   <model>.gguf
    bash evaluation/full_duplex_bench/run_v1_5.sh <model>.gguf
    ```

3. [evaluation/voicebench/](evaluation/voicebench/) is the evaluation of [VoiceBench](https://github.com/matthewcym/voicebench) about semantics capability:

    ```bash
    bash evaluation/voicebench/run.sh <checkpoint-dir>       # the trained model
    bash evaluation/voicebench/run.sh Qwen/Qwen3-4B chat     # the untrained baseline
    ```

4. [evaluation/fsm_bench/](evaluation/fsm_bench/) reproduces the evaluation of NFSM about interruption handling in both directions, machine-interrupts-user and user-interrupts-machine, on the synthetic test sets built by `scripts/prepare_data.sh`:

    ```bash
    python evaluation/fsm_bench/fsm_bench.py --model <checkpoint-dir> --output-dir outputs/fsm_bench/<model>
    ```

## Citation
```
@inproceedings{li-chu-2026-decoupling,
    title     = {Decoupling Turn-Taking from Semantics: A Decoupled Data Approach for
                 Finite-State-Machine-Based Full-Duplex Dialogue},
    author    = {Li, Yihang and Chu, Chenhui},
    booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural
                 Language Processing},
    year      = {2026},
}
```

## License

[Apache-2.0](LICENSE)
