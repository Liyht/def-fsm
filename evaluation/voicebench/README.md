# VoiceBench

Evaluates a trained DEF-FSM checkpoint on [VoiceBench](https://github.com/MatthewCYM/VoiceBench),
to check what the tape format and the fine-tuning cost the model in spoken-language
ability.

## Setup

Stage 1 answers the transcripts through vLLM, so it needs the `rewrite` extra in this
repository's environment: `uv pip install -e ".[rewrite]"`.

Clone the benchmark and install it as its README describes — it needs its own
environment, mainly for `qa_metrics`:

```bash
git clone https://github.com/MatthewCYM/VoiceBench
cd VoiceBench && git checkout da0621a
```

Set one entry in `configs/paths.yaml`:

```yaml
VOICEBENCH_PATH: /path/to/VoiceBench
```

Nothing in the checkout needs editing. The audio is pulled from the
`hlt-lab/voicebench` dataset on the Hugging Face Hub the first time you run.

## Run

```bash
export VOICEBENCH_PY=/path/to/VoiceBench/.venv/bin/python
export TENSOR_PARALLEL_SIZE=4

bash evaluation/voicebench/run.sh <checkpoint-dir>          # the trained model, tape prompt
bash evaluation/voicebench/run.sh Qwen/Qwen3-4B chat        # the untrained baseline
```

The second argument picks the prompt: `fsm` (default) uses the tape prompt the model
was trained on, cueing it with `[S.SPEAK]` and cutting the response at the first
`[S.LISTEN...]`; `chat` uses VoiceBench's own prompt, for an untrained baseline.

`ASR_TYPE` selects the perception module (`faster` by default, `simul` for a
checkpoint trained against SimulStreaming, `turbo` for plain Whisper). Both stages
cache and resume; the ASR cache is keyed by perception module, not by checkpoint, so
it is reused across checkpoints.

## Results

`outputs/voicebench/<model>_<asr>_<mode>/` holds one jsonl per subset plus
`score.json` (override the parent with `VOICEBENCH_RESULTS_ROOT`):

| Subset | Metric |
| --- | --- |
| `mmsu` | multiple-choice accuracy |
| `openbook` | multiple-choice accuracy |
| `sdqa` | open-ended answer match (PEDANT) |
| `advbench` | refusal rate |

`overall` is their mean.

Two things worth knowing when comparing numbers:

- **SD-QA is the PEDANT score only.** VoiceBench's QA evaluator averages PEDANT with
  the majority vote of a separate GPT judging pass; that pass needs its own run of
  the benchmark's `api_judge.py` and an API key. We report PEDANT alone.
  `score.py` subclasses the evaluator to do this, which is why the checkout stays
  unmodified.
- **The two multiple-choice scores are not deterministic.** When a response cannot be
  parsed into a choice, VoiceBench guesses at random with no seed
  (`src/evaluator/mcq.py`), so `mmsu` and `openbook` move by roughly ±0.2 between
  runs on identical generations. Only the scoring varies; the generations do not.
