# Bidirectional interruption

Evaluates how a trained DEF-FSM checkpoint handles interruptions in both
directions, following the NFSM protocol.

- **MiU** (machine interrupts user) — the user's closing statement carries a
  deliberate commonsense error. The tape is replayed position by position and the
  response is taken at the first point where the model proactively emits
  `[S.SPEAK]` instead of `[C.LISTEN]`.
- **UiM** (user interrupts machine) — the user cuts in mid-response for one of four
  reasons: denial, affirmation, environmental noise, or topic shift. The response is
  taken at the first point where the model yields the floor with `[S.LISTEN...]`.

## Setup

Both the model and the judge run through vLLM, so this evaluation needs the `rewrite`
extra: `uv pip install -e ".[rewrite]"`.

The two test sets come from the data pipeline:

```bash
bash scripts/prepare_data.sh nfsm-generate nfsm-tape
```

which writes `${DATA_ROOT}/FSM/test_MiU_tape.json` and `test_UiM_tape.json`.
`nfsm-generate` needs `OPENAI_API_KEY`.

## Run

```bash
python evaluation/fsm_bench/fsm_bench.py \
    --model <checkpoint-dir> \
    --output-dir outputs/fsm_bench/<model> \
    --tensor-parallel-size 4
```


The judge defaults to `Qwen/Qwen3.5-27B`. Change it with `--judge-model`.

## Results

`outputs/fsm_bench/<model>/metrics.json`, with the two headline figures printed at
the end of the run:

| Metric | Meaning |
| --- | --- |
| `MiU_F1` | harmonic mean of `MiU_Precision` and `MiU_Recall` |
| `UiM_PRR_avg` | mean Proper Response Rate, with a `UiM_PRR_<reason>` breakdown |

Alongside `metrics.json`, each test set gets a `*_with_eval.json` holding the
per-dialogue responses and the judge's verdicts, for inspecting a run.
