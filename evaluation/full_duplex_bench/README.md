# Full-Duplex-Bench

Evaluates a trained DEF-FSM checkpoint on [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
v1.0 and v1.5.

## Setup

Clone the benchmark and get its data:

```bash
git clone https://github.com/DanielLin94144/Full-Duplex-Bench
cd Full-Duplex-Bench && git checkout 3e799c4
```

Set two entries in `configs/paths.yaml`:

```yaml
FDB_PATH: /path/to/Full-Duplex-Bench        # the checkout above
FDB_DATA: /path/to/Full-Duplex-Bench-Data   # contains v1.0/ and v1.5/
```

The v1.0 interruption subset is rated by an LLM judge. Upstream hardcodes
`gpt-4-turbo`; our numbers use `gpt-4o-2024-08-06`. Change that one line in the
checkout to match:

```
v1_v1.5/evaluation/eval_user_interruption.py:32
-    MODEL_NAME = "gpt-4-turbo"
+    MODEL_NAME = "gpt-4o-2024-08-06"
```

Export `OPENAI_API_KEY` for the judge.

Two environments are needed. Inference uses this repository's environment. The
metrics need the benchmark's own environment (NeMo `parakeet-tdt-0.6b-v2`,
`silero-vad`, `utmosv2`) — install it as the benchmark's README describes and
point `BENCH_PY` at its interpreter.

## Run

```bash
export BENCH_PY=/path/to/benchmark-env/bin/python
export FSM_DEVICES="cuda:0 cuda:1 cuda:1"        # LLM, ASR, TTS

bash evaluation/full_duplex_bench/run_v1.sh   <model>.gguf
bash evaluation/full_duplex_bench/run_v1_5.sh <model>.gguf
```

Add a second argument to run one subset only, e.g. `run_v1.sh <model>.gguf icc_backchannel`.
Both scripts are resumable: a sample whose output already exists is skipped.

## Results

Each run writes one tree per checkpoint under `outputs/fdb_v1/<model>/` and
`outputs/fdb_v1_5/<model>/` (override the parent with `FDB_RESULTS_ROOT`). Per
sample it contains `output.wav`, its transcript `output.json`, and the judge's
verdict where a task uses one.



To listen to a run, `make_merge.py` writes a stereo `merge.wav` per sample with the
user on the left and the agent on the right:

```bash
python evaluation/full_duplex_bench/make_merge.py --base-dir outputs/fdb_v1_5/<model>
python evaluation/full_duplex_bench/make_merge.py --base-dir outputs/fdb_v1/<model> --protocol v1
```
