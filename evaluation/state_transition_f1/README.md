# State transition token F1

Scores turn-taking proficiency directly on the tape: at each teacher-forced decoding
step the model's output is treated as an independent binary classification for every
state transition token type, per-type F1 is computed within each evaluation set and
averaged across types, then averaged across Switchboard and Fisher.

Unlike the three benchmarks alongside it, this needs no external repository and no
GGUF export — it reads a training output directory directly.

## Run

```bash
bash evaluation/state_transition_f1/state_transition_f1.sh <training-output-dir>
```

The directory is the `training.output_dir` of a finished run, the one holding
`config.yaml` and the `best_loss_*` / `best_f1_*` checkpoints. That saved
`config.yaml` is reused, so evaluation sees exactly the tokenizer, prompts and data
paths the run trained with. Every split is restored to its full size regardless of
the training mixture ratios, so two runs trained on different mixtures are scored on
the same data.

`NUM_GPU` and `START_GPU` select the devices, as in `scripts/train.sh`. Extra
arguments are passed through to `training/eval.py`.

## Results

Written into the same training output directory:

| File | Contents |
| --- | --- |
| `best_loss_model_assistant_state_transition_scores.json` | `precision_`, `recall_` and `f1_` per token type, per split, alongside the losses |
| `best_loss_model_per_sample_scores.json` | the same, per sample |
| `vis/confusion_matrix_best_loss_model_assistant_<split>.pdf` | token-type confusion matrices, normalized against the ground-truth labels (also as `.png`) |
| `run_test.log` | the full console output |

The reported turn-taking figure is the macro average over token types, averaged
again over Switchboard and Fisher. The confusion matrices are what show the effect
of the SAC loss on the rare state-switch tokens.

Checkpoints that already have a score file are skipped, so the script can be rerun
over a directory as new checkpoints appear.
