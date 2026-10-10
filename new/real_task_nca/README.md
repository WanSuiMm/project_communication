# Real-task NCA v0: retinal vessel refinement

**Prepared, not trained. Official DRIVE training data is required.**
The official Download page rejected the anonymous session with HTTP403.
This is a website access condition; it is not local filesystem permission.
No real-data training has been launched and no task result exists yet.

1. Read [PROTOCOL.md](PROTOCOL.md) for the frozen split, qualification and gates.
2. [cells.py](cells.py) implements the common conditional transition and AU
   representation; [checks.py](checks.py) checks their algebra and gradient.
3. [prepare.py](prepare.py), [data.py](data.py), and [coarse.py](coarse.py)
   generate frozen, out-of-fold coarse model predictions without test labels.
4. [run.py](run.py) first runs the K64 positive control and a separate T8
   reference. It launches the paired K8 arms only after qualification.

Dependencies: Python3.12, PyTorch2.5+, NumPy, Pillow, SciPy, scikit-image.
Run commands from the repository root. Obtain the official training set via
the [DRIVE site](https://drive.grand-challenge.org/) and extract its standard
`training/{images,1st_manual,mask}` tree. The code does not download data or
register accounts. Do not substitute fabricated masks or predictions for
the real-data scientific experiment.

Software check (no data, no optimizer updates):

```powershell
python -X utf8 -B new/real_task_nca/run.py --check --out analyses/NEW_RETINA_CHECK.json
```

Once the official data is present under `data/drive/DRIVE`, prepare coarse
predictions and run qualification in one protected, on-demand job:

```powershell
& ./tools/start_protected_job.ps1 -JobName NEW_RETINA_RUN -Script new/real_task_nca/prepare.py -ScriptArguments @('--drive','data/drive/DRIVE','--out','data/retina_prepared_v0','--then-run','runs/NEW_RETINA_RUN')
```

Use a fresh name for a new experiment. If explicitly recovering the same
interrupted experiment, append `--resume` to the script arguments and use the
same prepared/run directories. Source/config/data hashes must match.
The worker has no runtime cap or recurring monitor. Teacher checkpoints save
every100 updates, NCA checkpoints every25; a restart recovers from the most
recent completed atomic checkpoint. No program can preserve unsaved work
through power loss. Keep launch receipts and model/data artifacts local.

Optimizations preserve the intended gradients: static RGB features are
computed once per rollout; K64 uses exact activation checkpointing; terminal
K8 evaluates its prefix without retaining activations. All inference uses
the common materialized16-channel state, since AU and the original transition
are equal at fixed parameters. The T32 hidden reset is evaluation-only and
uses the same subsequent firing plan as its unperturbed reference.

The outer four images and repeated firing plans provide developmental
measurements, not an independent test-set result. No DRIVE test labels or
official test submission are part of this screen. Runtime and GPU memory
have not yet been measured at the real training shape.
