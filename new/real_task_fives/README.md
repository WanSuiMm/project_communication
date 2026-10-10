# Real-task AU-NCA on FIVES

This directory contains the implementation and frozen protocol for a
single-block FIVES vessel-segmentation screen at **512 × 512 development
resolution**. FIVES is distributed at 2048 × 2048; this experiment does not
measure native-resolution performance. Read [PROTOCOL.md](PROTOCOL.md) first.
No result is claimed until the dispatched run and its saved evidence are
reviewed.

For the task and data source, see the official [Figshare record](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169)
and [FIVES paper](https://doi.org/10.1038/s41597-022-01564-3). For method
history, see the older [DRIVE protocol](../real_task_nca/PROTOCOL.md), the
[AU-NCA protocol](../au_nca/PROTOCOL.md), and its
[completed results](../../evidence/au_nca_20261009_01/RESULTS.md). Those
historical materials do not qualify the FIVES experiment.

Dependencies: Python 3.12 and [requirements.txt](requirements.txt). Official
RAR extraction also requires the operating system's `libarchive` shared
library, or a compatible 7-Zip executable. Install the appropriate CUDA
PyTorch build separately; the executed runtime is recorded in the results.

The official archive is Figshare item `19688169`, file `34969398`,
1,764,974,308 bytes, MD5 `789c80dd5376a82063e27fa49192bac9`, license CC BY 4.0.
The 600 official training images and masks define development. The 200 test
labels are not opened or decoded. A deterministic, diagnosis-balanced
120-image validation set is carved from the official training split; the
other 480 images supply OOF coarse predictions and NCA training. The protocol
makes no patient-level separation claim because no verified
image-to-patient map is available.

The main entry points are:

| File | Responsibility |
|---|---|
| [acquire.py](acquire.py) | Retrieve the official Figshare archive, verify size and MD5, and safely extract it. |
| [data.py](data.py), [coarse.py](coarse.py), [prepare.py](prepare.py) | Discover train-only pairs, apply preprocessing/splits, train OOF and full-training teachers, and freeze coarse predictions with provenance. |
| [cells.py](cells.py), [run.py](run.py) | Implement the shared conditional cell and train/evaluate the K64, T8, K8, and AU-K8 arms under the frozen gates. |
| [pipeline.py](pipeline.py) | Orchestrate acquisition, software checks, disposable GPU shape checks, teacher preparation, and the NCA qualification run. |
| [server_job.py](server_job.py) | Probe the server, bootstrap experiment-local dependencies, launch the pipeline as a detached job, and read its saved status. |

From the project root on the authorized server, replace `PROJECT_ROOT` with
the writable project checkout containing `new/real_task_fives/`. Probe and
bootstrap that checkout, then dispatch the pipeline to GPU 0:

```powershell
python -X utf8 -B new/real_task_fives/server_job.py --action probe --root "PROJECT_ROOT" --gpu 0
python -X utf8 -B new/real_task_fives/server_job.py --action bootstrap --root "PROJECT_ROOT"
python -X utf8 -B new/real_task_fives/server_job.py --action launch --root "PROJECT_ROOT" --gpu 0
```

The pipeline first acquires and verifies official FIVES, performs the CPU
software checks, then runs disposable GPU shape checks before any scientific
teacher or NCA update. The GPU smoke uses one real RGB/mask example and a
constant coarse input of `0.1` only to exercise the training and evaluation
shapes; it is not a scientific arm or a result. Acquisition extracts the
official archive, while dataset discovery and image decoding remain limited
to the official training tree; test masks are not opened or decoded. The main
pipeline has no runtime cap and creates a durable launch receipt. Use
`server_job.py --action status --root "PROJECT_ROOT" --gpu 0` for a one-shot
snapshot; it does not start a recurring monitor. The direct orchestrator
command is:

```powershell
python -X utf8 -u -B new/real_task_fives/pipeline.py --root "PROJECT_ROOT" --device cuda:0
```

Add `--resume` only when recovering the same bound run after checking its
source, config, and data hashes. The detached server helper launches this
pipeline for the normal deployment.

If the server cannot reach Figshare, retrieve the same official archive on
another machine and transfer it to
`data/downloads/FIVES_figshare_19688169_34969398.rar` under `PROJECT_ROOT`.
Acquisition reuses that cache only after checking the pinned size and MD5;
it then needs no metadata request. Recover a stopped worker with
`server_job.py --action recover --root "PROJECT_ROOT" --gpu 0`.
Recovery refuses a duplicate live worker and preserves the prior dispatch
receipt and status before resuming the pipeline.

The experiment compares Original K64, Original K8, and AU-K8. K64 must qualify
before the short-credit arms run. The independent T8 reference is only scored
at T8. AU-K8 preserves direct historical credit to the output projection
while perception and hidden-feature gradients remain K8-truncated. The
thresholds in [PROTOCOL.md](PROTOCOL.md) are design choices for one
developmental block. Do not train on CHASEDB1 or make cross-dataset
generalization claims under this screen.
