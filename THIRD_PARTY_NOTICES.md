# Third-party components

## Code included here

- **OpenUnlearning** (`https://github.com/locuslab/open-unlearning`, commit `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`). MIT License, copyright 2025 CMU Locus Lab.
  - Copies or adaptations of its objective, collator, tokenizer-configuration, evaluator and metric sources are in:
    - `cluster_experiments/org_m2_model_pilot_v2/` (files named `upstream_*.py`; notice in `UPSTREAM_LICENSE`)
    - `cluster_experiments/org_m2_metric_integration_v1/upstream/` (notice in `LICENSE`)
  - The notice bytes and source references are recorded in `provenance/source_selection/third_party_notices.json`.
  - This notice covers those upstream files only.

## Models and data used but not included

- Llama 2 7B and Llama 3 8B base models, under Meta's licences for those models.
- `Qwen/Qwen2.5-0.5B-Instruct`, under its published licence.
- The TOFU and MUSE-News unlearning benchmarks.
- Manufacturing Process Knowledge Graph-v1.0 (version 1, by Peihan Wen), CC BY 4.0, https://doi.org/10.17632/dfpv963n4w.1. The derived obsolete mappings and evaluation prompts in the supplementary package were constructed for this study.

Software packages used at run time (PyTorch, Transformers, PEFT, bitsandbytes and others) are not bundled. Their versions are recorded in the environment locks under `cluster_experiments/`.
