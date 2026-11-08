# Migration validation

The extraction was checked on the prepared `raw-annual-v2` IN_C view, seed 42, CPU, 4 threads, Python 3.8.13 and PyTorch 2.2.2.

1. Old entry point and independent entry point each trained one epoch, with `max_queries_per_year=128`, in separate output directories. Every model state tensor was exactly equal. Validation organization IDs, active masks, top-40 rankings and all 24 per-query metric arrays were exactly equal. Both produced NDCG@20 **0.46617668293872655**.
2. The independent evaluator loaded the already completed `IN_C/ablation_full/s42/best.pt`. All stored query arrays and metrics exactly matched the original full-candidate validation, including NDCG@20 **0.6470225902730652**. Evaluation was launched from `/tmp` with an explicit prepared workspace, independently of the current directory.
3. Two additional country checkpoints were replayed: `CN_C/ablation_full/s42` (91,534 queries, NDCG@20 **0.6808094673982319**) and `US_F/ablation_full/s42` (150,537 queries, NDCG@20 **0.6869804321343346**). Organization IDs, active masks, full saved rankings and all 24 per-query metric arrays were exactly equal to their original records.
4. After consistent Black formatting and loading defaults from `configs/v2_default.json`, the one-epoch test was repeated; all model tensors and query arrays remained exactly equal. The combined research/package regression suite passed **26 tests**.

The evidence is stored in the research workspace under `reports/main_migration/parity.json`, together with the original source hashes, a migration-only source snapshot and separate replay outputs. These are migration checks, not additional research results.

No model initialization, loss, optimizer, candidate filtering, training query selection or evaluation formula was intentionally changed. `classical.PairFeatures` was extracted as a neutral main-model feature component; baseline fitting and baseline prediction functions were excluded. Raw-data helpers were replaced by the minimal path/hash/JSON functions required for prepared artifact loading.

The historical `scripts/train_main.py` now delegates to this directory using process replacement. Thus older supervisors already holding a plan in memory also start the new implementation for their next job. Jobs already executing keep their loaded implementation until completion. Future source changes belong in `main_experiment/src/patentflow_main/`.
