"""Small command surface for the first implementation milestone."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .serialization import to_jsonable
from .specification import verify_specification


def _write_new(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(to_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def spec_lag(spec: dict, project: Path) -> int:
    protocol = json.loads((project / "config" / "protocol.json").read_text())
    return int(protocol["calendar"]["assumed_provider_lag_days"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fpp", description="International-to-NCAA reference implementation")
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="Project root with paper and protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-spec", help="Verify the frozen paper and decision pins")
    audit = sub.add_parser("audit-db", help="Inspect SQLite schema read-only; does not read outcome values")
    audit.add_argument("--db", type=Path, required=True)
    audit.add_argument("--out", type=Path)
    smoke = sub.add_parser("smoke", help="Sample and score synthetic seasons; no model training")
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=20260901)
    smoke.add_argument("--draws", type=int, default=200)
    smoke.add_argument("--score-first", type=int, default=3)
    smoke.add_argument("--schedule", type=int, default=32)
    grad = sub.add_parser("gradcheck", help="Validate the differentiable likelihood against the reference (needs torch)")
    grad.add_argument("--out", type=Path, required=True)
    grad.add_argument("--seed", type=int, default=20260901)
    grad.add_argument("--batch-size", type=int, default=256)
    grad.add_argument("--schedule", type=int, default=32)
    grad.add_argument("--nodes", type=int, default=32)
    rec = sub.add_parser("recovery", help="Intercept-only parameter recovery on simulated observations (needs torch)")
    rec.add_argument("--out", type=Path, required=True)
    rec.add_argument("--seed", type=int, default=20260901)
    rec.add_argument("--n-train", type=int, default=400)
    rec.add_argument("--n-test", type=int, default=300)
    rec.add_argument("--schedule", type=int, default=32)
    rec.add_argument("--nodes", type=int, default=32)
    rec.add_argument("--max-iter", type=int, default=200)
    build = sub.add_parser("build-tables", help="Assemble games/units/periods from the upstream stores (read-only) into parquet")
    build.add_argument("--out", type=Path, required=True, help="New directory for the tables and manifest")
    build.add_argument("--first-season", type=int, default=2003)
    build.add_argument("--last-season", type=int, default=2026)
    build.add_argument("--stores", type=Path, help="JSON mapping of store names to paths; defaults to the Dropbox locations")
    counts = sub.add_parser("fold-counts", help="Per-fold unit, window, and pool counts from assembled tables")
    counts.add_argument("--tables", type=Path, required=True)
    counts.add_argument("--out", type=Path, required=True)
    counts.add_argument("--first-development", type=int, default=2020)
    train = sub.add_parser("train", help="Train and score one arm on one rolling fold (needs torch, pandas)")
    train.add_argument("--tables", type=Path, required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--arm", choices=["A", "AR", "B", "C", "C_shuf"], default="A",
                       help="A forward; AR = A + reconstruction (no pretraining); B pretrained; C = B + reconstruction; C_shuf control")
    train.add_argument("--lambda-plus", type=float, default=1.0, help="reconstruction weight for C, C_shuf, AR (grid {0.1, 0.3, 1})")
    train.add_argument("--pretrain-epochs", type=int, default=3)
    train.add_argument("--pretrain-examples", type=int, default=60000)
    train.add_argument("--model", choices=["tower", "aggregate"], default="tower", help="reference tower or the aggregated-feature baseline")
    train.add_argument("--seed", type=int, default=20260901)
    train.add_argument("--out", type=Path, required=True, help="New run directory")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--nodes-train", type=int, default=24)
    train.add_argument("--nodes-eval", type=int, default=32)
    train.add_argument("--max-train-units", type=int)
    train.add_argument("--baseline-rows", type=int, default=5000)
    train.add_argument("--exclude-assumed-zeros", action="store_true",
                       help="sensitivity arm: drop provider-consistent zero-appearance labels from training (D-028)")
    train.add_argument("--recon-horizon-years", default="4",
                       help="reconstruction horizon after the NCAA season end in years (registered default 4, D-021); "
                            "'none' admits every post-NCAA game released by the training cutoff (D-038)")
    train.add_argument("--recon-k", default="1,2,3",
                       help="nested window sizes in post-NCAA seasons, comma-separated; 'all' adds one window over every admissible season")
    train.add_argument("--label-scope", choices=["all", "first_year"], default="all",
                       help="training labels: every NCAA season (registered, D-021/D-023) or first D1 seasons only, forward and reconstruction (D-060)")
    train.add_argument("--deploy", action="store_true",
                       help="deployment (D-051): fold = test season + 1, train through the test season - 1, early-stop on the test season, no evaluation")
    train.add_argument("--test-registration", type=Path,
                       help="one-shot test only: path to the written registration; its SHA-256 is recorded in the run manifest")
    train.add_argument("--resume", action="store_true", help="continue an unfinished run from its last epoch checkpoint (D-040)")
    train.add_argument("--recon-holdout-share", type=float, default=0.0,
                       help="D-042 translation test: share of reconstruction players held out entirely and scored on their windows after training")
    train.add_argument("--exclude-post-first-season-logs", action="store_true",
                       help="strict control (D-041): drop every non-NCAA game dated on/after the player's first NCAA season from "
                            "all input channels, the pool and reconstruction")
    report = sub.add_parser("report-run", help="Fold report for a finished run: paired comparison, calibration, reference rescoring")
    report.add_argument("--run", type=Path, required=True, help="Run directory with run.json, per_unit.csv, tower_state.pt")
    report.add_argument("--compare", type=Path, help="Second run directory on the same fold (e.g. the aggregated baseline)")
    report.add_argument("--tables", type=Path, required=True)
    report.add_argument("--draws", type=int, default=200)
    report.add_argument("--rescore", type=int, default=25, help="units rescored with the SciPy reference: random + largest disagreements")
    args = parser.parse_args(argv)
    try:
        spec = verify_specification(args.project)
        if args.command == "verify-spec":
            result = spec
        elif args.command == "audit-db":
            from .data.source import audit_database
            result = {"specification": spec, "audit": audit_database(args.db)}
            if args.out:
                _write_new(args.out, result)
        elif args.command == "build-tables":
            from .data.assemble import build_tables
            from .data.stores import Stores
            from .data.tables import write_tables
            paths = json.loads(args.stores.read_text()) if args.stores else None
            with (Stores(paths) if paths else Stores()) as stores:
                config = {"seasons": [args.first_season, args.last_season], "lag_days": spec_lag(spec, args.project),
                          "specification": spec}
                tables = build_tables(stores, seasons=(args.first_season, args.last_season), lag_days=config["lag_days"])
                payload = write_tables(tables, args.out, config=config, fingerprints=stores.fingerprints())
            result = {"saved": str(args.out), "tables": {k: v["rows"] for k, v in payload["tables"].items()},
                      "audit": payload["audit"]}
        elif args.command == "fold-counts":
            from .data.folds import fold_counts, fold_layout
            from .data.tables import read_tables
            protocol = json.loads((args.project / "config" / "protocol.json").read_text())
            tables = read_tables(args.tables)
            folds = fold_layout(protocol, first_development=args.first_development)
            test = tables["units"]
            test_players = frozenset(test.loc[(test["season"] == protocol["experiment"]["test_season"]) & test["intl_entrant"], "player_id"])
            frame = fold_counts(tables, folds, test_cohort_players=test_players)
            result = {"kind": "fold_counts", "specification": spec, "tables_manifest_sha256": tables["manifest"].get("tables"),
                      "folds": frame.to_dict(orient="records")}
            _write_new(args.out, result)
            result = {"saved": str(args.out), "folds": frame.to_dict(orient="records")}
        elif args.command == "train":
            from .experiments.train import run_arm
            horizon = None if str(args.recon_horizon_years).lower() == "none" else float(args.recon_horizon_years)
            if horizon is not None:
                if not horizon.is_integer():
                    raise SystemExit("--recon-horizon-years must be a whole number of years or 'none'")
                horizon = int(horizon)             # calendar arithmetic needs an integer year count
            recon_k = tuple(None if part.strip().lower() == "all" else int(part) for part in str(args.recon_k).split(",") if part.strip())
            payload = run_arm(tables=args.tables, fold_season=args.fold, arm=args.arm, seed=args.seed, out=args.out,
                              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                              nodes_train=args.nodes_train, nodes_eval=args.nodes_eval, max_train_units=args.max_train_units,
                              baseline_rows=args.baseline_rows, model_kind=args.model,
                              exclude_assumed_zeros=args.exclude_assumed_zeros, lambda_plus=args.lambda_plus,
                              pretrain_epochs=args.pretrain_epochs, pretrain_examples=args.pretrain_examples,
                              recon_max_years=horizon, recon_k=recon_k, resume=args.resume,
                              exclude_post_first_season_logs=args.exclude_post_first_season_logs,
                              recon_holdout_share=args.recon_holdout_share,
                              allow_test=args.test_registration is not None and not args.deploy, test_registration=args.test_registration,
                              deploy=args.deploy, label_scope=args.label_scope)
            result = {"saved": str(args.out), "arm": args.arm, "model": args.model, "fold": args.fold, "seed": args.seed,
                      "counts": payload["counts"], "training": {"epochs_run": payload["training"]["epochs_run"],
                                                                 "best_stop_nll": payload["training"]["best_stop_nll"],
                                                                 "baseline_stop_nll": payload["baseline"]["stop_nll"]},
                      "evaluation": payload["evaluation"], "timing": payload["timing"]}
        elif args.command == "report-run":
            from .experiments.report import report_run
            payload = report_run(run=args.run, tables=args.tables, compare=args.compare, draws=args.draws, n_rescore=args.rescore)
            result = {"saved": str(args.run / "report.json"), "summary": payload["summary"]}
        elif args.command == "recovery":
            from .experiments.recovery import run_recovery
            result = run_recovery(args.seed, args.n_train, args.n_test, args.schedule, args.nodes,
                                  max_iter=args.max_iter)
            result["specification"] = spec
            _write_new(args.out, result)
            result = {"saved": str(args.out), "kind": result["kind"],
                      "configurations": {k: {"fit": v["fit"], "n_weak": v["n_weak"],
                                             "max_abs_z_identified": v["max_abs_z_identified"],
                                             "excess_nll": v["predictive"]["excess_nll_fitted_minus_true"]}
                                         for k, v in result["configurations"].items()}}
        elif args.command == "gradcheck":
            from .experiments.gradcheck import run_gradcheck
            result = run_gradcheck(args.seed, args.batch_size, args.schedule, args.nodes)
            result["specification"] = spec
            _write_new(args.out, result)
            result = {"saved": str(args.out), "kind": result["kind"],
                      "value_agreement_by_nodes": result["value_agreement_by_nodes"],
                      "gradient_max_rel_err": result["gradient_check"]["max_rel_err"],
                      "timing": result["timing"]}
        else:
            from .experiments.synthetic import run_smoke
            result = run_smoke(args.seed, args.draws, args.score_first, args.schedule)
            result["specification"] = spec
            _write_new(args.out, result)
            result = {"saved": str(args.out), "kind": result["kind"],
                      "draws": args.draws, "scored_draws": args.score_first,
                      "summary": result["summary"]}
        print(json.dumps(to_jsonable(result), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, ArithmeticError, sqlite3.Error) as exc:
        print(f"fpp: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
