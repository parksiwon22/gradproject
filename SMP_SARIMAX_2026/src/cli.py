from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SARIMAX 72-hour land-SMP research pipeline")
    parser.add_argument("command", choices=["collect", "preprocess", "backtest", "visualize", "report"])
    args = parser.parse_args()
    if args.command == "collect":
        from src.collectors.collect import main as run
        run()
        from src.reports import write_collection_report
        write_collection_report()
    elif args.command == "preprocess":
        from src.preprocessing.build_dataset import build
        build()
        from src.reports import write_feature_report
        write_feature_report()
    elif args.command == "backtest":
        from src.evaluation.backtest import run_backtest
        try:
            run_backtest()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Backtest did not run: {type(exc).__name__}: {exc}")
            from src.reports import write_model_report, write_final_report
            write_model_report(); write_final_report()
            raise SystemExit(2)
        from src.reports import write_model_report, write_final_report
        write_model_report(); write_final_report()
    elif args.command == "visualize":
        from src.visualization.generate import generate
        generate()
    elif args.command == "report":
        from src.reports import write_all
        write_all()


if __name__ == "__main__":
    main()
