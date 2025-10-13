# main.py
import argparse
import json
import sys
from pathlib import Path
 
import pandas as pd
 
from etl import (
    DEFAULT_URL,
    extract_boston_salary,
    transform,
    load as load_to_csv,
    analyse,
)
 
 
def parse_args():
    p = argparse.ArgumentParser(
        description="ETL Boston Payroll 2018 – Extraction, transformation et analyse."
    )
    p.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="URL CKAN Boston (par défaut dataset Payroll 2018).",
    )
    p.add_argument(
        "-o",
        "--out",
        default="boston_salaries_clean.csv",
        help="Nom du fichier CSV de sortie (défaut: boston_salaries_clean.csv).",
    )
    p.add_argument(
        "--stats-json",
        default=None,
        help="Chemin d'un fichier pour sauvegarder les statistiques en JSON.",
    )
    p.add_argument(
        "--print-top",
        type=int,
        default=5,
        help="Afficher les N départements avec la moyenne la plus élevée (défaut: 5).",
    )
    p.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Affiche les N premières lignes après transformation (0 pour désactiver).",
    )
    p.add_argument(
        "--read-local",
        default=None,
        help="Chemin d'un CSV local déjà nettoyé (saute Extract/Transform et fait Analyse).",
    )
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    try:
        if args.read_local:
            csv_path = Path(args.read_local)
            if not csv_path.exists():
                print(f"[ERREUR] Fichier local introuvable: {csv_path}", file=sys.stderr)
                sys.exit(2)
            df_clean = pd.read_csv(csv_path)
            if not {"department", "total_earnings"}.issubset(df_clean.columns):
                print(
                    "[ERREUR] Le CSV local doit contenir les colonnes 'department' et 'total_earnings'.",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            print("[INFO] Extract depuis l'API…")
            df_raw = extract_boston_salary(args.url)
            print(f"[INFO] Lignes brutes: {len(df_raw)}")
 
            print("[INFO] Transform…")
            df_clean = transform(df_raw)
            print(f"[INFO] Lignes après nettoyage: {len(df_clean)}")
 
            if args.preview > 0:
                print("[INFO] Aperçu des données nettoyées:")
                print(df_clean.head(args.preview).to_string(index=False))
 
            print(f"[INFO] Sauvegarde CSV → {args.out}")
            load_to_csv(df_clean, args.out)
 
        print("[INFO] Analyse…")
        stats = analyse(df_clean)
        overall = stats["overall"]
        print("\n=== Résumé global ===")
        print(
            f"count={overall['count']} | "
            f"min={overall['min']:.2f} | median={overall['median']:.2f} | "
            f"mean={overall['mean']:.2f} | max={overall['max']:.2f}"
        )
 
        topN = stats["by_department"][: args.print_top]
        if topN:
            print(f"\n=== Top {args.print_top} départements par moyenne ===")
            for r in stats["top5_by_mean"][: args.print_top]:
                dept = r.get("department", "UNKNOWN")
                print(
                    f"- {dept}: mean={r['mean']:.2f}, median={r['median']:.2f}, "
                    f"min={r['min']:.2f}, max={r['max']:.2f}, n={r['count']}"
                )
 
        if args.stats_json:
            out_json = Path(args.stats_json)
            out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[INFO] Stats JSON sauvegardées → {out_json}")
 
        print("\n[OK] ETL terminé.")
        return 0
 
    except KeyboardInterrupt:
        print("\n[INFO] Interrompu par l'utilisateur.")
        return 130
    except Exception as e:
        print(f"[ERREUR] {e}", file=sys.stderr)
        return 1
 
 
if __name__ == "__main__":
    sys.exit(main())
