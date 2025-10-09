# etl.py
import json
import urllib.request
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
 
 
# Par défaut : Boston 2018 Payroll (CKAN)
DEFAULT_URL = (
    "https://data.boston.gov/api/3/action/datastore_search"
    "?resource_id=31358fd1-849a-48e0-8285-e813f6efbdf1&limit=50000"
)
 
 
def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    """
    Trouve une colonne par nom insensible à la casse/espaces/underscores.
    Ex: ['TOTAL EARNINGS', 'total_earnings'] -> 'Total Earnings'
    """
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None
 
 
def extract_boston_salary(url: str = DEFAULT_URL) -> pd.DataFrame:
    """Extrait les données brutes depuis l'API Boston (CKAN) et retourne un DataFrame."""
    with urllib.request.urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    records = payload.get("result", {}).get("records", [])
    df = pd.DataFrame.from_records(records)
 
    # Normalisation de colonnes utiles si elles existent
    # (on ne les crée pas si absentes, on laisse la transform gérer intelligemment)
    return df
 
 
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et transforme les données :
    - Détecte la colonne total earnings (plusieurs alias possibles).
    - Convertit en float (supprime $ et ,).
    - Optionnel: nettoie la colonne département.
    - Drop NaN sur total_earnings.
    """
    df = df.copy()
 
    total_col = _find_col(df, ["TOTAL EARNINGS", "Total Earnings", "total_earnings", "totalearnings"])
    if total_col is None:
        raise KeyError("Colonne 'TOTAL EARNINGS' introuvable dans le dataset.")
 
    # Nettoyage monétaire -> float
    def _to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x)
        s = s.replace("$", "").replace(",", "").strip()
        # gère " - " ou vide
        if s == "" or s == "-":
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
 
    df["TOTAL_EARNINGS_CLEAN"] = df[total_col].apply(_to_float).astype("float64")
 
    # Trouver une colonne département plausible
    dept_col = _find_col(
        df,
        [
            "DEPARTMENT",
            "Department",
            "department",
            "dept_name",
            "department_name",
            "Dept",
        ],
    )
 
    if dept_col is None:
        # Si pas de colonne département, on en crée une générique
        df["DEPARTMENT_CLEAN"] = "UNKNOWN"
        dept_col_clean = "DEPARTMENT_CLEAN"
    else:
        df["DEPARTMENT_CLEAN"] = df[dept_col].astype(str).str.strip()
        dept_col_clean = "DEPARTMENT_CLEAN"
 
    # Supprimer lignes sans salaire
    df = df.dropna(subset=["TOTAL_EARNINGS_CLEAN"])
 
    # Optionnel: enlever salaires négatifs incohérents
    df = df[df["TOTAL_EARNINGS_CLEAN"] >= 0]
 
    # Conserver uniquement colonnes utiles + id si présent
    keep_cols = [dept_col_clean, "TOTAL_EARNINGS_CLEAN"]
    if _find_col(df, ["EMPLOYEE ID", "employee_id", "EmpID", "id"]):
        keep_cols.append(_find_col(df, ["EMPLOYEE ID", "employee_id", "EmpID", "id"]))
    df = df[keep_cols]
 
    # Renommer colonnes finales
    df = df.rename(
        columns={
            dept_col_clean: "department",
            "TOTAL_EARNINGS_CLEAN": "total_earnings",
        }
    ).reset_index(drop=True)
 
    return df
 
 
def load(df: pd.DataFrame, filename: str = "boston_salaries_clean.csv") -> None:
    """Enregistre les données nettoyées dans un fichier CSV (UTF-8, index off)."""
    df.to_csv(filename, index=False, encoding="utf-8")
 
 
def analyse(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Réalise des calculs statistiques sur les salaires par département.
    Retourne un dict prêt à sérialiser (JSON friendly).
    """
    if not {"department", "total_earnings"}.issubset(df.columns):
        raise KeyError("Le DataFrame doit contenir 'department' et 'total_earnings'.")
 
    grouped = (
        df.groupby("department", dropna=False)["total_earnings"]
        .agg(["count", "min", "max", "median", "mean", "std"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
 
    overall = {
        "count": int(df["total_earnings"].count()),
        "min": float(df["total_earnings"].min()),
        "max": float(df["total_earnings"].max()),
        "median": float(df["total_earnings"].median()),
        "mean": float(df["total_earnings"].mean()),
        "std": float(df["total_earnings"].std(ddof=1)) if df["total_earnings"].count() > 1 else 0.0,
    }
 
    return {
        "overall": overall,
        "by_department": grouped.to_dict(orient="records"),
        "top5_by_mean": grouped.head(5).to_dict(orient="records"),
    }
 
 
# Exécution manuelle rapide (utile localement)
if __name__ == "__main__":
    url = DEFAULT_URL
    raw = extract_boston_salary(url)
    clean = transform(raw)
    load(clean, "boston_salaries_clean.csv")
    stats = analyse(clean)
    # Impression légère
    print("Overall:", stats["overall"])
    print("Top 5 departments by mean:")
    for r in stats["top5_by_mean"]:
        print(f"- {r['department']}: mean={r['mean']:.2f} (n={r['count']})")