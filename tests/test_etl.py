import json
import types
import pandas as pd
import numpy as np
import urllib.request
 
from etl import extract_boston_salary, transform, analyse
 
 
class _FakeResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")
 
    def __enter__(self):
        return self
 
    def __exit__(self, *exc):
        return False
 
    def read(self):
        return self._data
 
 
def _ckan_payload(records):
    """Construit un payload CKAN minimal valide."""
    return {
        "help": "mock",
        "success": True,
        "result": {"records": records},
    }
 
# ---Début des tests---
def test_extract_returns_dataframe(monkeypatch):
    # Prépare des enregistrements factices façon CKAN
    records = [
        {"Employee": "A", "Total Earnings": "$1,200.50", "Department": "IT"},
        {"Employee": "B", "Total Earnings": "$900.00", "Department": "HR"},
    ]
    payload = _ckan_payload(records)

    def _fake_urlopen(url):
        return _FakeResponse(payload)
 
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
 
    df = extract_boston_salary("https://fake-url/ignored")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Total Earnings" in df.columns
    assert "Department" in df.columns
 
 
def test_transform_converts_total_earnings():
    #---Gestion des cas variés--- 
    raw = pd.DataFrame(
        {
            "Total Earnings": ["$1,234.50", " - ", "", None, "$0.00", "$-10.00", "$2,000"],
            "Department": ["IT", "IT", "Finance", "Finance", "IT", "HR", "Finance"],
        }
    )
 
    clean = transform(raw)
 
    assert {"department", "total_earnings"}.issubset(clean.columns)
 
    # Tous les salaires doivent être >= 0 
    assert (clean["total_earnings"] >= 0).all()
    assert clean["total_earnings"].dtype == "float64"
 
    assert len(clean) == 3

    vals = sorted(clean["total_earnings"].tolist())
    assert vals == [0.0, 1234.50, 2000.0]
 

    assert set(clean["department"]) == {"IT", "Finance"}
 
 
def test_analyse_returns_dict():
    df_clean = pd.DataFrame(
        {
            "department": ["IT", "IT", "HR", "Finance", "Finance", "Finance"],
            "total_earnings": [1000.0, 3000.0, 2000.0, 500.0, 1500.0, 2500.0],
        }
    )
 
    stats = analyse(df_clean)
    assert isinstance(stats, dict)
    assert "overall" in stats and "by_department" in stats

    overall = stats["overall"]
    assert overall["count"] == 6
    assert overall["min"] == 500.0
    assert overall["max"] == 3000.0

    assert overall["median"] == 1750.0

    assert overall["mean"] == 1750.0
 
    # contient des enregistrements avec les clés attendues
    required_keys = {"department", "count", "min", "max", "median", "mean", "std"}
    assert all(required_keys.issubset(row.keys()) for row in stats["by_department"])
    row_it = next(r for r in stats["by_department"] if r["department"] == "IT")
    assert row_it["count"] == 2
    assert row_it["mean"] == 2000.0
    assert row_it["median"] == 2000.0
