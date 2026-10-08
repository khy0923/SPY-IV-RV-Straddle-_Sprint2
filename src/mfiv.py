"""MFIV 1단계: SPY 옵션 EOD 데이터 불러오기·정리"""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "options"

KEEP = ["quote_date", "expire_date", "dte", "underlying_last", "strike",
        "c_bid", "c_ask", "p_bid", "p_ask"]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """' [QUOTE_DATE]' -> 'quote_date'"""
    df.columns = df.columns.str.strip().str.strip("[]").str.lower()
    return df


def load_options(years=None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("spy_eod_*.parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem[-4:]) in years]
    if not files:
        raise FileNotFoundError(f"옵션 파일이 없습니다: {raw_dir}")

    frames = [_clean_columns(pd.read_parquet(f))[KEEP] for f in files]
    df = pd.concat(frames, ignore_index=True)

    for c in ["quote_date", "expire_date"]:
        df[c] = pd.to_datetime(df[c].astype(str).str.strip())
    num_cols = [c for c in KEEP if c not in ("quote_date", "expire_date")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # 호가가 비정상인 행 제거 (bid=0은 꼬리 절단 규칙에 필요하므로 남김)
    df = df.dropna(subset=num_cols)
    ok = ((df["c_ask"] >= df["c_bid"]) & (df["p_ask"] >= df["p_bid"])
          & (df["c_bid"] >= 0) & (df["p_bid"] >= 0))
    df = df[ok].copy()

    df["c_mid"] = (df["c_bid"] + df["c_ask"]) / 2
    df["p_mid"] = (df["p_bid"] + df["p_ask"]) / 2
    return df.sort_values(["quote_date", "expire_date", "strike"]).reset_index(drop=True)