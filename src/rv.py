"""2단계 준비: SPY 1분봉 -> 5분봉 -> 일별 실현분산(RV) 계산"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_1MIN = Path(__file__).resolve().parents[1] / "data" / "raw" / "SPY_raw_1min.parquet"
SOURCE_BREAK = pd.Timestamp("2022-03-07")   # pitrading -> iex 소스 전환일 (가격 레벨 불연속)


def load_5min(path=RAW_1MIN) -> pd.DataFrame:
    """1분봉을 읽어 5분봉으로 묶는다 (시가=첫 값, 고가=최대, 저가=최소, 종가=마지막 값)."""
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    bars = (df.resample("5min", label="left", closed="left").agg(agg)
              .dropna(subset=["Close"]).reset_index())
    bars["date"] = bars["datetime"].dt.normalize()
    return bars


def daily_rv(df: pd.DataFrame) -> pd.DataFrame:
    """일별 RV를 계산한다.

    rv_intraday : 장중 5분 로그수익률 제곱합 (첫 봉은 시가->종가)
    r_overnight : 전일 종가 -> 당일 시가 로그수익률
    rv          : rv_intraday + r_overnight^2  (옵션 IV와 같은 '종가->종가' 기준)
    r_cc        : 일별 종가 로그수익률 (HV 벤치마크용)
    """
    df = df.copy()
    df["logc"] = np.log(df["Close"])
    df["r"] = df.groupby("date")["logc"].diff()
    first = df.groupby("date").cumcount() == 0
    df.loc[first, "r"] = np.log(df.loc[first, "Close"] / df.loc[first, "Open"])   # 첫 봉: 시가 -> 종가

    g = df.groupby("date")
    out = pd.DataFrame({
        "rv_intraday": g["r"].apply(lambda x: np.sum(x**2)),
        "open": g["Open"].first(),
        "close": g["Close"].last(),
        "n_bars": g["r"].size(),
    })
    out["r_overnight"] = np.log(out["open"] / out["close"].shift(1))
    out["r_cc"] = np.log(out["close"] / out["close"].shift(1))
    if SOURCE_BREAK in out.index:                       # 소스 경계의 가짜 수익률 제거
        out.loc[SOURCE_BREAK, ["r_overnight", "r_cc"]] = np.nan
    out["rv"] = out["rv_intraday"] + out["r_overnight"].fillna(0) ** 2
    return out


if __name__ == "__main__":
    # 터미널에서 python -m src.rv 로 실행
    root = Path(__file__).resolve().parents[1]
    rv = daily_rv(load_5min())
    out_path = root / "data" / "processed" / "rv_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rv.to_parquet(out_path)
    ann = np.sqrt(rv[["rv_intraday", "rv"]] * 252) * 100
    print(ann.describe().round(2))
    print("봉 개수 분포:\n", rv["n_bars"].value_counts().head())
    print(f"저장 완료: {out_path} ({len(rv)}일, {rv.index.min().date()} ~ {rv.index.max().date()})")