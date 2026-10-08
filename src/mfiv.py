"""MFIV 1단계: SPY 옵션 EOD 데이터 불러오기·정리"""
from pathlib import Path
import numpy as np
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
def select_expiries(df: pd.DataFrame, target_days: int = 30, min_days: int = 7) -> pd.DataFrame:
    """날짜마다 MFIV 계산에 쓸 만기 2개(near, next)를 고른다.

    - min_days 이하 만기는 가격이 불안정하므로 제외
    - near: target_days 이하 중 가장 긴 만기
    - next: target_days 초과 중 가장 짧은 만기
    - target_days 이하 만기가 없으면 target_days 초과 만기 2개를 쓰고
      extrapolated=True로 표시 (보간 대신 외삽)
    """
    exp = (df[["quote_date", "expire_date", "dte"]]
           .drop_duplicates()
           .query("dte > @min_days")
           .sort_values(["quote_date", "dte"]))

    rows = []
    for date, g in exp.groupby("quote_date"):
        below = g[g["dte"] <= target_days]
        above = g[g["dte"] > target_days]
        if len(below) > 0 and len(above) > 0:
            near, nxt, extra = below.iloc[-1], above.iloc[0], False
        elif len(above) >= 2:
            near, nxt, extra = above.iloc[0], above.iloc[1], True
        else:
            continue  # 쓸 수 있는 만기가 부족한 날은 건너뜀
        rows.append({"quote_date": date,
                     "near_exp": near["expire_date"], "near_dte": near["dte"],
                     "next_exp": nxt["expire_date"], "next_dte": nxt["dte"],
                     "extrapolated": extra})
    return pd.DataFrame(rows)
def load_rates(path=None) -> pd.Series:
    """FRED DTB3(3개월 국채, %) CSV -> 날짜별 연이자율(소수) 시리즈"""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "raw" / "DTB3.csv"
    r = pd.read_csv(path)
    r.columns = ["date", "rate"]                       # FRED 컬럼명이 달라도 대응
    r["date"] = pd.to_datetime(r["date"])
    r["rate"] = pd.to_numeric(r["rate"], errors="coerce") / 100   # '.'(결측) -> NaN
    return r.set_index("date")["rate"].ffill()          # 휴일 결측은 직전 값으로 채움


def compute_forward(chain: pd.DataFrame, r: float, T: float):
    """한 날짜·한 만기의 옵션 체인으로 선도가격 F와 K0를 구한다.

    chain: 같은 quote_date, 같은 expire_date의 행들
    r: 연이자율 (소수), T: 만기까지 기간 (년)
    """
    both = chain[(chain["c_bid"] > 0) & (chain["p_bid"] > 0)]    # 콜·풋 모두 호가가 있는 행사가
    if both.empty:
        return float("nan"), float("nan")
    i = (both["c_mid"] - both["p_mid"]).abs().idxmin()           # 콜·풋 차이가 가장 작은 행사가
    k_star = both.loc[i, "strike"]
    F = k_star + np.exp(r * T) * (both.loc[i, "c_mid"] - both.loc[i, "p_mid"])   # 풋-콜 패리티
    below = chain.loc[chain["strike"] <= F, "strike"]
    K0 = below.max() if not below.empty else float("nan")       # F 바로 아래 행사가
    return F, K0
def _walk_otm(strikes, bids, mids):
    """K0에서 바깥쪽으로 이동하며 OTM 옵션을 모은다 (bid 0 연속 2개면 중단)."""
    out, zeros = [], 0
    for k, b, m in zip(strikes, bids, mids):
        if b <= 0:
            zeros += 1
            if zeros == 2:
                break
            continue                      # bid 0 한 개는 건너뛰고 계속
        zeros = 0
        out.append((k, m))
    return out


def select_otm(chain: pd.DataFrame, K0: float) -> pd.DataFrame:
    """MFIV 계산에 쓸 행사가별 가격 Q(K)를 고른다.

    K0 아래: OTM 풋 (K0에서 아래로), K0 위: OTM 콜 (K0에서 위로),
    K0: 콜·풋 중간값의 평균
    """
    ch = chain.sort_values("strike")
    puts = ch[ch["strike"] < K0].iloc[::-1]                 # K0에서 아래로 내려가는 순서
    calls = ch[ch["strike"] > K0]                           # K0에서 위로 올라가는 순서
    at = ch[ch["strike"] == K0]

    rows = _walk_otm(puts["strike"], puts["p_bid"], puts["p_mid"])
    rows += _walk_otm(calls["strike"], calls["c_bid"], calls["c_mid"])
    if not at.empty:
        rows.append((K0, (at["c_mid"].iloc[0] + at["p_mid"].iloc[0]) / 2))

    return pd.DataFrame(rows, columns=["strike", "Q"]).sort_values("strike").reset_index(drop=True)
def expiry_variance(otm: pd.DataFrame, F: float, K0: float, r: float, T: float) -> float:
    """한 만기의 내재분산(연율)을 VIX 산출법으로 계산한다.

    σ² = (2/T) Σ ΔK/K² · e^{rT} · Q(K)  −  (1/T)(F/K0 − 1)²
    """
    if len(otm) < 3 or T <= 0:
        return float("nan")
    K = otm["strike"].to_numpy()
    Q = otm["Q"].to_numpy()

    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2          # 가운데: 위아래 행사가 간격의 절반
    dK[0] = K[1] - K[0]                      # 맨 아래 끝
    dK[-1] = K[-1] - K[-2]                   # 맨 위 끝

    total = np.sum(dK / K**2 * np.exp(r * T) * Q)
    return (2 / T) * total - (1 / T) * (F / K0 - 1) ** 2
def interpolate_variance(v1, T1, v2, T2, T_target):
    """두 만기의 분산을 목표 기간으로 보간(또는 외삽)한다. 누적 분산 w = σ²·T를 직선으로 잇는다."""
    w1, w2 = v1 * T1, v2 * T2
    w = w1 + (w2 - w1) * (T_target - T1) / (T2 - T1)
    return w / T_target if w > 0 else float("nan")   # 외삽으로 음수가 나오면 무효 처리


def compute_mfiv(df: pd.DataFrame, rates: pd.Series,
                 target_days: int = 30, min_days: int = 7) -> pd.DataFrame:
    """전체 날짜에 대해 30일 MFIV 시계열을 계산한다."""
    exps = select_expiries(df, target_days, min_days)
    r_daily = rates.reindex(exps["quote_date"], method="ffill").to_numpy()
    groups = df.groupby(["quote_date", "expire_date"])

    out = []
    for (_, row), r in zip(exps.iterrows(), r_daily):
        var = []
        for exp, dte in [(row["near_exp"], row["near_dte"]), (row["next_exp"], row["next_dte"])]:
            chain = groups.get_group((row["quote_date"], exp))
            T = dte / 365
            F, K0 = compute_forward(chain, r, T)
            var.append(expiry_variance(select_otm(chain, K0), F, K0, r, T) if np.isfinite(K0) else np.nan)

        v30 = interpolate_variance(var[0], row["near_dte"] / 365, var[1], row["next_dte"] / 365,
                                   target_days / 365)
        out.append({"date": row["quote_date"], "mfiv_var": v30, "mfiv_vol": np.sqrt(v30),
                    "near_var": var[0], "next_var": var[1], "extrapolated": row["extrapolated"]})
    return pd.DataFrame(out).set_index("date")


if __name__ == "__main__":
    # 터미널에서 python -m src.mfiv 로 실행하면 전체 기간 MFIV를 계산해 저장한다
    root = Path(__file__).resolve().parents[1]
    options = load_options()
    mfiv = compute_mfiv(options, load_rates())
    out_path = root / "data" / "processed" / "mfiv_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mfiv.to_parquet(out_path)
    print(mfiv.describe())
    print(f"저장 완료: {out_path} ({len(mfiv)}일)")