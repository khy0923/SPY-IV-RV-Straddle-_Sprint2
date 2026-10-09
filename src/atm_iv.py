"""ATM IV: 30일 고정만기 ATM 내재변동성 (Black-76 역산)

메인 매매 신호용 IV. 만기 선택·선도가격·이자율·30일 보간은 mfiv.py의 함수를 재사용한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from src.mfiv import (compute_forward, interpolate_variance, load_options,
                      load_rates, select_expiries)


def black76_price(F, K, T, r, sigma, cp):
    """Black-76 옵션 가격. cp='c'(콜) 또는 'p'(풋)"""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if cp == "c":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, r, cp, lo=1e-4, hi=5.0):
    """시장 가격과 Black-76 가격이 같아지는 σ를 찾는다. 찾을 수 없으면 NaN."""
    disc = np.exp(-r * T)
    intrinsic = disc * max(F - K, 0) if cp == "c" else disc * max(K - F, 0)
    upper = disc * F if cp == "c" else disc * K
    if not (intrinsic < price < upper):          # 이론적으로 불가능한 가격
        return np.nan
    try:
        return brentq(lambda s: black76_price(F, K, T, r, s, cp) - price, lo, hi)
    except ValueError:
        return np.nan


def atm_iv_for_expiry(chain: pd.DataFrame, r: float, T: float) -> dict:
    """한 만기의 ATM IV. F에 가장 가까운 행사가의 콜·풋 IV를 평균한다."""
    F, _ = compute_forward(chain, r, T)
    both = chain[(chain["c_bid"] > 0) & (chain["p_bid"] > 0)]
    if not np.isfinite(F) or both.empty:
        return {"F": np.nan, "K": np.nan, "iv_c": np.nan, "iv_p": np.nan, "iv": np.nan}

    row = both.loc[(both["strike"] - F).abs().idxmin()]      # ATM 행사가
    K = row["strike"]
    iv_c = implied_vol(row["c_mid"], F, K, T, r, "c")
    iv_p = implied_vol(row["p_mid"], F, K, T, r, "p")
    iv = np.nanmean([iv_c, iv_p]) if np.isfinite([iv_c, iv_p]).any() else np.nan
    return {"F": F, "K": K, "iv_c": iv_c, "iv_p": iv_p, "iv": iv}


def compute_atm_iv(df: pd.DataFrame, rates: pd.Series,
                   target_days: int = 30, min_days: int = 7) -> pd.DataFrame:
    """전체 날짜에 대해 30일 ATM IV 시계열을 계산한다."""
    exps = select_expiries(df, target_days, min_days)
    r_daily = rates.reindex(exps["quote_date"], method="ffill").to_numpy()
    if np.isnan(r_daily).any():
        raise ValueError("이자율이 없는 날짜가 있습니다. DTB3.csv의 기간을 확인하세요.")
    groups = df.groupby(["quote_date", "expire_date"])

    out = []
    for (_, row), r in zip(exps.iterrows(), r_daily):
        near = atm_iv_for_expiry(groups.get_group((row["quote_date"], row["near_exp"])),
                                 r, row["near_dte"] / 365)
        nxt = atm_iv_for_expiry(groups.get_group((row["quote_date"], row["next_exp"])),
                                r, row["next_dte"] / 365)
        var30 = interpolate_variance(near["iv"]**2, row["near_dte"] / 365,
                                     nxt["iv"]**2, row["next_dte"] / 365, target_days / 365)
        out.append({"date": row["quote_date"],
                    "atm_iv": np.sqrt(var30), "atm_var": var30,
                    "near_iv": near["iv"], "next_iv": nxt["iv"],
                    "near_K": near["K"], "near_F": near["F"],
                    "near_cp_gap": near["iv_c"] - near["iv_p"],   # 콜·풋 IV 차이 (조기행사 점검용)
                    "extrapolated": row["extrapolated"]})
    return pd.DataFrame(out).set_index("date")


if __name__ == "__main__":
    # 터미널에서 python -m src.atm_iv 로 실행하면 전체 기간 ATM IV를 계산해 저장한다
    root = Path(__file__).resolve().parents[1]
    atm = compute_atm_iv(load_options(), load_rates())
    out_path = root / "data" / "processed" / "atm_iv_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atm.to_parquet(out_path)
    print(atm[["atm_iv", "near_iv", "next_iv", "near_cp_gap"]].describe())
    print(f"저장 완료: {out_path} ({len(atm)}일)")