# SPY-IV-RV-Straddle-_Sprint2
IV-RV spread straddle strategy on SPY options (ATM IV + HAR)
## IV 산출 방법
- **메인 신호: ATM IV** (Black-76, 30일 보간)
  - 스트래들은 ATM 옵션을 거래하므로, 실제 거래 가격에 가장 직접적인 기준
  - 백테스트와 실전매매에서 같은 방식으로 재현 가능
- **비교·검증: MFIV** (CBOE VIX 산출법)
  - VIX 대비 상관 0.995, 평균 차이 -0.19%p (2010~2023, 3,493일)
  - 스큐 프리미엄(MFIV - ATM IV) 분석 및 강건성 확인에 사용