import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 분석할 ETF 목록
etfs = ['SPY', 'QQQ', 'TQQQ', 'SQQQ']
start_date = '2020-01-01'
end_date = '2024-12-31'

# 가격 데이터 수집
data = yf.download(etfs, start=start_date, end=end_date)['Close']

# 결측치 제거
data.dropna(inplace=True)

# 일일 수익률 계산
returns = data.pct_change().dropna()

# 누적 수익률 계산
cumulative_returns = (1 + returns).cumprod()

# 누적 수익률 시각화
plt.figure(figsize=(14, 6))
for ticker in etfs:
    plt.plot(cumulative_returns[ticker], label=ticker)

plt.title("Cumulative Return Comparison (2020 ~ 2024)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("etf_result_plot.png")  # 이미지 저장
plt.show()

# 연간 수익률 (CAGR: Compound Annual Growth Rate)
years = (returns.index[-1] - returns.index[0]).days / 365.25
cagr = (cumulative_returns.iloc[-1] ** (1 / years)) - 1

# 연간 변동성 (Standard Deviation * sqrt(252))
volatility = returns.std() * np.sqrt(252)

# 최대 낙폭 (Maximum Drawdown)
max_drawdown = {}
for ticker in etfs:
    cum_max = cumulative_returns[ticker].cummax()
    drawdown = (cumulative_returns[ticker] - cum_max) / cum_max
    max_drawdown[ticker] = drawdown.min()

# 결과 출력
print("\n=== ETF Performance Summary ===")
for ticker in etfs:
    print(f"{ticker}")
    print(f"  CAGR:        {cagr[ticker]:.2%}")
    print(f"  Volatility:  {volatility[ticker]:.2%}")
    print(f"  Max Drawdown:{max_drawdown[ticker]:.2%}")
    print()
    
# Seaborn 스타일 적용 (좀 더 예쁘게)
sns.set(style="whitegrid")

# 박스플롯 그리기
plt.figure(figsize=(12, 6))
sns.boxplot(data=returns[etfs])
plt.title("Daily Return Distribution by ETF")
plt.ylabel("Daily Return")
plt.xlabel("ETF")
plt.grid(True)
plt.tight_layout()
plt.savefig("etf_daily_return_boxplot.png")  # 이미지 저장
plt.show()

#결과 Dataframe 저장후 CSV export
summary_df = pd.DataFrame({
    'CAGR': cagr,
    'Volatility': volatility,
    'Max Drawdown': pd.Series(max_drawdown)
})

summary_df.to_csv("etf_summary.csv")