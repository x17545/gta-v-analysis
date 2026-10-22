# bonus_timeseries.py
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 전처리 (analysis.py와 동일) =====
df = pd.read_csv("steamdb_chart_271590.csv", parse_dates=["DateTime"])
df = df.dropna(subset=["Players"])
df = df.set_index("DateTime")
daily = df["Players"].resample("D").mean().dropna()

# 최근 2년 필터링
last_date = daily.index.max()
start_date = last_date - pd.DateOffset(years=2)
recent = daily[daily.index >= start_date]
recent = recent[:-1]

print(f"✅ 전처리 완료: {len(recent)}개 데이터")
print(f"   기간: {recent.index.min().date()} ~ {recent.index.max().date()}")

# ============================================
# 보너스 ②-A: 시계열 분해
# ============================================
from statsmodels.tsa.seasonal import seasonal_decompose

# 주간 계절성(period=7)으로 분해
result = seasonal_decompose(recent, model='additive', period=7)

# 4단 그래프 생성
fig = result.plot()
fig.set_size_inches(14, 10)
fig.suptitle('GTA V 동접자 시계열 분해 (주간 주기)', fontsize=16, y=1.01)

plt.tight_layout()
plt.savefig('bonus_decompose.png', dpi=100, bbox_inches='tight')
plt.close()

print('✅ 시계열 분해 완료 → bonus_decompose.png 저장!')