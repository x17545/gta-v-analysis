import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# ============================================
# 1단계: 데이터 정제 (analysis.py와 동일)
# ============================================

df = pd.read_csv("steamdb_chart_271590.csv", parse_dates=["DateTime"])
df = df.dropna(subset=["Players"])
df = df.set_index("DateTime")
daily = df["Players"].resample("D").mean().dropna()

# 최근 2년 필터링 + 마지막 미완성일 제거
last_date = daily.index.max()
start_date = last_date - pd.DateOffset(years=2)
recent = daily[daily.index >= start_date]
recent = recent[:-1]   # 마지막 미완성일 제거 → 730행

print(f"분석 데이터: {len(recent)}행")
print(f"기간: {recent.index.min().date()} ~ {recent.index.max().date()}")

# ============================================
# 2단계: SARIMA 모델 학습
# ============================================
# SARIMA(p,d,q)(P,D,Q,s) — 각 숫자의 의미:
#   (p,d,q)     = 비계절 부분 (추세/흐름)
#   (P,D,Q,s)   = 계절 부분, s=7 → 7일(주간) 주기
#
# GTA 데이터에 맞춘 추천 파라미터:
#   order=(1,1,1)         → 기본적인 추세 반영
#   seasonal_order=(1,1,1,7) → 7일 주말 패턴 반영 (분해의 Seasonal!)

print("\nSARIMA 모델 학습 중... (10~30초 소요)")

model = SARIMAX(
    recent,
    order=(1, 1, 1),              # 비계절: (p,d,q)
    seasonal_order=(1, 1, 1, 7), # 계절: (P,D,Q,s) s=7일 주기
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)  # disp=False → 학습 과정 출력 생략

print("학습 완료! ✅")

# ============================================
# 3단계: 미래 30일 예측
# ============================================

forecast_days = 30
forecast = results.get_forecast(steps=forecast_days)

# 예측값 (평균)
forecast_mean = forecast.predicted_mean

# 신뢰구간 (95% 범위)
conf_int = forecast.conf_int()

# 미래 날짜 인덱스 만들기 (마지막 날 다음날부터 30일)
future_dates = pd.date_range(
    start=recent.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq='D'
)

print(f"\n--- 미래 {forecast_days}일 예측 ---")
print(f"예측 기간: {future_dates[0].date()} ~ {future_dates[-1].date()}")
print(f"예측 평균 동접자: {forecast_mean.mean():,.0f}명")

# ============================================
# 4단계: 예측 결과 그래프
# ============================================

plt.figure(figsize=(14, 6))

# 최근 90일 실제 데이터만 표시 (예측 부분 잘 보이게)
recent_90 = recent[-90:]
plt.plot(recent_90.index, recent_90.values,
         linewidth=1.5, color='green', label='실제 동접자 (최근 90일)')

# 예측값 (파란 점선)
plt.plot(future_dates, forecast_mean.values,
         linewidth=2, color='blue', linestyle='--', label='예측 (30일)')

# 신뢰구간 (파란 음영)
plt.fill_between(future_dates,
                 conf_int.iloc[:, 0],   # 하한
                 conf_int.iloc[:, 1],   # 상한
                 color='blue', alpha=0.15, label='95% 신뢰구간')

# 실제-예측 경계선 (세로 점선)
plt.axvline(x=recent.index[-1], color='gray',
            linestyle=':', linewidth=1.5, label='예측 시작점')

plt.title('GTA V 동접자 30일 예측 (SARIMA)')
plt.xlabel('날짜')
plt.ylabel('동접자 수')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bonus_forecast.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ bonus_forecast.png 저장 완료!")