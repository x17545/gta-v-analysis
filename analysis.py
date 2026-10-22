import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 1단계: 파일 읽기
df = pd.read_csv("steamdb_chart_271590.csv", parse_dates=["DateTime"])

# 결과 확인
print("행/열 개수:", df.shape)
print("\n--- 앞부분 미리보기 ---")
print(df.head())
print("\n--- 각 컬럼 타입 ---")
print(df.dtypes)

# ============================================
# 2단계: 결측치 제거
# ============================================

# 제거 전 개수 기록
before = len(df)

# Players 값이 비어있는(NaN) 행 제거
df = df.dropna(subset=["Players"])

# 제거 후 개수
after = len(df)

print("\n--- 2단계: 결측치 제거 ---")
print(f"제거 전: {before}행")
print(f"제거 후: {after}행")
print(f"삭제된 행: {before - after}행")

# ============================================
# 3단계: 일별 평균으로 리샘플링
# ============================================

# DateTime을 인덱스(기준축)로 설정
df = df.set_index("DateTime")

# 하루(D) 단위로 묶어서 Players 평균 계산
daily = df["Players"].resample("D").mean()

# 리샘플링 후에도 값이 없는 날 제거
daily = daily.dropna()

print("\n--- 3단계: 일별 평균 리샘플링 ---")
print(f"리샘플링 후 행 수: {len(daily)}")
print("\n앞부분 5개:")
print(daily.head())
print("\n뒷부분 5개:")
print(daily.tail())

# ============================================
# 4단계: 최근 2년 기간 필터링
# ============================================

# 데이터의 마지막 날짜 기준으로 2년 전 계산
last_date = daily.index.max()
start_date = last_date - pd.DateOffset(years=2)

print("\n--- 4단계: 기간 필터링 ---")
print(f"전체 마지막 날짜: {last_date.date()}")
print(f"필터 시작 날짜: {start_date.date()}")

# 시작 날짜 이후 데이터만 남기기
recent = daily[daily.index >= start_date]
recent = recent[:-1]

print(f"\n필터링 전: {len(daily)}행")
print(f"필터링 후: {len(recent)}행")
print("\n앞부분 3개:")
print(recent.head(3))
print("\n뒷부분 3개:")
print(recent.tail(3))

# ============================================
# 5단계: 2년치 일별 동접자 그래프
# ============================================

plt.figure(figsize=(14, 6))
plt.plot(recent.index, recent.values, linewidth=1, color='green')

plt.title('GTA V Daily Average Players (Recent 2 Years)')
plt.xlabel('Date')
plt.ylabel('Players')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_raw.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================
# 6단계: 이동평균으로 트렌드 확인 (Q2)
# ============================================

# 7일 이동평균 계산 (주간 진동 제거)
moving_avg_7 = recent.rolling(window=7).mean()

# 30일 이동평균 계산 (월간 큰 흐름)
moving_avg_30 = recent.rolling(window=30).mean()

# 그래프 그리기
plt.figure(figsize=(14, 6))

# 원본 데이터 (연하게)
plt.plot(recent.index, recent.values,
         linewidth=0.8, color='lightgray', label='Daily (raw)')

# 7일 이동평균
plt.plot(moving_avg_7.index, moving_avg_7.values,
         linewidth=1.5, color='blue', label='7-day average')

# 30일 이동평균
plt.plot(moving_avg_30.index, moving_avg_30.values,
         linewidth=2.5, color='red', label='30-day average')

plt.title('GTA V Player Trend with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Players')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_trend.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================
# 7단계: 요일별 평균 분석 (Q1)
# ============================================

# recent를 DataFrame으로 변환 (요일 정보 추가하려고)
df_weekday = recent.reset_index()
df_weekday.columns = ['date', 'players']

# 각 날짜에서 요일 뽑기 (0=월요일, 6=일요일)
df_weekday['weekday'] = df_weekday['date'].dt.dayofweek

# 확인용 출력
print(df_weekday.head(10))

# ============================================
# 8단계: 요일별 평균 계산
# ============================================

# weekday(0~6)로 묶어서 players 평균 구하기
weekday_avg = df_weekday.groupby('weekday')['players'].mean()

# 보기 좋게 요일 이름 붙이기
weekday_names = ['월', '화', '수', '목', '금', '토', '일']

print('--- 요일별 평균 동접자 ---')
for i, name in enumerate(weekday_names):
    print(f'{name}요일: {weekday_avg[i]:,.0f}명')

# ============================================
# 9단계: 요일별 평균 막대그래프
# ============================================

plt.figure(figsize=(10, 6))

# 막대그래프 (주말은 색 다르게!)
colors = ['skyblue', 'skyblue', 'skyblue', 'skyblue', 'skyblue', 'orange', 'orange']
plt.bar(weekday_names, weekday_avg, color=colors)

plt.title('GTA V 요일별 평균 동시 접속자')
plt.xlabel('요일')
plt.ylabel('평균 접속자 수')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('q1_weekday.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================
# 10단계: 급증한 날 자동 탐지 (Q3)
# ============================================

# recent를 DataFrame으로 변환
df_spike = recent.reset_index()
df_spike.columns = ['date', 'players']

# 전날 대비 증가율 계산 (%)
# pct_change(): 바로 앞 행과 비교해서 변화율을 구해줌
df_spike['change_pct'] = df_spike['players'].pct_change() * 100

# 급증 기준: 전날보다 15% 이상 오른 날
threshold = 15
spikes = df_spike[df_spike['change_pct'] >= threshold]

print(f'\n--- 10단계: 급증한 날 (전날 대비 +{threshold}% 이상) ---')
print(f'총 {len(spikes)}일 발견!\n')

# 급증한 날 출력 (증가율 높은 순으로 정렬)
for _, row in spikes.sort_values('change_pct', ascending=False).iterrows():
    date = row['date'].date()
    players = row['players']
    pct = row['change_pct']
    print(f'{date} : {players:,.0f}명 (전날보다 +{pct:.1f}%)')

# ============================================
# 11단계: 트렌드 그래프에 급증일 표시 (Q3 최종)
# ============================================

# 이벤트 이름 정리 (검색으로 찾은 것!)
event_labels = {
    '2024-11-09': ('습격 챌린지 2024\n(보상 최대 3배)',    (0,  38)),
    '2025-05-01': ('GTA+ 구독자 이벤트\n(보상 2배)',    (0, -48)),
    '2026-03-14': ('성 패트릭 데이 이벤트\n(접속 100만$)',  (0,  42)),
    '2026-05-01': ('GTA+ 구독자 이벤트\n(보상 2배)',    (0, -48)),
}

plt.figure(figsize=(14, 6))

# 원본 트렌드 (연하게)
plt.plot(recent.index, recent.values,
         linewidth=0.8, color='lightgray', label='Daily (raw)')

# 30일 이동평균 (큰 흐름)
plt.plot(moving_avg_30.index, moving_avg_30.values,
         linewidth=2.5, color='red', label='30-day average')

# 급증한 날에 마커 찍기 (빨간 점)
plt.scatter(spikes['date'], spikes['players'],
            color='darkorange', s=100, zorder=5,
            edgecolor='black', label='급증일')

# 주요 이벤트에 이름표 달기
for date_str, (label, offset) in event_labels.items():   # ← offset 추가!
    date = pd.to_datetime(date_str)
    row = spikes[spikes['date'] == date]
    if not row.empty:
        players = row['players'].values[0]
        plt.annotate(label,
                     xy=(date, players),
                     xytext=offset, textcoords='offset points',  # ← offset 사용!
                     ha='center', fontsize=9, color='black',
                     arrowprops=dict(arrowstyle='->', color='gray'))

plt.title('GTA V 급증일과 실제 이벤트 (Q3)')
plt.xlabel('Date')
plt.ylabel('Players')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q3_events.png', dpi=100, bbox_inches='tight')
plt.show()

# ========================================
#  데이터 요약 증거 파일 만들기
# ========================================

# 원본 파일 다시 읽기
raw = pd.read_csv("steamdb_chart_271590.csv")

# --- 여기부터 새로 정제! (증거파일 전용) ---
clean = raw.copy()
clean['DateTime'] = pd.to_datetime(clean['DateTime'])   # 날짜 변환
clean = clean.dropna(subset=['Players'])                # 빈 값 제거
clean = clean.set_index('DateTime')                     # 날짜를 인덱스로
daily_clean = clean['Players'].resample('D').mean().dropna()  # 일별 평균

# 최근 2년(731일)만 자르기
recent_clean = daily_clean.tail(731)

# ----------  마지막 미완성 날 제거 ----------
print(f"\n--- 마지막 날 제거 ---")
print(f"제거 전: {len(recent_clean)}개")
print(f"마지막 날: {recent_clean.index[-1].date()} = {recent_clean.iloc[-1]:,.0f}명 (오전만)")

recent_clean = recent_clean[:-1]   # ← 마지막 1개 제거!

print(f"제거 후: {len(recent_clean)}개")
print(f"새 최소값: {recent_clean.min():,.0f}명")
# --- 정제 끝! ---

# ---------- 샘플 데이터 만들기 (summary 위에!) ----------
result = recent_clean.reset_index()
result.columns = ['날짜', '접속자수']
result['접속자수'] = result['접속자수'].round(0).astype(int)
result['날짜'] = result['날짜'].dt.strftime('%Y-%m-%d')

#  여기서 CSV 저장! (딱 한 번만)
result.to_csv("cleaned_players_data.csv", index=False, encoding="utf-8-sig")
print("✅ cleaned_players_data.csv 저장 완료! (730개 전체)")

# 샘플 20개 텍스트 만들기
sample_lines = ["[실제 데이터 샘플 - 처음 10개]"]
for _, row in result.head(10).iterrows():
    sample_lines.append(f"  {row['날짜']} : {row['접속자수']:,}명")
sample_lines.append("")
sample_lines.append("  ... (중략, 전체는 cleaned_players_data.csv 참조) ...")
sample_lines.append("")
sample_lines.append("[실제 데이터 샘플 - 마지막 10개]")
for _, row in result.tail(10).iterrows():
    sample_lines.append(f"  {row['날짜']} : {row['접속자수']:,}명")
sample_text = "\n".join(sample_lines)

# 요약 내용 만들기
summary = f"""========================================
  GTA V 동접자 데이터 분석 - 데이터 요약
========================================

[원본 데이터]
  - 파일명: steamdb_chart_271590.csv
  - 총 행 수: {len(raw):,}행
  - 특징: 초기엔 일별, 최근엔 10분 간격 혼합

[정제 과정]
  1. 결측치(Players 빈 값) 제거
  2. 일별 평균으로 리샘플링 (10분 -> 하루 1개)
  3. 최근 2년(731일)만 필터링

[최종 분석 데이터]
  - 데이터 포인트: {len(recent_clean):,}개  (100개 이상 OK)
  - 분석 기간: {recent_clean.index.min().date()} ~ {recent_clean.index.max().date()}
  - 평균 접속자: {recent_clean.mean():,.0f}명
  - 최대 접속자: {recent_clean.max():,.0f}명
  - 최소 접속자: {recent_clean.min():,.0f}명

{sample_text}

========================================
  분석 질문 3가지
========================================
  Q1. 주말 vs 평일 동접자 차이
  Q2. 최근 2년 트렌드 증감
  Q3. 업데이트/세일 시점 동접자 급증

  ※ 데이터 출처: SteamDB (App ID 271590)
========================================
"""



# 파일로 저장!
with open("data_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print("data_summary.txt 저장 완료!")
