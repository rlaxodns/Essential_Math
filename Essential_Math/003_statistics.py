# Data
# 데이터 마이닝: 대규모 데이터에서 가치 있는 정보를 추출하는 과정

# 기술통계: 데이터의 간단한 요약하는 데 사용
# 추론통계: 표본을 기반으로 더 큰 모집단에 대한 속성을 발견하는 것

# 3.3 모집단, 표본, 편향
# 특정 그룹이 표본에 스스로를 포함시킬 가능성이 높은 유형의 편향 '자기 선택 편향' 
# --> 교란변수를 고려하지 않은 요인들이 연구에 영향

# 여러 종류의 편향
"""
확증편향:
자신의 신념을 뒷받침하는 데이터만 수집하는 것이며, 자신도 모르게 행할 수 있음

자기 선택 편향:
예를 들어 비행기 탑승 고객을 대상으로 다른 항공사보다 이번 비행의 항공사를 더 좋아하는 지 설문 시에 
해당 항공사의 재이용고객이 많을 경우 자기 선택 편향이 발생

생존 편향:
살아 있거나 살아남은 피험자만 포착하는 반면, 사망한 피험자는 고려하지 않는 경우
"""

# 머신러닝의 표본과 편향
# 데이터를 기반으로 추론하는 머신러닝에서도 데이터가 편향되어 있을 경우, 머신러닝 알고리즘도 편향된 결론을 내리게 된다.

# 3.4 기술 통계
# 3.4.1 평균과 가중평균

samples = [1,3,2,5,6,0,8,5,9]
mean = sum(samples) / len(samples)
print(mean)

# 평균은 익숙한 개념이지만, 가중 평균은 생소한 경우가 多
# 가중 평균: 일부 값이 다른 값보다 평균에 더 많이 기여하게 만드는 경우 유용 ex) 고교 내신: 1학년(20%), 2학년(30%), 3학년(50%)

samples = [90, 95, 100]
weights = [.2, .3, .5]
weighted_mean = sum(s*w for s, w in zip(samples, weights)) / sum(weights)
print(weighted_mean)

# 3.4.2. 중앙값
# 정렬된 값 집합에서 가장 가운데 있는 값
sample = [0, 1, 5, 7, 9, 10, 14]

def median(values):
    ordered = sorted(values)
    print(ordered)
    n = len(ordered)
    mid = int(n/2) - 1 if n % 2 == 0 else int(n/2)

    if n%2 == 0:
        return (ordered[mid] + ordered[mid+1])/2.0
    else:
        return ordered[mid]
    
print(median(sample))
# 중앙값은 이상치 또는 다른 값에 비해 매우 크거나 작은 값으로 인해 왜곡된 데이터에서 평균을 대신할 수 있는 유용한 대안
# 분위수--> 중앙값은 50% 분위수
# 25% 단위로 잘라내는 분위수를 '사분위수'

# 3.4.3 모드 (최빈값)
from collections import defaultdict

sample = [1, 3, 2, 5, 7, 0, 2, 3]

def mode(values):
    counts = defaultdict(lambda: 0)
    
    for s in values:
        counts[s] += 1

    max_count = max(counts.values())
    modes = [v for v in set(values) if counts[v] == max_count]

    return modes
print(mode(sample))

# 3.4.4 분산 & 표준 편차
# 모집단의 분산 = (x1 - mean)**2 + (x2 - mean)**2 +...+(xn-mean)**2 / N

# Code
data = [0, 1, 5, 7, 9, 10, 14]

def variance(values):
    mean = sum(values) / len(values)
    _variance = sum((v-mean)**2 for v in values) / len(values)
    return _variance

print(variance(data))

# 표준편차 --> 분산의 제곱근
# Code
from math import sqrt

data = [0, 1, 5, 7, 9, 10, 14]

def variance(values):
    mean = sum(values) / len(values)
    _variance = sum((v-mean)**2 for v in values) / len(values)
    return _variance

def std_dev(values):
    return sqrt(variance(values))

print(std_dev(data))

# 표본의 분산과 표준 편차
from math import sqrt
import math
def variance(values, is_sample: bool = False):
    mean = sum(values) / len(values)
    _variance = sum((v-mean)**2 for v in values)/(len(values) - (1 if is_sample else 0))
    return _variance

def std_dev(values, is_sample: bool = False):
    return sqrt(variance(values, is_sample))

print('분산', variance(data, is_sample=True))
print('표본의 표준편차', std_dev(data, is_sample=True))

# 3.4.5 정규 분포 (가우스 분포)
# 평균 근처가 가장 질량이 크고 대칭 형태를 띤 종 모양의 분포
# 이 분포의 퍼짐 정도는 표준 편차로 정의, 양쪽의 꼬리는 평균에서 멀어질수록 가늘어진다.

# 정규 분포의 속성
"""
1. 정규 분포는 대칭
2. 대부분의 질랴의 평균 부근
3. 퍼짐 정도가 있으며 표준편차로 이를 나타냄
4. 꼬리는 가능성이 가장 낮은 부분이며, 0에 수렴하지만, 0이 되지는 않는다.
5. 자연과 일상생활에서 일어나는 많은 현상과 유사합니다. 중심 극한 정리 덕분에 정규 분포가 아닌 문제에도 일반화 가능
"""

# 확률 밀도 함수(PDF) Probablity Density Function
def noraml_pdf(x: float, mean: float, std_dev: float) -> float:
    return (1.0/(2.0* math.pi * std_dev**2)**0.5) * math.exp(-1.0 *((x - mean)**2 / (2.0*std_dev**2)))

# 누적 분포 함수(CDF) Cumulative Distribution Function
# 정규 분포에서 세로축은 확률이 아니라 데이터에 대한 가능도, 확률을 얻을려면 주어진 범위의 곡선 아래의 면적
# 시그모이드 곡선인 CDF는 해당 위치까지의 PDF의 면적을 나타냄
from scipy.stats import norm
mean = 64.43
std_devx = 2.99
x2 = norm.cdf(64.43, mean, std_devx)
print(x2) # 0 ~ 64.43까지의 면적

x1 = norm.cdf(66, mean, std_devx) - norm.cdf(62, mean, std_devx)
print(x1) # 62~66까지의 범위 면적

# 3.4.6 역CDF
# 가설검정을 수행할때 CDF의 면적을 구한 다음, 이에 해당하는 x 값을 반환해야함.
# 이는 CDF를 역방향으로 사용하는 것, 이렇게 하면 확률을 찾아 해당하는 x 값을 반환할 수 있음

x2 = norm.ppf(0.95, loc = mean, scale = std_devx)
print(x2) # 골든 리트리버의 95%가 69.348

# 역CDF를 사용해 정규 분포를 따르는 난수를 생성할 수도 있음
import random
from scipy.stats import norm
for i in range(0, 1000):
    random_p = random.uniform(0.0, 1.0)
    random_weight = norm.ppf(random_p, loc = mean, scale = std_devx)
    print(i, ':', random_weight)


# 3.4.7 z-score
# 평균이 0이고 표준편차가 1이 되도록 정규 분포의 크기를 재조정하는 것이 일반적
# 이를 표준정규분포라고 함
 
# z = (x - mean) / std

def z_score(x, mean, std):
    return (x-mean) / std

def z_to_x(z, mean, std):
    return (z*std)+mean

mean = 140000
std_dev = 3000
x = 150000

z = z_score(x, mean, std_dev)
back_to_x = z_to_x(z, mean, std_dev)
print(z)
print(back_to_x)

# z_score() 함수는 평균과 표준편차가 주어지면 x 값을 받아 표준편차 단위로 변환

# 추론 통계
# 3.5.1 중심 극한 정리
# 균등 분포에서 뽑은 표본의 평균은 정규 분포가 된다
import random
import matplotlib.pyplot as plt

sample_size = 31
sample_count = 1000

# 중심극한정리, 크기가 31인 1000개의 표본
x_values = [(sum([random.uniform(0.0, 1.) for i in range(sample_size)]) / sample_size) for _ in range(sample_count)]

# 1. Matplotlib으로 히스토그램 생성
# plt.hist() 함수를 사용하고, bins=20으로 지정합니다.
# edgecolor='black'은 막대 구분을 명확하게 하기 위해 추가했습니다 (선택 사항).
plt.hist(x_values, bins=20, edgecolor='black')

# 2. (선택 사항) 그래프 제목과 축 레이블 추가
plt.title('Histogram of Sample Means (Sample Size = 31)')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency (Count)')

# 3. 그래프를 화면에 표시
# plt.show()

# 표본에 있는 하나의 숫자만으로는 정규 분포를 만들 수 없음
# 모든 숫자가 나올 가능성이 똑같은 평평한 분포가 된다(균등분포)
# 하지만 표본의 평균을 내면 정규 분포를 형성

# 모집단에서 충분히 많은 표본을 채취해 각각의 평균을 계산하고 이를 하나의 분포를 그리면 중심 극한 정리 현상이 나타남
"""
1. 표본 평균의 모집단 평균과 같음
2. 모집단이 정규 분포이면 표본평균도 정규 분포가 된다.
3. 모집단이 정규 분포가 아니지만 표본 크기가 30보다 큰 경우 표본 평균은 대략적으로 정규 분포를 따른다.
4. 표본 평균의 표준 편차는 모집단 표준 편차를 n의 제곱근으로 나눈 값과 같다.
"""
# 표본의 크기가 31 미만인 경우 정규 분포가 아닌 t-분포를 사용해야 한다. 

# 3.5.2 신뢰 구간
"""
신뢰구간은 표본 평균이 모집단 평균의 특정범위에 속한다고 얼마나 확실하게 믿는지를 보여주는 방법

먼저 모집단 평균 범위에 대한 확률을 포함할 수 있는 신뢰수준LOC을 선택
중앙의 95% 확률에 해당하는 표준 정규 분포의 대칭 범위인 임계 z 값이 필요

이를 통해 0.025 & 0.975를 제외한 95%의 중앙 범위를 도출 
그 후 최소 z값과 최대 z값을 반환
"""
# 임계 z값 검색
from scipy.stats import norm

def critical_z_value(p):
    norm_dist = norm(loc = 0.0, scale = 1.0)
    left_tail_area = (1.0-p)/2.0
    upper_area = 1.0 - ((1.0-p)/2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)
print(critical_z_value(p = 0.95))

# ±1.95996은 표준 정규 분포의 중심에서 확률의 95%에 해당하는 임계 z값
# 중심 극한 정리를 활용해 해당 신뢰 수준에서 모집단 평균을 포함하는 표본 평균의 범위인 허용 오차 계산

# E = critical_z_value * (Std/sqrt(n))

# 신뢰구간 계산하기
def critical_z_value(p):
    norm_dist = norm(loc=0, scale = 1)
    left_tail_area = (1.0 - p)/ 2
    upper_area = 1 - left_tail_area
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)

def confidence_interval(p, sample_mean, sample_std, n):
    # 표본의 크기가 최소 31이상 되어야함!!!
    lower, upper = critical_z_value(p)
    lower_ci = lower * (sample_std/sqrt(n))
    upper_ci = upper * (sample_std/sqrt(n))

    return sample_mean + lower_ci, sample_mean + upper_ci
print(confidence_interval(p=0.95, sample_mean=64.408, sample_std=2.05, n=31))

#3.5.3 p값 이해하기
"""
통계적 유의성이란?
- 대립가설을 직접 증명하지 않고, 귀무가설이 틀렸다는 것을 증명하는 것
"귀무가설"은 기본값 또는 현재의 사실을 의미하며, "효과가 없다.", "두 그룹 간에 차이가 없다"는 입장
ex) 새로 개발한 약은 효과가 없다

"대랍가설"은 연구자가 "새롭게 주장하고 싶은 것"을 의미
"어떤 효과가 있다.", "두 그룹간 차이가 있다"는 입장
ex) 새로 개발한 약은 효과가 있다

Q) 만약 귀무가설이 정말 사실이라면, 이 데이터가 단지 우연만으로 나올 확률은 얼마인가?
--> p-value

p-value < 0.05
- 통계적으로 유의미하다

p-value >= 0.05
-  통계적으로 유의미하지 않다.
"""

# 3.5.4 가설 검정
# Code
from scipy.stats import norm

mean = 18
std_dev = 1.5

# 역 cdf
x = norm.cdf(21, mean, std_dev) - norm.cdf(15, mean, std_dev)
print(x) # 16 ~ 21의 확률

# 단측검정
# 단측검정에서는 일반적으로 부등식을 사용해 귀무가설과 대립가설을 설정
"""
H0 = 모집단 평균 >= 18(귀무가설)
H1 = 모집단 평균 < 18 (대립가설)

귀무가설을 기각하려면 표본의 평균이 우연이 아닐 가능성이 높다는 것을 보여주어야함
일반적으로 0.05 이하의 p-value는 통계적으로 유의미하다고 판단
"""

from scipy.stats import norm
mean = 18
std_dev = 1.5

# 임계값에 해당하는 x값
x = norm.ppf(0.05, mean, std_dev)
print(x) # 15.53271955957279


from scipy.stats import norm

mean = 18
std_dev = 1.5

x = norm.cdf(16, mean, std_dev)
print(x) # p-value: 0.09121121972586788이므로 귀무가설을 기각하지 못함

# 양측검정
"""
양측검정의 실질적 의미
양측검정은 귀무가설을 기각하기 어렵게 만들고 검정을 통과하기 위해 더 강력한 증거를 요구한다.
"""
from scipy.stats import norm
mean = 18
std_dev = 1.5

p1 = norm.pdf(16, mean, std_dev)
p2 = norm.pdf(20, mean, std_dev)

p_value = p1+p2
print(p_value) # 0.21868009956799153


# t-분포: 소규모 표본 처리
"""표본의 크기가 30이하라면 정규 분포 대신 t분포를 사용
t-분포는 정규 분포와 비슷하지만, 더 많은 분산가 불확실성을 반영하기 위해 꼬리가 두껍다"""
from scipy.stats import t

# 표본의 크기가 25인 경우,
# 95% 신뢰도에 대한 임계값 범위 구하기

n = 25
lower = t.ppf(0.025, df = n-1)
upper = t.ppf(0.975, df = n-1)
print(lower, upper)
