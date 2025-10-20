"""
# 주변 확률
# - 단일확률 계산

# 2.2.1 결합확률
두 사건이 함께 일어날 확률 = AND 연산
ex) P(A AND B) = P(A) * P(B)

사건 A 또는 사건 B가 발생할 확률 = OR 연산, 합 확률
--> 상호 배타적 사건 
ex) P(A OR B) = P(A) + P(B) # 하지만 문제가 있음, 왜냐하면 중복해서 결과를 헤아리기 때문

## 확률의 덧셈 법칙 
P(A OR B) = P(A) + P(B) - P(A AND B)
P(A OR B) = P(A) + P(B) - P(A) * P(B)
: 두 개 이상의 사건 간에 합 확률을 계산할 때는 확률이 이중으로 계산되지 않도록 결확확률을 빼주어야 함

# 조건부 확률과 베이즈 정리 
-- 사건 B가 발생했을 때 사건 A가 발생할 확률 : P(A|B)

베이즈 정리: P(A|B) = (P(B|A)*P(A))/P(B)
"""
# 베이즈 정리 
p_coffee_drinker = 0.65
p_cancer = 0.005
p_coffer_drinker_given_cancer = 0.85 # p(커피|암)

p_cancer_given_coffee_drinker = (p_coffer_drinker_given_cancer*p_cancer)/p_coffee_drinker
print(p_cancer_given_coffee_drinker)

"""
ex)
전체 인구 중 암에 걸렸으면서 커피를 마시는 사람을 구한다면?
p(커피|암)*p(암)
"""

# 이항분포 : 확률이 p일때, n번의 시도 중 k번이 성공할 가능성 측정
from scipy.stats import binom

n = 10
p = 0.9

for k in range(n+1):
    probability = binom.pmf(k,n,p)
    print("{0}-{1}".format(k, probability))
"""
0.006538461538461539
0-9.999999999999977e-11
1-8.999999999999976e-09
2-3.6449999999999933e-07
3-8.747999999999988e-06
4-0.00013778099999999974
5-0.0014880347999999982
6-0.011160260999999989
7-0.05739562799999997
8-0.1937102444999998
9-0.38742048899999976
10-0.34867844010000015
"""

# 베타 분포: 알파 번의 성공과 베타 번의 실패가 주어졌을때, 사건이 발생할 수 있는 다양한 기본 확률의 가능성
from scipy.stats import beta

a = 8
b = 2

p = beta.cdf(.9, a, b)
print(p) # 0.7748409780000002

not_p = 1 - beta.cdf(.9, a, b)
print(not_p)

# 베파분포 중간 영역

p = beta.cdf(0.9, a, b) - beta.cdf(.8, a, b)
print(p)