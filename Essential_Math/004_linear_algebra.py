"""
선형대수학
선형계와 관련되지만 벡터공간과 행렬을 통해 표현하는 학문
"""
# 4.1 벡터란 무엇인가?
"""
벡터는 공간상에서 특정 방향과 길이를 가진 화살표이며 데이터의 종종 한 조각을 나타냄
벡터의 목적은 데이터를 시각적으로 표현하는 것
"""
v = [3, 2]
print(v)

import numpy as np
v = np.array([3, 2])
print(v)
print(v.shape)

"""
벡터의 용도는 多
물리학에서 벡터는 방향과 크기로 간주
수학에서는 xy평면에서의 방향과 스케일, 마치 어떤 움직임과 같은 개념으로 표시
컴퓨터 과학에서는 데이터를 저장하는 숫자 배열
"""
# 벡터는 2차원 이상을 가질 수 있음,
# 3차원 벡터

v3 = np.array([4,1,2])
print(v3)
print(v3.shape)

# 5차원
v5 = np.array([6,1,5,8,3])
print(v5)

# 4.1.1 덧셈과 결합
# 덧셈
v = np.array([3,2])
w = np.array([2,-1])
print(v+w) # 시각적으로는 두 벡터를 차례로 연결하고 마지막 벡터의 끝으로 이동을 의미
# 또한 v와 w의 순서는 중요하지 않은 "교환 법칙"이 성립하여 연산 순서는 중요하지 않음

# 4.1.2 스케일링
"""
스케일링은 벡터의 길이를 늘이거나 줄이는 것 
벡터에 스칼라라고 하는 하나의 값을 곱하거나 스케일링해서 벡터를 늘이거나 줄인다.
"""

v = np.array([4, 5])
scaled_v = 2*v
print(scaled_v)
# 벡터의 크기를 조정해도 벡터의 방향은 변하지 않고 크기만 변한다
# 하지만 음수를 곱할 경우에는 벡터의 방향이 점대칭으로 뒤집어 진다.

# 4.1.3 스팬과 선형종속
# 가능한 벡터의 전체 공간을 Span
# 서로 방향이 다른 벡터 두 개가 있을 때, 두 벡터는 선형 독립이며, 스팬이 무한

# 두 벡터가 같은 방향으로 존재하거나 같은 선상에 존재하면 스팬이 제한되고 어떻게 조작해도 같은 선위에 놓이게 됨: 선형 종속(lieanrly dependent)

# 4.2 선형 변환
# 함수와 같은 방식으로 한 벡터를 다른 벡터로 변환하는 선형 변환 Linear Transformation

# 4.2.1 기저벡터
# 두 개의 간단한 i, j를 통해 다른 벡터의 변환을 설명에 사용되는데 이를 '기저벡터'라함
import pandas as pd
i=np.array([1,0])
j=np.array([0,1])
print(i,j)

mat = np.array([i, j])
print(mat)
# [[1 0]
#  [0 1]]
# 행렬은 i,j와 같은 벡터의 모음

# 4.2.2 행렬 벡터 곱셈
"""[x_n, y_n]= [[a, b], [c, d]]*[x, y]
--> [ax+by, cx+dy]
이 공식은 앞서 두 벡터를 더하고 변환을 적용해 벡터 v를 만든 것 처럼 i와 j의 스케일을 조정하고 더하는 연산 공식

이러한 변환은 '점 곱'이라고 함
"""

from numpy import array
basis = array([[3, 0],
               [0, 2]])

v = array([1, 1])
new_v = basis.dot(v)
# dot product수행
print(new_v)
print(new_v.shape)

# 전치 Transpose

i_hat = array([2, 0])
j_hat = array([0, 3])

basis = array([i_hat, j_hat]).transpose() # transpose
print("transpose", basis)

v = array([1, 1])
new_v = basis.dot(v)
print(new_v)
"""
[2, 3] = [2*1+0*1, 0*1+3*1]
"""

# 더 복잡한 변환
i = array([2, 3])
j = array([2, -1])
basis = array([i, j]).T

v = array([2, 1])

new_v2 = basis.dot(v)
print(new_v2)
"""
[6, 5] = [2*2+2*1, 2*3+1*(-1)]
"""

# 4.3 행렬곱셈
# 두 변환 결합

i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

combined = transform2 @ transform1
print("combined matraix \n", combined)

print('\n')
v = array([1, 2])
print(combined.dot(v))

rotated = transform1.dot(v)
sheared = transform2.dot(rotated)
print(sheared)

# 변환을 반대로 적용하기

i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

# 변환 통합, 전단을 먼저 적용하고 그 다음 회전을 적용
combined = transform1 @ transform2
print('통합행렬\n', combined)
v = array([1,2])
print(combined.dot(v))

# 4.4 행렬식
# 공간의 확장 또는 축소

# 행렬식 계산하기
from numpy.linalg import det
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

basis = array([i_hat, j_hat]).T

determinant = det(basis)
print(determinant) # 행렬 공간의 면적

i_hat = array([1, 0])
j_hat = array([1, 1])
basis = array([i_hat, j_hat]).T

determinant = det(basis)
print(determinant)

i_hat = array([-2, 1])
j_hat = array([1, 2])
basis = array([i_hat, j_hat]).T

deter = det(basis)
print(deter) # 이 행렬식은 음수이므로 방향이 뒤집힌 것을 알 수 있다.
# 행렬식이 0이라면, 모든 공간이 더 작은 차원으로 줄어들었다는 것을 알 수 있다.
print(abs(deter))

# 2차원 공간을 1차원 직선으로 축소
from numpy.linalg import det
from numpy import array

i_hat = array([3, -1.5])
j_hat = array([-2, 1])
basis = array([i_hat, j_hat]).T
deter = det(basis)
print(basis)
print(deter) # 선형 종속

# 4.5 특수 행렬
# 4.5.1 정방행렬: 행과 열의 수가 같은 행렬
# 4.5.2 항등행렬: 대각선의 값이 1이고 나머지는 0인 정방행렬
# 4.5.3 역행렬: 다른 행렬의 변환을 취소하는 행렬 
# 행렬과 역행렬을 곱할 경우, 항등행렬이 나오게 된다.

# 4.5.4 대각행렬
# 항등행렬과 비슷하게 대각선에는 0이 아닌 값이 있고 나머지 값은 0인 행렬을 대각행렬이라함

# 4.5.5 삼각행렬: 대각선 위쪽 또는 아래쪽에는 원소값이 있고 나머지는 0인 행렬
#  --> 일반적으로 연립방정식으로 풀기 쉽기 때문에 많은 수치 분석 작업에 선호

# 4.5.6 희소행렬: 0이 대부분이고 -이 아닌 원소가 매우 적은 행렬

# 4.6 연립방정식과 역행렬
# 심파이를 활용한 역행렬과 항등 행렬 만들기 
from sympy import *
"""
4x + 2y + 4z = 44
5x + 3y + 7z = 56
9x + 3y + 6z = 72
"""
A = Matrix([
    [4, 2, 4],
    [5, 3, 7], 
    [9, 3, 6]
])

inverse = A.inv() # 역행렬
print(inverse)

indentity = A*inverse
print(indentity) # 항등행렬
print(inverse.shape)
# 연립방정식 풀기
B = Matrix([
    44,
    56, 
    72
])
print(B.shape)
X = inverse * B
print(X) # Matrix([[2], [34], [-8]])

# 4.7 고유 벡터와 고윳값
""" 
행렬 분해는 인수분해와 마찬가지로 행렬을 기본 구성요소로 분해하는 것
고윳값 분해에는 '람다'로 표시되는 고윳값과 v로 표시되는 고유 벡터 두 개의 구성 요소가 있음
정방행렬 A가 있다면, 
Av = 람다v
"""

# 넘파이에서 고윳값 분해 수행
from numpy import array, diag
from numpy.linalg import eig, inv

A = array([
    [1, 2], 
    [4, 5]
])
eigenvals, eigenvecs = eig(A)

print('고윳값', eigenvals)
print('고유벡터', eigenvecs)

# 행렬 재구성
Q = eigenvecs
R = inv(Q)

L = diag(eigenvals)
B = Q@L@R
print(B)