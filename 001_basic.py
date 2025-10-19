# 선형함수
def f(x):
    return 2*x +1

x_values = [0, 1, 2, 3, 4]

for x in x_values:
    y = f(x)
    print(y)

from sympy import *
x = symbols('x')
f = 2*x+1
# plot(f)
"""
     21 |                                                     ..
        |                                                  ...
        |                                                ..
        |                                             ...
        |                                          ...
        |                                        ..
        |                                     ...
        |                                  ...
        |                                ..
        |                             ...
      1 |--------------------------...--------------------------
        |                       ...
        |                     ..
        |                  ...
        |               ...
        |             ..
        |          ...
        |       ...
        |     ..
        |  ...
    -19 |_______________________________________________________
         -10                        0                          10
"""

x = symbols('x')
f = x**2 + 1
# plot(f)
"""
    100 |\                                                     /
        | .                                                   .
        |
        |  .                                                 .
        |   \                                               /
        |    .                                             .
        |
        |     .                                           .
        |      \                                         /
        |       \                                       /
     50 |--------\-------------------------------------/--------
        |         \                                   /
        |          \                                 /
        |           \                               /
        |            ..                           ..
        |              \                         /
        |               ..                     ..
        |                 \                   /
        |                  ...             ...
        |                     ...       ...
      0 |_______________________________________________________
         -10                        0                          10

"""

# 3d도 가능
from sympy.plotting import plot3d

x,y = symbols('x y')
f = 2*x + 3*y
# plot3d(f)

# 시그마
summation = sum(2*i for i in range(1, 6))
print(summation)

x = [1, 4, 6, 2]
n = len(x)
summation = sum(10*x[i] for i in range(0, n))
print(summation)

i, n = symbols('i n')
summation = Sum(2*i, (i, 1, n))
up_to_5 = summation.subs(n, 5)
print(up_to_5.doit())


# 거듭제곱
x = symbols('x') # sympy를 통해 x를 str로 인식이 아니라 변수로 인식할 수 있도록 변환
expr = x**2/ x**5
print(expr)

# 로그 
from math import log
x = log(8, 2)
print(x)

# 오일러 수 # e
# 이자율을 통한 오일러 수 e 이해하기

from math import exp

p = 100 # 원금
r = .2  # 이율
t = 2.0 # 기간
n = 12  # 1년 12달

a = p * (1 + (r/n)) ** (n*t) # 복리 이자
print(a)

a = p*exp(r*t) # 연속 이자
print(a) 

# 자연로그
from math import log

x = log(10) # e의 지수x의 값이 10이 되는 수
print(x)

# 극한
x = symbols('x')
f = 1/x
result = limit(f, x, oo)
print(result)

n = symbols('n')
f = (1+(1/n))**n
result = limit(f, n, oo)
print(result)
print(result.evalf())

# 미분

def derivative_x(f, x, step_size): # 함수 f, x 값, 변화율 step_size
    m = (f(x+step_size) - f(x))/((x + step_size) - x)
    return m

def my_funtion(x):
    return x**2

slope_at_2 = derivative_x(my_funtion, 2, 1e-5)
print(slope_at_2)

## 심파이 활용한 도함수 도출
x = symbols('x')
f = x**2
dx_f= diff(f)
print(dx_f)

### 파이썬 미분 계산기

def f(x):
    return x**2

def d_f(x):
    return 2*x

slope_at_2 = d_f(2.)
print(slope_at_2)

# 편도함수
x, y = symbols('x y')
f = 2*x**3 + 3*y**3

dx_f = diff(f, x)
dy_f = diff(f, y)
print(dx_f)
print(dy_f)

# plot3d(f)

# 극한을 활용한 기울기 계산
x, s = symbols('x s')

f = x**2

# 간격 's' 만큼 떨어진 두 점을 기울기 공식에 대입
slope_f = (f.subs(x, x + s) - f) / ((x+s) - x) # .subs 특정 변수나 값을 다른 값이나 표현식으로 바꾸는 역할 

# x에 2 대입
slope_2 = slope_f.subs(x, 2)

result = limit(slope_2, s, 0)
print(result)

### 극한을 사용한 도함수 도출
x, s = symbols('x s')

f = x**2

slope_f = (f.subs(x, x+s)-f)/ ((x+s) - x)

result = limit(slope_f, s, 0)
print(result)

# 연쇄법칙 Chain Rule
x = symbols('x')
z = (x**2 + 1)**3-2
dz_dx = diff(z, x)
print(dz_dx)

x, y = symbols('x y')
_y = x**2 + 1
dy_dx = diff(_y)

z = y**3-2
dz_dy = diff(z)

dz_dx_chain = (dy_dx * dz_dy).subs(y, _y)
dz_dx_no_chain = diff(z.subs(y, _y))

print(dz_dx_chain)
print(dz_dx_no_chain)

# 직사각형을 활용한 적분 근사값

def approximate_integral(a, b, n, f):
    delta_x = (b-a)/n
    total_sum = 0

    for i in range(1, n+1):
        midpoint = .5 * (2*a + delta_x * (2 * i-1))
        total_sum += f(midpoint)

    return total_sum * delta_x

def my_function(x):
    return x**2+1

area = approximate_integral(a=0, b=1, n=5, f=my_function)
print(area)


# 심파이를 활용한 적분 
x = symbols('x')
f = x**2+1

area = integrate(f, (x, 0, 1)) # 0~1 사이에 면적 구하기
print(area)

# 극한을 사용한 적분
x, i , n = symbols('x i n')
f = x**2 + 1
lower, upper = 0, 1

delta_x = ((upper-lower) / n)
x_i = (lower + delta_x * i)
fx_i = f.subs(x, x_i)

n_retangles = Sum(delta_x * fx_i, (i, 1, n)).doit()
area = limit(n_retangles, n, oo)
print(area)