Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

Question: Convert the point $(0, -3 \sqrt{3}, 3)$ in rectangular coordinates to spherical coordinates.  Enter your answer in the form $(\rho,\theta,\phi),$ where $\rho > 0,$ $0 \le \theta < 2 \pi,$ and $0 \le \phi \le \pi.$

Solution:
To convert the point $(x, y, z)$ in rectangular coordinates to spherical coordinates $(\rho, \theta, \phi)$, we use the following formulas:
1. $\rho = \sqrt{x^2 + y^2 + z^2}$
2. $\theta = \arctan{\frac{y}{x}}$
3. $\phi = \arccos{\frac{z}{\rho}}$

Let's write a function:

```python
from sympy import sqrt, atan2, acos, pi

def rectangular_to_spherical():
    x, y, z = 0, -3*sqrt(3), 3
    rho = sqrt(x**2 + y**2 + z**2)
    theta = atan2(y, x)
    phi = acos(z/rho)
    return rho, theta, phi

spherical_coordinates = rectangular_to_spherical()
print(spherical_coordinates)
```
```output
(6, -pi/2, pi/3)
```

The required form of the answer is $(\rho, \theta, \phi)$, where $\rho > 0$, $0 \le \theta < 2\pi$, and $0 \le \phi \le \pi$. Since our initial output has $\theta = -\pi/2$, which doesn't meet the condition for $\theta$, we add $2\pi$ to adjust its value.

Hence, the spherical coordinates for the point $(0, -3\sqrt{3}, 3)$ are $\boxed{(6, \frac{3\pi}{2}, \frac{\pi}{3})}$.

---

Question: Compute $1011_2 + 101_2 - 1100_2 + 1101_2$.  Express your answer in base 2.

Solution:
Let's write a single python function for binary computation:

```python
def binary_sum_diff():
    # Convert binary to decimal
    num1 = int("1011", 2)
    num2 = int("101", 2)
    num3 = int("1100", 2)
    num4 = int("1101", 2)

    # Perform the operations
    result = num1 + num2 - num3 + num4

    # Convert the result back to binary
    result_binary = format(result, "b")

    return result_binary

result = binary_sum_diff()
print(result)
```
```output
10001
```
The answer expressed in base 2 is $\boxed{10001_2}$.

---

Question: Find the matrix that corresponds to projecting onto the vector $\begin{pmatrix} 2 \\ -3 \end{pmatrix}.$

Solution:
```python
from sympy import Matrix, simplify

def projection_matrix():
    v = Matrix([2, -3])

    # Calculate the projection matrix
    P = (v * v.T) / (v.T * v)[0]

    # Simplify the elements
    P = P.applyfunc(simplify)

    return P

result = projection_matrix()
print(result)
```
```output
Matrix([[4/13, -6/13], [-6/13, 9/13]])
```

The matrix is $\boxed{\begin{pmatrix} \frac{4}{13} & -\frac{6}{13} \ -\frac{6}{13} & \frac{9}{13} \end{pmatrix}}.$

---

Question: Solve the inequality
\[\frac{x^2 - 25}{x + 5} < 0.\]

Solution:
Let's use `sympy` to solve the inequality:

```python
from sympy import symbols, simplify
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.core.relational import LessThan

def solve_inequality():
    x = symbols('x')
    expression = (x**2 - 25) / (x + 5)
    inequality = LessThan(expression, 0)
    solution = solve_univariate_inequality(inequality, x, relational=False)
    simplified_solution = simplify(solution)
    
    return simplified_solution

result = solve_inequality()
print(result)
```
```output
Union(Interval.open(-oo, -5), Interval.Lopen(-5, 5))
```
The answer is $\boxed{(-\infty,-5)\cup(-5,5)}$

---

Question: In the figure, triangles $ABC$ and $BCD$ are equilateral triangles. What is the value of $AD \div BC$ when expressed in simplest radical form?

[asy]
draw((0,0)--(5,8.7)--(10,0)--cycle);
draw((10,0)--(15,8.7)--(5,8.7));
label("$A$",(0,0),SW);
label("$B$",(5,8.7),N);
label("$C$",(10,0),SE);
label("$D$",(15,8.7),NE);
[/asy]

Solution:
```python
from sympy import Rational, sqrt, simplify

def ad_divided_by_bc():

    x = Rational(1, 1)  # Side length of equilateral triangles

    ad_squared = 2 * x**2 * (1 + Rational(1, 2))  # Using the law of cosines with cos(2*pi/3) = -1/2
    ad = sqrt(ad_squared)

    bc = x # BC is the side length of the equilateral triangles

    simplified_ratio = simplify(ad / bc)

    return simplified_ratio

result = ad_divided_by_bc()
print(result)
```
```output
sqrt(3)
```
The value of $AD \div BC$ is $\boxed{\sqrt{3}}$.

---
