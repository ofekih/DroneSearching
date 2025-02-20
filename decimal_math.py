from decimal import Decimal, getcontext

def pi():
    """Compute Pi to the current precision.

    >>> print(pi())
    3.141592653589793238462643383

    """
    getcontext().prec += 2  # extra digits for intermediate steps
    three = Decimal(3)      # substitute "three=3.0" for regular floats
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n+na, na+8
        d, da = d+da, da+32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return Decimal(+s)               # unary plus applies the new precision

def exp(x: Decimal):
    """Return e raised to the power of x.  Result type matches input type.

    >>> print(exp(Decimal(1)))
    2.718281828459045235360287471
    >>> print(exp(Decimal(2)))
    7.389056098930650227230427461
    >>> print(exp(2.0))
    7.38905609893
    >>> print(exp(2+0j))
    (7.38905609893+0j)

    """
    getcontext().prec += 2
    i, lasts, s, fact, num = 0, 0, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 1
        fact *= i
        num *= x
        s += num / fact
    getcontext().prec -= 2
    return +s

def cos(x: Decimal):
    """Return the cosine of x as measured in radians.

    The Taylor series approximation works best for a small value of x.
    For larger values, first compute x = x % (2 * pi).

    >>> print(cos(Decimal('0.5')))
    0.8775825618903727161162815826
    >>> print(cos(0.5))
    0.87758256189
    >>> print(cos(0.5+0j))
    (0.87758256189+0j)

    """
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return Decimal(+s)

def sin(x: Decimal):
    """Return the sine of x as measured in radians.

    The Taylor series approximation works best for a small value of x.
    For larger values, first compute x = x % (2 * pi).

    >>> print(sin(Decimal('0.5')))
    0.4794255386042030002732879352
    >>> print(sin(0.5))
    0.479425538604
    >>> print(sin(0.5+0j))
    (0.479425538604+0j)

    """
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s

def log2(x: Decimal):
	"""Return the base-2 logarithm of x.

	>>> print(log2(Decimal('0.5')))
	-1
	>>> print(log2(Decimal(1)))
	0
	>>> print(log2(Decimal(2)))
	1
	>>> print(log2(3))
	1.58496250072
	>>> print(log2(3+0j))
	(1.58496250072+0j)

	"""
	return x.ln() / Decimal(2).ln()

def asin(x: Decimal):
    """Return the arc sine (inverse sine) of x in radians.
    
    >>> print(asin(Decimal('0')))
    0
    >>> print(asin(Decimal('0.5')))
    0.5235987755982988730771072305
    >>> print(asin(Decimal('1')))
    1.570796326794896619231321692
    """
    if abs(x) > 1:
        raise ValueError("math domain error: asin(x) is only defined for |x| <= 1")
    
    if x == 1:
        return pi() / 2
    if x == -1:
        return -pi() / 2
    
    getcontext().prec += 2
    i, lasts, s = Decimal(0), 0, x
    while s != lasts:
        lasts = s
        i += 1
        term = Decimal(1)
        for j in range(1, 2 * int(i) + 1, 2):
            term *= Decimal(j) / Decimal(j + 1)
        s += term * (x ** (2 * i + 1)) / Decimal(2 * i + 1)
    
    getcontext().prec -= 2
    return +s

def acos(x: Decimal):
    """Return the arc cosine (inverse cosine) of x in radians.
    
    >>> print(acos(Decimal('1')))
    0
    >>> print(acos(Decimal('0')))
    1.570796326794896619231321692
    >>> print(acos(Decimal('-1')))
    3.141592653589793238462643383
    """
    if abs(x) > 1:
        raise ValueError("math domain error: acos(x) is only defined for |x| <= 1")
    return pi() / 2 - asin(x)

def atan(x: Decimal) -> Decimal:
    """Return the arc tangent (inverse tangent) of x in radians.
    
    >>> print(atan(Decimal('0')))
    0
    >>> print(atan(Decimal('1')))
    0.7853981633974483096156608458
    >>> print(atan(Decimal('-1')))
    -0.7853981633974483096156608458
    """
    if x == 0:
        return Decimal('0')
    
    # For better convergence, use the identity:
    # atan(x) = 2 * atan(x/(1 + sqrt(1 + x^2)))
    if abs(x) > Decimal('0.5'):
        getcontext().prec += 2
        x2 = x * x
        v = x / (Decimal(1) + (Decimal(1) + x2).sqrt())
        result = Decimal(2) * atan(v)
        getcontext().prec -= 2
        return +result
    
    getcontext().prec += 2
    i, lasts, s = 0, 0, x
    x2 = -x * x
    while s != lasts:
        lasts = s
        i += 1
        s += x * (x2 ** i) / (2 * i + 1)
    
    getcontext().prec -= 2
    return +s