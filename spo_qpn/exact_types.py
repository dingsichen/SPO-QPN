"""Exact scalar helpers for the symbolic stabilizer backend."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt


@dataclass(frozen=True)
class ExactComplex:
    real: Fraction = Fraction(0, 1)
    imag: Fraction = Fraction(0, 1)

    @staticmethod
    def from_int(value: int) -> "ExactComplex":
        return ExactComplex(Fraction(value, 1), Fraction(0, 1))

    @staticmethod
    def from_fraction(value: Fraction) -> "ExactComplex":
        return ExactComplex(value, Fraction(0, 1))

    def __add__(self, other: "ExactComplex") -> "ExactComplex":
        return ExactComplex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "ExactComplex") -> "ExactComplex":
        return ExactComplex(self.real - other.real, self.imag - other.imag)

    def __neg__(self) -> "ExactComplex":
        return ExactComplex(-self.real, -self.imag)

    def __mul__(self, other: "ExactComplex") -> "ExactComplex":
        return ExactComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def scale(self, scalar: Fraction) -> "ExactComplex":
        return ExactComplex(self.real * scalar, self.imag * scalar)

    def divide_fraction(self, scalar: Fraction) -> "ExactComplex":
        return ExactComplex(self.real / scalar, self.imag / scalar)

    def conjugate(self) -> "ExactComplex":
        return ExactComplex(self.real, -self.imag)

    def abs_squared(self) -> Fraction:
        return self.real * self.real + self.imag * self.imag

    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0

    def to_complex(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def pretty(self) -> str:
        if self.imag == 0:
            return _pretty_fraction(self.real)
        if self.real == 0:
            return _pretty_imag(self.imag)
        sign = "+" if self.imag >= 0 else "-"
        return f"{_pretty_fraction(self.real)}{sign}{_pretty_imag(abs(self.imag), signed=False)}"


ZERO = ExactComplex.from_int(0)
ONE = ExactComplex.from_int(1)
MINUS_ONE = ExactComplex.from_int(-1)
I = ExactComplex(Fraction(0, 1), Fraction(1, 1))
MINUS_I = ExactComplex(Fraction(0, 1), Fraction(-1, 1))


def sqrt_fraction_string(value: Fraction) -> str:
    if value < 0:
        raise ValueError("Radicand must be non-negative.")
    if value == 0:
        return "0"
    numerator = value.numerator
    denominator = value.denominator
    num_root = isqrt(numerator)
    den_root = isqrt(denominator)
    if num_root * num_root == numerator and den_root * den_root == denominator:
        return _pretty_fraction(Fraction(num_root, den_root))
    outside_num, inside_num = _split_square_factor(numerator)
    outside_den, inside_den = _split_square_factor(denominator)
    outside = Fraction(outside_num, outside_den)
    inside = Fraction(inside_num, inside_den)
    inside_text = _pretty_fraction(inside)
    if outside == 1:
        return f"sqrt({inside_text})"
    return f"{_pretty_fraction(outside)}*sqrt({inside_text})"


def _split_square_factor(value: int) -> tuple[int, int]:
    outside = 1
    inside = value
    factor = 2
    while factor * factor <= inside:
        square = factor * factor
        while inside % square == 0:
            outside *= factor
            inside //= square
        factor += 1
    return outside, inside


def _pretty_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _pretty_imag(value: Fraction, signed: bool = True) -> str:
    if value == 1:
        return "1j" if signed else "1j"
    return f"{_pretty_fraction(value)}j"
