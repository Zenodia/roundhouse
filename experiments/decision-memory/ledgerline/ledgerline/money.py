from __future__ import annotations


def format_cents(cents: int, currency: str = "USD") -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d} {currency}"


def split_cents(total: int, parts: int) -> list[int]:
    """Split without losing a cent: the remainder goes to the first parts."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]
