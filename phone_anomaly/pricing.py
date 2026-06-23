"""Simple resale-price estimator.

Price is a base value per model, depreciated by age, then discounted by the
condition grade. Amounts are in 만원 (10,000 KRW).
"""

# base price (만원) for a mint device at launch
MODEL_BASE_PRICE = {
    "iPhone 14 Pro": 135,
    "iPhone 13 Pro": 110,
    "iPhone 13": 90,
    "iPhone 12 Pro": 80,
    "iPhone 12": 65,
    "iPhone 11": 50,
    "Galaxy S23": 110,
    "Galaxy S22": 80,
    "Galaxy S21": 60,
}

DEPRECIATION_PER_YEAR = 0.18
GRADE_DISCOUNT = {"S": 0.0, "A": 0.10, "B": 0.25, "C": 0.45}


def estimate_price(model_name, years_old, grade):
    """Estimate resale price (만원) for a model / age / grade."""
    base = MODEL_BASE_PRICE.get(model_name, 80)
    price = base * ((1 - DEPRECIATION_PER_YEAR) ** years_old)
    price *= 1 - GRADE_DISCOUNT[grade]
    return max(5, int(price))


def price_range(price, margin=0.15):
    """Return a ``(low, high)`` band around ``price``."""
    return int(price * (1 - margin)), int(price * (1 + margin))
