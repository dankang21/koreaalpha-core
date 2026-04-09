"""Rule-based portfolio scoring (0-100) and grading (A+ to F).

Deterministic — same input always produces same output. No AI dependency.

Scoring breakdown:
- Sharpe Ratio: 30 points
- MDD: 25 points
- Benchmark comparison: 15 points
- CAGR: 15 points
- Sortino + Calmar: 15 points
"""

from __future__ import annotations


def calculate_portfolio_score(
    metrics: dict,
    benchmark_comparisons: list[dict] | None = None,
) -> tuple[int, str]:
    """Calculate portfolio score (0-100) and grade (A+ to F).

    Args:
        metrics: Portfolio metrics dict with keys:
            sharpe_ratio, mdd, cagr, sortino_ratio, calmar_ratio
        benchmark_comparisons: Optional list of benchmark comparison dicts with keys:
            sharpe_diff, cagr_diff

    Returns:
        (score, grade) tuple.
    """
    score = 0

    # 1. Sharpe Ratio (30 points)
    sharpe = metrics.get("sharpe_ratio", 0) or 0
    if sharpe >= 1.5:
        score += 30
    elif sharpe >= 1.0:
        score += 25
    elif sharpe >= 0.7:
        score += 20
    elif sharpe >= 0.4:
        score += 15
    elif sharpe >= 0.0:
        score += 8
    else:
        score += 3

    # 2. MDD (25 points) — closer to 0 is better
    mdd = abs(metrics.get("mdd", 0) or 0)
    if mdd <= 0.05:
        score += 25
    elif mdd <= 0.10:
        score += 22
    elif mdd <= 0.15:
        score += 18
    elif mdd <= 0.20:
        score += 14
    elif mdd <= 0.30:
        score += 8
    else:
        score += 3

    # 3. Benchmark comparison (15 points)
    if benchmark_comparisons:
        best_grade_score = 0
        for comp in benchmark_comparisons:
            sharpe_diff = comp.get("sharpe_diff", 0) or 0
            cagr_diff = comp.get("cagr_diff", 0) or 0
            gs = 0
            if sharpe_diff >= 0.3:
                gs += 8
            elif sharpe_diff >= 0:
                gs += 5
            else:
                gs += 2
            if cagr_diff >= 0.02:
                gs += 7
            elif cagr_diff >= 0:
                gs += 4
            else:
                gs += 2
            best_grade_score = max(best_grade_score, gs)
        score += best_grade_score
    else:
        score += 8  # default middle value when no benchmark comparison

    # 4. CAGR (15 points)
    cagr = metrics.get("cagr", 0) or 0
    if cagr >= 0.15:
        score += 15
    elif cagr >= 0.10:
        score += 12
    elif cagr >= 0.05:
        score += 9
    elif cagr >= 0.0:
        score += 5
    else:
        score += 2

    # 5. Sortino + Calmar (15 points)
    sortino = metrics.get("sortino_ratio", 0) or 0
    calmar = metrics.get("calmar_ratio", 0) or 0

    sub = 0
    if sortino >= 2.0:
        sub += 8
    elif sortino >= 1.0:
        sub += 6
    elif sortino >= 0.5:
        sub += 4
    else:
        sub += 2

    if calmar >= 1.0:
        sub += 7
    elif calmar >= 0.5:
        sub += 5
    elif calmar >= 0.2:
        sub += 3
    else:
        sub += 1

    score += sub

    # Grade conversion
    score = max(0, min(100, score))
    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B+"
    elif score >= 60:
        grade = "B"
    elif score >= 50:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    return score, grade
