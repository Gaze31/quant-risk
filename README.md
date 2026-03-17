# Quantitative Risk & Portfolio Optimization

A Python toolkit covering portfolio optimization, risk measurement, and factor models — from Modern Portfolio Theory to CVaR and Fama-French. Built to understand how institutional risk management and portfolio construction actually works under the hood.

---

## Modules

| Module | File | Description |
|---|---|---|
| Efficient Frontier | `eff_fronplot.py` | MPT optimization with Max Sharpe, Min Vol |
| Portfolio Optimization | `optimze_port.py` | Constrained optimization via scipy/cvxpy |
| CAPM | `capm_mod.py` | Beta estimation, SML, alpha calculation |
| Fama-French 3-Factor | `fama_french3.py` | Factor loading and regression |
| Multi-Factor Model | `pymulti_fac.py` | Custom factor construction |
| VaR & CVaR | `cvar_var.py` | Historical, parametric, Monte Carlo |
| Risk Metrics | `riskpy_metrices.py` | Sharpe, Sortino, Calmar, Max Drawdown |
| Covariance Matrix | `covariance_matrix.py` | Ledoit-Wolf shrinkage estimation |
| Credit Scoring | `credit_scoring.py` | PD estimation, KMV model |
| Historical Simulation | `historocal_simulation.py` | Non-parametric VaR |

---

## Sample Output

### Efficient Frontier and Optimal Portfolios
![Efficient Frontier](results/efficient_frontier.png)

Shows the mean-variance efficient frontier with individual assets plotted. Key portfolios marked: Maximum Sharpe Ratio (red star), Minimum Volatility (green star), and Equal Weight (orange diamond).

### Portfolio Optimization Exercise
![Portfolio Optimization](results/exercise1_portfolio_optimization.png)

---

## Key Concepts Covered

**Modern Portfolio Theory**
- Mean-variance optimization
- Efficient frontier construction
- Maximum Sharpe ratio portfolio
- Minimum volatility portfolio
- Portfolio constraints (long-only, weight bounds)

**Risk Measurement**
- Value at Risk (VaR) — historical, parametric, Monte Carlo
- Conditional VaR (CVaR/Expected Shortfall)
- Maximum drawdown and recovery analysis
- Stress testing

**Factor Models**
- CAPM: single-factor beta regression
- Fama-French 3-factor: market, size (SMB), value (HML)
- Multi-factor: custom factor construction and loading estimation

**Credit Risk**
- KMV model for default probability
- Credit scoring with logistic regression
- Distance to default calculation

---

## Setup

```bash
git clone https://github.com/Gaze31/quant-risk.git
cd quant-risk
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python eff_fronplot.py
```

---

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
cvxpy>=1.3.0
scikit-learn>=1.3.0
yfinance>=0.2.0
```

---

## Known Limitations

- Efficient frontier uses sample covariance — sensitive to estimation error in small samples
- CAPM assumes single-factor world — does not capture size, value, or momentum premia
- VaR models assume return distributions that may not hold during tail events
- Credit scoring model trained on sample data — not validated on real loan data

---

## Next Steps

- [ ] Black-Litterman model for incorporating views
- [ ] Risk parity portfolio construction
- [ ] Regime-switching risk models
- [ ] Backtested factor portfolio performance

---

## Author

**Sumedha Hundekar** — Finance graduate building quantitative risk tools in Python.  
Contact: velvetgazeze@gmail.com
