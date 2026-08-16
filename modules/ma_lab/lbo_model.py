"""
LBO Valuation Engine
=====================
Interactive LBO model for teaching: enter assumptions → project cash flows → compute IRR/MOIC.

Based on Kaplan & Strömberg (2009) "Leveraged Buyouts and Private Equity".
"""

import numpy as np
import pandas as pd
from typing import Dict


class LBOModel:
    """
    Simplified LBO valuation engine.
    
    Steps:
    1. Project revenue/EBITDA over hold period.
    2. Capital structure: debt + equity at entry.
    3. Annual debt paydown from FCF.
    4. Exit at exit EV/EBITDA multiple.
    5. Compute equity value at exit → IRR/MOIC.
    """
    
    def __init__(self, assumptions: Dict):
        self.a = assumptions
    
    def project(self) -> pd.DataFrame:
        """Build LBO projection DataFrame year by year."""
        a = self.a
        years = list(range(a["hold_years"] + 1))
        
        # Revenue grows annually
        revenue = [a["entry_revenue_b"]]
        for y in range(1, a["hold_years"] + 1):
            revenue.append(revenue[-1] * (1 + a["revenue_growth"]))
        
        # EBITDA margin expands
        margins = [a["entry_ebitda_margin"] + a["margin_expansion"] * y
                   for y in range(a["hold_years"] + 1)]
        ebitda = [r * m for r, m in zip(revenue, margins)]
        
        # Entry capital structure
        entry_ebitda = a["entry_revenue_b"] * a["entry_ebitda_margin"]
        entry_ev = entry_ebitda * a["entry_ev_ebitda"]
        debt_entry = entry_ev * a["debt_pct"]
        equity_entry = entry_ev - debt_entry
        
        # Annual debt schedule
        debt_balance = [debt_entry]
        interest = []
        fcf_available = []
        
        for y in range(1, a["hold_years"] + 1):
            # Interest on opening balance
            i = debt_balance[-1] * a["interest_rate"]
            interest.append(i)
            
            # FCF approximation: EBITDA - interest - taxes (simplified)
            taxable_income = ebitda[y] - i
            tax = max(0, taxable_income) * a["tax_rate"]
            fcf = ebitda[y] - i - tax
            fcf_available.append(fcf)
            
            # Pay down debt with FCF (assume 100% sweep for simplicity)
            new_balance = max(0, debt_balance[-1] - fcf)
            debt_balance.append(new_balance)
        
        # Exit valuation
        exit_ebitda = ebitda[-1]
        exit_ev = exit_ebitda * a["exit_ev_ebitda"]
        exit_debt = debt_balance[-1]
        exit_equity = exit_ev - exit_debt
        
        # Returns
        moic = exit_equity / equity_entry if equity_entry > 0 else 0
        irr = (moic ** (1 / a["hold_years"]) - 1) if moic > 0 else -1
        
        df = pd.DataFrame({
            "Year": years,
            "Revenue ($B)": [round(r, 2) for r in revenue],
            "EBITDA Margin (%)": [round(m * 100, 1) for m in margins],
            "EBITDA ($B)": [round(e, 2) for e in ebitda],
            "Debt Balance ($B)": [round(d, 2) for d in debt_balance],
            "Interest ($B)": [round(i, 2) if i else 0 for i in [0] + interest],
            "FCF ($B)": [round(f, 2) if f else 0 for f in [0] + fcf_available],
        })
        
        self.summary = {
            "entry_ev_b": round(entry_ev, 2),
            "entry_debt_b": round(debt_entry, 2),
            "entry_equity_b": round(equity_entry, 2),
            "entry_debt_pct": round(a["debt_pct"] * 100, 1),
            "exit_ev_b": round(exit_ev, 2),
            "exit_debt_b": round(exit_debt, 2),
            "exit_equity_b": round(exit_equity, 2),
            "moic": round(moic, 2),
            "irr_pct": round(irr * 100, 1),
            "hold_years": a["hold_years"],
        }
        return df
