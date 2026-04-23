"""
Paper Replication Lab Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import PAPERS as CLASSIC_PAPERS
from utils.helpers import generate_synthetic_returns


def render():
    st.markdown('<div class="main-header"><h1>📚 Paper Replication Lab</h1><p>One-click replication of classic monetary policy papers</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    Replicate foundational papers in the monetary policy announcement literature.
    Each replication includes the original methodology, key results, and code.
    """)
    
    # ── Paper Selection ──
    papers = list(CLASSIC_PAPERS.keys())
    selected = st.selectbox("Select Paper to Replicate", papers)
    
    paper = CLASSIC_PAPERS[selected]
    
    # ── Paper Info ──
    st.markdown(f"### 📄 {selected}")
    st.markdown(f"**{paper['title']}**")
    st.markdown(f"*{paper['journal']}*")
    st.markdown(f"**Method**: {paper['method']}")
    st.markdown(f"**Key Result**: {paper['key_result']}")
    
    st.markdown("---")
    
    # ── Replication Steps ──
    if "Kuttner" in selected:
        _replicate_kuttner()
    elif "Bernanke" in selected:
        _replicate_bernanke_kuttner()
    elif "Gürkaynak" in selected:
        _replicate_gurkaynak()
    elif "Jarociński" in selected:
        _replicate_jarocinski()
    elif "Nakamura" in selected:
        _replicate_nakamura()


def _replicate_kuttner():
    """Replicate Kuttner (2001): Monetary policy surprises from Fed funds futures."""
    st.markdown("## Replication: Kuttner (2001)")
    
    st.markdown("""
    ### Step 1: Data Requirements
    - Fed funds futures prices (CME, daily)
    - FOMC meeting dates
    - Target rate changes (FRED: DFF)
    
    ### Step 2: Compute Surprise
    The surprise is the change in the implied rate from the day before to the day of the FOMC meeting:
    
    $$S_t = \\frac{f_{t,d} - f_{t-1,d}}{100}$$
    
    where $f_{t,d}$ is the futures price for the month of the FOMC meeting.
    
    ### Step 3: Regression
    $$\\Delta y_i = \\alpha + \\beta \\cdot S_t + \\epsilon_t$$
    
    where $\\Delta y_i$ is the change in yield for maturity $i$.
    
    ### Expected Results
    | Maturity | $\\beta$ (bp/bp) | t-stat |
    |----------|:---:|:---:|
    | 3-month | ~1.0 | >5.0 |
    | 2-year | ~0.8 | >5.0 |
    | 5-year | ~0.5 | >3.0 |
    | 10-year | ~0.3 | >2.0 |
    """)
    
    st.markdown("### Python Code")
    st.code("""
import pandas as pd
import numpy as np
from statsmodels.api import OLS

# Step 1: Load Fed funds futures data
# Source: CME via Bloomberg/WRDS
futures = pd.read_csv('fed_funds_futures.csv', parse_dates=['date'])

# Step 2: Identify FOMC dates
fomc_dates = pd.to_datetime([
    '2001-01-31', '2001-03-20', '2001-05-15', ...
])

# Step 3: Compute surprise
surprises = []
for fomc in fomc_dates:
    # Find the contract for the FOMC month
    contract_month = fomc + pd.DateOffset(months=1)
    # Day-before and day-of futures prices
    day_before = futures[
        (futures['date'] == fomc - pd.Timedelta(days=1)) &
        (futures['contract'] == contract_month)
    ]['price'].values[0]
    day_of = futures[
        (futures['date'] == fomc) &
        (futures['contract'] == contract_month)
    ]['price'].values[0]
    surprise = (day_of - day_before) / 100
    surprises.append({'date': fomc, 'surprise': surprise})

surprise_df = pd.DataFrame(surprises)

# Step 4: Regression
yields = pd.read_csv('treasury_yields.csv', parse_dates=['date'])
merged = surprise_df.merge(yields, on='date')

for maturity in ['3M', '2Y', '5Y', '10Y']:
    y = merged[f'delta_{maturity}']
    X = merged[['surprise']]
    X = sm.add_constant(X)
    model = OLS(y, X).fit()
    print(f"{maturity}: beta = {model.params['surprise']:.4f}, "
          f"t = {model.tvalues['surprise']:.2f}")
    """, language="python")
    
    st.info("💡 **Data Source**: Fed funds futures from CME. Free via FRED API (limited) or Bloomberg/WRDS (full).")


def _replicate_bernanke_kuttner():
    """Replicate Bernanke & Kuttner (2005): Stock market reaction decomposition."""
    st.markdown("## Replication: Bernanke & Kuttner (2005)")
    
    st.markdown("""
    ### Key Innovation
    Decompose stock market reaction into:
    1. **Discount rate channel** (~75% of reaction)
    2. **Cash flow / expected future dividends** (~25%)
    
    ### Methodology
    Use the Campbell-Shiller (1988) log-linear present value model:
    
    $$r_{t+1} \\approx \\Delta d_{t+1} - \\rho \\cdot \\Delta p_{t+1} + \\rho \\cdot r_{t+1}$$
    
    ### Regression Specification
    $$\\Delta S\\&P_t = \\alpha + \\beta_1 \\cdot S_t + \\beta_2 \\cdot \\Delta FF_t + \\epsilon_t$$
    
    where $S_t$ is the Kuttner surprise and $\\Delta FF_t$ is the actual target change.
    
    ### Expected Results
    - Unexpected rate increase of 25bp → S&P 500 declines ~1%
    - Discount rate channel explains ~75% of the reaction
    - Cash flow channel explains ~25%
    """)
    
    st.code("""
import pandas as pd
import numpy as np
from statsmodels.api import OLS
import statsmodels.api as sm

# Load data
surprises = pd.read_csv('kuttner_surprises.csv', parse_dates=['date'])
sp500 = pd.read_csv('sp500_daily.csv', parse_dates=['date'])

# Merge
df = surprises.merge(sp500, on='date')

# Event window: day of FOMC
df['sp500_return'] = df['sp500_close'].pct_change()

# Regression: stock return on surprise
y = df['sp500_return']
X = sm.add_constant(df['surprise'])
model = OLS(y, X, missing='drop').fit()
print(model.summary())

# Key coefficient: beta ≈ -40 to -50
# Interpretation: 25bp surprise → -1.0% to -1.25% stock return
    """, language="python")


def _replicate_gurkaynak():
    """Replicate Gürkaynak et al. (2005): Target vs Path factor."""
    st.markdown("## Replication: Gürkaynak et al. (2005)")
    
    st.markdown("""
    ### Key Innovation
    Decompose monetary policy surprises into TWO factors:
    1. **Target Factor**: Unexpected change in current FFR target
    2. **Path Factor**: Unexpected change in future policy path
    
    ### Methodology
    Principal Component Analysis (PCA) on changes in Eurodollar futures 
    across multiple maturities around FOMC announcements.
    
    ### Identification
    - Target factor: loads most on near-term futures (1-3 months)
    - Path factor: loads most on longer-term futures (6-24 months)
    
    ### Expected Results
    - Path factor > Target factor for long-term bond yields
    - Equity prices respond more to path factor
    - Exchange rates respond to both
    """)
    
    st.code("""
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Load Eurodollar futures changes around FOMC
# Maturities: 1M, 2M, 3M, 6M, 9M, 12M, 18M, 24M
eurodollar_changes = pd.read_csv('eurodollar_fomc.csv')

# PCA with 2 components
pca = PCA(n_components=2)
factors = pca.fit_transform(eurodollar_changes.iloc[:, 1:])

# Factor 1 ≈ Target (loads on short end)
# Factor 2 ≈ Path (loads on long end)
print("Explained variance:", pca.explained_variance_ratio_)
print("Loadings:")
for i, loading in enumerate(pca.components_):
    print(f"Factor {i+1}: {loading.round(3)}")

# Regression: asset returns on both factors
for asset in ['sp500', '10y_yield', 'dxy']:
    y = df[asset]
    X = sm.add_constant(pd.DataFrame(factors, columns=['target', 'path']))
    model = OLS(y, X).fit()
    print(f"\\n{asset}:")
    print(f"  Target: {model.params['target']:.4f} (t={model.tvalues['target']:.2f})")
    print(f"  Path:   {model.params['path']:.4f} (t={model.tvalues['path']:.2f})")
    """, language="python")


def _replicate_jarocinski():
    """Replicate Jarociński & Karadi (2020): Two-shocks decomposition."""
    st.markdown("## Replication: Jarociński & Karadi (2020)")
    
    st.markdown("""
    ### Key Innovation
    FOMC surprises contain TWO types of shocks:
    1. **Policy Shock**: Pure monetary policy change
    2. **Information Shock**: Central bank reveals info about the economy
    
    ### Identification (High-Frequency)
    Use 30-minute window around FOMC:
    - Equity return = f(policy_shock, info_shock)
    - Bond yield change = g(policy_shock, info_shock)
    
    System:
    $$\\begin{bmatrix} \\Delta r_{eq} \\\\ \\Delta y_{bond} \\end{bmatrix} = 
    \\begin{bmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{bmatrix}
    \\begin{bmatrix} \\epsilon^{policy} \\\\ \\epsilon^{info} \\end{bmatrix}$$
    
    Sign restrictions:
    - Policy shock: equity↓, bond yield↑
    - Info shock: equity↑, bond yield↓ (good news about economy)
    
    ### Expected Results
    - Information shocks explain why equities sometimes RISE on tightening
    - Policy shocks dominate for bonds
    - Information shocks dominate for equities
    """)
    
    st.code("""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# High-frequency data: 30-min window around FOMC
# Columns: equity_return, bond_yield_change, fed_funds_surprise
hf_data = pd.read_csv('fomc_highfreq.csv')

# Step 1: Estimate VAR
model = VAR(hf_data[['equity_return', 'bond_yield_change']])
results = model.fit(maxlags=0)  # No lags for contemporaneous identification

# Step 2: Identify with sign restrictions
# Policy shock: equity↓, bond yield↑
# Info shock: equity↑, bond yield↓

# Reduced-form residuals
residuals = results.resid
cov_matrix = np.cov(residuals.T)

# Cholesky decomposition (order: equity, bond)
# This gives one identification; for sign restrictions, use Uhlig (2005)
P = np.linalg.cholesky(cov_matrix)

# Structural shocks
structural_shocks = np.linalg.solve(P, residuals.T).T
structural_shocks.columns = ['shock1', 'shock2']

# Check sign patterns to label shocks
# shock1: if equity↓ and bond↑ → policy shock
# shock2: if equity↑ and bond↓ → information shock
    """, language="python")


def _replicate_nakamura():
    """Replicate Nakamura & Steinsson (2018): High-frequency monetary non-neutrality."""
    st.markdown("## Replication: Nakamura & Steinsson (2018)")
    
    st.markdown("""
    ### Key Innovation
    Show that monetary policy is **non-neutral** even at high frequency:
    - Real interest rates respond to monetary policy
    - Expected inflation responds
    - These responses are persistent
    
    ### Methodology
    Use TIPS (Treasury Inflation-Protected Securities) to separate:
    - Nominal rate response
    - Real rate response  
    - Break-even inflation response
    
    ### Key Equation
    $$i_t = r_t + \\pi_t^e$$
    
    where $i_t$ = nominal rate, $r_t$ = real rate, $\\pi_t^e$ = expected inflation.
    
    ### Expected Results
    - 25bp tightening → real rate rises ~15bp, inflation expectations fall ~10bp
    - Responses are persistent (not just transitory)
    - Challenges the "money neutrality" view
    """)
    
    st.code("""
import pandas as pd
import numpy as np
from statsmodels.api import OLS
import statsmodels.api as sm

# Load TIPS and nominal Treasury data around FOMC
tips_data = pd.read_csv('tips_fomc.csv', parse_dates=['date'])
nominal_data = pd.read_csv('nominal_fomc.csv', parse_dates=['date'])

# Merge
df = tips_data.merge(nominal_data, on='date')
df = df.merge(surprise_df, on='date')

# Break-even inflation = nominal yield - TIPS yield
df['breakeven_10y'] = df['nominal_10y'] - df['tips_10y']

# Regression: each component on surprise
for component in ['nominal_10y', 'tips_10y', 'breakeven_10y']:
    y = df[component]
    X = sm.add_constant(df['surprise'])
    model = OLS(y, X, missing='drop').fit()
    print(f"{component}: beta = {model.params['surprise']:.4f}, "
          f"t = {model.tvalues['surprise']:.2f}")
    """, language="python")
