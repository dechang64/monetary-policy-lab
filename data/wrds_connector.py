"""
WRDS Data Connector for Monetary Policy Research Lab
=====================================================

Integrates Wharton Research Data Services (WRDS) into the research pipeline,
providing access to institutional-grade financial data that FRED cannot offer.

WRDS provides PostgreSQL access via the `wrds` Python package:
    pip install wrds

Connection:
    import wrds
    db = wrds.Connection(wrds_username='your_username')
    # First run prompts for password, stored in ~/.wrds.cfg

Key databases for monetary policy research:
    - CRSP: Daily/monthly stock returns, market indices, mutual funds
    - Compustat: Fundamental financial data (earnings, assets, debt)
    - TAQ / TAQMQS: Intraday trades and quotes (high-frequency identification)
    - IBES: Analyst earnings forecasts (expectations data)
    - OptionMetrics: Options data (implied volatility, VIX construction)
    - FRED (via WRDS): Same FRED data but with SQL flexibility
    - Board of Governors: H.15 rates, G.17 industrial production
    - Philadelphia Fed: Survey of Professional Forecasters
    - Chicago Fed: Financial Conditions Index
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime


class WRDSConnector:
    """
    Connect to WRDS PostgreSQL and fetch monetary policy research data.

    Usage:
        wrds_conn = WRDSConnector(username='your_username')
        # Or set WRDS_USERNAME env var

        # Get intraday data around FOMC
        df = wrds_conn.fetch_taq_window(
            fomc_date='2024-03-20',
            ticker='SPY',
            pre_minutes=30,
            post_minutes=60,
        )

        # Get CRSP daily returns for event study
        df = wrds_conn.fetch_crsp_returns(
            permnos=[14593, 10107],  # SPY, etc.
            start='2020-01-01',
            end='2024-12-31',
        )

        # Get Fed Funds futures from WRDS (more granular than FRED)
        df = wrds_conn.fetch_fed_funds_futures(
            start='2020-01-01',
            end='2024-12-31',
        )
    """

    def __init__(self, username: str = None, auto_connect: bool = True):
        self.username = username or os.environ.get('WRDS_USERNAME', '')
        self._db = None
        self._connected = False

        if auto_connect and self.username:
            self.connect()

    def connect(self):
        """Establish WRDS connection. Prompts for password on first use."""
        try:
            import wrds
            self._db = wrds.Connection(wrds_username=self.username)
            self._connected = True
        except ImportError:
            raise ImportError(
                "wrds package not installed. Run: pip install wrds"
            )
        except Exception as e:
            raise ConnectionError(f"WRDS connection failed: {e}")

    def close(self):
        """Close the WRDS connection."""
        if self._db:
            self._db.close()
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── CRSP: Stock Returns & Market Data ──

    def fetch_crsp_returns(
        self,
        permnos: List[int] = None,
        ticker: str = None,
        start: str = '2000-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch CRSP daily stock returns.

        CRSP provides the gold-standard daily return data used in virtually
        every asset pricing paper. Key advantage over yfinance:
        - Correct delisting returns (yfinance drops delisted stocks)
        - Distribution-adjusted returns (dividends, splits)
        - Full PERMNO/PERMCO tracking

        Args:
            permnos: CRSP permanent numbers (e.g., [14593] for SPY)
            ticker: Ticker symbol (will be resolved to PERMNO)
            start/end: Date range
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        # Resolve ticker to PERMNO if needed
        if ticker and not permnos:
            permnos = self._resolve_ticker(ticker, start)

        if not permnos:
            return pd.DataFrame()

        permno_list = ','.join(str(p) for p in permnos)

        query = f"""
        SELECT
            date,
            permno,
            permco,
            ret,
            prc,
            vol,
            shrout,
            cfacpr,
            cfacshr
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date >= '{start}'
          AND date <= '{end}'
        ORDER BY date, permno
        """

        return self._db.raw_sql(query)

    def fetch_crsp_index(
        self,
        index_type: str = 'vwretd',
        start: str = '2000-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch CRSP market index returns (value-weighted or equal-weighted).

        Available indices:
        - vwretd: Value-weighted return (incl. distributions)
        - ewretd: Equal-weighted return (incl. distributions)
        - vwretx: Value-weighted return (excl. distributions)
        - ewretx: Equal-weighted return (excl. distributions)
        - sprtrn: S&P 500 total return
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        query = f"""
        SELECT date, vwretd, ewretd, vwretx, ewretx, sprtrn
        FROM crsp.dsi
        WHERE date >= '{start}'
          AND date <= '{end}'
        ORDER BY date
        """

        return self._db.raw_sql(query)

    def _resolve_ticker(self, ticker: str, date: str = None) -> List[int]:
        """Resolve ticker symbol to CRSP PERMNO(s)."""
        date_filter = f"AND namedt <= '{date}' AND nameenddt >= '{date}'" if date else ""
        query = f"""
        SELECT DISTINCT permno
        FROM crsp.stocknames
        WHERE ticker = '{ticker.upper()}'
        {date_filter}
        """
        result = self._db.raw_sql(query)
        return result['permno'].tolist() if not result.empty else []

    # ── TAQ: Intraday Data for High-Frequency Identification ──

    def fetch_taq_window(
        self,
        fomc_date: str,
        ticker: str = 'SPY',
        pre_minutes: int = 30,
        post_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch intraday trades and quotes around FOMC announcements.

        This is the KEY upgrade for the Two-Shocks decomposition.
        Current implementation uses daily data → cannot distinguish
        policy vs information shocks precisely.

        With TAQ intraday data:
        - Measure asset response in narrow windows (e.g., 30min post-FOMC)
        - Replicate Jarociński & Karadi (2020) methodology exactly
        - Compute high-frequency surprises (Gürkaynak et al. 2005)

        Args:
            fomc_date: FOMC announcement date
            ticker: Security ticker (SPY, TLT, etc.)
            pre_minutes: Minutes before announcement
            post_minutes: Minutes after announcement
        """
        # FOMC announcements typically at 14:00 ET
        # TAQ timestamps are in EST/EDT
        query = f"""
        SELECT
            date,
            time_m,
            sym_root as ticker,
            price,
            size,
            ex,
            tr_scond
        FROM taqmsec.ctm_{fomc_date.replace('-', '')}
        WHERE sym_root = '{ticker}'
          AND time_m >= '13:{60-pre_minutes:02d}:00'
          AND time_m <= '15:{post_minutes:02d}:00'
        ORDER BY time_m
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            # TAQ data may not exist for all dates
            return pd.DataFrame()

    def fetch_taq_quotes_window(
        self,
        fomc_date: str,
        ticker: str = 'SPY',
        pre_minutes: int = 30,
        post_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch intraday NBBO quotes around FOMC announcements.

        Quotes provide bid-ask spreads → measure liquidity impact
        of FOMC announcements (a la Fleming & Remolona 1999).
        """
        query = f"""
        SELECT
            date,
            time_m,
            sym_root as ticker,
            bid,
            bidsiz,
            ask,
            asksiz,
            qu_cond
        FROM taqmsec.cqm_{fomc_date.replace('-', '')}
        WHERE sym_root = '{ticker}'
          AND time_m >= '13:{60-pre_minutes:02d}:00'
          AND time_m <= '15:{post_minutes:02d}:00'
        ORDER BY time_m
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            return pd.DataFrame()

    # ── Fed Funds Futures (More Granular than FRED) ──

    def fetch_fed_funds_futures(
        self,
        start: str = '2000-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch Fed Funds futures from CME via WRDS.

        Advantage over FRED DFF:
        - Individual contract prices (not just effective rate)
        - Enables proper Kuttner (2001) surprise calculation
        - Multiple maturities → path factor decomposition
        - Intraday prices → high-frequency identification

        Tables: cme.ff (fed funds futures), cme.ef (eurodollar futures)
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        query = f"""
        SELECT
            date,
            symbol,
            settle,
            volume,
            open_interest
        FROM cme.ff
        WHERE date >= '{start}'
          AND date <= '{end}'
        ORDER BY date, symbol
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            return pd.DataFrame()

    def fetch_eurodollar_futures(
        self,
        start: str = '2000-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch Eurodollar futures for longer-horizon rate expectations.

        Used in Gürkaynak, Sack, Swanson (2005) to decompose
        policy surprises into target and path factors.
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        query = f"""
        SELECT
            date,
            symbol,
            settle,
            volume,
            open_interest
        FROM cme.ef
        WHERE date >= '{start}'
          AND date <= '{end}'
        ORDER BY date, symbol
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            return pd.DataFrame()

    # ── OptionMetrics: Implied Volatility ──

    def fetch_option_implied_vol(
        self,
        ticker: str = 'SPY',
        start: str = '2020-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch option-implied volatility from OptionMetrics.

        Key use cases:
        - Construct FOMC-specific implied volatility (Kelly et al. 2016)
        - Measure uncertainty changes around FOMC
        - VIX decomposition (spot vs. term structure)
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        query = f"""
        SELECT
            s.date,
            s.symbol,
            s.impl_volatility,
            s.delta,
            s.days,
            s.strike_price,
            s.best_bid,
            s.best_offer,
            o.cpnrate,
            o.exdate
        FROM optionm.opprcd{end[:4]} s
        JOIN optionm.secprd o ON s.secid = o.secid AND s.date = o.date
        WHERE o.symbol = '{ticker}'
          AND s.date >= '{start}'
          AND s.date <= '{end}'
          AND s.impl_volatility IS NOT NULL
        ORDER BY s.date, s.days
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            return pd.DataFrame()

    # ── IBES: Analyst Expectations ──

    def fetch_ibes_surprises(
        self,
        ticker: str = None,
        start: str = '2020-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch analyst earnings forecast surprises from IBES.

        Use case: Distinguish information shock from policy shock.
        If FOMC reveals information about the economy, we should see
        simultaneous revision in analyst forecasts (Campbell et al. 2012).
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        ticker_filter = f"AND ticker = '{ticker}'" if ticker else ""

        query = f"""
        SELECT
            statpers,
            ticker,
            fpi,
            meanest,
            medest,
            stdev,
            numest
        FROM ibes.statsum_epsus
        WHERE statpers >= '{start}'
          AND statpers <= '{end}'
          {ticker_filter}
        ORDER BY statpers, ticker
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            return pd.DataFrame()

    # ── Survey of Professional Forecasters ──

    def fetch_spf_forecasts(
        self,
        start: str = '2000-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch Philadelphia Fed Survey of Professional Forecasters.

        Key for monetary policy research:
        - Expected inflation (CPI, GDP deflator)
        - Expected GDP growth
        - Expected unemployment
        - Probability of recession

        Use case: Measure "expected" vs "unexpected" policy changes
        relative to market expectations (Romer & Romer 2004 approach).
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        query = f"""
        SELECT *
        FROM philfed.spf
        WHERE quarter >= '{start}'
        ORDER BY quarter
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            return pd.DataFrame()

    # ── Compustat: Fundamentals ──

    def fetch_compustat_quarterly(
        self,
        gvkeys: List[str] = None,
        start: str = '2020-01-01',
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch Compustat quarterly fundamentals.

        Use case: Information shock validation.
        If FOMC reveals economic information, firms sensitive to
        monetary policy should see earnings revisions.
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        gvkey_filter = ""
        if gvkeys:
            gvkey_list = ','.join(f"'{g}'" for g in gvkeys)
            gvkey_filter = f"AND gvkey IN ({gvkey_list})"

        query = f"""
        SELECT
            gvkey,
            datadate,
            fyearq,
            fqtr,
            atq,
            ltq,
            saleq,
            ibq,
            dlttq,
            dlcq,
            cheq,
            niq
        FROM comp.fundq
        WHERE datadate >= '{start}'
          AND datadate <= '{end}'
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND consol = 'C'
          AND popsrc = 'D'
          {gvkey_filter}
        ORDER BY gvkey, datadate
        """

        try:
            return self._db.raw_sql(query)
        except Exception:
            return pd.DataFrame()

    # ── Convenience: Bulk Fetch for Research ──

    def fetch_research_dataset(
        self,
        start: str = '2000-01-01',
        end: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data needed for the Two-Shocks research pipeline.

        Returns dict of DataFrames:
        - crsp_index: CRSP daily market returns
        - fed_funds_futures: CME FF futures for Kuttner surprise
        - eurodollar_futures: CME ED futures for path factor
        - spf_forecasts: Survey of Professional Forecasters
        """
        datasets = {}

        print("Fetching CRSP market index...")
        datasets['crsp_index'] = self.fetch_crsp_index(start=start, end=end)

        print("Fetching Fed Funds futures...")
        datasets['fed_funds_futures'] = self.fetch_fed_funds_futures(start=start, end=end)

        print("Fetching Eurodollar futures...")
        datasets['eurodollar_futures'] = self.fetch_eurodollar_futures(start=start, end=end)

        print("Fetching SPF forecasts...")
        datasets['spf_forecasts'] = self.fetch_spf_forecasts(start=start, end=end)

        return datasets

    # ── Utility ──

    def list_libraries(self) -> List[str]:
        """List all available WRDS data libraries."""
        return self._db.list_libraries()

    def list_tables(self, library: str) -> List[str]:
        """List tables in a WRDS library."""
        return self._db.list_tables(library=library)

    def describe_table(self, library: str, table: str) -> pd.DataFrame:
        """Describe a WRDS table schema."""
        return self._db.describe_table(library=library, table=table)

    def raw_sql(self, query: str) -> pd.DataFrame:
        """Execute arbitrary SQL query on WRDS."""
        return self._db.raw_sql(query)
