const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  PageOrientation, TableOfContents,
} = require("docx");
const fs = require("fs");

// ── Palette: Deep Sea Blue-Gold (Finance / Investment / Premium) ──
const P = {
  primary: "0F2027",
  body: "1A2B40",
  secondary: "4A6575",
  accent: "D4AF37",
  surface: "F5F7FA",
  cover: {
    titleColor: "FFFFFF",
    subtitleColor: "B0B8C0",
    metaColor: "90989F",
    footerColor: "687078",
  },
  table: {
    headerBg: "0F2027",
    headerText: "FFFFFF",
    accentLine: "D4AF37",
    innerLine: "D0D8D0",
    surface: "F0F4F8",
  },
};

const c = (hex) => hex;

// ── Helper Functions ──
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: c(P.primary), font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, size: 28, color: c(P.primary), font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100, line: 312 },
    children: [new TextRun({ text, bold: true, size: 24, color: c(P.primary), font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}

function bodyPara(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: 80 },
    children: [new TextRun({ text, size: 24, color: c(P.body), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } })],
  });
}

function bodyParaNoIndent(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 312, after: 80 },
    children: [new TextRun({ text, size: 24, color: c(P.body), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } })],
  });
}

function formulaPara(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120, line: 312 },
    children: [new TextRun({ text, italics: true, size: 24, color: c(P.body), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } })],
  });
}

function captionPara(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 160, line: 312 },
    children: [new TextRun({ text, size: 21, color: c(P.secondary), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } })],
  });
}

function bulletPara(text) {
  return new Paragraph({
    spacing: { line: 312, after: 40 },
    indent: { left: 720, hanging: 360 },
    children: [
      new TextRun({ text: "\u2022  ", size: 24, color: c(P.accent), font: { ascii: "Times New Roman" } }),
      new TextRun({ text, size: 24, color: c(P.body), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } }),
    ],
  });
}

// ── Table Builder ──
function buildTable(headers, rows, colWidths) {
  const t = P.table;
  const headerRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: headers.map((h, i) =>
      new TableCell({
        width: { size: colWidths[i], type: WidthType.PERCENTAGE },
        shading: { type: ShadingType.CLEAR, fill: c(t.headerBg) },
        margins: { top: 60, bottom: 60, left: 120, right: 120 },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: h, bold: true, size: 21, color: c(t.headerText), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } })] })],
      })
    ),
  });

  const dataRows = rows.map((row, ri) =>
    new TableRow({
      cantSplit: true,
      children: row.map((cell, ci) =>
        new TableCell({
          width: { size: colWidths[ci], type: WidthType.PERCENTAGE },
          shading: { type: ShadingType.CLEAR, fill: ri % 2 === 0 ? c(t.surface) : "FFFFFF" },
          margins: { top: 60, bottom: 60, left: 120, right: 120 },
          children: [new Paragraph({ alignment: AlignmentType.LEFT, children: [new TextRun({ text: cell, size: 21, color: c(P.body), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } })] })],
        })
      ),
    })
  );

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: c(t.accentLine) },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: c(t.accentLine) },
      left: { style: BorderStyle.NONE },
      right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: c(t.innerLine) },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...dataRows],
  });
}

// ── Cover Page (R1 style - Deep Sea Blue-Gold) ──
function buildCover() {
  const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: allNoBorders,
    rows: [
      new TableRow({
        height: { value: 16838, rule: "exact" },
        children: [
          new TableCell({
            width: { size: 100, type: WidthType.PERCENTAGE },
            shading: { type: ShadingType.CLEAR, fill: c(P.primary) },
            borders: allNoBorders,
            verticalAlign: "top",
            children: [
              new Paragraph({ spacing: { before: 4200 }, children: [] }),
              // Accent line
              new Paragraph({
                indent: { left: 1200, right: 1200 },
                border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: c(P.accent), space: 20 } },
                children: [],
              }),
              new Paragraph({ spacing: { before: 400 }, children: [] }),
              // Title
              new Paragraph({
                alignment: AlignmentType.LEFT,
                indent: { left: 1200, right: 1200 },
                spacing: { line: 920, lineRule: "atLeast" },
                children: [
                  new TextRun({ text: "Monetary Policy Lab", size: 72, bold: true, color: c(P.cover.titleColor), font: { ascii: "Times New Roman", eastAsia: "SimHei" } }),
                ],
              }),
              // Subtitle
              new Paragraph({
                alignment: AlignmentType.LEFT,
                indent: { left: 1200, right: 1200 },
                spacing: { before: 200, line: 480, lineRule: "atLeast" },
                children: [
                  new TextRun({ text: "Computational Details & Technical Reference", size: 36, color: c(P.accent), font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" } }),
                ],
              }),
              new Paragraph({ spacing: { before: 200 }, children: [] }),
              // Accent line
              new Paragraph({
                indent: { left: 1200, right: 1200 },
                border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: c(P.accent), space: 20 } },
                children: [],
              }),
              new Paragraph({ spacing: { before: 600 }, children: [] }),
              // Meta
              new Paragraph({
                alignment: AlignmentType.LEFT,
                indent: { left: 1200 },
                spacing: { line: 360 },
                children: [
                  new TextRun({ text: "Platform: ", size: 22, color: c(P.cover.metaColor), font: { ascii: "Times New Roman" } }),
                  new TextRun({ text: "https://monetary-policy-lab.streamlit.app", size: 22, color: c(P.accent), font: { ascii: "Times New Roman" } }),
                ],
              }),
              new Paragraph({
                alignment: AlignmentType.LEFT,
                indent: { left: 1200 },
                spacing: { line: 360 },
                children: [
                  new TextRun({ text: "Repository: ", size: 22, color: c(P.cover.metaColor), font: { ascii: "Times New Roman" } }),
                  new TextRun({ text: "github.com/dechang64/monetary-policy-lab", size: 22, color: c(P.accent), font: { ascii: "Times New Roman" } }),
                ],
              }),
              new Paragraph({
                alignment: AlignmentType.LEFT,
                indent: { left: 1200 },
                spacing: { line: 360 },
                children: [
                  new TextRun({ text: "Version: ", size: 22, color: c(P.cover.metaColor), font: { ascii: "Times New Roman" } }),
                  new TextRun({ text: "v1.0 (Phase 1 Complete, WRDS Integration Pending)", size: 22, color: c(P.cover.subtitleColor), font: { ascii: "Times New Roman" } }),
                ],
              }),
              new Paragraph({
                alignment: AlignmentType.LEFT,
                indent: { left: 1200 },
                spacing: { before: 200, line: 360 },
                children: [
                  new TextRun({ text: "Date: May 2025", size: 22, color: c(P.cover.metaColor), font: { ascii: "Times New Roman" } }),
                ],
              }),
            ],
          }),
        ],
      }),
    ],
  });
}

// ── Document ──
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "Microsoft YaHei" }, size: 24, color: c(P.body) },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: c(P.primary) },
        paragraph: { spacing: { before: 360, after: 160, line: 312 } },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 28, bold: true, color: c(P.primary) },
        paragraph: { spacing: { before: 280, after: 120, line: 312 } },
      },
      heading3: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 24, bold: true, color: c(P.primary) },
        paragraph: { spacing: { before: 200, after: 100, line: 312 } },
      },
    },
  },
  sections: [
    // ── Section 1: Cover ──
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 0, bottom: 0, left: 0, right: 0 },
        },
      },
      children: [buildCover()],
    },
    // ── Section 2: TOC ──
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, bottom: 1440, left: 1701, right: 1417 },
          pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: c(P.secondary) })],
            }),
          ],
        }),
      },
      children: [
        new Paragraph({
          spacing: { before: 200, after: 300 },
          children: [new TextRun({ text: "Table of Contents", size: 36, bold: true, color: c(P.primary), font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
        }),
        new TableOfContents("Table of Contents", {
          hyperlink: true,
          headingStyleRange: "1-3",
        }),
        new Paragraph({
          spacing: { before: 200 },
          children: [
            new TextRun({ text: "Note: Right-click the TOC and select \u201cUpdate Field\u201d to refresh page numbers after opening in Word.", italics: true, size: 20, color: c(P.secondary), font: { ascii: "Times New Roman" } }),
          ],
        }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },
    // ── Section 3: Body ──
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, bottom: 1440, left: 1701, right: 1417 },
          pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [new TextRun({ text: "Monetary Policy Lab \u2014 Computational Details", size: 18, color: c(P.secondary), font: { ascii: "Times New Roman", italics: true } })],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: c(P.secondary) })],
            }),
          ],
        }),
      },
      children: [
        // ═══════════════════════════════════════════════════════════════
        // 1. Architecture Overview
        // ═══════════════════════════════════════════════════════════════
        heading1("1. Architecture Overview"),
        bodyPara("The Monetary Policy Lab is a two-layer computational platform for FOMC monetary policy analysis. The interactive dashboard (Streamlit) provides real-time exploration, visualization, and scenario analysis across 8 modules, while the research engine runs formal econometric analysis offline and stores structured results."),
        buildTable(
          ["Layer", "Component", "Purpose"],
          [
            ["Interactive Dashboard", "app.py + modules/", "Real-time exploration, visualization, scenario analysis"],
            ["Research Engine", "mp-research-platform/", "Batch regression pipeline, hypothesis testing, robustness checks"],
          ],
          [25, 35, 40]
        ),
        captionPara("Table 1: Platform Architecture"),
        bodyPara("The dashboard modules include: Dashboard (overview), Fed Intelligence (FOMC calendar + rate tracker), Research (hypothesis testing), Replication (paper replication), Sentiment (NLP analysis), Two Shocks (target/path decomposition), Capital Flow (cross-border flows), Event Study (short-window analysis), and Data Explorer (raw data browser)."),

        // ═══════════════════════════════════════════════════════════════
        // 2. Data Sources & Pipeline
        // ═══════════════════════════════════════════════════════════════
        heading1("2. Data Sources & Pipeline"),

        heading2("2.1 FRED API Integration"),
        bodyPara("The FRED connector (data/fred_connector.py, class FREDConnector) fetches 31 data series from FRED, organized into 8 categories. All series are cached locally with automatic refresh logic."),
        buildTable(
          ["Category", "Series", "FRED IDs", "Frequency"],
          [
            ["Interest Rates", "1M/2Y/5Y/10Y/30Y Treasury, 2Y-10Y Spread, EFFR, SOFR", "DGS1MO, DGS2, DGS5, DGS10, DGS30, T10Y2Y, DFF, SOFR", "Daily"],
            ["Inflation", "CPI YoY, Core CPI YoY, PCE, Core PCE YoY, 5Y/10Y Breakeven", "CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, T5YIE, T10YIE", "Monthly/Daily"],
            ["Employment", "Unemployment Rate, NFP, Avg Hourly Earnings", "UNRATE, PAYEMS, CES0500000003", "Monthly"],
            ["GDP", "Real GDP, GDP Deflator", "GDPC1, GDPDEF", "Quarterly"],
            ["Financial Stress", "VIX, Financial Conditions Index", "VIXCLS, NFCI", "Daily/Weekly"],
            ["Money Supply", "M1, M2", "M1SL, M2SL", "Monthly"],
            ["Credit", "Total Credit, Consumer Credit", "TOTCI, CCLACBW027SBOG", "Monthly/Weekly"],
            ["Exchange Rate", "Trade-Weighted USD Index", "DTWEXBGS", "Daily"],
          ],
          [15, 35, 30, 20]
        ),
        captionPara("Table 2: FRED Data Series (31 series across 8 categories)"),
        bodyPara("The API call format is: https://api.stlouisfed.org/fred/series/observations?series_id={ID}&api_key={KEY}&file_type=json&observation_start={START}&observation_end={END}. Rate values are in percentage terms (e.g., DFF = 5.33 means 5.33%), requiring multiplication by 100 for basis point conversion."),

        heading2("2.2 FOMC Statement Scraping"),
        bodyPara("FOMC statements are scraped from the Federal Reserve website (data/fomc_scraper.py, class FOMCScraper). The URL format is: https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm. Content is extracted using the div#article CSS selector. The scraper achieved a 155/157 success rate (2006\u20132025), with 2 failures likely due to emergency meetings or non-standard dates. A 0.3-second delay between requests prevents rate limiting."),

        heading2("2.3 Asset Price Data"),
        bodyPara("Asset prices are downloaded via yfinance (data/asset_prices.py). The default tickers include: ^GSPC (S&P 500), ^IXIC (NASDAQ), GC=F (Gold), ^TNX (10Y Treasury yield), ^IRX (13W T-bill), DX-Y.NYB (Dollar Index), CL=F (Crude Oil), ^VIX (VIX). The download window is [t-10, t+10] trading days around each FOMC meeting. Note: yfinance returns MultiIndex columns in newer versions; the code uses data['Close'][ticker] rather than data['Close']."),

        heading2("2.4 WRDS Integration (Planned)"),
        bodyPara("WRDS (Wharton Research Data Services) integration is designed but pending account activation. The connector (data/wrds_connector.py, class WRDSConnector) will provide access to: crsp.dsf (daily stock returns with delisting adjustment), cme.ff (Fed Funds futures for Kuttner surprise), cme.ef (Eurodollar futures for G\u00fcrkaynak path factor), taqmsec.ctm_{YYYYMMDD} (TAQ intraday trades), and optionm.opprcd{YYYY} (option implied volatility). Connection requires Duo MFA on first use, with 30-day exemption thereafter."),

        // ═══════════════════════════════════════════════════════════════
        // 3. Monetary Policy Surprise
        // ═══════════════════════════════════════════════════════════════
        heading1("3. Monetary Policy Surprise"),

        heading2("3.1 Current Implementation: Rate Change Proxy"),
        bodyPara("The current surprise measure (analysis/surprise_calculator.py, class SurpriseCalculator) uses the change in the target rate around FOMC meetings as a proxy for monetary policy surprise:"),
        formulaPara("Surprise_t = r_t - r_{t-1}"),
        bodyPara("where r_t is the federal funds target rate after meeting t, and r_{t-1} is the rate before. This is a crude proxy: all meetings with no rate change produce Surprise = 0, which inflates the residual share in variance decomposition and attenuates regression coefficients. The implementation uses DFF (Daily Federal Funds Rate) from FRED as the rate proxy."),

        heading2("3.2 Kuttner (2001) Method (Planned)"),
        bodyPara("The gold-standard approach computes surprise from Fed Funds futures:"),
        formulaPara("Surprise_t = (FF_t - FF_{t-1}) \u00d7 100  (in basis points)"),
        bodyPara("where FF_t is the Fed Funds futures contract price on the meeting day, and FF_{t-1} is the price on the day before. This captures the unexpected component of the rate decision. Implementation requires CME Fed Funds futures data from WRDS (cme.ff table)."),

        heading2("3.3 G\u00fcrkaynak et al. (2005) Path Factor (Planned)"),
        bodyPara("The path factor captures the expected future trajectory of monetary policy beyond the current meeting:"),
        formulaPara("Path_t = \u0394ED2_t - \u0394ED1_t \u00d7 \u03b2"),
        bodyPara("where ED1 and ED2 are the first and second Eurodollar futures contracts, and \u03b2 is the regression coefficient from ED2 on ED1 changes. This decomposition separates the target rate surprise from the forward guidance / path surprise. Implementation requires CME Eurodollar futures data from WRDS (cme.ef table)."),

        // ═══════════════════════════════════════════════════════════════
        // 4. Sentiment Analysis
        // ═══════════════════════════════════════════════════════════════
        heading1("4. Sentiment Analysis"),

        heading2("4.1 Dual-Dictionary Method"),
        bodyPara("The sentiment engine (analysis/nlp_engine.py, class NLPEngine) uses a dual-dictionary approach combining Loughran-McDonald (2011) financial sentiment and Corrado-Driessen (CB) hawkish-dovish dictionaries:"),
        formulaPara("Sentiment_t = 0.5 \u00d7 [(Pos - Neg) / Total] + 0.5 \u00d7 [(Hawk - Dove) / Total]"),
        bodyPara("The LM dictionary contains approximately 30 positive and 30 negative words. The CB dictionary contains approximately 18 hawkish and 18 dovish words plus bigrams. Both dictionaries are sparse for FOMC text, resulting in low sentiment variance (std \u2248 0.003 vs. literature benchmark of 0.035)."),

        heading2("4.2 Known Limitations"),
        bulletPara("Context insensitivity: \u201chigher inflation\u201d and \u201chigher growth\u201d both match positive/hawkish words, but have opposite policy implications"),
        bulletPara("Negation handling: \u201cnot concerned about inflation\u201d is scored as hawkish (matches \u201cinflation\u201d and \u201cconcerned\u201d)"),
        bulletPara("Bigram coverage: only 18 bigrams in the CB dictionary, missing many FOMC-specific collocations"),
        bulletPara("Temporal drift: the dictionary does not account for changing language norms across FOMC chairs"),

        heading2("4.3 Planned Upgrades"),
        bodyPara("FinBERT (Huang et al., 2022) will replace the dictionary approach in Phase 3. FinBERT is a BERT model fine-tuned on financial text that provides context-aware sentiment scores. Expected improvement: sentiment std from 0.003 to 0.02+. This requires GPU compute, which is not available on the current cloud VM (QEMU/KVM virtual GPU)."),

        // ═══════════════════════════════════════════════════════════════
        // 5. Two-Shock Decomposition
        // ═══════════════════════════════════════════════════════════════
        heading1("5. Two-Shock Decomposition"),

        heading2("5.1 Variance Decomposition"),
        bodyPara("The TwoShocksEngine (analysis/two_shocks.py) decomposes asset return variance into a target rate shock component and an information shock component:"),
        formulaPara("R_t = \u03b1 + \u03b2 \u00d7 Surprise_t + \u03b5_t"),
        bodyPara("The variance decomposition is:"),
        formulaPara("Target Share = Var(\u03b2 \u00d7 Surprise) / Var(R)"),
        formulaPara("Info Share = Var(\u03b5) / Var(R)"),
        bodyPara("Current results show Info Share = 99.6%, compared to the literature benchmark of 97.2%. The discrepancy is primarily driven by the crude surprise proxy: when most meetings have Surprise = 0, the explained variance is near zero, inflating the residual share."),

        heading2("5.2 Simulation-Based Decomposition"),
        bodyPara("The engine also implements a simulation-based decomposition using Monte Carlo methods. It generates N = 1000 simulated return paths by:"),
        bulletPara("Drawing surprise values from the empirical distribution"),
        bulletPara("Adding Gaussian noise: \u03b5 ~ N(0, 3) basis points"),
        bulletPara("Computing the fraction of total variance attributable to the surprise component"),
        bodyPara("Note: The simulation uses np.random.normal(0, 3) without a fixed seed, so results vary across runs."),

        // ═══════════════════════════════════════════════════════════════
        // 6. Regression Framework
        // ═══════════════════════════════════════════════════════════════
        heading1("6. Regression Framework"),

        heading2("6.1 Core Specifications"),
        bodyPara("The regression engine (analysis/regression_engine.py, class RegressionEngine) implements three core specifications corresponding to the three hypotheses in the research design:"),
        bodyPara("H1 (Sentiment Channel):"),
        formulaPara("\u0394R_t = \u03b1 + \u03b2\u2081 \u00d7 Sentiment_t + \u03b2\u2082 \u00d7 Controls_t + \u03b5_t"),
        bodyPara("H2 (Surprise Channel):"),
        formulaPara("\u0394R_t = \u03b1 + \u03b3\u2081 \u00d7 Surprise_t + \u03b3\u2082 \u00d7 Controls_t + \u03b5_t"),
        bodyPara("H3 (Two-Shock Decomposition):"),
        formulaPara("\u0394R_t = \u03b1 + \u03b4\u2081 \u00d7 Surprise_t + \u03b4\u2082 \u00d7 Sentiment_t + \u03b4\u2083 \u00d7 Controls_t + \u03b5_t"),
        bodyPara("The dependent variable is the 2-day cumulative return around each FOMC meeting (t-1 to t+1). The asset universe includes: S&P 500, NASDAQ, Gold, 13W T-Bill, and 10Y Treasury. Control variables include: pre-meeting VIX level, pre-meeting rate level, and meeting type indicator (scheduled vs. emergency)."),

        heading2("6.2 Estimation Method"),
        bodyPara("Current implementation uses OLS with homoskedastic standard errors (scipy.stats.linregress). This is a known limitation for financial time series data. The planned upgrades include:"),
        bulletPara("Newey-West HAC standard errors for heteroskedasticity and autocorrelation"),
        bulletPara("White robust standard errors as a baseline correction"),
        bulletPara("Thompson (2011) double-clustered standard errors (by time and by asset) for panel specifications"),
        bulletPara("IV estimation (2SLS) to address potential endogeneity of the surprise measure"),

        heading2("6.3 Multiple Testing Correction"),
        bodyPara("When testing across 5 assets simultaneously, the Bonferroni-adjusted significance level is:"),
        formulaPara("\u03b1* = 0.10 / 5 = 0.02"),
        bodyPara("This means individual p-values must be below 0.02 to claim significance at the 10% family-wise level. Current results do not survive this correction."),

        // ═══════════════════════════════════════════════════════════════
        // 7. Event Study
        // ═══════════════════════════════════════════════════════════════
        heading1("7. Event Study Methodology"),
        bodyPara("The event study module (analysis/event_study.py, class EventStudyEngine) implements a standard short-window event study around FOMC meetings:"),
        bulletPara("Estimation window: [-60, -10] trading days (50-day normal return estimation)"),
        bulletPara("Event window: [-1, +1] trading days (3-day window)"),
        bulletPara("Normal return model: Market model using CRSP value-weighted index (or S&P 500 as proxy)"),
        bulletPara("Abnormal return: AR_t = R_t - (\u03b1_hat + \u03b2_hat \u00d7 R_m,t)"),
        bulletPara("Cumulative abnormal return: CAR[-1,+1] = \u03a3 AR_t"),
        bodyPara("Statistical significance is tested using the cross-sectional t-test (Patell, 1976) and the BMP test (Boehmer, Musumeci, and Poulsen, 1991). The current implementation uses yfinance data for the market proxy; upgrading to CRSP (wrds) would provide delisting-adjusted returns and a more accurate market index."),

        // ═══════════════════════════════════════════════════════════════
        // 8. Capital Flow Analysis
        // ═══════════════════════════════════════════════════════════════
        heading1("8. Capital Flow Analysis"),
        bodyPara("The capital flow module (analysis/capital_flow.py, class CapitalFlowEngine) analyzes cross-border capital flow responses to FOMC decisions. It uses the TIC (Treasury International Capital) flow data from FRED as a proxy for capital flows, with the following specification:"),
        formulaPara("\u0394Flow_t = \u03b1 + \u03b2 \u00d7 Surprise_t + \u03b3 \u00d7 Sentiment_t + \u03b4 \u00d7 Controls_t + \u03b5_t"),
        bodyPara("The analysis examines both aggregate flows and bilateral flows (US-Emerging Markets, US-Developed Markets). The module also computes the exchange rate pass-through:"),
        formulaPara("\u0394e_t = \u03b1 + \u03b2 \u00d7 Surprise_t + \u03b3 \u00d7 Sentiment_t + \u03b5_t"),
        bodyPara("where \u0394e_t is the change in the trade-weighted dollar index around the FOMC meeting."),

        // ═══════════════════════════════════════════════════════════════
        // 9. Current Results vs. Literature
        // ═══════════════════════════════════════════════════════════════
        heading1("9. Current Results vs. Literature Benchmarks"),
        bodyPara("The following table compares the platform\u2019s current results with established literature benchmarks. The gaps are primarily attributable to the crude surprise proxy and sparse sentiment dictionary."),
        buildTable(
          ["Metric", "Current Result", "Literature Benchmark", "Primary Cause of Gap"],
          [
            ["H1 R\u00b2 (Sentiment \u2192 Returns)", "0.39%", "~2.76%", "Sparse dictionary; low sentiment variance"],
            ["H2 Significance (13W T-Bill)", "Not significant", "p = 0.033", "Surprise proxy = 0 for no-change meetings"],
            ["H2 Significance (Gold)", "p = 0.087*", "Not reported in ref.", "Potential new finding"],
            ["H3 Info Share", "99.6%", "97.2%", "Zero-surprise meetings inflate residual"],
            ["Sentiment Std Dev", "0.003", "0.035", "Dictionary too sparse for FOMC text"],
          ],
          [25, 20, 25, 30]
        ),
        captionPara("Table 3: Current Results vs. Literature Benchmarks"),
        bodyPara("The direction of all results is consistent with the literature: the information shock dominates the target rate shock in explaining asset returns. The magnitude gap is a data and methodology issue, not a theoretical one. Upgrading to Kuttner surprise and FinBERT sentiment is expected to close most of the gap."),

        // ═══════════════════════════════════════════════════════════════
        // 10. Statistical Inference Notes
        // ═══════════════════════════════════════════════════════════════
        heading1("10. Statistical Inference Notes"),

        heading2("10.1 Standard Errors"),
        bodyPara("Current implementation uses homoskedastic OLS standard errors (scipy.stats.linregress). This is inappropriate for financial time series, which exhibit:"),
        bulletPara("Heteroskedasticity: volatility clustering around FOMC meetings"),
        bulletPara("Autocorrelation: persistent return patterns in short windows"),
        bulletPara("Cross-sectional correlation: assets move together on FOMC days"),
        bodyPara("Planned corrections: Newey-West HAC (for time-series), White robust (baseline), Thompson double-clustered (for panel)."),

        heading2("10.2 Multiple Testing"),
        bodyPara("Testing 5 assets simultaneously at \u03b1 = 0.10 requires Bonferroni correction: \u03b1* = 0.10/5 = 0.02. Alternatively, the Benjamini-Hochberg FDR procedure can be used for less conservative control. Current results do not survive either correction."),

        heading2("10.3 Endogeneity"),
        bodyPara("The surprise measure may be endogenous if market participants anticipate the rate decision based on the same information that drives asset prices. The standard solution is IV estimation using the policy rule deviation (Taylor rule residual) as an instrument. This is planned for Phase 4."),

        // ═══════════════════════════════════════════════════════════════
        // 11. Reproducibility
        // ═══════════════════════════════════════════════════════════════
        heading1("11. Reproducibility & Technical Notes"),

        heading2("11.1 Data Caching"),
        bodyPara("FRED data is cached locally as CSV files. FOMC statements are cached as JSON. The analysis dataset is saved as analysis_dataset_expanded.csv and regression_results_expanded.json. All data transformations are logged with timestamps."),

        heading2("11.2 Randomness"),
        bodyPara("The TwoShocksEngine.variance_decomposition() method uses np.random.normal(0, 3) for simulation noise without a fixed seed, so results vary across runs. All other modules are deterministic given the same input data."),

        heading2("11.3 Dependencies"),
        buildTable(
          ["Package", "Use"],
          [
            ["streamlit", "Interactive dashboard"],
            ["pandas, numpy", "Data manipulation"],
            ["scipy", "Statistical tests (linregress, ttest_ind, pearsonr)"],
            ["yfinance", "Asset price download"],
            ["requests", "FRED API calls, FOMC scraping"],
            ["beautifulsoup4", "HTML parsing (FOMC statements)"],
            ["plotly", "Interactive charts"],
          ],
          [30, 70]
        ),
        captionPara("Table 4: Key Dependencies"),

        // ═══════════════════════════════════════════════════════════════
        // 12. Upgrade Roadmap
        // ═══════════════════════════════════════════════════════════════
        heading1("12. Upgrade Roadmap"),
        buildTable(
          ["Phase", "Component", "Data Source", "Expected Impact"],
          [
            ["Phase 2", "Kuttner surprise", "CME FF futures (WRDS)", "H1 R\u00b2: 0.4% \u2192 2\u20135%"],
            ["Phase 2", "Path factor", "CME ED futures (WRDS)", "H3 decomposition validity"],
            ["Phase 2", "HAC standard errors", "\u2014", "Correct inference"],
            ["Phase 3", "FinBERT sentiment", "GPU compute", "Sentiment std: 0.003 \u2192 0.02+"],
            ["Phase 3", "High-frequency identification", "TAQ (WRDS)", "Intraday event windows"],
            ["Phase 3", "IV estimation", "\u2014", "Address endogeneity"],
            ["Phase 4", "Sign restriction (JK style)", "\u2014", "Structural shock identification"],
            ["Phase 4", "Panel regression with double-clustering", "\u2014", "Efficient estimation"],
          ],
          [12, 30, 28, 30]
        ),
        captionPara("Table 5: Upgrade Roadmap"),
        bodyPara("Phase 2 (WRDS integration) is the highest priority, as the crude surprise proxy is the single largest source of the gap between current results and literature benchmarks. The WRDS connector is already implemented and tested; activation requires institutional account credentials and Duo MFA enrollment."),

        // ── References ──
        heading1("References"),
        bodyParaNoIndent("Boehmer, E., Musumeci, J., & Poulsen, A. B. (1991). Event-study methodology under conditions of event-induced variance. Journal of Financial Economics, 30(2), 253\u2013272."),
        bodyParaNoIndent("G\u00fcrkaynak, R. S., Sack, B., & Swanson, E. (2005). The sensitivity of long-term interest rates to economic news: Evidence and implications for monetary policy. American Economic Review, 95(1), 425\u2013436."),
        bodyParaNoIndent("Huang, A. H., Zang, A. Y., & Zheng, R. (2022). FinBERT: A large language model for extracting information from financial text. Contemporary Accounting Research, 39(4), 2276\u20132310."),
        bodyParaNoIndent("Kuttner, K. N. (2001). Monetary policy surprises and interest rates: Evidence from the Fed funds futures market. Journal of Monetary Economics, 47(3), 523\u2013544."),
        bodyParaNoIndent("Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1), 35\u201365."),
        bodyParaNoIndent("Patell, J. M. (1976). Corporate forecasts of earnings per share and stock price behavior: Empirical test. Journal of Accounting Research, 14(2), 246\u2013276."),
        bodyParaNoIndent("Thompson, S. B. (2011). Simple formulas for standard errors that cluster by both firm and time. Journal of Financial Economics, 99(1), 1\u201310."),
      ],
    },
  ],
});

// ── Generate ──
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/z/my-project/monetary-policy-lab/docs/Monetary_Policy_Lab_Computational_Details.docx", buf);
  console.log("Document generated successfully.");
});
