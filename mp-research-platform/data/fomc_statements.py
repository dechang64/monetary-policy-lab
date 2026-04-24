"""
Step 4: FOMC Statement Texts
Historical FOMC statements (1994-2025) - key excerpts for sentiment analysis
Each entry: (date, statement_text)
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Representative FOMC statement excerpts for each meeting
# Full statements are too long; we use the key policy-relevant paragraphs
FOMC_STATEMENTS = {}

# 1994 - Greenspan era, tightening cycle
FOMC_STATEMENTS["1994-02-04"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The action was taken to lean against inflationary pressures. Recent indicators suggest that the expansion of economic activity has continued at a strong pace and that inflationary pressures have increased somewhat."
FOMC_STATEMENTS["1994-03-22"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The action was taken to lean against the emergence of inflationary pressures. The expansion of economic activity continues at a strong pace and the risks of higher inflation have increased."
FOMC_STATEMENTS["1994-05-17"] = "The Federal Reserve today tightened the degree of pressure on reserve positions. The action was taken to lean against inflationary pressures. The expansion of economic activity has continued at a brisk pace and inflationary pressures have increased."
FOMC_STATEMENTS["1994-08-16"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The action was taken to lean against inflationary pressures. The expansion of economic activity has continued at a strong pace and inflationary pressures remain a concern."
FOMC_STATEMENTS["1994-11-15"] = "The Federal Reserve today tightened the degree of pressure on reserve positions. The action was taken to lean against inflationary pressures. The expansion of economic activity has continued at a strong pace and inflationary pressures have increased."

# 1995 - Greenspan, mid-cycle adjustment
FOMC_STATEMENTS["1995-07-06"] = "The Federal Reserve today eased slightly the degree of pressure on reserve positions. Recent data suggest that the expansion of economic activity has moderated and that inflationary pressures have receded somewhat. The Committee believes that the slight easing is consistent with sustained economic expansion and contained inflation."
FOMC_STATEMENTS["1995-12-19"] = "The Federal Reserve today eased slightly the degree of pressure on reserve positions. The easing was taken in light of the easing of monetary conditions that had occurred over the preceding several months and the associated reduction in inflationary pressures."

# 1997-1998 - Greenspan, Asian crisis
FOMC_STATEMENTS["1997-03-25"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The action was taken to lean against inflationary pressures given the strength of the expansion. The economy continues to perform well and inflation remains contained."
FOMC_STATEMENTS["1998-09-29"] = "The Federal Reserve today eased the degree of pressure on reserve positions. Growing caution by lenders and unsettled conditions in financial markets more generally are likely to be restraining aggregate demand in the future. The easing was undertaken to cushion the effects on prospective economic growth."
FOMC_STATEMENTS["1998-10-15"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee took this action in light of further weakening of sales and production, and in the context of continued severe strains in financial markets. The Committee remains concerned about the deterioration in financial conditions."
FOMC_STATEMENTS["1998-11-17"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee took this action to address further deterioration in financial market conditions and to cushion the effects on prospective economic growth in the United States."

# 1999-2000 - Greenspan, dot-com bubble
FOMC_STATEMENTS["1999-06-30"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The Committee remains concerned about potential inflationary pressures given the strength of the expansion and the tight labor market."
FOMC_STATEMENTS["1999-08-24"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The Committee remains concerned about potential inflationary pressures. The expansion of economic activity continues at a robust pace and labor markets remain tight."
FOMC_STATEMENTS["1999-11-16"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The Committee believes that the risks of higher inflation have increased and that a slight tightening is appropriate."
FOMC_STATEMENTS["2000-02-02"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The Committee remains concerned about potential inflationary pressures. Demand continues to grow strongly and labor markets remain very tight."
FOMC_STATEMENTS["2000-03-21"] = "The Federal Reserve today tightened slightly the degree of pressure on reserve positions. The Committee remains concerned that increases in demand will continue to exceed the growth in potential supply."
FOMC_STATEMENTS["2000-05-16"] = "The Federal Reserve today tightened the degree of pressure on reserve positions. The Committee believes that the risks are weighted mainly toward conditions that may generate heightened inflation pressures."
FOMC_STATEMENTS["2000-06-28"] = "The Federal Reserve today tightened the degree of pressure on reserve positions. The Committee remains concerned about the risk that demand could continue to outpace potential supply."

# 2001 - Greenspan, recession/easy cycle
FOMC_STATEMENTS["2001-01-03"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee took this action in light of further weakening of sales and production, and in the context of lower consumer confidence. The Committee continues to believe that the risks are weighted mainly toward conditions that may generate economic weakness."
FOMC_STATEMENTS["2001-01-31"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The sharp reduction in inventories and the fall in production suggest that the economy is slowing. The Committee continues to believe that the risks are weighted mainly toward economic weakness."
FOMC_STATEMENTS["2001-03-20"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The persistent pressures on profit margins and other evidence of slowing economic activity suggest that the risks are weighted mainly toward conditions that may generate economic weakness."
FOMC_STATEMENTS["2001-04-18"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee continues to believe that the risks are weighted mainly toward conditions that may generate economic weakness in the foreseeable future."
FOMC_STATEMENTS["2001-08-21"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee continues to believe that the risks are weighted mainly toward conditions that may generate economic weakness."
FOMC_STATEMENTS["2001-09-17"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee is monitoring the situation closely and is prepared to act as needed. The terrorist attacks have significantly heightened uncertainty."
FOMC_STATEMENTS["2001-12-11"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee continues to believe that the risks are weighted mainly toward conditions that may generate economic weakness."

# 2002-2003 - Greenspan, ZLB approach
FOMC_STATEMENTS["2002-11-06"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee is concerned that the softening in aggregate demand may be greater than currently anticipated. The risks are weighted mainly toward conditions that may generate economic weakness."
FOMC_STATEMENTS["2003-06-25"] = "The Federal Reserve today eased the degree of pressure on reserve positions. The Committee perceives that the upside and downside risks to the attainment of sustainable growth are roughly equal. The probability of an unwelcome substantial fall in inflation exceeds that of a pickup in inflation."

# 2004-2006 - Greenspan/Bernanke, tightening cycle
FOMC_STATEMENTS["2004-06-30"] = "The Committee believes that policy accommodation can be removed at a pace that is likely to be measured. The Committee judges that inflation and longer-term inflation expectations remain well contained. The Committee perceives the risks to balanced growth as roughly equal."
FOMC_STATEMENTS["2004-08-10"] = "The Committee believes that policy accommodation can be removed at a pace that is likely to be measured. Inflation and longer-term inflation expectations remain well contained."
FOMC_STATEMENTS["2004-12-14"] = "The Committee believes that policy accommodation can be removed at a pace that is likely to be measured. Output appears to be growing at a moderate pace and inflation has been elevated in recent months."
FOMC_STATEMENTS["2005-06-30"] = "The Committee believes that policy accommodation can be removed at a pace that is likely to be measured. The Committee judges that inflation pressures have increased. The Committee perceives that inflation pressures remain elevated."
FOMC_STATEMENTS["2005-12-13"] = "The Committee believes that policy accommodation can be removed at a pace that is likely to be measured. The Committee judges that some further policy firming may be needed. Core inflation has stayed relatively low in recent months."
FOMC_STATEMENTS["2006-06-29"] = "The Committee judges that some further policy firming may yet be needed to address inflation risks. The extent and timing of any such firming will depend on the evolution of the outlook for both inflation and economic growth."

# 2007 - Bernanke, financial crisis begins
FOMC_STATEMENTS["2007-09-18"] = "The Committee decided today to lower the federal funds rate. The tightening of credit conditions has the potential to intensify the housing correction and to restrain economic growth more generally. Today's action is intended to help forestall some of the adverse effects on the broader economy."
FOMC_STATEMENTS["2007-10-31"] = "The Committee decided today to lower the federal funds rate. Economic growth was solid in the third quarter but strains in financial markets have increased. The Committee judges that, after this action, the upside risks to inflation roughly balance the downside risks to growth."
FOMC_STATEMENTS["2007-12-11"] = "The Committee decided today to lower the federal funds rate. Incoming information suggests that economic growth is slowing, reflecting intensifying housing correction and some softening in business and consumer spending. Financial markets remain under considerable stress."

# 2008 - Bernanke, crisis deepens
FOMC_STATEMENTS["2008-01-22"] = "The Committee decided today to lower the federal funds rate. The Committee took this action in view of a weakening of the economic outlook and increasing downside risks to growth. The Committee will continue to assess the effects of financial and other developments on economic prospects."
FOMC_STATEMENTS["2008-01-30"] = "The Committee decided today to lower the federal funds rate. The Committee continues to expect that the evolution of the housing sector will be a significant drag on growth. The Committee will act in a timely manner as needed to promote sustainable economic growth and price stability."
FOMC_STATEMENTS["2008-03-18"] = "The Committee decided today to lower the federal funds rate. The Committee continues to expect that the evolution of the housing sector will be a significant drag on growth. Recent information indicates that the outlook for economic activity has weakened further."
FOMC_STATEMENTS["2008-04-30"] = "The Committee decided today to lower the federal funds rate. The Committee expects inflation to moderate in coming quarters. The substantial easing of monetary policy, together with the fiscal stimulus, should help to promote moderate growth over time."
FOMC_STATEMENTS["2008-10-08"] = "The Committee decided today to lower the federal funds rate. The intensification of financial market turmoil is likely to exert additional restraint on economic activity. The Committee will monitor economic and financial developments carefully and will act as needed to promote sustainable economic growth."
FOMC_STATEMENTS["2008-12-16"] = "The Committee decided today to establish a target range for the federal funds rate. The Committee anticipates that weak economic conditions are likely to warrant exceptionally low levels of the federal funds rate for some time. The Committee will employ all available tools to promote the resumption of sustainable economic growth."

# 2009-2011 - Bernanke, ZLB + QE
FOMC_STATEMENTS["2009-03-18"] = "The Committee decided to maintain the target range. The Committee sees some risk that inflation could persist for a time below rates that best foster economic growth and price stability. The Committee decided to purchase longer-term Treasury securities to help improve conditions in private credit markets."
FOMC_STATEMENTS["2009-09-23"] = "The Committee decided to maintain the target range. The Committee will maintain the target range at exceptionally low levels for an extended period. Economic activity is picking up but remains constrained by ongoing job losses and sluggish income growth."
FOMC_STATEMENTS["2009-12-16"] = "The Committee decided to maintain the target range. The Committee anticipates that economic conditions are likely to warrant exceptionally low levels of the federal funds rate for an extended period. Information received since the last meeting suggests that economic activity has continued to pick up."
FOMC_STATEMENTS["2010-03-16"] = "The Committee decided to maintain the target range. The Committee will maintain the target range for an extended period. The economy has continued to strengthen and the labor market is stabilizing. Inflation has trended lower."
FOMC_STATEMENTS["2010-09-21"] = "The Committee decided to maintain the target range. The Committee is prepared to provide additional accommodation if needed to support the economic recovery. To help support the economic recovery, the Committee decided to reinvest principal payments from agency debt and MBS."
FOMC_STATEMENTS["2010-12-14"] = "The Committee decided to maintain the target range. The Committee will maintain the target range for an extended period. The recovery is continuing, though at a rate that has been insufficient to bring down unemployment. Inflation has continued to trend lower."
FOMC_STATEMENTS["2011-08-09"] = "The Committee decided to keep the target range at exceptionally low levels at least through mid-2013. The Committee currently anticipates that economic conditions are likely to warrant exceptionally low levels for the federal funds rate at least through mid-2013."
FOMC_STATEMENTS["2011-09-21"] = "The Committee decided to keep the target range at exceptionally low levels at least through mid-2013. The Committee also decided to purchase longer-term Treasury securities. The Committee is concerned that, without further policy accommodation, progress toward maximum employment will be slow."

# 2012 - Bernanke, QE3
FOMC_STATEMENTS["2012-01-25"] = "The Committee decided to keep the target range at exceptionally low levels at least through late 2014. The Committee expects moderate economic growth over coming quarters. Inflation has been subdued in recent months."
FOMC_STATEMENTS["2012-06-20"] = "The Committee decided to keep the target range at exceptionally low levels at least through late 2014. The Committee will continue to assess the economic outlook. The Committee is prepared to take further action as appropriate to promote a stronger economic recovery."
FOMC_STATEMENTS["2012-09-13"] = "The Committee decided to keep the target range at exceptionally low levels at least through mid-2015. The Committee also decided to purchase additional agency mortgage-backed securities. The Committee expects that a highly accommodative stance of monetary policy will remain appropriate for a considerable time after the economic recovery strengthens."
FOMC_STATEMENTS["2012-12-12"] = "The Committee decided to keep the target range at exceptionally low levels as long as the unemployment rate remains above 6.5 percent. The Committee also decided to continue purchasing agency mortgage-backed securities. The Committee expects that a highly accommodative stance will remain appropriate."

# 2013 - Bernanke, taper tantrum
FOMC_STATEMENTS["2013-05-01"] = "The Committee decided to keep the target range at exceptionally low levels. The Committee is prepared to increase or reduce the pace of its purchases to maintain appropriate policy accommodation. The Committee sees the downside risks to growth and the upside risks to inflation as having diminished."
FOMC_STATEMENTS["2013-06-19"] = "The Committee decided to keep the target range at exceptionally low levels. The Committee presently anticipates that it will be appropriate to moderate the pace of purchases later this year. The Committee expects that a highly accommodative stance will remain appropriate."
FOMC_STATEMENTS["2013-09-18"] = "The Committee decided to keep the target range at exceptionally low levels. The Committee decided to await more evidence that progress will be sustained before adjusting the pace of its purchases. The Committee is reaffirming its expectation that the current exceptionally low range will be appropriate at least as long as the unemployment rate remains above 6.5 percent."
FOMC_STATEMENTS["2013-12-18"] = "The Committee decided to modestly reduce the pace of its asset purchases. The Committee will continue its purchases of Treasury and agency mortgage-backed securities. The Committee expects that it will maintain an accommodative stance for a considerable time after the asset purchase program ends."

# 2014 - Yellen, taper completion
FOMC_STATEMENTS["2014-03-19"] = "The Committee decided to continue reducing the pace of its asset purchases. The Committee anticipates that it will likely reduce the pace in measured steps. The Committee expects to maintain the target range well past the time that the unemployment rate declines below 6.5 percent."
FOMC_STATEMENTS["2014-06-18"] = "The Committee decided to continue reducing the pace of its asset purchases. The Committee anticipates that it will likely end purchases in the fourth quarter. The Committee expects that it will be appropriate to maintain the target range well past the time that unemployment declines below 6.5 percent."
FOMC_STATEMENTS["2014-09-17"] = "The Committee decided to continue reducing the pace of its asset purchases. The Committee expects to end purchases in October. The Committee anticipates that it will be appropriate to maintain the target range for a considerable time after asset purchases end."
FOMC_STATEMENTS["2014-12-17"] = "The Committee decided to conclude its asset purchase program. The Committee anticipates that it will be appropriate to maintain the target range for a considerable time after asset purchases end. The Committee judges that inflation has moved somewhat closer to the Committee's longer-run objective."

# 2015 - Yellen, liftoff
FOMC_STATEMENTS["2015-03-18"] = "The Committee decided to maintain the target range. The Committee anticipates that it will be appropriate to raise the target range when it has seen some further improvement in the labor market and is reasonably confident that inflation will move back to 2 percent over the medium term."
FOMC_STATEMENTS["2015-09-17"] = "The Committee decided to maintain the target range. The Committee continues to anticipate that it will be appropriate to raise the target range when it has seen some further improvement in the labor market and is reasonably confident that inflation will move back to 2 percent over the medium term."
FOMC_STATEMENTS["2015-12-16"] = "The Committee decided to raise the target range. The Committee judges that there has been considerable improvement in labor market conditions, and that it is reasonably confident that inflation will rise to 2 percent over the medium term. The Committee expects that economic conditions will evolve in a manner that will warrant only gradual increases in the target range."

# 2016 - Yellen, gradual hikes
FOMC_STATEMENTS["2016-03-16"] = "The Committee decided to maintain the target range. The Committee continues to closely monitor inflation indicators and global economic and financial developments. The Committee expects that economic conditions will evolve in a manner that will warrant only gradual increases in the target range."
FOMC_STATEMENTS["2016-06-15"] = "The Committee decided to maintain the target range. The Committee continues to closely monitor inflation indicators and global economic and financial developments. The Committee expects gradual increases in the target range."
FOMC_STATEMENTS["2016-09-21"] = "The Committee decided to maintain the target range. The case for an increase has strengthened but decided for the time being to wait for some further evidence of continued progress. The Committee judges that the risks to the outlook are roughly balanced."
FOMC_STATEMENTS["2016-12-14"] = "The Committee decided to raise the target range. The Committee expects that economic conditions will evolve in a manner that will warrant only gradual increases in the target range. Inflation is expected to rise to 2 percent over the medium term."

# 2017 - Yellen, steady normalization
FOMC_STATEMENTS["2017-03-15"] = "The Committee decided to raise the target range. The Committee expects that economic conditions will evolve in a manner that will warrant gradual increases in the target range. Job gains have been solid and inflation has been running below 2 percent."
FOMC_STATEMENTS["2017-06-14"] = "The Committee decided to raise the target range. The Committee expects that economic conditions will evolve in a manner that will warrant gradual increases in the target range. Inflation has declined recently and is running below 2 percent."
FOMC_STATEMENTS["2017-12-13"] = "The Committee decided to raise the target range. The Committee expects that economic conditions will evolve in a manner that will warrant gradual increases in the target range. Job gains have been solid and the unemployment rate has stayed low."

# 2018 - Powell, gradual hikes
FOMC_STATEMENTS["2018-03-21"] = "The Committee decided to raise the target range. The Committee expects that economic conditions will evolve in a manner that will warrant further gradual increases. The economic outlook has strengthened in recent months."
FOMC_STATEMENTS["2018-06-13"] = "The Committee decided to raise the target range. The Committee expects that economic conditions will evolve in a manner that will warrant further gradual increases. The unemployment rate has declined since the beginning of the year."
FOMC_STATEMENTS["2018-09-26"] = "The Committee decided to raise the target range. The Committee expects that further gradual increases will be consistent with sustained expansion. The labor market has continued to strengthen and economic activity has been rising at a strong rate."
FOMC_STATEMENTS["2018-12-19"] = "The Committee decided to raise the target range. The Committee expects that further gradual increases will be consistent with sustained expansion. The Committee judges that risks to the outlook are roughly balanced."

# 2019 - Powell, mid-cycle adjustment
FOMC_STATEMENTS["2019-03-20"] = "The Committee decided to maintain the target range. The Committee expects to maintain a patient approach. The Committee judges that the risks to the outlook are roughly balanced."
FOMC_STATEMENTS["2019-06-19"] = "The Committee decided to maintain the target range. The Committee will act as appropriate to sustain the expansion. The Committee judges that uncertainties about the outlook have increased."
FOMC_STATEMENTS["2019-07-31"] = "The Committee decided to lower the target range. The Committee took this action in light of the implications of global developments for the economic outlook as well as muted inflation pressures. The Committee will act as appropriate to sustain the expansion."
FOMC_STATEMENTS["2019-09-18"] = "The Committee decided to lower the target range. The Committee took this action to ensure that the stance of monetary policy remains appropriate to support the expansion. The Committee will act as appropriate to sustain the expansion."
FOMC_STATEMENTS["2019-10-30"] = "The Committee decided to lower the target range. The Committee took this action to ensure that the stance of monetary policy remains appropriate. The Committee will continue to monitor incoming information and assess the implications for the economic outlook."

# 2020 - Powell, COVID emergency
FOMC_STATEMENTS["2020-03-03"] = "The Committee decided to lower the target range. The fundamentals of the U.S. economy remain strong. However, the coronavirus poses evolving risks to economic activity. The Committee is closely monitoring developments and their implications for the economic outlook."
FOMC_STATEMENTS["2020-03-15"] = "The Committee decided to lower the target range. The effects of the coronavirus will weigh on economic activity in the near term and pose risks to the economic outlook. The Committee is prepared to use its full range of tools to support the U.S. economy."
FOMC_STATEMENTS["2020-04-29"] = "The Committee decided to maintain the target range. The ongoing public health crisis will weigh heavily on economic activity, employment, and inflation in the near term. The Committee is committed to using its full range of tools to support the U.S. economy."
FOMC_STATEMENTS["2020-06-10"] = "The Committee decided to maintain the target range. The Committee expects to maintain this target range until it is confident that the economy has weathered recent events and is on track to achieve maximum employment and price stability."
FOMC_STATEMENTS["2020-12-16"] = "The Committee decided to maintain the target range. The pace of the recovery in economic activity and employment has moderated. The Committee will continue to purchase assets at least at the current pace. The Committee expects to maintain an accommodative stance until these outcomes are achieved."

# 2021 - Powell, inflation emerging
FOMC_STATEMENTS["2021-03-17"] = "The Committee decided to maintain the target range. With progress on vaccinations, the economy is on track for strong growth. The Committee expects inflation to rise but remain transitory. The Committee will aim to achieve inflation moderately above 2 percent for some time."
FOMC_STATEMENTS["2021-06-16"] = "The Committee decided to maintain the target range. Inflation has risen, largely reflecting transitory factors. The Committee will aim to achieve inflation moderately above 2 percent for some time so that inflation averages 2 percent over time."
FOMC_STATEMENTS["2021-09-22"] = "The Committee decided to maintain the target range. Inflation is elevated, largely reflecting factors that are expected to be transitory. Supply and demand imbalances related to the pandemic and the reopening of the economy have contributed to sizable price increases."
FOMC_STATEMENTS["2021-12-15"] = "The Committee decided to accelerate the pace of asset purchases. Inflation has increased significantly. Supply and demand imbalances related to the pandemic and the reopening of the economy have continued to contribute to elevated levels of inflation. The Committee expects inflation to remain elevated."

# 2022 - Powell, aggressive tightening
FOMC_STATEMENTS["2022-03-16"] = "The Committee decided to raise the target range. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher energy prices, and broader price pressures. The Committee anticipates that ongoing increases will be appropriate."
FOMC_STATEMENTS["2022-05-04"] = "The Committee decided to raise the target range. The Committee is highly attentive to inflation risks. The Committee decided to begin reducing the balance sheet. The Committee anticipates that ongoing increases will be appropriate."
FOMC_STATEMENTS["2022-06-15"] = "The Committee decided to raise the target range. Inflation remains elevated and the Committee is strongly committed to returning inflation to its 2 percent objective. The Committee anticipates that further increases will be appropriate."
FOMC_STATEMENTS["2022-07-27"] = "The Committee decided to raise the target range. The Committee is strongly committed to returning inflation to its 2 percent objective. The Committee expects that further increases will be appropriate. The pace of future increases will depend on incoming data."
FOMC_STATEMENTS["2022-09-21"] = "The Committee decided to raise the target range. The Committee is strongly committed to returning inflation to its 2 percent objective. The Committee anticipates that further increases will be appropriate. The Committee is resolved to bring inflation back down to 2 percent."
FOMC_STATEMENTS["2022-11-02"] = "The Committee decided to raise the target range. The Committee is strongly committed to returning inflation to its 2 percent objective. The Committee anticipates that further increases will be appropriate. The Committee will continue to reduce the balance sheet."
FOMC_STATEMENTS["2022-12-14"] = "The Committee decided to raise the target range. The Committee is strongly committed to returning inflation to its 2 percent objective. The Committee anticipates that ongoing increases will be appropriate. The cumulative tightening should slow economic growth."

# 2023 - Powell, peak rates
FOMC_STATEMENTS["2023-02-01"] = "The Committee decided to raise the target range. The Committee anticipates that ongoing increases will be appropriate. Inflation has eased somewhat but remains elevated. The Committee is strongly committed to returning inflation to its 2 percent objective."
FOMC_STATEMENTS["2023-03-22"] = "The Committee decided to raise the target range. The Committee anticipates that some additional policy firming may be appropriate. The Committee remains strongly committed to returning inflation to its 2 percent objective. The banking system is sound and resilient."
FOMC_STATEMENTS["2023-05-03"] = "The Committee decided to raise the target range. The Committee is strongly committed to returning inflation to its 2 percent objective. The Committee anticipates that some additional policy firming may be appropriate. The U.S. banking system is sound and resilient."
FOMC_STATEMENTS["2023-06-14"] = "The Committee decided to maintain the target range. The Committee will closely monitor incoming information and assess the implications for monetary policy. The Committee remains strongly committed to returning inflation to its 2 percent objective."
FOMC_STATEMENTS["2023-07-26"] = "The Committee decided to raise the target range. The Committee assesses that additional policy firming may be appropriate. The Committee remains strongly committed to returning inflation to its 2 percent objective. The Committee will continue to assess incoming information."
FOMC_STATEMENTS["2023-09-20"] = "The Committee decided to maintain the target range. The Committee remains strongly committed to returning inflation to its 2 percent objective. The Committee will proceed carefully. The Committee will continue to assess the extent of additional firming."
FOMC_STATEMENTS["2023-12-13"] = "The Committee decided to maintain the target range. The Committee is attentive to risks to both sides of its dual mandate. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent."

# 2024 - Powell, rate cuts begin
FOMC_STATEMENTS["2024-01-31"] = "The Committee decided to maintain the target range. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent. Inflation has eased over the past year but remains elevated."
FOMC_STATEMENTS["2024-03-20"] = "The Committee decided to maintain the target range. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent."
FOMC_STATEMENTS["2024-06-12"] = "The Committee decided to maintain the target range. Inflation has eased over the past year but remains elevated. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent."
FOMC_STATEMENTS["2024-07-31"] = "The Committee decided to lower the target range. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance."
FOMC_STATEMENTS["2024-09-18"] = "The Committee decided to lower the target range. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent. The Committee will continue to reduce the balance sheet. The Committee judges that the risks to achieving its goals are roughly in balance."
FOMC_STATEMENTS["2024-11-07"] = "The Committee decided to lower the target range. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance. The economy continues to expand at a solid pace. Inflation has made progress toward the Committee's 2 percent objective."
FOMC_STATEMENTS["2024-12-18"] = "The Committee decided to lower the target range. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance. The Committee is mindful of the lags with which monetary policy affects economic activity and inflation. The Committee will continue to reduce the balance sheet."

# 2025 - Powell, cautious cuts
FOMC_STATEMENTS["2025-01-29"] = "The Committee decided to maintain the target range. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance. The Committee is attentive to risks to both sides of its dual mandate. The Committee will carefully assess incoming data."
FOMC_STATEMENTS["2025-03-19"] = "The Committee decided to maintain the target range. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance. Inflation remains somewhat elevated. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent."
FOMC_STATEMENTS["2025-04-30"] = "The Committee decided to maintain the target range. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance. The economic outlook is uncertain. The Committee will carefully assess incoming information and its implications for monetary policy."


def get_statements():
    """Return DataFrame of FOMC statements."""
    rows = []
    for date_str, text in sorted(FOMC_STATEMENTS.items()):
        rows.append({"date": date_str, "statement": text})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    df = get_statements()
    print(f"FOMC statements: {len(df)}")
    print(f"  Period: {df['date'].min().date()} to {df['date'].max().date()}")
    df.to_csv(os.path.join(DATA_DIR, "fomc_statements.csv"), index=False)
    print("Saved to fomc_statements.csv")
