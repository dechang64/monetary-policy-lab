# -*- coding: utf-8 -*-
"""
查看CRSP Mutual Fund关键表的列名
用法: python inspect_tables.py 你的WRDS用户名
"""
import sys
import wrds

def inspect(username):
    db = wrds.Connection(wrds_username=username)

    tables_to_check = [
        ("crsp_q_mutualfunds", "fund_hdr"),
        ("crsp_q_mutualfunds", "fund_hdr_hist"),
        ("crsp_q_mutualfunds", "monthly_tna_ret_nav"),
        ("crsp_q_mutualfunds", "fund_names"),
        ("crsp_q_mutualfunds", "fund_style"),
        ("crsp_q_mutualfunds", "fund_summary2"),
    ]

    for lib, tbl in tables_to_check:
        print(f"\n{'='*60}")
        print(f"{lib}.{tbl}")
        print(f"{'='*60}")
        try:
            # 列出列名
            cols = db.describe_table(library=lib, table=tbl)
            print("列名:")
            for col in cols:
                print(f"  {col}")
            # 拉2行样本数据
            sample = db.raw_sql(f"SELECT * FROM {lib}.{tbl} LIMIT 2")
            print(f"\n样本数据 (2行):")
            print(sample.to_string())
        except Exception as e:
            print(f"  ❌ {e}")

    db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_tables.py 你的WRDS用户名")
        sys.exit(1)
    inspect(sys.argv[1])
