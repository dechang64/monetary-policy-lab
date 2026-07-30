# -*- coding: utf-8 -*-
"""
WRDS最小连接测试脚本
用法: python test_wrds_connection.py 你的WRDS用户名
"""

import sys
import time

def test_wrds(username):
    print("=" * 50)
    print(f"WRDS连接测试 | 用户: {username}")
    print("=" * 50)

    # Step 1: 检查wrds包是否安装
    print("\n[1/4] 检查wrds包...")
    try:
        import wrds
        print("  ✅ wrds包已安装")
    except ImportError:
        print("  ❌ wrds包未安装")
        print("  → 运行: pip install wrds")
        return False

    # Step 2: 检查网络连通性
    print("\n[2/4] 检查WRDS服务器网络连通性...")
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        result = sock.connect_ex(("wrds-cloud.wharton.upenn.edu", 22))
        if result == 0:
            print("  ✅ WRDS SSH端口(22)可达")
        else:
            print(f"  ❌ SSH端口不可达 (错误码: {result})")
            print("  → 网络/防火墙可能挡了SSH隧道，建议用WRDS JupyterHub")
            sock.close()
            return False
    except Exception as e:
        print(f"  ❌ 网络连接失败: {e}")
        print("  → 建议: 登录 wrds-www.wharton.upenn.edu → Compute → JupyterHub")
        sock.close()
        return False
    finally:
        sock.close()

    # Step 3: 尝试连接WRDS
    print("\n[3/4] 尝试连接WRDS（会提示输入密码）...")
    try:
        start = time.time()
        db = wrds.Connection(wrds_username=username)
        elapsed = time.time() - start
        print(f"  ✅ 连接成功! (耗时 {elapsed:.1f}s)")
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        print("  → 检查用户名/密码是否正确")
        return False

    # Step 4: 列出可用的CRSP mutual fund表
    print("\n[4/4] 查找CRSP Mutual Fund表...")
    try:
        # 用wrds内置方法列出库
        libs = db.list_libraries()
        crsp_libs = [l for l in libs if 'crsp' in l.lower()]
        print(f"  可访问的CRSP库: {crsp_libs}")

        # 在crsp mutual funds库里列出表
        for lib in crsp_libs:
            if 'mutual' in lib.lower() or 'mf' in lib.lower():
                tbls = db.list_tables(library=lib)
                print(f"\n  {lib} 库中的表 ({len(tbls)} 个):")
                for t in tbls[:20]:
                    print(f"    {t}")
                if len(tbls) > 20:
                    print(f"    ... 还有 {len(tbls)-20} 个")

        db.close()
        print("\n" + "=" * 50)
        print("🎉 连接成功! 把上面列出的表名贴给我")
        print("   我来确认pipeline代码里的表名是否匹配")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"  ❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        db.close()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_wrds_connection.py 你的WRDS用户名")
        sys.exit(1)
    test_wrds(sys.argv[1])
