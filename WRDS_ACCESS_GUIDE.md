# WRDS 访问方案 — 完整研究笔记

## 核心发现：WRDS 认证流程

WRDS 的所有访问方式都受 **Duo Mobile MFA（双因素认证）** 保护。这不是可选的，是强制的。

### 认证机制

1. **网页登录**：用户名 + 密码 → Duo Push → 手机点 Approve → 登录成功
2. **PostgreSQL 直连**：`wrds.Connection()` → 发起连接 → 手机收到 Duo Push → 点 Approve → 连接建立
3. **SSH 连接**：`ssh dechang@wrds-cloud.wharton.upenn.edu` → 输入密码 → 选择 MFA 方式（1=Duo Push, 2=SMS）→ 手机 Approve → 登录成功

**关键限制**：
- PostgreSQL 连接**只支持 Duo Mobile Push**，不支持 passcode/SMS/电话
- SSH 连接支持 Duo Push 和 SMS passcode
- MFA 会话可保留 30 天（IP 不变的前提下）
- **多次认证失败会自动锁号**（这就是我们之前触发的问题）

---

## 三种访问方式详解

### 方式 1：PostgreSQL 直连（推荐用于数据拉取）

**连接参数**：
```
Host: wrds-pgdata.wharton.upenn.edu
Port: 9737
Database: wrds
SSL: require
Auth: PAM + Duo Push
```

**Python 代码**：
```python
import wrds
db = wrds.Connection(wrds_username='dechang')
# 此时手机会收到 Duo Push，点 Approve
# 连接成功后可保留 30 天
```

**前提条件**：
1. WRDS 账号已激活
2. Duo Mobile 已注册（在网页登录时完成）
3. 手机在身边，能收到 Push 通知
4. IP 地址不变（云 VM 的公网 IP 需要稳定）

**从云 VM 访问的问题**：
- ✅ TCP 9737 端口可达
- ✅ SSL TLSv1.3 握手成功
- ❌ PAM 认证失败 → 可能是账号被锁（之前暴力尝试导致）
- ⚠️ Duo Push 需要人工在手机上点 Approve，无法完全自动化

**自动化方案**：
- 首次连接需要人工 Approve Duo Push
- 连接建立后，`wrds` 包会自动创建 `~/.pgpass` 文件
- 后续 30 天内（IP 不变）不需要再次 MFA
- 30 天后需要重新 Approve 一次

---

### 方式 2：WRDS Cloud SSH（推荐用于计算密集型任务）

**连接参数**：
```
密码认证: ssh dechang@wrds-cloud.wharton.upenn.edu
SSH Key:  ssh dechang@wrds-cloud-sshkey.wharton.upenn.edu
```

**SSH Key 设置**（免密码 + 仍需 MFA）：
1. 本地生成密钥：`ssh-keygen -t ed25519`
2. 在 WRDS 网页 Account → SSH Keys 上传公钥
3. 连接时使用 `wrds-cloud-sshkey.wharton.upenn.edu`
4. 仍需 Duo Push 认证（但不需要输入密码）

**WRDS Cloud 上的环境**：
- Python 3.6+（已预装 `wrds` 包）
- R 3.5
- SAS 9.4
- Stata 15（需额外许可）
- PostgreSQL 直连无需 MFA（已在 WRDS 内网）

**优势**：
- 在 WRDS 内网，PostgreSQL 连接无需 MFA
- 高性能计算集群
- 数据不出 WRDS 网络

**劣势**：
- SSH 连接仍需 Duo Push
- 计算环境受限（不能装自定义包）
- 需要把代码传上去运行

---

### 方式 3：WRDS Web Query（最简单，适合一次性数据拉取）

**流程**：
1. 登录 https://wrds-www.wharton.upenn.edu
2. 选择数据库（如 CRSP → Daily Stock）
3. 填写查询表单（日期范围、变量、筛选条件）
4. 提交查询，下载 CSV/SAS/Excel

**优势**：
- 只需浏览器
- 不需要编程
- 适合一次性大批量数据拉取

**劣势**：
- 不能自动化
- 查询灵活性有限
- 不适合频繁查询

---

## 从云 VM 访问的推荐方案

### 短期方案（账号解锁后立即可用）

**Step 1**：解锁账号
- 发邮件 wrds@wharton.upenn.edu 请求解锁
- 或在 https://wrds-www.wharton.upenn.edu/users/password/reset/ 重置密码

**Step 2**：网页登录验证
- 用浏览器登录 WRDS 网站
- 完成 Duo Push 认证
- 确认账号正常

**Step 3**：Python 直连
```python
import wrds
db = wrds.Connection(wrds_username='dechang')
# 手机点 Approve → 连接成功
# 自动创建 ~/.pgpass，30 天内免 MFA
```

**Step 4**：拉取数据
```python
# CME Fed Funds 期货
ff = db.raw_sql("""
    SELECT date, symbol, open, high, low, close, volume, oi
    FROM cme.ff
    WHERE date >= '2000-01-01'
    ORDER BY date
""")

# CRSP 市场指数
crsp = db.raw_sql("""
    SELECT date, vwretd, ewretd, sprtrn
    FROM crsp.dsi
    WHERE date >= '2000-01-01'
""")
```

### 中期方案（SSH Key + 自动化）

1. 生成 SSH Key 并上传到 WRDS
2. 通过 SSH 隧道连接 PostgreSQL（绕过 IP 限制）
3. 或直接在 WRDS Cloud 上运行 Python 脚本

### 长期方案（数据本地化）

1. 一次性从 WRDS 拉取所有需要的数据
2. 存为本地 Parquet/CSV
3. 日常分析用本地数据，定期增量更新

---

## 重要注意事项

1. **绝对不要暴力尝试认证**——多次失败会锁号
2. **Duo Push 必须在手机上手动 Approve**——无法自动化绕过
3. **IP 变化需要重新 MFA**——云 VM 重启后 IP 可能变化
4. **30 天 MFA 会话**——在 ~/.pgpass 有效期内免 MFA
5. **WRDS Cloud 内网免 MFA**——如果能在 Cloud 上跑脚本，PostgreSQL 连接不需要 Duo

---

*研究版本：v2.0 | 2026-05-20 | 曼卿*
