import pandas as pd
import numpy as np

CSV = "E:/Hackathon/archive/PS_20174392719_1491204439457_log.csv"

# 采样50万行进行账户级分析（已知总量630万，比例约8%）
df = pd.read_csv(CSV)
ft = df[df['type'].isin(['TRANSFER','CASH_OUT'])].copy()
print(f"样本: {len(df):,} 行 | TRANSFER+CASH_OUT: {len(ft):,}")

# -- 欺诈 vs 正常账户特征 --
stats = ft.groupby('nameOrig').agg(
    total_sent=('amount','sum'),
    tx_count=('amount','count'),
    avg_amount=('amount','mean'),
    drain_rate=('newbalanceOrig', lambda x: (x==0).mean()),
    is_fraud=('isFraud','max')
).reset_index()
fs = stats[stats['is_fraud']==1]
ns = stats[stats['is_fraud']==0]

print(f"\n=== 欺诈 vs 正常账户特征对比（样本） ===")
print(f"欺诈账户: {len(fs)}  |  正常账户: {len(ns):,}")
print(f"{'指标':22s}  {'欺诈':>14s}  {'正常':>14s}")
print("-"*54)
for col, label in [
    ('total_sent',  '总发送金额(中位)'),
    ('tx_count',    '交易笔数(中位)'),
    ('avg_amount',  '平均金额(中位)'),
    ('drain_rate',  '余额清零比(均值)'),
]:
    fv = fs[col].median() if col!='drain_rate' else float(fs[col].mean())
    nv = ns[col].median() if col!='drain_rate' else float(ns[col].mean())
    print(f"{label:22s}  {fv:>14,.2f}  {nv:>14,.2f}")

# -- 收款账户入度 Top10 --
print("\n=== 收款账户入度 Top10（含欺诈率）===")
recv = ft.groupby('nameDest').agg(
    in_degree=('nameOrig','count'),
    fraud_recv=('isFraud','sum'),
    total_recv_K=('amount', lambda x: x.sum()/1000)
).sort_values('in_degree', ascending=False).head(10)
recv['fraud_rate'] = (recv['fraud_recv']/recv['in_degree']).round(4)
print(recv.round(1).to_string())

# -- 时间周期 --
print("\n=== 每小时交易量周期 ===")
step_counts = df.groupby('step').size()
hourly = step_counts.reset_index()
hourly.columns = ['step','tx']
hourly['hour'] = hourly['step'] % 24
pat = hourly.groupby('hour')['tx'].mean()
print(f"峰值: {pat.idxmax():2d}:00  均值={pat.max():.0f}")
print(f"低谷: {pat.idxmin():2d}:00  均值={pat.min():.0f}")
print(f"峰谷比: {pat.max()/pat.min():.1f}x")
for h, v in pat.items():
    bar = '#' * max(1, int(v/20))
    print(f"  {h:2d}h {v:6.0f} {bar}")

# -- 金额分布对比 --
print("\n=== 金额分布：欺诈 vs 正常（TRANSFER+CASHOUT中）===")
fraud_amt  = ft[ft['isFraud']==1]['amount']
normal_amt = ft[ft['isFraud']==0]['amount']
print(f"{'':10s} {'欺诈':>15s} {'正常':>15s}")
for label, fn in [('均值','mean'),('中位数','median'),('P75',''),('P95',''),('P99','')]:
    if fn:
        fv = getattr(fraud_amt, fn)()
        nv = getattr(normal_amt, fn)()
    else:
        p = int(label[1:])
        fv = np.percentile(fraud_amt, p)
        nv = np.percentile(normal_amt, p)
    print(f"  {label:8s} {fv:>15,.2f} {nv:>15,.2f}")
