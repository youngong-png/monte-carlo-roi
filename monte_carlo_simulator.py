import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import os, warnings, subprocess, sys
from pathlib import Path

warnings.filterwarnings("ignore")


def setup_korean_font():
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "fonts-nanum", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        nanum_paths = list(Path(sys.prefix).rglob("NanumGothic.ttf"))
        if not nanum_paths:
            nanum_paths = list(Path("/usr").rglob("NanumGothic.ttf"))
        if nanum_paths:
            fp = str(nanum_paths[0])
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()

BINS = [0, 30, 60, 90, 120, 180, 360, 540, 720, 900, 9999]
LABELS = ["0-30일","31-60일","61-90일","91-120일","121-180일",
          "181-360일","361-540일","541-720일","721-900일","901일+"]
EXCLUDE_STAGES_DEFAULT = ["특수채권(회생/파산 등)"]

DEFAULTS = dict(
    cost_per=80000, n_sim=10000,
    partial_mean=0.95, partial_std=0.05,
    objection_rate=15.0, objection_extra_cost=300000,
    enforcement_rate=30.0, enforcement_cost=150000,
    min_value=800000, max_value=30000000,
    min_overdue=30, max_overdue=927,
)


@st.cache_data
def load_all(file):
    hist = pd.read_excel(file, sheet_name="실행관리")
    daily = pd.read_excel(file, sheet_name="연체현황_daily", header=2)
    return hist, daily


def compute_hist_rates(hist):
    ch = hist[hist["거래 - 파이프라인"] == "C(젠트)-추심"].copy()
    ch["발생일"] = pd.to_datetime(ch["채권 발생일"], errors="coerce")
    ch["성사일"] = pd.to_datetime(ch["거래 - 성사 시간"], errors="coerce")
    ch["ok"] = (ch["거래 - 상태"] == "성사됨").astype(int)
    today = pd.Timestamp.now().normalize()
    ch["연체일"] = np.where(ch["ok"]==1,
        (ch["성사일"]-ch["발생일"]).dt.days, (today-ch["발생일"]).dt.days)
    v = ch[ch["발생일"].notna() & ch["연체일"].notna()].copy()
    v["구간"] = pd.cut(v["연체일"], bins=BINS, labels=LABELS)
    g = v.groupby("구간", observed=False)["ok"].agg(["count","sum"])
    g.columns = ["전체건수","성사건수"]
    g["회수율(%)"] = (g["성사건수"]/g["전체건수"]*100).round(1).fillna(0)
    return g.reset_index().rename(columns={"구간":"연체구간"})


def fmt_krw(x, _=None):
    if abs(x) >= 1e8: return f"{x/1e8:.1f}억"
    if abs(x) >= 1e4: return f"{x/1e4:.0f}만"
    return f"{x:,.0f}"


def run_mc(targets, rate_series, cost_per, n_sim, partial_mean, partial_std,
           objection_rate, objection_extra_cost, enforcement_rate, enforcement_cost):
    amounts = targets["거래가치"].values.astype(float)
    overdue = targets["연체일수"].values.astype(float)
    n = len(amounts)
    base_cost = n * cost_per
    rate_lookup = dict(zip(LABELS, rate_series / 100.0))

    def prob(d):
        for i in range(len(BINS)-1):
            if BINS[i] < d <= BINS[i+1]: return rate_lookup[LABELS[i]]
        return rate_lookup[LABELS[-1]]

    probs = np.array([prob(d) for d in overdue])
    rng = np.random.default_rng(seed=42)
    obj_r = objection_rate / 100.0
    enf_r = enforcement_rate / 100.0

    net = np.zeros(n_sim)
    gross = np.zeros(n_sim)
    total_costs = np.zeros(n_sim)

    for s in range(n_sim):
        hit = rng.random(n) < probs
        part = np.clip(rng.normal(partial_mean, partial_std, n), 0.1, 1.0)
        recovery = np.sum(amounts[hit] * part[hit])

        obj_mask = rng.random(n) < obj_r
        n_objections = obj_mask.sum()
        enf_mask = hit & (rng.random(n) < enf_r)
        n_enforcements = enf_mask.sum()

        sim_cost = base_cost + n_objections * objection_extra_cost + n_enforcements * enforcement_cost
        gross[s] = recovery
        total_costs[s] = sim_cost
        net[s] = recovery - sim_cost

    return dict(net=net, gross=gross, total_costs=total_costs,
                base_cost=base_cost, avg_cost=total_costs.mean(),
                n=n, total_amt=amounts.sum(), probs=probs, amounts=amounts)


# ══════════════════════ 차트 ══════════════════════

def chart_scenario(r):
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = ["현재 채권 총액", "시나리오 A\n(미조치)", "시나리오 B\n(기대 회수)", "시나리오 B\n(순이익)"]
    vals = [r["total_amt"], 0, r["gross"].mean(), r["net"].mean()]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    bars = ax.bar(cats, vals, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, max(v,0)+r["total_amt"]*0.015,
                fmt_krw(v)+"원", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("법적 조치 전 vs 후 — 기대가치 비교", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_krw)); plt.tight_layout()
    return fig

def chart_dist(r):
    fig, ax = plt.subplots(figsize=(10, 5))
    d = r["net"]; m = d.mean()
    p5, p95 = np.percentile(d, [5,95])
    ax.hist(d, bins=80, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(m, color="#C44E52", ls="--", lw=2, label=f"평균: {fmt_krw(m)}원")
    ax.axvline(p5, color="#DD8452", ls=":", lw=1.5, label=f"5%ile: {fmt_krw(p5)}원")
    ax.axvline(p95, color="#55A868", ls=":", lw=1.5, label=f"95%ile: {fmt_krw(p95)}원")
    ax.axvline(0, color="black", lw=1, alpha=0.4)
    ax.set_title("순이익(Net Profit) 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("순이익 (원)"); ax.set_ylabel("빈도")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_krw))
    ax.legend(fontsize=10); plt.tight_layout()
    return fig

def chart_roi(r):
    fig, ax = plt.subplots(figsize=(10, 5))
    roi = r["net"] / r["total_costs"] * 100
    ax.hist(roi, bins=80, color="#8172B2", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(roi.mean(), color="#C44E52", ls="--", lw=2, label=f"평균 ROI: {roi.mean():.1f}%")
    ax.axvline(0, color="black", lw=1, alpha=0.4, label="BEP (0%)")
    ax.set_title("ROI 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("ROI (%)"); ax.set_ylabel("빈도")
    ax.legend(fontsize=10); plt.tight_layout()
    return fig

def chart_rates(rate_df):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(rate_df))
    ax1.bar(x, rate_df["전체건수"], color="#4C72B0", alpha=0.5, label="이력 건수")
    ax1.set_ylabel("건수", color="#4C72B0")
    ax1.set_xticks(x); ax1.set_xticklabels(rate_df["연체구간"], fontsize=8, rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(x, rate_df["회수율(%)"], "o-", color="#C44E52", lw=2, ms=7, label="회수율(%)")
    for i, v in enumerate(rate_df["회수율(%)"]):
        ax2.annotate(f"{v:.1f}%", (i,v), textcoords="offset points", xytext=(0,10),
                     ha="center", fontsize=9, fontweight="bold", color="#C44E52")
    ax2.set_ylabel("회수율 (%)", color="#C44E52")
    ax1.set_title("연체구간별 건수 & 회수율 (채권발생→종결 실데이터)", fontsize=13, fontweight="bold")
    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right"); plt.tight_layout()
    return fig

def chart_asset_value(bs, avg_cost):
    fig, ax = plt.subplots(figsize=(12, 6))
    bv = bs[bs["건수"]>0].copy()
    if bv.empty: return fig
    x = np.arange(len(bv)); w = 0.35
    ax.bar(x-w/2, [0]*len(bv), w, label="시나리오 A (미조치→0원)", color="#C44E52", alpha=0.7)
    ax.bar(x+w/2, bv["기대회수액"], w, label="시나리오 B (지급명령→기대회수)", color="#55A868", alpha=0.85)
    for i, (_,row) in enumerate(bv.iterrows()):
        ax.text(x[i]+w/2, row["기대회수액"]+bv["기대회수액"].max()*0.02,
                fmt_krw(row["기대회수액"])+"원", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2d6a2e")
    ax.set_xticks(x); ax.set_xticklabels(bv["연체구간"], fontsize=9, rotation=15)
    ax.set_title("법적 조치 유무에 따른 자산 가치 변화 (연체구간별)", fontsize=14, fontweight="bold")
    ax.set_ylabel("기대 자산가치 (원)"); ax.yaxis.set_major_formatter(FuncFormatter(fmt_krw))
    ax.legend(fontsize=10, loc="upper right")
    total_b = bv["기대회수액"].sum()
    ax.text(0.98, 0.85, f"합계 기대회수: {fmt_krw(total_b)}원\n평균 총비용: {fmt_krw(avg_cost)}원\n순이익: {fmt_krw(total_b-avg_cost)}원",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f7f0", edgecolor="#55A868", alpha=0.9))
    plt.tight_layout()
    return fig

def chart_sensitivity(r):
    fig, ax = plt.subplots(figsize=(11, 6))
    base_gross = r["gross"].mean()
    avg_cost = r["avg_cost"]
    total_amt = r["total_amt"]
    bep_rate = avg_cost / total_amt * 100

    mults = np.linspace(0.0, 2.5, 60)
    eff_rates = []; nets = []
    for m in mults:
        g = base_gross * m
        eff_rates.append(g / total_amt * 100 if total_amt else 0)
        nets.append(g - avg_cost)

    ax.fill_between(eff_rates, nets, 0, where=[n>0 for n in nets], color="#55A868", alpha=0.15, label="수익 구간")
    ax.fill_between(eff_rates, nets, 0, where=[n<=0 for n in nets], color="#C44E52", alpha=0.15, label="손실 구간")
    ax.plot(eff_rates, nets, color="#4C72B0", lw=2.5, zorder=5)

    cur_rate = base_gross / total_amt * 100 if total_amt else 0
    cur_net = base_gross - avg_cost
    ax.plot(cur_rate, cur_net, "o", color="#C44E52", ms=12, zorder=10)
    ax.annotate(f"현재: {cur_rate:.1f}%\n순이익: {fmt_krw(cur_net)}원",
                (cur_rate, cur_net), textcoords="offset points", xytext=(15,15),
                fontsize=10, fontweight="bold", color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))
    ax.axvline(bep_rate, color="#FF6B35", ls="--", lw=2, zorder=6)
    ax.annotate(f"BEP: {bep_rate:.2f}%", (bep_rate, min(nets)*0.3),
                fontsize=11, fontweight="bold", color="#FF6B35",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#FF6B35"))
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)

    safety = cur_rate / bep_rate if bep_rate > 0 else 0
    ax.text(0.02, 0.95,
            f"안전마진: 현재 회수율({cur_rate:.1f}%)은\nBEP({bep_rate:.2f}%)의 {safety:.1f}배\n→ 회수율이 {100-(bep_rate/cur_rate*100):.0f}% 하락해도 손실 없음",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor="#4CAF50", alpha=0.9))
    ax.set_title("민감도 분석: 회수율 변동에 따른 순이익 변화", fontsize=14, fontweight="bold")
    ax.set_xlabel("실효 회수율 (%)"); ax.set_ylabel("순이익 (원)")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_krw))
    ax.legend(fontsize=10, loc="lower right"); plt.tight_layout()
    return fig

def chart_cost_breakdown(r, params):
    fig, ax = plt.subplots(figsize=(8, 5))
    base = r["base_cost"]
    obj_cost = r["avg_cost"] - base - (params["enf_r"]/100 * (r["gross"].mean()/r["total_amt"]) * r["n"] * params["enf_cost"])
    enf_cost_est = r["avg_cost"] - base - max(obj_cost, 0)
    obj_cost = max(obj_cost, 0)
    enf_cost_est = max(enf_cost_est, 0)

    cats = ["지급명령\n기본비용", "이의신청\n추가비용", "강제집행\n비용", "총 비용\n(평균)"]
    vals = [base, obj_cost, enf_cost_est, r["avg_cost"]]
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#333333"]
    bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x()+b.get_width()/2, v+r["avg_cost"]*0.02,
                    fmt_krw(v)+"원", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("비용 구조 분석 (평균)", fontsize=14, fontweight="bold")
    ax.set_ylabel("비용 (원)"); ax.yaxis.set_major_formatter(FuncFormatter(fmt_krw))
    plt.tight_layout()
    return fig


# ══════════════════════ CSS ══════════════════════

CUSTOM_CSS = """
<style>
.report-card { background: linear-gradient(135deg, #667eea11, #764ba211); border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px 24px; margin: 8px 0; }
.report-card h4 { margin: 0 0 12px 0; color: #333; }
.report-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f0f0f0; }
.report-row:last-child { border-bottom: none; }
.report-label { color: #555; font-size: 14px; }
.report-value { font-weight: 700; font-size: 15px; color: #1a1a1a; }
.report-value.positive { color: #0e8a16; }
.report-value.negative { color: #cb2431; }
.big-number { text-align: center; padding: 16px; background: white; border-radius: 10px; border: 1px solid #e8e8e8; margin: 6px; }
.big-number .num { font-size: 28px; font-weight: 800; color: #1a1a1a; }
.big-number .label { font-size: 12px; color: #888; margin-top: 4px; }
.big-number .delta { font-size: 13px; margin-top: 2px; }
.delta-up { color: #0e8a16; } .delta-down { color: #cb2431; }
.conclusion-box { background: linear-gradient(135deg, #d4edda, #c3e6cb); border-left: 5px solid #28a745; border-radius: 8px; padding: 20px; margin: 16px 0; }
.conclusion-box.warn { background: linear-gradient(135deg, #fff3cd, #ffeeba); border-left-color: #ffc107; }
.conclusion-box h3 { margin: 0 0 10px 0; }
.filter-badge { display: inline-block; background: #e8f4f8; color: #0366d6; padding: 2px 10px; border-radius: 12px; font-size: 12px; margin: 2px 3px; border: 1px solid #79b8ff; }
.explain-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px 24px; margin: 12px 0; line-height: 1.8; }
.explain-box h4 { color: #495057; margin: 16px 0 8px 0; } .explain-box h4:first-child { margin-top: 0; }
.highlight { background: #fff3cd; padding: 2px 6px; border-radius: 4px; font-weight: 700; }
.safe { background: #d4edda; padding: 2px 6px; border-radius: 4px; font-weight: 700; }
.danger { background: #f8d7da; padding: 2px 6px; border-radius: 4px; font-weight: 700; }
.limit-box { background: #fff8e1; border: 1px solid #ffcc02; border-radius: 10px; padding: 16px 20px; margin: 12px 0; }
</style>
"""

def big_card(label, value, delta=None, delta_dir="up"):
    dcls = f"delta-{delta_dir}" if delta else ""
    dh = f'<div class="delta {dcls}">{delta}</div>' if delta else ""
    return f'<div class="big-number"><div class="num">{value}</div><div class="label">{label}</div>{dh}</div>'


# ══════════════════════ 메인 ══════════════════════

def main():
    st.set_page_config(page_title="지급명령 ROI 시뮬레이터", page_icon="⚖️", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("⚖️ 소멸시효 연장 지급명령 ROI 시뮬레이터")
    st.caption("몬테카를로 시뮬레이션 · 법적 조치 비용 대비 기대 회수액 분석 · 이의신청/강제집행 비용 반영")

    # 데이터
    st.sidebar.header("📂 데이터")
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "실행관리_데이터.xlsx")
    uploaded = st.sidebar.file_uploader("엑셀 업로드 (선택)", type=["xlsx"],
        help="시트1: 실행관리, 시트2: 연체현황_daily")

    if uploaded:
        src = uploaded
    elif os.path.exists(default_path):
        src = default_path
    else:
        st.warning("데이터 파일을 업로드해 주세요.")
        st.markdown("---")
        st.subheader("📋 업로드 파일 요구사항")
        st.markdown("엑셀 파일(.xlsx)에 **2개 시트**가 있어야 합니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 시트 1: `실행관리`")
            st.markdown("역사적 회수율 산출용")
            st.dataframe(pd.DataFrame({
                "필수 컬럼": ["거래 - 파이프라인", "거래 - 상태", "채권 발생일", "거래 - 성사 시간"],
                "설명": ["추심 파이프라인 구분", "성사됨/진행중 등", "채권 시작일 (날짜)", "회수 완료일 (날짜)"],
                "예시": ["C(젠트)-추심", "성사됨", "2023-05-15", "2024-01-20"],
            }), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("#### 시트 2: `연체현황_daily`")
            st.markdown("시뮬레이션 대상 채권 (헤더: 3번째 행)")
            st.dataframe(pd.DataFrame({
                "필수 컬럼": ["거래가치", "연체일수", "단계", "소멸시효 완성일"],
                "설명": ["채권 금액 (원)", "현재 연체일수", "채권 단계", "시효 만료일"],
                "예시": ["5,000,000", "120", "추심진행", "2027-06-30"],
            }), hide_index=True, use_container_width=True)
            st.markdown("**권장 컬럼**: 추심콜 담당자, 추심위임, 사건유형(법조치1), No., 이름(거래명), 잔액")
        return

    hist, daily = load_all(src)
    all_stages = sorted(daily["단계"].dropna().unique().tolist())
    all_managers = sorted(daily["추심콜 담당자"].dropna().unique().tolist())
    default_stages = [s for s in all_stages if s not in EXCLUDE_STAGES_DEFAULT]

    # 초기화
    if "rc" not in st.session_state: st.session_state["rc"] = 0
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 전체 초기화", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k != "rc": del st.session_state[k]
        st.session_state["rc"] += 1; st.rerun()
    rc = st.session_state["rc"]

    # ── 시뮬레이션 설정 ──
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 시뮬레이션 설정")
    cost_per = st.sidebar.number_input("건당 지급명령 비용 (원)", 10000, 500000, DEFAULTS["cost_per"], 5000, key=f"c_{rc}")
    n_sim = st.sidebar.slider("시뮬레이션 횟수", 1000, 20000, DEFAULTS["n_sim"], 1000, key=f"ns_{rc}")
    partial_mean = st.sidebar.slider("회수 시 평균 회수비율", 0.10, 1.0, DEFAULTS["partial_mean"], 0.05,
        help="실데이터: 성사 건의 99.5%가 전액회수 (용역수수료 특성)", key=f"pm_{rc}")
    partial_std = st.sidebar.slider("회수비율 표준편차", 0.0, 0.30, DEFAULTS["partial_std"], 0.05, key=f"ps_{rc}")

    # ── 추가 비용 (현실 반영) ──
    st.sidebar.markdown("---")
    st.sidebar.header("⚠️ 추가 비용 (현실 반영)")
    st.sidebar.caption("지급명령 후 발생 가능한 추가 비용")
    obj_rate = st.sidebar.slider("이의신청 예상 비율 (%)", 0.0, 50.0, DEFAULTS["objection_rate"], 1.0,
        help="채무자가 지급명령에 이의 제기 → 소송 전환", key=f"or_{rc}")
    obj_cost = st.sidebar.number_input("이의신청 시 추가비용 (원/건)", 0, 2000000, DEFAULTS["objection_extra_cost"], 50000,
        help="소송 전환 시 변호사비·인지대 등", key=f"oc_{rc}")
    enf_rate = st.sidebar.slider("강제집행 필요 비율 (%)", 0.0, 80.0, DEFAULTS["enforcement_rate"], 5.0,
        help="회수 성공 건 중 강제집행(압류 등) 필요한 비율", key=f"er_{rc}")
    enf_cost = st.sidebar.number_input("강제집행 비용 (원/건)", 0, 1000000, DEFAULTS["enforcement_cost"], 10000,
        help="압류·추심 등 집행비용", key=f"ec_{rc}")

    # ── 대상 필터 ──
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 대상 채권 필터")
    min_val = st.sidebar.number_input("최소 거래가치 (원)", 0, 50000000, DEFAULTS["min_value"], 50000, key=f"mv_{rc}")
    max_val = st.sidebar.number_input("최대 거래가치 (원)", 0, 100000000, DEFAULTS["max_value"], 1000000, key=f"xv_{rc}")
    min_od = st.sidebar.number_input("최소 연체일수", 0, 9999, DEFAULTS["min_overdue"], 10, key=f"mo_{rc}")
    max_od = st.sidebar.number_input("최대 연체일수", 0, 9999, DEFAULTS["max_overdue"], 10, key=f"xo_{rc}")

    st.sidebar.markdown("##### 단계 필터")
    st.sidebar.caption("기본: 특수채권(회생/파산) 제외")
    sel_stages = st.sidebar.multiselect("포함 단계", all_stages, default=default_stages, key=f"stg_{rc}")

    st.sidebar.markdown("##### 담당자")
    mgr_all = st.sidebar.checkbox("전체 담당자", True, key=f"ma_{rc}")
    if not mgr_all:
        sel_mgrs = st.sidebar.multiselect("선택", all_managers, default=all_managers, key=f"mg_{rc}")

    st.sidebar.markdown("##### 추심위임")
    out_all = st.sidebar.checkbox("전체", True, key=f"oa_{rc}")
    if not out_all:
        out_opt = st.sidebar.radio("추심위임", ["위임건만","미위임건만"], key=f"oo_{rc}")

    excl_legal = st.sidebar.checkbox("기존 법적조치 건 제외", False, key=f"el_{rc}")

    st.sidebar.markdown("##### 소멸시효")
    sihyo_opt = st.sidebar.radio("소멸시효", ["전체","기간 내 만료건만"], key=f"so_{rc}")
    sihyo_deadline = None
    if sihyo_opt == "기간 내 만료건만":
        sihyo_deadline = st.sidebar.date_input("만료 기한", pd.Timestamp("2027-12-31"), key=f"sd_{rc}")

    # 필터 적용
    df = daily.copy()
    df = df[(df["거래가치"]>=min_val)&(df["거래가치"]<=max_val)]
    df = df[(df["연체일수"]>=min_od)&(df["연체일수"]<=max_od)]
    if sel_stages: df = df[df["단계"].isin(sel_stages)]
    if not mgr_all: df = df[df["추심콜 담당자"].isin(sel_mgrs)]
    if not out_all:
        df = df[df["추심위임"].notna()] if out_opt=="위임건만" else df[df["추심위임"].isna()]
    if excl_legal: df = df[df["사건유형(법조치1)"].isna()]
    if sihyo_deadline:
        df = df[pd.to_datetime(df["소멸시효 완성일"], errors="coerce") <= pd.Timestamp(sihyo_deadline)]
    filtered = df.reset_index(drop=True)

    badges = [f"가치: {min_val:,}~{max_val:,}원", f"연체: {min_od}~{max_od}일"]
    excl = [s for s in all_stages if s not in sel_stages]
    if excl: badges.append(f"제외: {', '.join(excl)}")
    if not mgr_all: badges.append(f"담당자: {len(sel_mgrs)}명")
    if not out_all: badges.append(f"추심위임: {out_opt}")
    if excl_legal: badges.append("기존법조치 제외")
    if sihyo_deadline: badges.append(f"시효≤{sihyo_deadline}")

    # ══════════════ 회수율 ══════════════
    hist_rates = compute_hist_rates(hist)
    st.markdown("---")
    st.subheader("📊 연체구간별 회수율 — 실데이터 기반 (채권발생→종결)")
    st.caption("성사 건=실제 회수 소요일, 진행 중=현재 연체일수 기준. **회수율(%)** 직접 수정 가능.")
    cc, ct = st.columns([1.3, 1])
    with ct:
        edited_rates = st.data_editor(hist_rates,
            column_config={"연체구간": st.column_config.TextColumn("연체구간", disabled=True),
                           "전체건수": st.column_config.NumberColumn("전체건수", disabled=True),
                           "성사건수": st.column_config.NumberColumn("성사건수", disabled=True),
                           "회수율(%)": st.column_config.NumberColumn("✏️ 회수율(%)", min_value=0, max_value=100, step=0.5)},
            hide_index=True, use_container_width=True, key=f"re_{rc}")
    with cc:
        st.pyplot(chart_rates(edited_rates)); plt.close()

    # ══════════════ 대상 채권 ══════════════
    st.markdown("---")
    st.subheader("✏️ 시뮬레이션 대상 채권")
    st.markdown("적용 필터: " + " ".join(f'<span class="filter-badge">{b}</span>' for b in badges), unsafe_allow_html=True)

    display = filtered[["No.","단계","이름(거래명)","거래가치","잔액","연체일수","소멸시효 완성일","추심콜 담당자","추심위임","사건유형(법조치1)"]].copy()
    display["시뮬대상"] = True
    edited_targets = st.data_editor(display,
        column_config={"시뮬대상": st.column_config.CheckboxColumn("대상", default=True),
            "No.": st.column_config.NumberColumn("No.", disabled=True),
            "단계": st.column_config.TextColumn("단계", disabled=True),
            "이름(거래명)": st.column_config.TextColumn("이름", disabled=True),
            "거래가치": st.column_config.NumberColumn("거래가치(원)", min_value=0, format="%d"),
            "잔액": st.column_config.NumberColumn("잔액(원)", disabled=True, format="%d"),
            "연체일수": st.column_config.NumberColumn("연체일수", min_value=0),
            "소멸시효 완성일": st.column_config.DateColumn("소멸시효", disabled=True),
            "추심콜 담당자": st.column_config.TextColumn("담당자", disabled=True),
            "추심위임": st.column_config.TextColumn("추심위임", disabled=True),
            "사건유형(법조치1)": st.column_config.TextColumn("법적조치", disabled=True)},
        hide_index=True, use_container_width=True, height=420, num_rows="dynamic", key=f"te_{rc}")
    sim_targets = edited_targets[edited_targets["시뮬대상"]==True].copy()

    # ══════════════ 요약 ══════════════
    st.markdown("---"); st.subheader("📋 대상 요약")
    sihyo_s = pd.to_datetime(sim_targets["소멸시효 완성일"], errors="coerce")
    n_tgt = len(sim_targets)
    total_val = sim_targets["거래가치"].sum() if n_tgt else 0
    avg_val = sim_targets["거래가치"].mean() if n_tgt else 0
    avg_od = sim_targets["연체일수"].mean() if n_tgt else 0

    r1 = st.columns(5)
    r1[0].markdown(big_card("대상 건수", f"{n_tgt:,}건", f"전체 {len(daily):,}건 중", "up"), unsafe_allow_html=True)
    r1[1].markdown(big_card("채권 총액", f"{fmt_krw(total_val)}원"), unsafe_allow_html=True)
    r1[2].markdown(big_card("평균 채권금액", f"{fmt_krw(avg_val)}원"), unsafe_allow_html=True)
    r1[3].markdown(big_card("평균 연체일수", f"{avg_od:.0f}일"), unsafe_allow_html=True)
    r1[4].markdown(big_card("기본 법적비용", f"{fmt_krw(n_tgt*cost_per)}원", f"건당 {cost_per:,}원", "down"), unsafe_allow_html=True)

    r2 = st.columns(4)
    for col, lbl, cond in [
        (r2[0], "2026년 시효만료", sihyo_s < "2027-01-01"),
        (r2[1], "2027 상반기", (sihyo_s >= "2027-01-01") & (sihyo_s < "2027-07-01")),
        (r2[2], "2027 하반기", (sihyo_s >= "2027-07-01") & (sihyo_s < "2028-01-01")),
        (r2[3], "2028 이후", sihyo_s >= "2028-01-01")]:
        cnt = int(cond.sum())
        d = ("긴급","down") if "2026" in lbl else (None, "up")
        col.markdown(big_card(lbl, f"{cnt}건", d[0], d[1]), unsafe_allow_html=True)

    bucket_summary = None
    if n_tgt:
        stc = sim_targets.copy()
        stc["연체구간"] = pd.cut(stc["연체일수"], bins=BINS, labels=LABELS)
        bucket_summary = stc.groupby("연체구간", observed=False).agg(건수=("거래가치","count"), 금액합계=("거래가치","sum")).reset_index()
        bucket_summary["적용회수율(%)"] = edited_rates["회수율(%)"].values
        bucket_summary["기대회수액"] = (bucket_summary["금액합계"] * bucket_summary["적용회수율(%)"] / 100 * partial_mean).astype(int)
        st.dataframe(bucket_summary.style.format({"금액합계":"{:,.0f}","적용회수율(%)":"{:.1f}","기대회수액":"{:,.0f}"}),
                     use_container_width=True, hide_index=True)

    # ══════════════ 실행 ══════════════
    st.markdown("---")
    if st.button("🚀 몬테카를로 시뮬레이션 실행", type="primary", use_container_width=True):
        if not n_tgt: st.error("대상 0건입니다."); return
        with st.spinner(f"몬테카를로 {n_sim:,}회 시뮬레이션 중..."):
            r = run_mc(sim_targets, edited_rates["회수율(%)"].values,
                       cost_per, n_sim, partial_mean, partial_std,
                       obj_rate, obj_cost, enf_rate, enf_cost)
        st.session_state["results"] = r
        st.session_state["params"] = dict(cost=cost_per, n_sim=n_sim, pm=partial_mean, ps=partial_std,
                                          obj_r=obj_rate, obj_c=obj_cost, enf_r=enf_rate, enf_cost=enf_cost, n_targets=n_tgt)
        st.session_state["bucket_summary"] = bucket_summary

    if "results" not in st.session_state: return
    r = st.session_state["results"]; p = st.session_state["params"]; bs_saved = st.session_state.get("bucket_summary")

    # ══════════════ 결과 ══════════════
    st.markdown("---"); st.subheader("📈 시뮬레이션 결과")
    net_m = r["net"].mean(); gross_m = r["gross"].mean()
    avg_total_cost = r["avg_cost"]
    roi_m = net_m / avg_total_cost * 100 if avg_total_cost else 0
    prob_pos = (r["net"]>0).mean()*100
    p5,p25,p50,p75,p95 = np.percentile(r["net"],[5,25,50,75,95])
    bep = avg_total_cost / r["total_amt"] * 100 if r["total_amt"] else 0

    rc1 = st.columns(4)
    rc1[0].markdown(big_card("평균 순이익", f"{fmt_krw(net_m)}원", f"ROI {roi_m:+.1f}%", "up" if roi_m>0 else "down"), unsafe_allow_html=True)
    rc1[1].markdown(big_card("평균 기대 회수액", f"{fmt_krw(gross_m)}원"), unsafe_allow_html=True)
    rc1[2].markdown(big_card("평균 총비용", f"{fmt_krw(avg_total_cost)}원", f"기본{fmt_krw(r['base_cost'])}+추가비용", "down"), unsafe_allow_html=True)
    rc1[3].markdown(big_card("수익 확률", f"{prob_pos:.1f}%", "양(+)수익", "up" if prob_pos>50 else "down"), unsafe_allow_html=True)

    tabs = st.tabs(["📊 시나리오 비교","🏢 자산가치 변화","🔬 민감도 분석","📈 순이익 분포","💰 ROI 분포","💸 비용 구조","📑 상세 리포트"])

    with tabs[0]: st.pyplot(chart_scenario(r)); plt.close()
    with tabs[1]:
        if bs_saved is not None: st.pyplot(chart_asset_value(bs_saved, avg_total_cost)); plt.close()
    with tabs[2]:
        st.pyplot(chart_sensitivity(r)); plt.close()
        cur_rate = gross_m/r["total_amt"]*100 if r["total_amt"] else 0
        safety = cur_rate/bep if bep else 0
        st.markdown(f"""<div class="explain-box"><h4>🔬 민감도 분석 해석</h4>
        <p>BEP(손익분기점) = 총비용을 회수하기 위한 최소 회수율입니다.</p>
        <ul><li>BEP: <span class="safe">{bep:.2f}%</span> — 전체 채권의 {bep:.2f}%만 회수하면 본전</li>
        <li>현재 기대 회수율: <span class="highlight">{cur_rate:.1f}%</span></li>
        <li>안전마진: BEP의 <span class="safe">{safety:.1f}배</span></li></ul>
        <p>회수율이 현재의 <b>1/{safety:.0f}</b>로 떨어져야 비로소 손실이 발생합니다.</p></div>""", unsafe_allow_html=True)
    with tabs[3]: st.pyplot(chart_dist(r)); plt.close()
    with tabs[4]: st.pyplot(chart_roi(r)); plt.close()
    with tabs[5]: st.pyplot(chart_cost_breakdown(r, p)); plt.close()
    with tabs[6]:
        for title, items in [
            ("💼 투자 개요", [("대상 채권 수",f"{r['n']:,}건"),("미회수 원금 합계",f"{r['total_amt']:,.0f}원"),
                            ("기본 법적 비용",f"{r['base_cost']:,.0f}원"),("평균 총비용 (추가비용 포함)",f"{avg_total_cost:,.0f}원"),
                            ("시뮬레이션 횟수",f"{p['n_sim']:,}회")]),
            ("📊 시나리오 비교", [("시나리오 A (미조치)","0원 — 전액손실","negative"),
                               ("시나리오 B 기대 회수",f"{gross_m:,.0f}원","positive"),
                               ("시나리오 B 순이익",f"{net_m:,.0f}원","positive" if net_m>0 else "negative")]),
            ("📈 핵심 지표", [("평균 ROI",f"{roi_m:+.1f}%","positive" if roi_m>0 else "negative"),
                            ("수익 확률",f"{prob_pos:.1f}%","positive" if prob_pos>50 else "negative"),
                            ("BEP 필요 회수율",f"{bep:.2f}%","")])]:
            st.markdown(f'<div class="report-card"><h4>{title}</h4>', unsafe_allow_html=True)
            for item in items:
                lbl, val = item[0], item[1]
                cls = item[2] if len(item)>2 else ""
                st.markdown(f'<div class="report-row"><span class="report-label">{lbl}</span><span class="report-value {cls}">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card"><h4>📉 리스크 분석</h4>', unsafe_allow_html=True)
        for lbl, val in [("5%ile (최악)",f"{p5:,.0f}원"),("25%ile",f"{p25:,.0f}원"),("50%ile (중앙값)",f"{p50:,.0f}원"),("75%ile",f"{p75:,.0f}원"),("95%ile (최선)",f"{p95:,.0f}원")]:
            cls = "positive" if float(val.replace(",","").replace("원",""))>0 else "negative"
            st.markdown(f'<div class="report-row"><span class="report-label">{lbl}</span><span class="report-value {cls}">{val}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if roi_m > 0:
            st.markdown(f"""<div class="conclusion-box"><h3>✅ 결론: 지급명령 진행이 경제적으로 유리합니다</h3>
            <ul><li>투자금 <b>{fmt_krw(avg_total_cost)}원</b>(추가비용 포함) 대비 평균 <b>{fmt_krw(net_m)}원</b> 순이익</li>
            <li>ROI <b>{roi_m:+.1f}%</b> · 수익 확률 <b>{prob_pos:.1f}%</b></li>
            <li>미조치 시 <b>{fmt_krw(r['total_amt'])}원 전액 손실</b></li>
            <li>BEP 필요 회수율 <b>{bep:.2f}%</b> (안전마진 {safety:.1f}배)</li></ul></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="conclusion-box warn"><h3>⚠️ ROI 음수</h3><p>ROI <b>{roi_m:+.1f}%</b> — 조건 조정 필요</p></div>""", unsafe_allow_html=True)

        # ══════════════ 쉬운 설명 + 현업 분석 ══════════════
        st.markdown("---")
        cur_rate = gross_m/r["total_amt"]*100 if r["total_amt"] else 0
        safety = cur_rate/bep if bep else 0
        bep_cases = max(1, int(r["n"] * bep / 100 / partial_mean)) if partial_mean else 1

        st.subheader("📖 이 분석이 뭔가요? (채권을 모르는 분도 읽어보세요)")
        st.info(
            '우리 회사가 갖고 있는 **"받을 돈(채권)"**에는 유효기간(소멸시효 3년)이 있습니다. '
            '기간이 지나면 **돈을 받을 권리 자체가 사라집니다**. '
            '법원에 **"지급명령"**이라는 서류를 내면, 유효기간을 **10년 연장**할 수 있습니다.'
        )

        st.markdown("#### 💡 왜 해야 하나요?")
        col_a, col_b = st.columns(2)
        with col_a:
            st.error(f"**시나리오 A: 아무것도 안 하면**\n\n"
                     f"대상 채권 {r['n']:,}건, {fmt_krw(r['total_amt'])}원이\n\n"
                     f"시효 만료로 **전부 사라집니다** (되돌릴 수 없음)")
        with col_b:
            st.success(f"**시나리오 B: 지급명령을 넣으면**\n\n"
                       f"비용을 쓰고 평균 **{fmt_krw(gross_m)}원을 회수**\n\n"
                       f"시효를 10년 연장하여 회수 기회 확보")

        st.markdown("#### 💰 그래서 남는 게 있나요?")
        st.markdown(
            f"기본 비용(건당 {cost_per//10000}만원) + 이의신청·강제집행 추가비용까지 합쳐서 "
            f"평균 **{fmt_krw(avg_total_cost)}원**이 들고, "
            f"빼고 나면 **{fmt_krw(net_m)}원의 순이익**이 남습니다.\n\n"
            f"ROI **{roi_m:+.1f}%**"
            + (f" — **1원을 쓰면 {roi_m/100:.1f}원이 돌아온다**는 뜻입니다." if roi_m > 0 else "")
        )

        st.markdown("#### 📉 혹시 손해 볼 수도 있나요?")
        st.markdown(
            f"손해를 보려면 전체 회수율이 **{bep:.2f}%** 이하여야 합니다.\n\n"
            f"이것은 {r['n']:,}건 중 **약 {bep_cases}건**도 회수 못해야 한다는 뜻입니다.\n\n"
            f"현재 기대 회수율({cur_rate:.1f}%)은 BEP의 **{safety:.1f}배**이므로, "
            f"**손실 가능성은 극히 낮습니다**.\n\n"
            f"시뮬레이션 {p['n_sim']:,}회 중 수익 확률 **{prob_pos:.1f}%**, "
            f"최악(하위 5%)에도 순이익 **{fmt_krw(p5)}원**입니다."
        )

        # ── 현업 분석 ──
        st.markdown("---")
        st.subheader("🏢 현업 분석: 세무/회계 용역 수수료 채권의 특수성")

        st.markdown("#### 1. 우리 채권의 특징 (일반 채권과 다른 점)")
        st.dataframe(pd.DataFrame({
            "항목": ["채권 성격", "회수 시 금액", "평균 채권 금액", "채무자 유형", "소멸시효"],
            "일반 소비자 대출": ["원금+이자", "분할상환/일부감면 빈번", "수백~수천만원", "개인/가계", "5~10년"],
            "우리 (용역 수수료)": [
                "용역 대가 (세무/회계 서비스 비용)",
                "✅ 전액 회수 99.5% (내거나 안 내거나)",
                f"{fmt_krw(avg_val)}원 (상대적 소액)",
                "개인사업자/법인 (사업활동 중이면 유리)",
                "⚠️ 3년 (짧음 → 조기 대응 중요)",
            ],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### 2. 시뮬레이션 정확도 체크")
        check_pm = "✅ 실데이터 반영" if partial_mean >= 0.9 else "⚠️ 보수적"
        check_cost = "✅ 현실적" if 50000 <= cost_per <= 100000 else "⚠️ 확인"
        check_obj = "✅ 현실적" if 10 <= obj_rate <= 20 else "⚠️ 확인"
        check_enf = "✅ 현실적" if 20 <= enf_rate <= 40 else "⚠️ 확인"
        check_safe = "✅ 충분" if safety >= 3 else ("✅ 양호" if safety >= 2 else "⚠️ 부족")
        st.dataframe(pd.DataFrame({
            "항목": ["회수율 산출 방식", "회수 시 부분회수율", "건당 법적 비용", "이의신청률", "강제집행 비율", "BEP 안전마진"],
            "업계 기준": ["보통 추정치 사용", "소비자대출 40~70%", "인지대+송달료 5~10만원", "소액채권 10~20%", "회수건의 20~40%", "3배 이상 권장"],
            "이 시뮬레이션": [
                "실제 12,112건 이력 기반",
                f"{partial_mean*100:.0f}% (실데이터 99.5%)",
                f"{cost_per//10000}만원",
                f"{obj_rate:.0f}%",
                f"{enf_rate:.0f}%",
                f"{safety:.1f}배",
            ],
            "판단": ["✅ 우수", check_pm, check_cost, check_obj, check_enf, check_safe],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### 3. 이 시뮬레이션이 반영하지 못하는 것 (한계점)")
        st.warning(
            "- **화폐의 시간가치**: 10년에 걸쳐 회수되는 금액을 현재가치로 할인하지 않았습니다.\n"
            "- **내부 인건비**: 서류 작성·법원 방문·관리에 드는 직원 시간 비용이 미포함.\n"
            "- **채무자 무자력**: 사업 폐업·파산으로 재산이 없으면 판결 받아도 회수 불가.\n"
            "- **주소불명/송달불능**: 주소 변경으로 지급명령 송달 불가 시 공시송달 추가비용 발생.\n"
            "- **회수 시점 불확실성**: '언제' 회수되는지는 예측하지 않습니다."
        )

        st.markdown("#### 4. 그럼에도 진행해야 하는 이유")
        st.markdown("위 한계점들을 모두 고려해도 핵심 논리는 변하지 않습니다:")
        st.success(
            f"**\"안 하면 확정 손실 {fmt_krw(r['total_amt'])}원. "
            f"하면 비용 {fmt_krw(avg_total_cost)}원으로 회수 가능성을 10년 확보. "
            f"비용은 전체 채권액의 {bep:.1f}%에 불과하므로, 극소수만 회수해도 본전.\"**"
        )

        st.markdown("#### 📌 한 줄 요약")
        st.markdown(
            f"> **\"건당 {cost_per//10000}만원(추가비용 포함 평균 {fmt_krw(avg_total_cost/r['n'])}원)을 써서 시효를 연장하면, "
            f"{r['n']:,}건 중 극소수만 회수해도 본전이고, 현실적으로 약 {fmt_krw(net_m)}원의 수익이 기대됩니다. "
            f"안 하면 {fmt_krw(r['total_amt'])}원을 버리는 것입니다.\"**"
        )

        st.markdown("#### 적용된 회수율 기준")
        st.dataframe(edited_rates, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
