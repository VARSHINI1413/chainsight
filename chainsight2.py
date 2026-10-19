"""
ChainSight 2: AI-Driven Cross-Border Supply Chain Risk & ESG Analytics (near-real-time version), ONE file.

SETUP
    pip install pandas numpy scikit-learn joblib streamlit plotly requests
    (optional, better NLP: pip install transformers torch, then set USE_TRANSFORMERS=1)

COMMANDS (run in the terminal, inside this folder)
    python chainsight2.py            -> first-time build: companies, simulated shipments, model, risk table
    python chainsight2.py refresh    -> fetch LIVE news once, re-score everything
    python chainsight2.py loop       -> refresh every 60 minutes (leave the terminal open; Ctrl+C stops)
    python chainsight2.py loop 30    -> refresh every 30 minutes
    streamlit run chainsight2.py     -> open the dashboard

WHAT IS REAL AND WHAT IS SIMULATED (say this in your application)
    REAL      : news headlines about real companies, pulled from the free GDELT news API.
    SIMULATED : shipment records and annual spend (companies do not publish these).
    ASSUMED   : 60/40 weights in the composite score, the 15% loss rate, and the small
                "logistics news" bump to delay probability. All are yours to justify.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data_chainsight2")   # separate folder, so your original project's data is never touched
FEATURES = ["lead_time_days", "defect_rate", "volume", "port_congestion", "weather_risk", "geo_risk"]
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_WINDOW_DAYS = 30

# ISO3 code for the map, baseline geopolitical risk 0-1 (an assumption)
COUNTRIES = {
    "Taiwan": ("TWN", 0.45), "South Korea": ("KOR", 0.25), "Japan": ("JPN", 0.15),
    "China": ("CHN", 0.50), "India": ("IND", 0.35), "Germany": ("DEU", 0.10),
    "United States": ("USA", 0.10),
}
# name, country, part, news search query, word that must appear in a headline to count it
COMPANIES = [
    ("TSMC", "Taiwan", "Chips", '"TSMC"', "tsmc"),
    ("Foxconn", "Taiwan", "Assembly", '"Foxconn"', "foxconn"),
    ("MediaTek", "Taiwan", "Chips", '"MediaTek"', "mediatek"),
    ("Samsung Electronics", "South Korea", "Memory", '"Samsung Electronics"', "samsung"),
    ("SK Hynix", "South Korea", "Memory", '"SK Hynix"', "hynix"),
    ("LG Energy Solution", "South Korea", "Batteries", '"LG Energy Solution"', "lg energy"),
    ("Hyundai Mobis", "South Korea", "Auto parts", '"Hyundai Mobis"', "mobis"),
    ("Sony", "Japan", "Sensors", '"Sony Group"', "sony"),
    ("Murata", "Japan", "Components", '"Murata Manufacturing"', "murata"),
    ("CATL", "China", "Batteries", '"CATL"', "catl"),
    ("BYD", "China", "Batteries", '"BYD"', "byd"),
    ("BOE Technology", "China", "Displays", '"BOE Technology"', "boe"),
    ("Tata Electronics", "India", "Assembly", '"Tata Electronics"', "tata"),
    ("Infineon", "Germany", "Chips", '"Infineon"', "infineon"),
    ("Intel", "United States", "Chips", '"Intel Corporation"', "intel"),
    ("Texas Instruments", "United States", "Chips", '"Texas Instruments"', "texas instruments"),
]

# Extra well-known companies offered in the dashboard's dropdown (live look-up only, not tracked)
EXTRA_CHOICES = ["Apple", "Nvidia", "Toyota", "Tesla", "Microsoft", "Amazon", "Google", "Qualcomm", "AMD",
                 "Broadcom", "Micron", "ASML", "Nokia", "Ericsson", "Siemens", "Bosch", "Volkswagen", "BMW",
                 "Panasonic", "Toshiba", "Hitachi", "Huawei", "Xiaomi", "Lenovo", "Alibaba", "Tencent",
                 "Reliance Industries", "Infosys", "Tata Motors", "Maersk", "DHL", "FedEx", "Nike", "Walmart"]
COMPANY_CHOICES = sorted(set([c[0] for c in COMPANIES] + EXTRA_CHOICES))


# ============================================================ 1. DATA (companies real, shipments simulated)
def generate_data():
    rng = np.random.default_rng(42)
    DATA.mkdir(exist_ok=True)

    n = len(COMPANIES)
    sup = pd.DataFrame(COMPANIES, columns=["supplier_name", "country", "part", "query", "match"])
    sup.insert(0, "supplier_id", [f"S{i:03d}" for i in range(1, n + 1)])
    sup["tier"] = rng.choice([1, 2], n)
    sup["annual_spend_usd"] = rng.lognormal(16, 0.5, n).round(-3)   # SIMULATED
    sup["base_lead_time"] = rng.integers(7, 35, n)                   # SIMULATED
    sup["iso3"] = sup.country.map(lambda c: COUNTRIES[c][0])
    sup["geo_base"] = sup.country.map(lambda c: COUNTRIES[c][1])
    sup.to_csv(DATA / "suppliers.csv", index=False)

    rows = []  # SIMULATED weekly shipments
    for _, s in sup.iterrows():
        for week in range(1, 53):
            port = np.clip(rng.normal(0.35, 0.2), 0, 1)
            weather = np.clip(rng.normal(0.25, 0.2), 0, 1)
            geo = np.clip(s.geo_base + rng.normal(0, 0.08), 0, 1)
            lead = s.base_lead_time + rng.normal(0, 2) + 6 * port
            defect = np.clip(rng.normal(0.02, 0.01) + 0.02 * geo, 0, 0.2)
            vol = int(rng.integers(200, 2000))
            logit = (-3.2 + 2.6 * port + 1.6 * weather + 2.2 * geo + 25 * defect
                     + 0.04 * (lead - s.base_lead_time))
            p = 1 / (1 + np.exp(-logit))
            rows.append([s.supplier_id, week, round(lead, 1), round(defect, 4), vol,
                         round(port, 3), round(weather, 3), round(geo, 3), int(rng.random() < p)])
    pd.DataFrame(rows, columns=["supplier_id", "week"] + FEATURES + ["delayed"]).to_csv(
        DATA / "shipments.csv", index=False)

    # Starter news (SIMULATED) so the dashboard works offline. Live news replaces it per company.
    templates = [
        "{n} fined for exceeding emission limits at its plant", "Workers at {n} strike over unpaid overtime",
        "{n} executives under investigation for fraud", "Port congestion delays shipments from {n}",
        "{n} commits to cut carbon emissions by 30% by 2030", "{n} signs agreement to improve worker conditions",
        "{n} publishes transparent annual sustainability report", "{n} announces new product line",
    ]
    news = []
    for _, s in sup.iterrows():
        for t in rng.choice(templates, rng.integers(2, 6), replace=False):
            d = (pd.Timestamp.today().normalize() - pd.Timedelta(days=int(rng.integers(0, 20)))).date()
            news.append([s.supplier_id, t.format(n=s.supplier_name), d, "simulated"])
    pd.DataFrame(news, columns=["supplier_id", "headline", "date", "source"]).to_csv(DATA / "news.csv", index=False)


# ============================================================ 2. LIVE NEWS (GDELT)
MIN_GAP = 6          # seconds between GDELT requests (it asks for at most 1 request per 5 seconds)
_last_call = [0.0]


class NewsError(Exception):
    """Raised when the news service cannot be reached or keeps refusing requests."""


def _gdelt_get(params, tries=3):
    """One GDELT request that waits its turn, and retries when the service says 'too many requests'."""
    import requests
    reason = "unknown"
    for attempt in range(1, tries + 1):
        wait = MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = requests.get(GDELT_URL, params=params, timeout=45,
                             headers={"User-Agent": "ChainSight2/1.0 (student analytics project)"})
            if r.status_code == 429 or r.status_code >= 500:
                reason = f"service busy (HTTP {r.status_code})"
            else:
                r.raise_for_status()
                try:
                    return r.json().get("articles", [])
                except ValueError:                     # GDELT answers in plain text when something is wrong
                    if "limit" not in r.text.lower():
                        return []                      # e.g. "phrase too short": nothing to retry
                    reason = "rate limited"
        except requests.exceptions.RequestException as ex:
            reason = type(ex).__name__
        if attempt < tries:
            time.sleep(10 * attempt)                   # back off, then try again
    raise NewsError(reason)


def fetch_live_news(query, match, max_records=15, timespan="7d", tries=3):
    """Latest English headlines for one company from GDELT. Returns [(headline, date), ...]."""
    import re
    articles = _gdelt_get({"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
                           "maxrecords": max_records, "timespan": timespan, "sort": "datedesc"}, tries)
    out, seen = [], set()
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title or title.lower() in seen or not re.search(r"\b" + re.escape(match.lower()) + r"\b", title.lower()):
            continue   # skip duplicates and headlines that are not really about the company
        seen.add(title.lower())
        out.append((title, pd.to_datetime(a["seendate"][:8], format="%Y%m%d").date()))
    return out


def refresh():
    """Fetch live news for every company, merge into history, re-score. Takes a few minutes (rate limits)."""
    sup = pd.read_csv(DATA / "suppliers.csv")
    new, failed = [], 0
    for _, s in sup.iterrows():
        try:
            found = fetch_live_news(s["query"], s["match"])
            for title, d in found:
                new.append([s.supplier_id, title, d, "gdelt"])
            print(f"  {s.supplier_name}: ok ({len(found)} headlines)")
        except Exception as e:
            failed += 1
            print(f"  {s.supplier_name}: failed ({e})")
    if failed:
        print(f"{failed} of {len(sup)} companies failed. Run refresh again in a few minutes to fill the gaps.")

    hist = pd.read_csv(DATA / "news.csv")
    if "source" not in hist:
        hist["source"] = "simulated"
    if new:
        hist = pd.concat([hist, pd.DataFrame(new, columns=hist.columns)], ignore_index=True)
        hist = hist.drop_duplicates(["supplier_id", "headline"])
        keep = pd.to_datetime(hist.date) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=NEWS_WINDOW_DAYS)
        hist = hist[keep]
        hist.to_csv(DATA / "news.csv", index=False)
        print(f"Added live headlines ({len(new)} fetched). Re-scoring...")
    else:
        print("No live news retrieved (offline, blocked or rate-limited). Existing scores kept.")
    score_esg()
    build_risk_table()
    print("Refresh complete:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))


# ============================================================ 3. NEWS TAGGING + ESG SCORE
KEYWORDS = {
    "environmental": ["emission", "carbon", "wastewater", "pollution", "leak", "climate", "spill",
                      "environmental", "fire", "explosion", "toxic", "waste"],
    "labor": ["worker", "strike", "labor", "labour", "overtime", "safety", "wage", "union",
              "layoff", "layoffs", "fatigue", "injury", "protest"],
    "governance": ["fraud", "audit", "accounting", "investigation", "probe", "executive", "transparent",
                   "lawsuit", "bribery", "scandal", "sanction", "sanctions", "antitrust", "penalty", "fined"],
    "logistics": ["port", "ports", "typhoon", "shipment", "shipments", "delay", "delays", "distribution",
                  "congestion", "shortage", "shutdown", "halt", "disruption", "tariff", "tariffs", "export"],
}
NEG = ["fined", "leak", "strike", "violation", "fraud", "investigation", "probe", "irregularit", "delay",
       "disrupt", "exceeding", "unpaid", "flag", "congestion", "lawsuit", "scandal", "shortage", "halt",
       "shutdown", "sanction", "penalty", "explosion", "fire", "layoff", "protest", "recall", "ban", "injury"]
POS = ["commits", "agreement", "improve", "transparent", "opens", "speed", "record", "growth", "invest", "wins"]


def _tokens(text):
    import re
    return re.findall(r"[a-z]+", text.lower())


def _hit(token, word):
    return token == word or token == word + "s" or (len(word) >= 5 and token.startswith(word))


def rule_category(text):
    toks = _tokens(text)
    scores = {c: sum(any(_hit(t, w) for t in toks) for w in ws) for c, ws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


def rule_sentiment(text):
    toks = _tokens(text)
    neg = sum(any(_hit(t, w) for t in toks) for w in NEG)
    pos = sum(any(_hit(t, w) for t in toks) for w in POS)
    return (pos - neg) / max(1, pos + neg)  # -1 .. +1


def classify_news(df):
    if df.empty:
        df["category"], df["sentiment"] = [], []
        return df
    if os.getenv("USE_TRANSFORMERS") == "1":   # optional zero-shot model: slower, usually smarter
        from transformers import pipeline
        zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        sa = pipeline("sentiment-analysis")
        labels = ["environmental", "labor", "governance", "logistics", "neutral"]
        df["category"] = [zs(h, labels)["labels"][0] for h in df.headline]
        df["sentiment"] = [(1 if r["label"] == "POSITIVE" else -1) * r["score"]
                           for r in sa(df.headline.tolist())]
    else:
        df["category"] = df.headline.map(rule_category)
        df["sentiment"] = df.headline.map(rule_sentiment)
    return df


def score_esg():
    news = pd.read_csv(DATA / "news.csv")
    if "source" not in news:
        news["source"] = "simulated"
    news = news[pd.to_datetime(news.date) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=NEWS_WINDOW_DAYS)]
    live = news[news.source != "simulated"]
    # per company: use live headlines when there are any, otherwise the simulated starter news
    news = pd.concat([live, news[(news.source == "simulated") & ~news.supplier_id.isin(set(live.supplier_id))]])
    news = classify_news(news.copy())
    news.to_csv(DATA / "news_scored.csv", index=False)

    esg = news[news.category.isin(["environmental", "labor", "governance"])]
    penalty = esg[esg.sentiment < 0].groupby("supplier_id").sentiment.apply(lambda s: (-s).sum() * 12)
    sup = pd.read_csv(DATA / "suppliers.csv")[["supplier_id"]]
    sup["esg_penalty"] = sup.supplier_id.map(penalty).fillna(0)
    sup["esg_score"] = (100 - sup.esg_penalty).clip(0, 100).round(1)
    sup["logistics_news"] = sup.supplier_id.map(
        news[(news.category == "logistics") & (news.sentiment < 0)].groupby("supplier_id").size()).fillna(0).astype(int)
    sup["live_headlines"] = sup.supplier_id.map(live.groupby("supplier_id").size()).fillna(0).astype(int)
    sup.to_csv(DATA / "supplier_esg.csv", index=False)


# ============================================================ 4. DELAY MODEL + COMPOSITE RISK
def train_model():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    ship = pd.read_csv(DATA / "shipments.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(
        ship[FEATURES], ship.delayed, test_size=0.25, random_state=42, stratify=ship.delayed)
    model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    print("ROC-AUC:", round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 3))
    print(classification_report(y_te, model.predict(X_te)))
    print(pd.Series(model.feature_importances_, FEATURES).sort_values(ascending=False))
    joblib.dump(model, DATA / "delay_model.joblib")


def build_risk_table():
    import joblib
    model = joblib.load(DATA / "delay_model.joblib")
    ship = pd.read_csv(DATA / "shipments.csv")
    recent = ship[ship.week > ship.week.max() - 8].copy()
    recent["p"] = model.predict_proba(recent[FEATURES])[:, 1]
    delay = recent.groupby("supplier_id").p.mean().rename("delay_prob_model")

    sup = pd.read_csv(DATA / "suppliers.csv").merge(pd.read_csv(DATA / "supplier_esg.csv"), on="supplier_id")
    sup = sup.merge(delay, on="supplier_id")
    # assumption: each negative logistics headline adds 5 points of delay probability
    sup["delay_prob"] = (sup.delay_prob_model + 0.05 * sup.logistics_news).clip(0, 1).round(3)
    sup["esg_risk"] = 1 - sup.esg_score / 100
    sup["composite_risk"] = (0.6 * sup.delay_prob + 0.4 * sup.esg_risk).round(3)       # assumption: 60/40
    sup["value_at_risk"] = (sup.delay_prob * sup.annual_spend_usd * 0.15).round(0)     # assumption: 15% loss
    sup.to_csv(DATA / "suppliers_scored.csv", index=False)

    snap = sup[["supplier_id", "supplier_name", "composite_risk", "esg_score", "delay_prob"]].copy()
    snap.insert(0, "timestamp", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    hist = DATA / "risk_history.csv"
    snap.to_csv(hist, mode="a", header=not hist.exists(), index=False)


def build_all():
    hist = DATA / "risk_history.csv"
    if hist.exists():
        hist.unlink()
    generate_data(); score_esg(); train_model(); build_risk_table()
    print("Build done. Next: python chainsight2.py refresh   (live news)   then   streamlit run chainsight2.py")


def lookup_company(name, timespan="7d"):
    """Live look-up for ANY company name (used by the dashboard's 'Watch any company' tab). Nothing is saved."""
    known = {c[0]: (c[3], c[4]) for c in COMPANIES}
    if name in known:
        query, match = known[name]
    else:
        words = [w for w in name.lower().split() if w not in ("the", "a", "an")] or [name.lower()]
        query, match = f'"{name}"', words[0]
    items = fetch_live_news(query, match, max_records=25, timespan=timespan, tries=2)
    df = pd.DataFrame(items, columns=["headline", "date"])
    if df.empty:
        return df, None
    df = classify_news(df)
    esg = df[df.category.isin(["environmental", "labor", "governance"]) & (df.sentiment < 0)]
    score = round(max(0.0, 100 - (-esg.sentiment).sum() * 12), 1)
    return df, score


# ============================================================ 5. DASHBOARD
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.stAppDeployButton { display: none; }
.block-container { padding-top: 1.4rem; max-width: 1300px; }
.hero { background: linear-gradient(120deg, #1e1b4b 0%, #4338ca 55%, #0ea5e9 100%); border-radius: 18px;
        padding: 28px 34px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(67,56,202,.28); }
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 700; letter-spacing: -.5px; color: #fff !important; padding: 0; }
.hero p { margin: 4px 0 14px; opacity: .9; font-size: 1.02rem; color: #fff !important; }
.chip { display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
        padding: 4px 12px; border-radius: 999px; font-size: .8rem; margin: 0 8px 6px 0; color: #fff; }
.kpi { border: 1px solid rgba(127,127,127,.25); background: rgba(127,127,127,.07); border-radius: 14px;
       padding: 15px 18px; border-left: 5px solid var(--accent); height: 100%; }
.kpi .l { font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; opacity: .7; font-weight: 600; }
.kpi .v { font-size: 1.65rem; font-weight: 700; line-height: 1.3; margin-top: 2px; }
.kpi .s { font-size: .8rem; opacity: .7; margin-top: 2px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 600; }
.note { opacity: .7; font-size: .9rem; margin: -4px 0 10px; }
</style>
"""
RISK_SCALE = ["#22C55E", "#F59E0B", "#EF4444"]


def kpi(label, value, sub="", accent="#4F46E5"):
    return (f'<div class="kpi" style="--accent:{accent}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def dashboard():
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="ChainSight 2", page_icon="🛰️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if not (DATA / "suppliers_scored.csv").exists():
        with st.spinner("First run: building data and model..."):
            build_all()

    @st.cache_data(ttl=300)   # re-reads the files every 5 minutes
    def load():
        return (pd.read_csv(DATA / "suppliers_scored.csv"), pd.read_csv(DATA / "news_scored.csv"),
                pd.read_csv(DATA / "risk_history.csv"))

    @st.cache_data(ttl=600, show_spinner="Fetching live headlines...")
    def cached_lookup(name, look):
        return lookup_company(name, look)

    sup_all, news_all, hist_all = load()
    updated = pd.Timestamp(os.path.getmtime(DATA / "suppliers_scored.csv"), unit="s").tz_localize("UTC") \
        .tz_convert("Asia/Kolkata").strftime("%d %b %Y, %H:%M IST")

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.markdown("### 🛰️ ChainSight 2")
        st.caption("Supply chain risk & ESG monitor")
        all_countries = sorted(sup_all.country.unique())
        countries = st.multiselect("Filter by country", all_countries, default=all_countries)
        if st.button("🔄 Fetch live news now", width="stretch"):
            with st.spinner("Fetching live news (about 2 minutes)..."):
                refresh()
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.markdown("**How to read it**")
        st.caption("Risk score = 60% disruption probability + 40% ESG risk. "
                   "Green means lower risk, red means higher. Bubble size is annual spend.")
        st.markdown("**Data**")
        st.caption("News: live headlines (GDELT).  \nShipments and spend: simulated.")

    sup = sup_all[sup_all.country.isin(countries)].copy()
    if sup.empty:
        st.warning("Select at least one country in the sidebar.")
        st.stop()
    news = news_all[news_all.supplier_id.isin(sup.supplier_id)]
    hist = hist_all[hist_all.supplier_id.isin(sup.supplier_id)]
    hi, lo = sup.composite_risk.quantile(0.75), sup.composite_risk.quantile(0.25)
    sup["level"] = sup.composite_risk.map(lambda v: "🔴 High" if v >= hi else ("🟢 Low" if v <= lo else "🟠 Medium"))

    # ---------------- hero + KPI cards ----------------
    st.markdown(
        '<div class="hero"><h1>ChainSight 2</h1>'
        '<p>AI-driven cross-border supply chain risk &amp; ESG analytics</p>'
        f'<span class="chip">🕒 Updated {updated}</span>'
        f'<span class="chip">📰 {int(sup.live_headlines.sum())} live headlines in use</span>'
        '<span class="chip">🧪 Shipments &amp; spend simulated</span></div>', unsafe_allow_html=True)

    top = sup.sort_values("composite_risk", ascending=False).iloc[0]
    low_esg = sup.sort_values("esg_score").iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Companies monitored", len(sup), f"{sup.country.nunique()} countries", "#4F46E5"), unsafe_allow_html=True)
    k2.markdown(kpi("Highest risk", top.supplier_name, f"risk score {top.composite_risk:.2f}", "#EF4444"), unsafe_allow_html=True)
    k3.markdown(kpi("Value at risk", f"${sup.value_at_risk.sum() / 1e6:,.1f}M", "based on simulated spend", "#F59E0B"), unsafe_allow_html=True)
    k4.markdown(kpi("Average ESG score", f"{sup.esg_score.mean():.0f} / 100", f"lowest: {low_esg.supplier_name}", "#14B8A6"), unsafe_allow_html=True)
    st.write("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Risk matrix", "🌍 ESG map", "📈 Risk trend", "🧮 What-if simulator", "🔎 Watch any company"])

    # ---------------- tab 1: risk matrix ----------------
    with tab1:
        st.markdown('<div class="note">Top-right companies are likely to be disrupted <b>and</b> costly if they are. '
                    'Look there first.</div>', unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        fig = px.scatter(sup, x="delay_prob", y="value_at_risk", size="annual_spend_usd", size_max=36,
                         color="composite_risk", color_continuous_scale=RISK_SCALE, text="supplier_name",
                         hover_data=["country", "part", "esg_score"],
                         labels={"delay_prob": "Disruption probability", "value_at_risk": "Value at risk (USD)",
                                 "composite_risk": "Risk"})
        fig.update_traces(textposition="top center", textfont_size=10, marker=dict(line=dict(width=1, color="white")))
        fig.add_vline(x=sup.delay_prob.median(), line_dash="dot", opacity=0.35)
        fig.add_hline(y=sup.value_at_risk.median(), line_dash="dot", opacity=0.35)
        for x, y, txt, xa, ya in [(0.99, 0.99, "ACT FIRST", "right", "top"), (0.01, 0.99, "HIGH IMPACT", "left", "top"),
                                  (0.99, 0.01, "LIKELY DELAYS", "right", "bottom"), (0.01, 0.01, "LOW CONCERN", "left", "bottom")]:
            fig.add_annotation(xref="paper", yref="paper", x=x, y=y, text=txt, showarrow=False,
                               xanchor=xa, yanchor=ya, font=dict(size=10, color="gray"))
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickprefix="$")
        left.plotly_chart(fig, width="stretch")

        right.markdown("**Risk ranking**")
        rank = sup.sort_values("composite_risk", ascending=False)[["level", "supplier_name", "composite_risk", "esg_score"]]
        right.dataframe(rank, hide_index=True, width="stretch", height=450, column_config={
            "level": "Level", "supplier_name": "Company",
            "composite_risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
            "esg_score": st.column_config.ProgressColumn("ESG", min_value=0, max_value=100, format="%.0f")})

        st.markdown("#### Why is a company rated this way?")
        pick = st.selectbox("Company", sup.sort_values("composite_risk", ascending=False).supplier_name,
                            label_visibility="collapsed")
        sid = sup[sup.supplier_name == pick].supplier_id.iloc[0]
        st.dataframe(news[news.supplier_id == sid][["date", "headline", "category", "sentiment", "source"]]
                     .sort_values("date", ascending=False), hide_index=True, width="stretch")

    # ---------------- tab 2: ESG map ----------------
    with tab2:
        st.markdown('<div class="note">Average ESG score of the monitored companies in each country '
                    '(100 = no negative ESG news).</div>', unsafe_allow_html=True)
        by_c = sup.groupby(["country", "iso3"], as_index=False).agg(
            esg_score=("esg_score", "mean"), companies=("supplier_id", "count")).round(1)
        m_left, m_right = st.columns([3, 1])
        fig = px.choropleth(by_c, locations="iso3", color="esg_score", hover_name="country",
                            color_continuous_scale=["#EF4444", "#F59E0B", "#22C55E"], range_color=(0, 100),
                            labels={"esg_score": "Avg ESG"})
        fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                        bgcolor="rgba(0,0,0,0)", landcolor="rgba(127,127,127,0.15)")
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0))
        m_left.plotly_chart(fig, width="stretch")
        m_right.dataframe(by_c.sort_values("esg_score")[["country", "esg_score", "companies"]], hide_index=True,
                          width="stretch", column_config={
                              "country": "Country", "companies": "Cos.",
                              "esg_score": st.column_config.ProgressColumn("ESG", min_value=0, max_value=100, format="%.0f")})

    # ---------------- tab 3: trend ----------------
    with tab3:
        st.markdown('<div class="note">Every refresh saves a snapshot, so you can watch risk move over time.</div>',
                    unsafe_allow_html=True)
        names = st.multiselect("Companies", sorted(sup.supplier_name),
                               default=list(sup.sort_values("composite_risk", ascending=False).supplier_name[:4]))
        h = hist[hist.supplier_name.isin(names)]
        if h.timestamp.nunique() < 2:
            st.info("Only one snapshot so far. Run `python chainsight2.py refresh` (or `loop`) a few times, "
                    "or click 'Fetch live news now' in the sidebar, and this chart will show the changes.")
        else:
            fig = px.line(h, x="timestamp", y="composite_risk", color="supplier_name", markers=True,
                          labels={"composite_risk": "Composite risk", "timestamp": "", "supplier_name": "Company"})
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

    # ---------------- tab 4: what-if ----------------
    with tab4:
        st.markdown('<div class="note">Pick a supplier and a delay length to estimate the damage to your margin '
                    'and see lower-risk alternatives.</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        pick2 = a.selectbox("Supplier", sup.supplier_name, key="wi")
        delay_days = a.slider("Delay (days)", 1, 60, 10)
        buffer_days = a.slider("Safety stock on hand (days)", 0, 30, 5)
        rev_mult = b.slider("Revenue per $1 of supplier spend", 1.0, 6.0, 3.0)
        margin = b.slider("Gross margin (%)", 5, 60, 25) / 100
        expedite = b.slider("Expedite / alt-sourcing premium (%)", 0, 40, 15) / 100

        s = sup[sup.supplier_name == pick2].iloc[0]
        daily_spend = s.annual_spend_usd / 365
        effective = max(0, delay_days - buffer_days)
        lost = daily_spend * rev_mult * margin * effective
        extra = daily_spend * expedite * delay_days
        r1, r2, r3 = st.columns(3)
        r1.markdown(kpi("Estimated margin impact", f"${lost + extra:,.0f}", "lost margin + expedite cost", "#EF4444"), unsafe_allow_html=True)
        r2.markdown(kpi("Uncovered days", effective, f"{buffer_days} days covered by safety stock", "#F59E0B"), unsafe_allow_html=True)
        r3.markdown(kpi("Supplier risk score", f"{s.composite_risk:.2f}", f"{s.country} · {s.part}", "#4F46E5"), unsafe_allow_html=True)
        st.write("")
        c_l, c_r = st.columns([2, 3])
        bar = px.bar(pd.DataFrame({"Cost": ["Lost margin", "Expedite premium"], "USD": [lost, extra]}),
                     x="Cost", y="USD", color="Cost", color_discrete_sequence=["#EF4444", "#F59E0B"])
        bar.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="")
        c_l.plotly_chart(bar, width="stretch")
        alts = sup[(sup.supplier_id != s.supplier_id) & (sup.part == s.part)].sort_values("composite_risk").head(3)
        if alts.empty:
            alts = sup[sup.supplier_id != s.supplier_id].sort_values("composite_risk").head(3)
        c_r.markdown("**Suggested alternatives (same part, lowest risk)**")
        c_r.dataframe(alts[["supplier_name", "country", "part", "composite_risk", "base_lead_time"]],
                      hide_index=True, width="stretch", column_config={
                          "supplier_name": "Company", "country": "Country", "part": "Part",
                          "base_lead_time": "Lead time (days)",
                          "composite_risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f")})

    # ---------------- tab 5: watch any company ----------------
    with tab5:
        st.markdown('<div class="note">Pick a company from the list (or choose "Other" to type any name). '
                    'ChainSight fetches its latest headlines and scores them right now. Not saved to the dashboard.</div>',
                    unsafe_allow_html=True)
        c_a, c_b = st.columns([3, 1])
        choice = c_a.selectbox("Choose a company", COMPANY_CHOICES + ["Other (type a name)"], index=None,
                               placeholder="Click to choose, or start typing to search")
        look = c_b.selectbox("Look back", ["1d", "3d", "7d", "14d"], index=2)
        who = None
        if choice == "Other (type a name)":
            who = st.text_input("Company name", placeholder="e.g. Hyundai Motor").strip() or None
        elif choice:
            who = choice
        if who:
            try:
                df, score = cached_lookup(who, look)
            except Exception as e:
                st.error(f"Could not fetch news: {e}. The news service limits how fast it can be asked; wait a minute and pick the company again.")
            else:
                if df.empty:
                    st.warning(f"No headlines found for '{who}'. Try a longer look-back or a fuller company name.")
                else:
                    st.markdown(f"### {who}")
                    g, rt = st.columns([1, 2])
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=score, title={"text": "ESG score"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4F46E5"},
                               "steps": [{"range": [0, 50], "color": "#fecaca"}, {"range": [50, 75], "color": "#fde68a"},
                                         {"range": [75, 100], "color": "#bbf7d0"}]}))
                    gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
                    g.plotly_chart(gauge, width="stretch")
                    n_neg = int((df.sentiment < 0).sum())
                    q1, q2 = rt.columns(2)
                    q1.markdown(kpi("Headlines found", len(df), f"last {look}", "#4F46E5"), unsafe_allow_html=True)
                    q2.markdown(kpi("Negative headlines", n_neg, f"{n_neg / len(df):.0%} of the total", "#EF4444"), unsafe_allow_html=True)
                    cats = df.category.value_counts().rename_axis("topic").reset_index(name="headlines")
                    topic = px.bar(cats, x="headlines", y="topic", orientation="h", color="topic",
                                   color_discrete_sequence=px.colors.qualitative.Set2)
                    topic.update_layout(height=160, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="", xaxis_title="")
                    rt.plotly_chart(topic, width="stretch")
                    show = df.sort_values("date", ascending=False)[["date", "headline", "category", "sentiment"]]
                    colour = lambda v: "color:#EF4444" if v < 0 else ("color:#22C55E" if v > 0 else "")
                    styler = show.style.map(colour, subset=["sentiment"]) if hasattr(show.style, "map") \
                        else show.style.applymap(colour, subset=["sentiment"])
                    st.dataframe(styler.format({"sentiment": "{:+.2f}"}), hide_index=True, width="stretch")


# ============================================================ ENTRY POINT
def _under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _under_streamlit():
    dashboard()
elif __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build_all()
    elif cmd == "refresh":
        refresh()
    elif cmd == "loop":
        minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        print(f"Refreshing every {minutes} minutes. Press Ctrl+C to stop.")
        while True:
            refresh()
            time.sleep(minutes * 60)
    else:
        print("Commands: build | refresh | loop [minutes]")
