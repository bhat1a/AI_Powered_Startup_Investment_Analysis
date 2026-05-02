import hashlib
import os
import html as _html
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "advisory_app")
HC_CORE   = os.path.join(BASE_DIR, "csv_visualizer", "static", "export", "highcharts.js")

_llm_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.cerebras.ai/v1",
)

FEATURE_LABELS = {
    "revenue_usd_mn":              "Annual Revenue",
    "profit_loss_usd_mn":          "Profit / Loss",
    "market_size_usd_bn":          "Market Size",
    "growth_rate_pct":             "Growth Rate",
    "total_funding_usd_mn":        "Total Funding",
    "valuation_usd_mn":            "Valuation",
    "founder_exp_yrs":             "Founder Experience",
    "employees":                   "Employees",
    "app_downloads_mn":            "App Downloads",
    "website_traffic_mn_monthly":  "Website Traffic",
    "burn_rate_usd_mn_monthly":    "Monthly Burn Rate",
    "market_growth_rate_pct":      "Market Growth Rate",
    "social_media_followers_mn":   "Social Following",
    "num_founders":                "No. of Founders",
    "prev_startups":               "Previous Startups",
    "tech_founder":                "Tech Founder",
    "funding_rounds":              "Funding Rounds",
    "seed_usd_mn":                 "Seed Funding",
    "series_a_usd_mn":             "Series A",
    "series_b_usd_mn":             "Series B",
    "competitor_count":            "Competitor Count",
}

CHECKMARK_SVG = (
    '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;">'
    '<path d="M15 4.5L7 12.5L3 8.5" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
X_MARK_SVG = (
    '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;">'
    '<path d="M13.5 4.5L4.5 13.5M4.5 4.5L13.5 13.5" stroke="currentColor" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


@st.cache_resource
def load_artifacts():
    model     = joblib.load(os.path.join(MODEL_DIR, "startup_model.pkl"))
    selector  = joblib.load(os.path.join(MODEL_DIR, "feature_selector.pkl"))
    sel_feats = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    all_feats = joblib.load(os.path.join(MODEL_DIR, "all_features.pkl"))
    imp       = joblib.load(os.path.join(MODEL_DIR, "imputation_defaults.pkl"))
    le        = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, selector, sel_feats, all_feats, imp, le


@st.cache_resource
def load_shap_explainer(_model):
    return shap.Explainer(_model)


@st.cache_resource
def load_hc_core():
    try:
        with open(HC_CORE, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _build_feature_vector(inputs, all_features, imp_defaults):
    row = {col: 0.0 for col in all_features}

    for col in ["revenue_usd_mn", "profit_loss_usd_mn", "market_size_usd_bn",
                "growth_rate_pct", "total_funding_usd_mn",
                "founder_exp_yrs", "employees"]:
        if col in row and col in inputs:
            row[col] = float(inputs[col])

    opt_numeric = [
        "valuation_usd_mn", "app_downloads_mn",
        "website_traffic_mn_monthly", "burn_rate_usd_mn_monthly",
        "market_growth_rate_pct", "social_media_followers_mn",
        "num_founders", "prev_startups", "tech_founder",
        "funding_rounds", "seed_usd_mn", "series_a_usd_mn", "series_b_usd_mn",
        "competitor_count",
        "year_founded", "company_age", "last_funding_year",
        "time_between_rounds_months", "is_unicorn",
    ]
    for col in opt_numeric:
        val = inputs.get(col)
        if val is not None and col in row:
            row[col] = float(val)
        elif col in imp_defaults.get("numeric", {}) and col in row:
            row[col] = imp_defaults["numeric"][col]

    for cat_col in ["industry", "city", "education_level", "ipo_status"]:
        val = imp_defaults.get("categorical", {}).get(cat_col)
        if val:
            dummy_col = f"{cat_col}_{val}"
            if dummy_col in row:
                row[dummy_col] = 1.0

    return pd.DataFrame([row])[all_features]


def _render_shap_chart(shap_values, sel_feats, pred_idx):
    hc = load_hc_core()
    if not hc:
        return

    if len(shap_values.values.shape) == 3:
        flat = shap_values.values[0, :, pred_idx]
    else:
        flat = shap_values.values[0]

    pairs = sorted(zip(sel_feats, flat), key=lambda x: abs(x[1]), reverse=True)[:8]
    pairs = list(reversed(pairs))

    features = [FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f, _ in pairs]
    data = [{"y": round(float(v), 6), "color": "#7C3AED" if v >= 0 else "#C9A227"} for _, v in pairs]

    cats_json = json.dumps(features)
    data_json = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
html, body {{ margin:0; padding:0; background:transparent; font-family:'Outfit',sans-serif; }}
#shap {{ width:100%; height:320px; }}
</style></head>
<body>
<div id="shap"></div>
<script>{hc}</script>
<script>
Highcharts.chart('shap', {{
    chart: {{ type: 'bar', backgroundColor: 'transparent',
              style: {{ fontFamily: 'Outfit, sans-serif' }} }},
    title: {{ text: 'What drove this prediction',
              style: {{ color: '#F0EEFF', fontSize: '15px', fontWeight: '500' }} }},
    xAxis: {{ categories: {cats_json},
              labels: {{ style: {{ color: '#A89BC2', fontSize: '13px' }} }},
              lineColor: 'rgba(255,255,255,0.08)', tickColor: 'rgba(255,255,255,0.08)' }},
    yAxis: {{ title: null, gridLineColor: 'rgba(255,255,255,0.06)',
              labels: {{ style: {{ color: '#A89BC2' }} }} }},
    legend: {{ enabled: false }},
    tooltip: {{
        backgroundColor: 'rgba(13,8,32,0.95)',
        borderColor: 'rgba(124,58,237,0.40)',
        borderRadius: 8,
        style: {{ color: '#F0EEFF', fontSize: '13px' }},
        formatter: function() {{
            var dir = this.y >= 0 ? 'towards' : 'against';
            return '<b>' + this.x + '</b><br/>Pushes ' + dir + ' predicted class<br/>SHAP: ' + this.y.toFixed(4);
        }}
    }},
    plotOptions: {{ bar: {{ borderRadius: 3, dataLabels: {{ enabled: false }} }} }},
    series: [{{ name: 'SHAP', data: {data_json} }}],
    credits: {{ enabled: false }}
}});
</script>
</body></html>"""

    components.html(html, height=340, scrolling=False)


def _llm_narrative(pred_class, probs, classes, top_features):
    feature_text = " ".join(
        f"{FEATURE_LABELS.get(f, f)} had a {'positive' if v > 0 else 'negative'} impact."
        for f, v in top_features
    )

    prompt = f"""You are an AI investment analyst writing for an academic audience.

A startup was evaluated. Model prediction: {pred_class.upper()}
Confidence: {max(probs)*100:.1f}%

Class probabilities — Avoid: {probs[list(classes).index('Avoid')]*100:.1f}%, Hold: {probs[list(classes).index('Hold')]*100:.1f}%, Invest: {probs[list(classes).index('Invest')]*100:.1f}%

Key influencing factors: {feature_text}

Write a concise 4-5 sentence investment analysis. Be direct and professional. Avoid jargon.
Do not mention SHAP or machine learning internals."""

    for attempt in range(4):
        try:
            resp = _llm_client.chat.completions.create(
                model=os.getenv("MODEL", "qwen-3-235b-a22b-instruct-2507"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                timeout=30,
            )
            return resp.choices[0].message.content
        except Exception as e:
            is_429 = hasattr(e, "status_code") and e.status_code == 429
            if is_429 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            if is_429:
                return "__RATE_LIMITED__"
            return f"LLM narrative unavailable: {e}"


_COOLDOWN_SECS = 30

def render_advisor():
    model, selector, sel_feats, all_feats, imp, le = load_artifacts()

    if "narrative_cache" not in st.session_state:
        st.session_state["narrative_cache"] = {}
    if "last_submit_time" not in st.session_state:
        st.session_state["last_submit_time"] = 0

    accent     = "#7C3AED"
    success    = "#10B981"
    warning    = "#F59E0B"
    danger     = "#EF4444"
    text_sec   = "#A89BC2"
    text_muted = "#5A4E72"

    req_pill = (
        "font-family:'Outfit',sans-serif; font-size:11px; font-weight:500; "
        "color:#A855F7; background:rgba(124,58,237,0.15); "
        "border:1px solid rgba(124,58,237,0.35); border-radius:999px; padding:3px 10px;"
    )

    st.markdown(
        f"<p style='font-family:Syne,sans-serif;font-size:12px;font-weight:600;"
        f"color:#F0EEFF;letter-spacing:0.08em;text-transform:uppercase;margin:0 0 0.75rem;'>"
        f"Financial Metrics &nbsp;<span style='{req_pill}'>Required</span></p>",
        unsafe_allow_html=True,
    )

    growth_rate = st.slider("Growth Rate % YoY", -100, 500, 20, key="adv_growth")

    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        profit_loss = st.number_input("Profit / Loss (USD Million)",
                                      min_value=-10000.0, max_value=10000.0, value=0.0, step=1.0, key="adv_pl")
    with r1c2:
        market_size = st.number_input("Market Size (USD Billion)",
                                      min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="adv_mkt")

    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1:
        revenue = st.number_input("Annual Revenue (USD Million)",
                                  min_value=0.0, max_value=10000.0, value=0.0, step=1.0, key="adv_rev")
    with r2c2:
        founder_exp = st.number_input("Founder Experience (Years)",
                                      min_value=0.0, max_value=50.0, value=0.0, step=1.0, key="adv_fexp")

    r3c1, r3c2 = st.columns(2, gap="medium")
    with r3c1:
        employees = st.number_input("Employees",
                                    min_value=1.0, max_value=100000.0, value=10.0, step=10.0, key="adv_emp")
    with r3c2:
        total_fund = st.number_input("Total Funding Raised (USD Million)",
                                     min_value=0.0, max_value=5000.0, value=0.0, step=1.0, key="adv_fund")

    with st.expander("Optional — Advanced Metrics", expanded=False):
        st.markdown(
            f"<p style='font-family:Outfit,sans-serif;font-size:11px;font-weight:300;"
            f"color:{text_muted};margin:0 0 0.75rem;'>Blank fields use dataset averages.</p>",
            unsafe_allow_html=True,
        )
        oc1, oc2 = st.columns(2, gap="medium")
        with oc1:
            app_downloads = st.number_input("App Downloads (Million)",
                                            min_value=0.0, max_value=10000.0, value=0.0, step=1.0, key="adv_apps")
            burn_rate     = st.number_input("Monthly Burn Rate (USD Million)",
                                            min_value=0.0, max_value=1000.0, value=0.0, step=1.0, key="adv_burn")
            social_follow = st.number_input("Social Media Following (Million)",
                                            min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="adv_social")
        with oc2:
            web_traffic   = st.number_input("Website Traffic (Million visits/month)",
                                            min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="adv_web")
            mkt_growth    = st.number_input("Market Growth Rate (%)",
                                            min_value=-100.0, max_value=500.0, value=0.0, step=1.0, key="adv_mktg")
            valuation     = st.number_input("Valuation (USD Million)",
                                            min_value=0.0, max_value=50000.0, value=0.0, step=10.0, key="adv_val")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    elapsed = time.time() - st.session_state["last_submit_time"]
    in_cooldown = elapsed < _COOLDOWN_SECS
    cooldown_remaining = max(0, int(_COOLDOWN_SECS - elapsed))
    btn_label = f"Please wait {cooldown_remaining}s..." if in_cooldown else "Analyse Startup"
    analyse = st.button(btn_label, type="primary", use_container_width=True,
                        key="adv_analyse", disabled=in_cooldown)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    if analyse:
        st.session_state["last_submit_time"] = time.time()
        inputs = {
            "growth_rate_pct":             float(growth_rate),
            "profit_loss_usd_mn":          profit_loss,
            "market_size_usd_bn":          market_size,
            "revenue_usd_mn":              revenue,
            "founder_exp_yrs":             founder_exp,
            "employees":                   employees,
            "total_funding_usd_mn":        total_fund,
            "app_downloads_mn":            app_downloads if app_downloads > 0 else None,
            "website_traffic_mn_monthly":  web_traffic if web_traffic > 0 else None,
            "burn_rate_usd_mn_monthly":    burn_rate if burn_rate > 0 else None,
            "market_growth_rate_pct":      mkt_growth if mkt_growth != 0 else None,
            "social_media_followers_mn":   social_follow if social_follow > 0 else None,
            "valuation_usd_mn":            valuation if valuation > 0 else None,
        }
        _input_hash = hashlib.md5(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()

        X_raw    = _build_feature_vector(inputs, all_feats, imp)
        X_sel    = selector.transform(X_raw)
        X_df     = pd.DataFrame(X_sel, columns=sel_feats)

        probs      = model.predict_proba(X_df)[0]
        pred_idx   = int(np.argmax(probs))
        classes    = le.classes_
        pred_class = classes[pred_idx]

        avoid_p  = probs[list(classes).index("Avoid")]
        hold_p   = probs[list(classes).index("Hold")]
        invest_p = probs[list(classes).index("Invest")]

        st.markdown(
"<div style='height:1px;background:rgba(255,255,255,0.08);margin:2rem 0;'></div>",
unsafe_allow_html=True)

        res_left, res_right = st.columns([1, 1], gap="large")

        with res_left:
            badge_color = {"Invest": success, "Hold": warning, "Avoid": danger}[pred_class]
            if pred_class == "Invest":
                icon = CHECKMARK_SVG
                label = "INVEST"
            elif pred_class == "Avoid":
                icon = X_MARK_SVG
                label = "AVOID"
            else:
                icon = ""
                label = "~ HOLD"

            st.markdown(
f"<div style='margin-bottom:1rem;animation:fadeUp 0.4s ease-out both;'>"
f"<div style='display:inline-flex;align-items:center;gap:8px;"
f"background:{badge_color}22;color:{badge_color};"
f"border:1px solid {badge_color};border-radius:8px;"
f"padding:10px 20px;font-family:Outfit,sans-serif;"
f"font-size:16px;font-weight:600;letter-spacing:0.05em;'>"
f"{icon} {label}</div></div>",
unsafe_allow_html=True)

            for cls, prob, color in [
                ("Invest", invest_p, success),
                ("Hold",   hold_p,   warning),
                ("Avoid",  avoid_p,  danger),
            ]:
                pct = prob * 100
                st.markdown(
f"<div style='margin-bottom:0.75rem;'>"
f"<div style='display:flex;justify-content:space-between;margin-bottom:5px;'>"
f"<span style='font-family:Outfit,sans-serif;font-size:14px;font-weight:500;color:{text_sec};'>{cls}</span>"
f"<span style='font-family:JetBrains Mono,monospace;font-size:14px;font-weight:500;color:{text_sec};'>{pct:.1f}%</span>"
f"</div>"
f"<div style='background:rgba(255,255,255,0.08);border-radius:4px;height:8px;'>"
f"<div style='background:{color};width:{pct:.1f}%;height:8px;border-radius:4px;transition:width 0.6s ease;'></div>"
"</div></div>",
unsafe_allow_html=True)

        with res_right:
            shap_values = None
            try:
                explainer   = load_shap_explainer(model)
                shap_values = explainer(X_df)
                _render_shap_chart(shap_values, sel_feats, pred_idx)
            except Exception as e:
                st.warning(f"SHAP chart unavailable: {e}")

        top_features = []
        if shap_values is not None:
            sv_vals = shap_values.values
            flat = sv_vals[0, :, pred_idx] if len(sv_vals.shape) == 3 else sv_vals[0]
            top_features = sorted(
                zip(sel_feats, flat), key=lambda x: abs(x[1]), reverse=True
            )[:5]

        if _input_hash in st.session_state["narrative_cache"]:
            narrative = st.session_state["narrative_cache"][_input_hash]
        else:
            with st.spinner("Generating analysis via Cerebras..."):
                narrative = _llm_narrative(pred_class, probs, classes, top_features)
            if narrative != "__RATE_LIMITED__":
                st.session_state["narrative_cache"][_input_hash] = narrative

        if narrative == "__RATE_LIMITED__":
            st.markdown(
f"<div style='background:rgba(239,68,68,0.06);backdrop-filter:blur(20px);"
f"border:1px solid rgba(239,68,68,0.25);border-left:3px solid #EF4444;"
f"border-radius:16px;padding:24px 28px;margin-top:1.5rem;'>"
f"<p style='font-family:Outfit,sans-serif;font-size:11px;font-weight:500;"
f"color:#EF4444;letter-spacing:0.1em;text-transform:uppercase;margin:0 0 12px;'>Free Tier Limit Reached</p>"
f"<p style='font-family:Outfit,sans-serif;font-size:15px;font-weight:400;"
f"color:{text_sec};line-height:1.8;margin:0;'>The AI analysis could not be generated right now — "
f"the Cerebras free tier rate limit was reached. Please wait a moment and try again.</p></div>",
unsafe_allow_html=True)
        else:
            st.markdown(
f"<div style='background:rgba(124,58,237,0.06);backdrop-filter:blur(20px);"
f"border:1px solid rgba(124,58,237,0.25);border-left:3px solid {accent};"
f"border-radius:16px;padding:24px 28px;margin-top:1.5rem;'>"
f"<p style='font-family:Outfit,sans-serif;font-size:11px;font-weight:500;"
f"color:{accent};letter-spacing:0.1em;text-transform:uppercase;margin:0 0 12px;'>AI Analysis</p>"
f"<p style='font-family:Outfit,sans-serif;font-size:15px;font-weight:400;"
f"color:{text_sec};line-height:1.8;margin:0;'>{_html.escape(narrative)}</p></div>",
unsafe_allow_html=True)
