import json
import sys
import os
import logging
logging.basicConfig(level=logging.WARNING)
sys.path.append(os.path.abspath("."))

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="VentureAI",
    layout="wide",
    page_icon="V",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Outfit:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --bg-base:        #080812;
  --bg-deep:        #0D0820;
  --bg-card:        rgba(255, 255, 255, 0.04);
  --bg-card-hover:  rgba(255, 255, 255, 0.07);
  --border:         rgba(255, 255, 255, 0.08);
  --border-accent:  rgba(124, 58, 237, 0.40);
  --accent:         #7C3AED;
  --accent-light:   #A855F7;
  --accent-glow:    rgba(124, 58, 237, 0.35);
  --gold:           #C9A227;
  --success:        #10B981;
  --warning:        #F59E0B;
  --danger:         #EF4444;
  --text-primary:   #F0EEFF;
  --text-secondary: #A89BC2;
  --text-muted:     #5A4E72;
}

*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; }

body, .stApp {
    background: linear-gradient(135deg, #080812 0%, #0D0820 40%, #130A2E 70%, #080812 100%) !important;
    min-height: 100vh !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

[data-testid="stMainBlockContainer"] {
    padding-left: 3.5rem !important;
    padding-right: 3.5rem !important;
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.block-container {
    padding-top: 64px !important;
    padding-bottom: 3rem !important;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* Section-level element-containers: no margin so sections don't drift apart */
.element-container { margin-top: 0 !important; padding: 0 !important; }
/* Allow a small gap between individual widget rows */
.element-container + .element-container { margin-top: 0.5rem !important; }
.stMarkdown { margin: 0 !important; padding: 0 !important; }

/* Widget labels — consistent font + colour */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stFileUploader"] label,
[data-testid="stTextArea"] label {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    margin-bottom: 0.2rem !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080812; }
::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.40); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.65); }

/* ── Blobs ── */
.vc-blob {
    position: fixed;
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
}
.vc-blob-1 {
    top: -100px; left: -200px;
    width: 600px; height: 400px;
    background: radial-gradient(ellipse 600px 400px, rgba(124, 58, 237, 0.12), transparent);
    animation: float 6s ease-in-out infinite;
}
.vc-blob-2 {
    bottom: 0; right: -150px;
    width: 500px; height: 500px;
    background: radial-gradient(ellipse 500px 500px, rgba(168, 85, 247, 0.08), transparent);
    animation: float 6s ease-in-out infinite;
    animation-delay: 2s;
}
.vc-blob-3 {
    top: 40%; right: 5%;
    width: 400px; height: 300px;
    background: radial-gradient(ellipse 400px 300px, rgba(201, 162, 39, 0.06), transparent);
    animation: float 6s ease-in-out infinite;
    animation-delay: 4s;
}

/* ── Page wrapper ── */
.vc-page { position: relative; z-index: 1; }

/* ── Navbar ── */
.vc-navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    height: 64px;
    background: rgba(8, 8, 18, 0.75);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid rgba(124, 58, 237, 0.25);
    box-shadow: 0 1px 0 rgba(124, 58, 237, 0.15), 0 4px 24px rgba(0, 0, 0, 0.50);
    transition: background 300ms ease, border-bottom-color 300ms ease;
}
.vc-navbar.navbar-scrolled {
    background: rgba(8, 8, 18, 0.92);
    border-bottom-color: rgba(124, 58, 237, 0.45);
}

.vc-brand {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    text-decoration: none;
    text-shadow: 0 0 30px rgba(124, 58, 237, 0.50);
}

.vc-nav-links { display: flex; align-items: center; gap: 2rem; }

.vc-nav-links a {
    color: var(--text-secondary);
    text-decoration: none;
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    font-weight: 400;
    transition: color 200ms ease;
    position: relative;
    padding-bottom: 2px;
}

.vc-nav-links a::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 0; right: 0;
    height: 2px;
    background: var(--gold);
    transform: scaleX(0);
    transition: transform 200ms ease;
}

.vc-nav-links a:hover { color: var(--text-primary); }
.vc-nav-links a:hover::after { transform: scaleX(1); }

/* ── Hero ── */
.vc-hero {
    min-height: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4.5rem 2rem 3.5rem;
    margin-left: -3.5rem;
    margin-right: -3.5rem;
    background: radial-gradient(ellipse 800px 500px at 50% 55%, rgba(124, 58, 237, 0.18), transparent);
}

.vc-hero-badge {
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: var(--accent-light);
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.40);
    border-radius: 999px;
    padding: 5px 16px;
    margin-bottom: 1.25rem;
    text-transform: uppercase;
    display: inline-block;
    backdrop-filter: blur(8px);
    animation: fadeUp 0.4s ease-out both;
}

.vc-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 54px;
    font-weight: 800;
    line-height: 1.1;
    color: var(--text-primary);
    margin: 0 0 1rem;
    animation: fadeUp 0.5s ease-out 0.1s both;
}

.vc-hero-accent {
    color: var(--accent-light);
    display: block;
    text-shadow: 0 0 40px rgba(168, 85, 247, 0.40);
}

.vc-hero-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 16px;
    font-weight: 300;
    color: var(--text-secondary);
    margin: 0 0 1.75rem;
    animation: fadeUp 0.5s ease-out 0.2s both;
}

.vc-hero-ctas {
    display: flex;
    gap: 1rem;
    justify-content: center;
    animation: fadeUp 0.5s ease-out 0.3s both;
}

.vc-btn-primary {
    background: rgba(124, 58, 237, 0.20);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(124, 58, 237, 0.50);
    color: var(--text-primary) !important;
    text-decoration: none !important;
    border-radius: 12px;
    height: 52px;
    padding: 0 32px;
    line-height: 52px;
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    display: inline-block;
    animation: glowPulse 3s ease-in-out infinite;
    transition: all 250ms ease;
}
.vc-btn-primary:visited, .vc-btn-primary:active { color: var(--text-primary) !important; }

.vc-btn-primary:hover {
    background: rgba(124, 58, 237, 0.38);
    border-color: rgba(168, 85, 247, 0.75);
    box-shadow: 0 0 50px rgba(124, 58, 237, 0.45), 0 8px 32px rgba(124, 58, 237, 0.25);
    transform: translateY(-2px);
}

.vc-btn-secondary {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: var(--text-secondary) !important;
    text-decoration: none !important;
    border-radius: 12px;
    height: 52px;
    padding: 0 32px;
    line-height: 52px;
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    display: inline-block;
    transition: all 250ms ease;
}
.vc-btn-secondary:visited, .vc-btn-secondary:active { color: var(--text-secondary) !important; }

.vc-btn-secondary:hover {
    border-color: rgba(124, 58, 237, 0.45);
    color: var(--text-primary);
    background: rgba(124, 58, 237, 0.08);
}

/* ── Sections ── */
.vc-section {
    padding: 36px 0 8px;
}

.vc-section-alt-inner {
    padding: 36px 0 8px;
}

/* ── Headings ── */
.vc-playfair {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.75rem;
    line-height: 1.2;
    letter-spacing: -0.02em;
}

.vc-section-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 400;
    color: var(--text-secondary);
    margin: 0 0 1.5rem;
    line-height: 1.6;
    max-width: 560px;
}

/* ── Glass card system ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 20px;
    position: relative;
    overflow: hidden;
    transition: background 250ms ease, border-color 250ms ease,
                transform 250ms ease, box-shadow 250ms ease;
}

.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-accent);
    transform: translateY(-3px);
    box-shadow: 0 20px 60px rgba(124, 58, 237, 0.15), 0 0 0 1px rgba(124, 58, 237, 0.10);
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.60), transparent);
    transition: left 300ms ease, right 300ms ease;
}

.glass-card:hover::before { left: 0; right: 0; }

/* ── Stat cards ── */
.vc-stat-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    position: relative;
    overflow: hidden;
    height: auto;
    cursor: default;
    transition: background 250ms ease, border-color 250ms ease,
                transform 250ms ease, box-shadow 250ms ease;
}

.vc-stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 25%; right: 25%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.70), transparent);
    transition: left 300ms ease, right 300ms ease;
}

.vc-stat-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-accent);
    transform: translateY(-4px);
    box-shadow: 0 24px 64px rgba(124, 58, 237, 0.18),
                0 0 0 1px rgba(124, 58, 237, 0.12);
}

.vc-stat-card:hover::before { left: 0; right: 0; }

.vc-stat-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: var(--accent-light);
    margin: 0 0 8px;
    line-height: 1.2;
}

.vc-stat-label {
    font-family: 'Outfit', sans-serif;
    font-size: 13px;
    font-weight: 400;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.4;
}

/* ── Flow pills ── */
.vc-flow {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin: 1rem 0;
}

.vc-flow-pill {
    font-family: 'Outfit', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: var(--accent-light);
    background: rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 999px;
    padding: 7px 16px;
    white-space: nowrap;
    transition: all 250ms ease;
    cursor: default;
}

.vc-flow-pill:hover {
    background: rgba(124, 58, 237, 0.15);
    border-color: rgba(124, 58, 237, 0.55);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.18);
}

.vc-flow-arrow { color: var(--text-muted); font-size: 14px; }

/* ── Step cards ── */
.vc-steps {
    list-style: none;
    padding: 0; margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.vc-step { display: flex; align-items: baseline; gap: 0.75rem; }

.vc-step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 400;
    color: var(--accent);
    opacity: 0.70;
    flex-shrink: 0;
    padding-top: 2px;
    min-width: 18px;
}

.vc-step-title {
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.4;
}

/* ── Metric cards ── */
.vc-metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: background 250ms ease, border-color 250ms ease,
                transform 250ms ease, box-shadow 250ms ease;
    animation: fadeUp 0.5s ease-out both;
}

.vc-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.60), transparent);
    transition: left 300ms ease, right 300ms ease;
}

.vc-metric-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-accent);
    transform: translateY(-3px);
    box-shadow: 0 20px 60px rgba(124, 58, 237, 0.15);
}

.vc-metric-card:hover::before { left: 0; right: 0; }

.vc-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: var(--accent-light);
    margin: 0 0 4px;
    display: block;
}

.vc-metric-label {
    font-family: 'Outfit', sans-serif;
    font-size: 12px;
    font-weight: 300;
    color: var(--text-muted);
    margin: 0;
    display: block;
}

/* ── Footer ── */
.vc-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(8, 8, 18, 0.80);
    backdrop-filter: blur(12px);
    padding: 2rem 0;
    margin-top: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}

.vc-footer-name {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
}

.vc-footer-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 12px;
    font-weight: 300;
    color: var(--text-muted);
    margin-top: 4px;
}

.vc-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }

.vc-badge {
    background: rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 999px;
    padding: 4px 12px;
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    color: var(--text-secondary);
    transition: border-color 200ms ease, color 200ms ease;
}

.vc-badge:hover {
    border-color: rgba(124, 58, 237, 0.50);
    color: var(--text-primary);
}

/* ── Streamlit overrides ── */
.stButton button {
    font-family: 'Outfit', sans-serif !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    transition: all 250ms ease !important;
}

.stButton button[kind="primary"] {
    background: rgba(124, 58, 237, 0.20) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(124, 58, 237, 0.50) !important;
    height: 56px !important;
    font-size: 16px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    max-width: 360px !important;
    margin: 0 auto !important;
    display: block !important;
    backdrop-filter: blur(12px) !important;
    animation: glowPulse 3s ease-in-out infinite !important;
}

.stButton button[kind="primary"]:hover {
    background: rgba(124, 58, 237, 0.38) !important;
    border-color: rgba(168, 85, 247, 0.75) !important;
    box-shadow: 0 0 50px rgba(124, 58, 237, 0.45),
                0 8px 32px rgba(124, 58, 237, 0.25) !important;
    transform: translateY(-2px) !important;
}

.stNumberInput input, .stTextInput input, .stTextArea textarea,
.stSelectbox select, div[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 200ms ease, box-shadow 200ms ease !important;
}

.stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(124, 58, 237, 0.60) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.12) !important;
}

.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.60) !important;
}

details summary {
    color: var(--text-secondary) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
}

[data-testid="stExpander"],
[data-testid="stExpander"]:focus,
[data-testid="stExpander"]:focus-within,
[data-testid="stExpander"]:hover {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(124, 58, 237, 0.25) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px) !important;
    margin-bottom: 0.75rem !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    outline: none !important;
    border-color: transparent !important;
}

[data-testid="stFileUploader"] {
    background: rgba(124, 58, 237, 0.04) !important;
    border: 1.5px dashed rgba(124, 58, 237, 0.30) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 200ms ease, background 200ms ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(124, 58, 237, 0.60) !important;
    background: rgba(124, 58, 237, 0.07) !important;
}

/* Centre the upload button and shrink the size hint */
[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    min-height: 90px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 6px !important;
}

[data-testid="stFileUploaderDropzone"] small {
    font-family: 'Outfit', sans-serif !important;
    font-size: 11px !important;
    color: var(--text-muted) !important;
    margin-top: 2px !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }
.stCaption { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* ── Animations ── */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-10px); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(124, 58, 237, 0.20),
                            0 4px 16px rgba(124, 58, 237, 0.10); }
    50%       { box-shadow: 0 0 40px rgba(124, 58, 237, 0.45),
                            0 8px 32px rgba(124, 58, 237, 0.20); }
}

.vc-divider { height: 1px; background: rgba(255, 255, 255, 0.06); margin: 0; }

@media (max-width: 768px) {
    .vc-hero-title { font-size: 38px !important; }
    .vc-hero-sub { font-size: 15px !important; }
    .vc-hero-ctas { flex-direction: column; align-items: center; }
    .vc-btn-primary, .vc-btn-secondary { width: 260px; }
    .vc-section, .vc-section-alt-inner { padding: 48px 1.5rem !important; }
    .vc-footer { padding: 2rem 1.5rem; }
    [data-testid="column"] { width: 100% !important; flex: none !important; }
    .stButton button[kind="primary"] { max-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Floating blobs ──────────────────────────────────────────────────────
st.markdown("""
<div class="vc-blob vc-blob-1"></div>
<div class="vc-blob vc-blob-2"></div>
<div class="vc-blob vc-blob-3"></div>
""", unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="vc-navbar" id="vc-navbar">
  <span class="vc-brand">VentureAI</span>
  <nav class="vc-nav-links">
    <a href="#advisor">Advisor</a>
    <a href="#visualiser">Visualiser</a>
    <a href="#overview">The Platform</a>
  </nav>
  <div style="width:120px;"></div>
</div>
<script>
window.addEventListener('scroll', function() {
  var nav = document.getElementById('vc-navbar');
  if (nav) nav.classList.toggle('navbar-scrolled', window.scrollY > 80);
});
</script>
""", unsafe_allow_html=True)

st.markdown('<div class="vc-page">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="vc-hero">
<div class="vc-hero-badge">AI-Powered Investment Intelligence</div>
<h1 class="vc-hero-title">Smarter startup<span class="vc-hero-accent">decisions.</span></h1>
<p class="vc-hero-sub">Predict investment potential using machine learning and explainable AI.</p>
<div class="vc-hero-ctas">
<a class="vc-btn-primary" href="#advisor">Analyse a Startup</a>
<a class="vc-btn-secondary" href="#visualiser">Explore Visualiser</a>
</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# STARTUP ADVISOR
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div id="advisor" class="vc-section">
<h2 class="vc-playfair">Startup Advisor</h2>
<p class="vc-section-sub">Enter your startup&#39;s details to receive an AI-powered investment verdict.</p>
</div>
""", unsafe_allow_html=True)

try:
    from advisory_app import render_advisor
    render_advisor()
except Exception as e:
    st.error(f"Advisor module error: {e}")
    import traceback; st.code(traceback.format_exc())

st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CSV VISUALISER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div id="visualiser" class="vc-section">
<h2 class="vc-playfair">Data Visualiser</h2>
<p class="vc-section-sub">Upload any CSV and describe the chart you want.</p>
</div>
""", unsafe_allow_html=True)

try:
    from visualizer_app import render_visualizer
    render_visualizer()
except Exception as e:
    st.error(f"Visualiser module error: {e}")
    import traceback; st.code(traceback.format_exc())

st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div id="overview"></div>', unsafe_allow_html=True)

_overview_html = (
'<div style="padding:36px 0 32px;">'
'<div style="display:grid;grid-template-columns:3fr 2fr;gap:48px;align-items:start;">'
'<div>'
'<h2 style="font-family:\'Syne\',sans-serif;font-size:38px;font-weight:700;color:#F0EEFF;margin:0 0 1.5rem;line-height:1.2;letter-spacing:-0.02em;">The platform.</h2>'
'<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;">'
'<span class="vc-flow-pill">CSV Upload</span>'
'<span class="vc-flow-arrow">&#x2192;</span>'
'<span class="vc-flow-pill">LangGraph Router</span>'
'<span class="vc-flow-arrow">&#x2192;</span>'
'<span class="vc-flow-pill">Planner / Patch</span>'
'<span class="vc-flow-arrow">&#x2192;</span>'
'<span class="vc-flow-pill">Highcharts Output</span>'
'</div>'
'<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">'
'<span class="vc-flow-pill">Startup Inputs</span>'
'<span class="vc-flow-arrow">&#x2192;</span>'
'<span class="vc-flow-pill">LightGBM Model</span>'
'<span class="vc-flow-arrow">&#x2192;</span>'
'<span class="vc-flow-pill">SHAP Analysis</span>'
'<span class="vc-flow-arrow">&#x2192;</span>'
'<span class="vc-flow-pill">LLM Narrative</span>'
'</div>'
'</div>'
'<div style="display:flex;flex-direction:column;gap:12px;">'
'<div class="vc-stat-card" style="padding:18px 22px;">'
'<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;font-weight:500;color:#A855F7;margin:0 0 4px;line-height:1.2;">LightGBM</div>'
'<div style="font-family:\'Outfit\',sans-serif;font-size:12px;font-weight:400;color:#A89BC2;margin:0;line-height:1.4;">Gradient Boosted Classifier</div>'
'</div>'
'<div class="vc-stat-card" style="padding:18px 22px;">'
'<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;font-weight:500;color:#A855F7;margin:0 0 4px;line-height:1.2;">371</div>'
'<div style="font-family:\'Outfit\',sans-serif;font-size:12px;font-weight:400;color:#A89BC2;margin:0;line-height:1.4;">Indian startups in training data</div>'
'</div>'
'<div class="vc-stat-card" style="padding:18px 22px;">'
'<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;font-weight:500;color:#A855F7;margin:0 0 4px;line-height:1.2;">4-node</div>'
'<div style="font-family:\'Outfit\',sans-serif;font-size:12px;font-weight:400;color:#A89BC2;margin:0;line-height:1.4;">LangGraph agentic pipeline</div>'
'</div>'
'</div>'
'</div>'
'</div>'
)
st.markdown(_overview_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HOW IT WORKS + MODEL METRICS (collapsed expander)
# ══════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

with st.expander("How It Works & Model Performance", expanded=False):
    st.markdown("""
<div style="padding:0.75rem 0 0.5rem;">
<p style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:#F0EEFF;margin:0 0 0.25rem;letter-spacing:-0.02em;">How It Works</p>
<p style="font-family:'Outfit',sans-serif;font-size:14px;font-weight:400;color:#A89BC2;margin:0 0 1.25rem;line-height:1.6;">Two pipelines — one for investment analysis, one for data visualisation.</p>
</div>
""", unsafe_allow_html=True)

    hw1, hw2 = st.columns(2, gap="large")
    with hw1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px 24px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:20%;right:20%;height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.60),transparent);"></div>
<p style="font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:#F0EEFF;margin:0 0 0.875rem;letter-spacing:0.03em;">Startup Advisor</p>
<ol class="vc-steps">
<li class="vc-step"><span class="vc-step-num">01</span><span class="vc-step-title">Enter startup financials</span></li>
<li class="vc-step"><span class="vc-step-num">02</span><span class="vc-step-title">LightGBM predicts investment class</span></li>
<li class="vc-step"><span class="vc-step-num">03</span><span class="vc-step-title">SHAP ranks feature impact</span></li>
<li class="vc-step"><span class="vc-step-num">04</span><span class="vc-step-title">LLM explains the verdict</span></li>
</ol>
</div>""", unsafe_allow_html=True)

    with hw2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px 24px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:20%;right:20%;height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.60),transparent);"></div>
<p style="font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:#F0EEFF;margin:0 0 0.875rem;letter-spacing:0.03em;">CSV Visualiser</p>
<ol class="vc-steps">
<li class="vc-step"><span class="vc-step-num">01</span><span class="vc-step-title">Upload your CSV</span></li>
<li class="vc-step"><span class="vc-step-num">02</span><span class="vc-step-title">Describe your chart in plain English</span></li>
<li class="vc-step"><span class="vc-step-num">03</span><span class="vc-step-title">LangGraph builds the query pipeline</span></li>
<li class="vc-step"><span class="vc-step-num">04</span><span class="vc-step-title">Highcharts renders the result</span></li>
</ol>
</div>""", unsafe_allow_html=True)

    # ── Model metrics ─────────────────────────────────────────────────
    try:
        import json as _json
        with open("advisory_app/model_metrics.json") as _f:
            _m = _json.load(_f)
        _wf1   = f"{_m['weighted_f1']*100:.1f}%"
        _acc   = f"{_m['test_accuracy']*100:.1f}%"
        _cv    = (f"{_m['cv_f1_mean']*100:.1f}% ±{_m['cv_f1_std']*100:.1f}%"
                  if "cv_f1_mean" in _m else "—")
        _inv_p = f"{_m['per_class']['Invest']['precision']*100:.1f}%"
        _hld_r = f"{_m['per_class']['Hold']['recall']*100:.1f}%"
        _avd_r = f"{_m['per_class']['Avoid']['recall']*100:.1f}%"
    except Exception:
        _wf1 = _acc = _cv = _inv_p = _hld_r = _avd_r = "—"

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#F0EEFF;margin:0 0 0.75rem;letter-spacing:-0.02em;">Model Performance</p>
""", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3, gap="medium")
    mc4, mc5, mc6 = st.columns(3, gap="medium")
    for i, (col, val, label) in enumerate([
        (mc1, _wf1,   "Weighted F1 Score"),
        (mc2, _acc,   "Test Accuracy"),
        (mc3, _cv,    "CV F1 Mean ± Std"),
        (mc4, _inv_p, "Invest Precision"),
        (mc5, _hld_r, "Hold Recall"),
        (mc6, _avd_r, "Avoid Recall"),
    ]):
        with col:
            delay = 0.1 + i * 0.05
            st.markdown(
f"<div class=\"vc-metric-card\" style=\"animation-delay:{delay:.2f}s;\">"
f"<span class=\"vc-metric-value\">{val}</span>"
f"<span class=\"vc-metric-label\">{label}</span>"
"</div>",
unsafe_allow_html=True)

    st.markdown(
"<p style=\"font-family:'Outfit',sans-serif;font-size:12px;font-weight:300;"
"color:#5A4E72;text-align:center;margin-top:0.5rem;\">"
"Evaluated on held-out test set &nbsp;&middot;&nbsp; 80/20 stratified split &nbsp;&middot;&nbsp; 5-fold cross-validation"
"</p>",
unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="vc-footer">
  <div>
    <div class="vc-footer-name">VentureAI</div>
    <div class="vc-footer-sub">MITADT University, Pune &nbsp;&middot;&nbsp; Final Year Project</div>
  </div>
  <div class="vc-badges">
    <span class="vc-badge">LightGBM</span>
    <span class="vc-badge">LangGraph</span>
    <span class="vc-badge">Cerebras</span>
    <span class="vc-badge">Highcharts</span>
    <span class="vc-badge">Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close vc-page
