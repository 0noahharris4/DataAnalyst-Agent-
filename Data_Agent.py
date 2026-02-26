import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
try:
    from data_model import (
        initial_analysis,
        clean_data,
        numeric_stats,
        categorical_stats,
        distribution_plots,
        correlation_analysis,
        time_series_analysis,
        generate_profile,
        build_ai_prompt,
        detect_problematic_columns,
        sanitize_for_display,
    )
except ImportError:
    initial_analysis = clean_data = numeric_stats = categorical_stats = distribution_plots = correlation_analysis = time_series_analysis = generate_profile = build_ai_prompt = detect_problematic_columns = sanitize_for_display = None

# Ensure HAS_PROFILE is always defined
try:
    from ydata_profiling import ProfileReport
    HAS_PROFILE = True
except Exception:
    HAS_PROFILE = False

# Ensure HAS_OPENAI and openai are always defined
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False
    openai = None

# Ensure model_choice and temperature_choice are always defined
model_options = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
    "gpt-4o-mini-instruct",
    "gpt-4o-realtime-preview"
]
default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
try:
    default_index = model_options.index(default_model) if default_model in model_options else 0
except Exception:
    default_index = 0
model_choice = model_options[default_index]
try:
    temp_default = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
except Exception:
    temp_default = 0.2
temperature_choice = temp_default

st.set_page_config(layout="wide", page_title="AI Data Analyst")

# Sidebar: Settings (visible before upload)

# --- Load OpenAI API key from Streamlit secrets if available ---
if not os.getenv("OPENAI_API_KEY") and hasattr(st, "secrets"):
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
        if key:
            os.environ["OPENAI_API_KEY"] = key
    except Exception:
        pass

def call_openai(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 400, temperature: float = 0.2):
    if not HAS_OPENAI:
        return "openai package not available. Set up openai SDK to enable AI insights."
    # prefer explicit env var, fall back to Streamlit secrets if available
    key = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # If env var missing, try Streamlit secrets safely (don't evaluate st.secrets in boolean context)
    if not key and hasattr(st, 'secrets'):
        try:
            secrets_obj = st.secrets  # may raise if no secrets file; catch below
            try:
                raw_keys = list(secrets_obj.keys())
            except Exception:
                raw_keys = []
            for raw_k in raw_keys:
                if raw_k is None:
                    continue
                k = str(raw_k).strip().lstrip('\ufeff').lower()
                try:
                    val = secrets_obj[raw_k]
                except Exception:
                    val = None
                if k in ("openai_api_key", "openaiapikey", "openai_api", "api_key"):
                    key = val
                    break
                if k == 'openai' and isinstance(val, dict):
                    for inner_k, inner_v in val.items():
                        ik = str(inner_k).strip().lower()
                        if ik in ("api_key", "openai_api_key"):
                            key = inner_v
                            break
                    if key:
                        break
        except Exception:
            key = key

    if not key:
        return "OPENAI API key not found. Set OPENAI_API_KEY env var or add it to Streamlit secrets." 
    # set for legacy clients that read openai.api_key
    try:
        openai.api_key = key
    except Exception:
        pass

    # Prefer the modern OpenAI client (openai.OpenAI) when available
    try:
        if hasattr(openai, 'OpenAI'):
            try:
                client = openai.OpenAI(api_key=key)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "You are an expert data analyst skilled in generating valuable insights from CSV files."}, {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # handle mapping-like or object-like responses
                try:
                    return resp['choices'][0]['message']['content'].strip()
                except Exception:
                    try:
                        return resp.choices[0].message.content.strip()
                    except Exception:
                        return str(resp)
            except Exception as e:
                return f"OpenAI call failed (modern client): {e}"

        # Fall back to legacy openai package interface if present
        if hasattr(openai, 'ChatCompletion'):
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role":"system","content":"You are a data analyst."},{"role":"user","content":prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return resp['choices'][0]['message']['content'].strip()
            except Exception as e:
                return f"OpenAI call failed (legacy ChatCompletion): {e}"

        return "openai package does not expose expected client API (OpenAI or ChatCompletion). Check your openai package version."
    except Exception as e:
        return f"OpenAI call failed: {e}"

import streamlit as st

# Title (no black box)
import pathlib
topbar_css_path = pathlib.Path(__file__).parent / "topbar_styles.css"
if topbar_css_path.exists():
    with open(topbar_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_topbar = (
    "<div class='blue-top-bar' style='position: relative;'>"
    "<div style='display: flex; align-items: center; justify-content: center;'>"
    "<h1 style='margin-left: 0; position: relative; z-index: 1; text-align: center; display: flex; align-items: center; justify-content: center;'>"
        "<h1 style='margin-left: 0; position: relative; z-index: 1; text-align: center; display: flex; align-items: center; justify-content: center; font-size: 3.5rem;'>"
        " AI Data Analyst Assistant"
    "<span style='display: inline-flex; align-items: center; height: 90px; margin-left: 38px;'>"
    "<svg width='150' height='90' viewBox='0 0 150 90' fill='none' xmlns='http://www.w3.org/2000/svg'>"
    "<rect x='10' y='50' width='25' height='30' rx='6' fill='#4285F4'/><rect x='45' y='30' width='25' height='50' rx='6' fill='#34A853'/><rect x='80' y='15' width='25' height='65' rx='6' fill='#FBBC05'/><rect x='115' y='5' width='20' height='75' rx='6' fill='#EA4335'/><circle cx='22.5' cy='65' r='7' fill='#fff' fill-opacity='0.7'/><circle cx='57.5' cy='55' r='7' fill='#fff' fill-opacity='0.7'/><circle cx='92.5' cy='35' r='7' fill='#fff' fill-opacity='0.7'/><circle cx='125' cy='25' r='7' fill='#fff' fill-opacity='0.7'/><polyline points='22.5,65 57.5,55 92.5,35 125,25' stroke='#4285F4' stroke-width='3' fill='none' stroke-linecap='round' stroke-linejoin='round'/><circle cx='125' cy='25' r='10' fill='#34A853' stroke='#fff' stroke-width='3'/><text x='125' y='30' text-anchor='middle' font-size='13' font-family='Arial' fill='#fff' font-weight='bold'>AI</text>"
    "</svg>"
    "</span>"
    "</h1>"
    "</div>"
        "<div class='subtitle' style='display: flex; align-items: center; justify-content: center; position: relative; z-index: 1; text-align: center; font-size: 1.35rem;'>Fast, intelligent data analysis — from upload to insight in seconds, powered by AI.</div>"
    "</div>"
)


st.markdown(css_topbar, unsafe_allow_html=True)
# Add spacing below subtitle and above info text
st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
# Add spacing above the info text

info_html = '''
<div style="
    background: linear-gradient(90deg, #4285F4 0%, #34A853 50%, #FBBC05 100%);
    color: #fff;
    font-weight: 600;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 18px;
    font-size: 1.08rem;
    box-shadow: 0 2px 8px rgba(66,133,244,0.08);
    text-align: center;">
    Upload a CSV file to begin an analysis with the AI Data Analyst Assistant.
</div>
'''
st.markdown(info_html, unsafe_allow_html=True)
uploaded = st.file_uploader("Upload CSV", type=['csv'], label_visibility='collapsed')
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(io.StringIO(uploaded.getvalue().decode('utf-8', errors='ignore')))
    # detect and sanitize problematic columns for display
    problematic = detect_problematic_columns(df)
    if problematic:
        st.warning(f"Detected {len(problematic)} column(s) with complex objects that were sanitized for display. See samples below.")
        for c, info in problematic.items():
            st.write(f"- `{c}` (type: {info.get('type')}) - samples: {info.get('samples')[:5]}")
    sanitized_head = sanitize_for_display(df.head(50))
    # (Sidebar settings are defined above so they are visible before upload)
    st.markdown("<span style='font-size:2rem; font-weight:700; color:#4285F4;'>Initial Analysis</span>", unsafe_allow_html=True)
    ia = initial_analysis(df)
    st.markdown("<span style='font-size:1.4rem; font-weight:600; color:#34A853;'>Previewing Dataset</span>", unsafe_allow_html=True)
    # show sanitized preview to avoid serialization errors
    st.dataframe(sanitized_head)
    st.write("Shape:", ia['shape'])
    st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#FBBC05;'>Column data types</span>", unsafe_allow_html=True)
    st.write(ia['dtypes'])
    # Present flagged/problematic columns in a user-friendly format
    problematic = ia.get('problematic')
    st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#EA4335;'>Flagged columns</span>", unsafe_allow_html=True)
    if problematic:
        for kind, cols in problematic.items():
            try:
                cols_list = list(cols)
            except Exception:
                # single value
                cols_list = [cols]
            if not cols_list:
                continue
            title = kind.replace('_', ' ').title()
            st.markdown(f"**{title}** — {len(cols_list)} column(s)")
            # show as inline code list, truncated if long
            display_list = []
            for c in cols_list:
                s = str(c)
                if len(s) > 40:
                    s = s[:37] + '...'
                display_list.append(f"`{s}`")
            st.write(", ".join(display_list))
    else:
        st.write("No flagged/problematic columns detected.")
    st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#4285F4;'>Missing values (top 20)</span>", unsafe_allow_html=True)
    st.table(ia['missing'].head(20))
    # Nicely format fully-empty and >50% missing columns for end users
    fully_empty = ia.get('fully_empty', [])
    more50 = ia.get('more50', [])

    st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#4285F4;'>Missing Columns and Duplicated Rows</span>", unsafe_allow_html=True)
    if fully_empty:
        st.markdown(f"**Fully empty columns ({len(fully_empty)}):**")
        fe_display = []
        for c in fully_empty:
            s = str(c)
            if len(s) > 40:
                s = s[:37] + '...'
            fe_display.append(f"`{s}`")
        st.write(", ".join(fe_display))
    else:
        st.markdown("**Fully empty columns:** None")

    if more50:
        st.markdown(f"**Columns with >50% missing ({len(more50)}):**")
        m50_display = []
        for c in more50:
            s = str(c)
            if len(s) > 40:
                s = s[:37] + '...'
            m50_display.append(f"`{s}`")
        st.write(", ".join(m50_display))
    else:
        st.markdown("**Columns with >50% missing:** None")
    st.write("Duplicate rows count:", ia['duplicates'])
    # Nicely format inconsistent formats for end users
    inconsistent = ia.get('inconsistent')
    if inconsistent:
        st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#EA4335;'>Inconsistent formats detected</span>", unsafe_allow_html=True)
        for kind, cols in inconsistent.items():
            try:
                cols_list = list(cols)
            except Exception:
                cols_list = [cols]
            st.write(f"- **{kind}**: {', '.join(cols_list)}")
    else:
        st.write("Inconsistent formats detected: None")

    st.markdown("<span style='font-size:2rem; font-weight:700; color:#34A853;'>Cleaning</span>", unsafe_allow_html=True)
    if st.button("Clean Dataset"):
        with st.spinner("Cleaning..."):
            clean_df, clean_summary = clean_data(df.copy())
        st.success("Cleaning done")
        # User-friendly cleaning summary output
        st.markdown("**Cleaning Summary:**")
        st.write(f"Rows/columns before cleaning: {clean_summary['before_shape'][0]} rows × {clean_summary['before_shape'][1]} columns")
        st.write(f"Rows/columns after cleaning: {clean_summary['after_shape'][0]} rows × {clean_summary['after_shape'][1]} columns")
        if clean_summary['dropped_fully_empty_cols']:
            st.write(f"Dropped fully empty columns: {', '.join(clean_summary['dropped_fully_empty_cols'])}")
        else:
            st.write("No fully empty columns were dropped.")
        st.write(f"Dropped duplicate rows: {clean_summary['dropped_duplicates']}")
        st.info(clean_summary['note'])
        st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#FBBC05;'>Resulting shape</span>", unsafe_allow_html=True)
        st.write(clean_df.shape)
        st.session_state['clean_df'] = clean_df
        # store sanitized preview as well
        st.session_state['clean_preview'] = sanitize_for_display(clean_df.head(50))

    if 'clean_df' in st.session_state:
        working = st.session_state['clean_df']
        sanitized_working_preview = st.session_state.get('clean_preview', sanitize_for_display(working.head(50)))
    else:
        working = df
        sanitized_working_preview = sanitize_for_display(df.head(50))

    st.markdown("<span style='font-size:2rem; font-weight:700; color:#FBBC05;'>Descriptive Statistics</span>", unsafe_allow_html=True)
    ns = numeric_stats(working)
    # numeric_stats returns a DataFrame when numeric columns exist, otherwise returns {}
    if isinstance(ns, pd.DataFrame) and not ns.empty:
            st.dataframe(ns.head(50))
    else:
        st.info("No numeric columns detected.")

    st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#EA4335;'>Categorical stats</span>", unsafe_allow_html=True)
    cats = categorical_stats(working, top_n=10)
    if cats:
        cat_rows = []
        for col, info in cats.items():
            unique = info.get('unique') if isinstance(info, dict) else None
            top_cats = info.get('top_categories', {}) if isinstance(info, dict) else {}
            try:
                top_str = ", ".join(f"{k} ({v})" for k, v in list(top_cats.items()))
            except Exception:
                top_str = str(top_cats)
            cat_rows.append({
                'column': col,
                'unique': unique,
                'top_categories': top_str,
            })
        st.dataframe(pd.DataFrame(cat_rows))
    else:
        st.info("No categorical columns detected.")

    st.markdown("<span style='font-size:2rem; font-weight:700; color:#4285F4;'>Distributions</span>", unsafe_allow_html=True)
    plots, cat_plots = distribution_plots(working)
    for c, fig in plots.items():
        st.pyplot(fig)
    for c, fig in cat_plots.items():
        st.pyplot(fig)

    st.markdown("<span style='font-size:2rem; font-weight:700; color:#34A853;'>Correlation & Relationships</span>", unsafe_allow_html=True)
    corr_res = correlation_analysis(working)
    if corr_res['corr'] is not None:
        st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#4285F4;'>Correlation matrix</span>", unsafe_allow_html=True)
        st.dataframe(corr_res['corr'])
        st.markdown("<span style='font-size:1.2rem; font-weight:600; color:#FBBC05;'>Top correlations</span>", unsafe_allow_html=True)
        top_pairs = corr_res.get('top_pairs', [])
        # Normalize to list of dicts for display
        rows = []
        for p in top_pairs[:10]:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                a, b, c = p[0], p[1], p[2]
            elif isinstance(p, dict):
                a = p.get('col_a') or p.get('a')
                b = p.get('col_b') or p.get('b')
                c = p.get('corr') or p.get('value')
            else:
                continue
            try:
                corr_val = float(c)
            except Exception:
                corr_val = None
            rows.append({'col_a': a, 'col_b': b, 'corr': round(corr_val, 3) if corr_val is not None else ''})
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No correlation pairs to display.")
        if corr_res['multicollinear']:
            st.warning(f"Potential multicollinearity: {corr_res['multicollinear']}")
    else:
        st.info("Not enough numeric columns for correlation analysis.")

    st.markdown("<span style='font-size:2rem; font-weight:700; color:#EA4335;'>Time-series (if datetime detected)</span>", unsafe_allow_html=True)
    ts_res = time_series_analysis(working)
    if ts_res:
        st.write("Datetime column used:", ts_res.get('datetime_column'))
        for freq, res in (ts_res.get('resampled') or {}).items():
            if res is not None:
                st.subheader(f"Resampled ({freq}) - preview")
                st.dataframe(res.head())
        # spike detection removed from data model; nothing to display here
    else:
        st.info("No datetime-based time-series analysis available.")



    st.markdown("<span style='font-size:2rem; font-weight:700; color:#4285F4;'>AI Analysis Report</span>", unsafe_allow_html=True)
    st.markdown('''
        <style>
        .stButton > button {
            background: linear-gradient(90deg, #4285F4 0%, #34A853 33%, #FBBC05 66%, #EA4335 100%);
            color: #fff !important;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-size: 1.1rem;
            box-shadow: 0 2px 8px rgba(66,133,244,0.08);
            transition: background 0.3s;
        }
        .stButton > button:hover {
            filter: brightness(1.08);
            box-shadow: 0 4px 16px rgba(66,133,244,0.18);
        }
        </style>
    ''', unsafe_allow_html=True)
    if st.button("Generate AI Insights", key="ai_insights_btn"):
        spinner_html = '''
        <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
            <span style="color:#4285F4; font-weight:600; font-size:1.2rem;">Analyzing with AI</span>
            <span class="google-dot" style="background:#4285F4;"></span>
            <span class="google-dot" style="background:#34A853;"></span>
            <span class="google-dot" style="background:#FBBC05;"></span>
            <span class="google-dot" style="background:#EA4335;"></span>
        </div>
        <style>
        .google-dot {
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-left: 4px;
            animation: google-bounce 1.2s infinite both;
        }
        .google-dot:nth-child(2) { animation-delay: 0s; }
        .google-dot:nth-child(3) { animation-delay: 0.2s; }
        .google-dot:nth-child(4) { animation-delay: 0.4s; }
        .google-dot:nth-child(5) { animation-delay: 0.6s; }
        @keyframes google-bounce {
            0%, 80%, 100% { transform: scale(0.8); }
            40% { transform: scale(1.3); }
        }
        </style>
        '''
        spinner_placeholder = st.empty()
        spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
        import time
        try:
            ns_for_ai = ns
        except Exception:
            ns_for_ai = numeric_stats(working)

        if isinstance(ns_for_ai, pd.DataFrame) and not ns_for_ai.empty:
            top_variances = list(ns_for_ai.head(10).index)
            numeric_stats_head = ns_for_ai.head().to_dict()
        else:
            top_variances = []
            numeric_stats_head = {}

        summary = {
            'shape': working.shape,
            'missing': working.isna().sum().sort_values(ascending=False),
            'correlation_top': [list(x) for x in corr_res.get('top_pairs', [])[:10]],
            'top_variances': top_variances,
            'outliers_summary': {}, # could be filled with more detail
            'categorical_sample': {k:list(v['top_categories'].items()) for k,v in categorical_stats(working, top_n=5).items()},
            'numeric_stats_head': numeric_stats_head
        }
        prompt = build_ai_prompt(summary)
        try:
            model = model_choice
        except Exception:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            temperature = float(temperature_choice)
        except Exception:
            temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        ai_response = call_openai(prompt, model=model, temperature=temperature)
        spinner_placeholder.empty()
        st.markdown("<span style='font-size:1.5rem; font-weight:700; color:#4285F4;'>AI Insights</span>", unsafe_allow_html=True)
        st.write(ai_response)

