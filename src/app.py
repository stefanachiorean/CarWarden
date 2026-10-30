"""CarWarden dashboard.   Run with:   streamlit run src/app.py

Reads the files written by `python src/carwarden.py train` (or `demo`):
outputs/scored_windows.csv, outputs/metrics.json, outputs/evasion.csv
"""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CarWarden", layout="wide")

LEVEL_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "NORMAL": "🟢"}


@st.cache_data
def load_results(out_dir):
    scored = pd.read_csv(os.path.join(out_dir, "scored_windows.csv")).fillna({"reasons": ""})
    with open(os.path.join(out_dir, "metrics.json")) as f:
        metrics = json.load(f)
    evasion_path = os.path.join(out_dir, "evasion.csv")
    try:
        evasion = pd.read_csv(evasion_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        evasion = pd.DataFrame()
    return scored, metrics, evasion


st.title("CarWarden")
st.caption("AI security analyst for in-vehicle CAN networks")

out_dir = st.sidebar.text_input("Results folder", os.environ.get("CARWARDEN_OUT", "outputs"))
if not os.path.exists(os.path.join(out_dir, "scored_windows.csv")):
    st.error(f"No results found in '{out_dir}'. Run `python src/carwarden.py train` "
             f"(or `python src/carwarden.py demo`) first.")
    st.stop()

scored, metrics, evasion = load_results(out_dir)
overall = [r for r in metrics["detection"] if r["capture"] == "ALL"][0]

tab_incidents, tab_perf, tab_evasion = st.tabs(["Incidents", "Model performance", "Evasion test"])

# ---------------------------------------------------------------- incidents
with tab_incidents:
    captures = list(scored["capture"].unique())
    capture = st.selectbox("Capture", captures)
    part = scored[scored["capture"] == capture]

    left, mid, right = st.columns(3)
    left.metric("Windows", len(part))
    mid.metric("Flagged", int(part["flagged"].sum()))
    right.metric("Highest risk", f"{part['risk'].max():.0f} / 100")

    st.subheader("Risk over time")
    st.line_chart(part.set_index("t_rel")["risk"], y_label="risk (0-100)", x_label="seconds into capture")

    st.subheader("Flagged windows")
    min_risk = st.slider("Minimum risk", 50, 100, 50)
    flagged = part[part["flagged"] & (part["risk"] >= min_risk)].sort_values("risk", ascending=False)
    if len(flagged) == 0:
        st.info("No windows at or above this risk.")
    else:
        labels = {i: f"t={r.t_rel:.1f}s   risk {r.risk:.0f}   {r.predicted_type}" for i, r in flagged.iterrows()}
        chosen = st.selectbox("Pick a window", list(labels), format_func=lambda i: labels[i])
        row = flagged.loc[chosen]

        confidence = f"{row['confidence']:.0%}" if pd.notna(row["confidence"]) else "n/a"
        headline = (f"{LEVEL_ICON[row['level']]} {row['level']} RISK — {row['predicted_type']} — "
                    f"risk {row['risk']:.0f}/100 — confidence {confidence}")
        (st.error if row["level"] == "HIGH" else st.warning)(headline)

        st.markdown("**Why:**")
        for reason in [r for r in row["reasons"].split(" | ") if r]:
            st.markdown(f"- {reason}")
        st.caption(f"Ground truth from the dataset labels: {row['attack_type']}")

        st.dataframe(flagged[["t_rel", "risk", "level", "predicted_type", "confidence", "reasons"]],
                     use_container_width=True)

# -------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Detection on the held-out test part")
    a, b, c, d = st.columns(4)
    a.metric("Precision", f"{overall['precision']:.3f}")
    b.metric("Recall", f"{overall['recall']:.3f}")
    c.metric("False positive rate", f"{overall['false_positive_rate']:.3f}")
    d.metric("AUC", f"{overall['auc']:.3f}")
    st.dataframe(pd.DataFrame(metrics["detection"]), use_container_width=True)

    st.subheader("Attack-type classifier")
    st.write(f"Accuracy on all test windows: **{metrics['classifier_accuracy']:.3f}**")
    cm = metrics["confusion_matrix"]
    st.dataframe(pd.DataFrame(cm["data"], index=cm["index"], columns=cm["columns"]))

    st.subheader("Which features matter to the classifier")
    importance = pd.Series(metrics["feature_importance"]).sort_values()
    st.bar_chart(importance, horizontal=True)

    st.caption("The detector was trained on normal windows only, from the first part of each capture. "
               "Everything shown here comes from the later part, which it never saw.")

# ------------------------------------------------------------------ evasion
with tab_evasion:
    st.subheader("Attacking the detector")
    st.write("A smarter attacker injects fewer frames so the attack hides in normal traffic. "
             "Each row removes a random share of the injected frames and measures how many of the "
             "original attack windows are still caught.")
    if len(evasion) == 0:
        st.info("No evasion results were saved (training was run with --no-evasion).")
    else:
        recall_table = evasion.pivot(index="keep_fraction", columns="attack_type", values="recall")
        st.line_chart(recall_table, y_label="recall", x_label="share of injected frames kept")
        st.dataframe(evasion, use_container_width=True)
        st.caption("Trade-off for the attacker: fewer frames also means less effect on the vehicle. "
           "Caveat: DoS and fuzzy attacks also disturb the legitimate traffic, and removing injected "
           "frames does not undo that, so recall stays high for them. The spoofing curves are the "
           "meaningful ones.")
