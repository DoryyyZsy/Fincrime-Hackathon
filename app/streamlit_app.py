import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from src.io import PROCESSED, read_json
from src.match import waterfall

st.set_page_config(page_title="Right Person Screening", page_icon="◈", layout="wide")

ORDER = {"likely_true_hit": 0, "human_review": 1, "auto_dismissed": 2}
LABEL = {"likely_true_hit": "Likely hit", "human_review": "Human review", "auto_dismissed": "Auto-dismissed"}
COLORS = {"Likely hit": "#a42d25", "Human review": "#8a5b09", "Auto-dismissed": "#196656"}
AUDIT = PROCESSED / "audit_log.jsonl"


def record_decision(alert, action, reason):
    if not reason.strip():
        return False
    entry = {
        "alert_id": alert["alert_id"], "customer_id": alert["customer"]["customer_id"],
        "entity_id": alert["entity"]["entity_id"], "decision": action,
        "reason": reason.strip(), "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    fd = os.open(AUDIT, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(entry) + "\n").encode())
    finally:
        os.close(fd)
    return True


def audit_history(alert):
    if not AUDIT.exists():
        return []
    with AUDIT.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.endswith("\n")]
    return [row for row in rows if (row["customer_id"], row["entity_id"]) ==
            (alert["customer"]["customer_id"], alert["entity"]["entity_id"])]


def overview():
    st.title("Right Person Screening")
    metrics = read_json("metrics")
    funnel = read_json("funnel")
    if not metrics or not funnel:
        st.info("Generate the screening results with make data, make match, make alerts and make eval.")
        return
    cols = st.columns(4)
    for column, key, label in zip(cols[:3], ["alerts", "missed", "false_positives"],
                                 ["Alerts raised", "Missed true hits", "False positives"]):
        ours, baseline = metrics["ours"][key], metrics["baseline"][key]
        column.metric(label, f"{ours:,}", help=f"Name-only baseline: {baseline:,}")
        column.caption(f"Baseline: {baseline:,}")
    reduction = metrics["alert_reduction_pct"]
    cols[3].metric("Alert reduction", f"{reduction:.1f}%" if reduction is not None else "—")
    st.caption(f"True hits caught · {metrics['ours']['caught']} linkage · {metrics['baseline']['caught']} baseline")
    st.subheader("Screening funnel")
    names = ["Customers", "Candidate rows", "Scored pairs", "Grey zone", "Human queue"]
    keys = ["customers", "candidates", "scored", "grey_zone", "human_queue"]
    bars = [{"Stage": name, "Count": funnel.get(key, 0)} for name, key in zip(names, keys)]
    st.altair_chart(
        alt.Chart(alt.Data(values=bars)).mark_bar(color="#176b68", cornerRadiusEnd=4).encode(
            y=alt.Y("Stage:N", sort=names, title=None), x=alt.X("Count:Q", title=None),
            tooltip=["Stage:N", "Count:Q"],
        ).properties(height=220),
        width="stretch",
    )
    if demos := read_json("demo_cases", []):
        st.subheader("Demo cases")
        for demo in demos:
            if st.button(f"{demo['title']} · {demo['alert_id']}", key=f"demo_{demo['alert_id']}"):
                st.session_state["selected_alert"] = demo["alert_id"]
                st.switch_page(detail_page)


def queue():
    st.title("Alert queue")
    alerts = read_json("alerts", [])
    choices = st.multiselect("Routing outcomes", list(LABEL.values()), default=list(LABEL.values()))
    search = st.text_input("Filter by name or alert ID").strip().casefold()
    selected = [
        alert for alert in sorted(alerts, key=lambda a: (ORDER[a["route"]], -a["match_probability"]))
        if LABEL[alert["route"]] in choices and (not search or any(
            search in field.casefold() for field in
            (alert["alert_id"], alert["customer"]["name"], alert["entity"]["name"])))
    ]
    if not selected:
        st.info("No alerts match the current filters.")
        return
    frame = pd.DataFrame([
        {"Alert": a["alert_id"], "Outcome": LABEL[a["route"]], "Identity score": a["match_probability"],
         "Customer": a["customer"]["name"], "Listed entity": a["entity"]["name"]} for a in selected
    ])
    styler = frame.style.map(lambda v: f"color: {COLORS[v]}; font-weight: bold", subset=["Outcome"])
    event = st.dataframe(styler, width="stretch", hide_index=True,
                         on_select="rerun", selection_mode="single-row",
                         column_config={"Identity score": st.column_config.NumberColumn(format="percent")})
    if event.selection.rows and st.button("Open selected alert"):
        st.session_state["selected_alert"] = selected[event.selection.rows[0]]["alert_id"]
        st.switch_page(detail_page)


def display(value):
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value) if value is not None else "—"


def detail():
    st.title("Alert detail")
    alerts = {a["alert_id"]: a for a in read_json("alerts", [])}
    if not alerts:
        st.info("Generate alerts with make alerts.")
        return
    selected = st.session_state.get("selected_alert")
    key = st.selectbox("Alert", list(alerts), index=list(alerts).index(selected) if selected in alerts else 0,
                       format_func=lambda k: f"{k} · {alerts[k]['customer']['name']} / {alerts[k]['entity']['name']}")
    alert = alerts[key]
    customer, entity = alert["customer"], alert["entity"]
    st.metric(LABEL[alert["route"]], f"{alert['match_probability']:.2%}")
    left, right = st.columns(2)
    left.subheader("Bank customer")
    left.write(customer["name"])
    right.subheader("Listed person")
    right.write(entity["name"])
    right.caption(f"Compared alias · {entity['compared_name']}")
    if entity["source_url"]:
        right.link_button("Open list entry", entity["source_url"])
    if entity.get("countries"):
        right.caption(f"List country affiliations · {display(entity['countries'])}")
    symbols = {"exact": "✓", "overlap": "✓", "mismatch": "✗", "none shared": "✗", "missing": "—"}
    st.table([{"Field": f["field"], "Customer": display(f["customer"]), "Listed person": display(f["entity"]),
               "Comparison": f"{symbols.get(f['result'], '≈')} {f['result']}"}
              for f in alert["field_comparisons"]])
    st.altair_chart(waterfall(alert["field_comparisons"], alert["prior_weight"]), width="stretch")
    st.subheader("Recommendation")
    if method := alert["agent"].get("method"):
        st.caption(method)
    st.text(alert["agent"]["reasoning"])
    if note := alert.get("public_identity_note"):
        st.text(note["text"])
        st.link_button(note["title"], note["url"])
    for evidence in alert["agent"].get("missing_info", []):
        st.text(evidence)
    for source in alert["agent"]["sources"]:
        st.link_button(source["title"], source["url"])
    st.subheader("Analyst decision")
    with st.form(f"decision_{key}"):
        reason = st.text_area("Reason")
        a, b, c = st.columns(3)
        confirm = a.form_submit_button("Confirm match")
        dismiss = b.form_submit_button("Dismiss")
        request = c.form_submit_button("Request info")
    if confirm or dismiss or request:
        action = "confirm_match" if confirm else "dismiss" if dismiss else "request_info"
        if record_decision(alert, action, reason):
            st.success("Decision recorded.")
        else:
            st.error("Enter a reason before recording a decision.")
    for event in reversed(audit_history(alert)):
        st.text(f"{event['timestamp']} · {event['decision']} · {event['reason']}")


def live():
    from src.live import live_screen

    st.title("Live screen")
    with st.form("live"):
        name = st.text_input("Name")
        dob = st.text_input("Date of birth (optional)", placeholder="YYYY-MM-DD or YYYY")
        nationality = st.text_input("Nationality (optional)", placeholder="GB or United Kingdom")
        submitted = st.form_submit_button("Screen")
    if submitted:
        if not name.strip():
            st.error("Enter a name.")
            return
        if not (PROCESSED / "model.json").exists():
            st.info("Train the model with make match.")
            return
        with st.spinner("Screening against the list"):
            try:
                rows = live_screen(name, dob, nationality)
            except ValueError as exc:
                st.error(str(exc))
                return
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True,
                         column_config={"Identity score": st.column_config.NumberColumn(format="percent")})
        else:
            st.info("No candidates passed blocking.")


overview_page = st.Page(overview, title="Overview", default=True)
queue_page = st.Page(queue, title="Alert queue")
detail_page = st.Page(detail, title="Alert detail")
live_page = st.Page(live, title="Live screen")
st.navigation([overview_page, queue_page, detail_page, live_page]).run()
