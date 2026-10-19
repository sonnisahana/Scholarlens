"""ScholarLens UI.   streamlit run app/streamlit_app.py"""
import streamlit as st

from scholarlens.pipeline import DISCLAIMER, ScholarLens

st.set_page_config(page_title="ScholarLens", page_icon="🎓", layout="wide")
st.title("🎓 ScholarLens")
st.caption("Describe your situation in plain words (English, ಕನ್ನಡ or mixed). "
           "Every answer cites the source sentence from the official scheme document.")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retriever", ["hybrid_rerank", "hybrid", "dense", "bm25", "tfidf"])
    verify = st.selectbox("Claim verification", ["nli", "light", "off"])
    if st.button("Reset conversation"):
        st.session_state.clear()
        st.rerun()
    st.info(DISCLAIMER)


@st.cache_resource(show_spinner="Loading models and index...")
def engine(mode, verify):
    return ScholarLens(retrieval_mode=mode, verify=verify)


def cites(sids, sources):
    return " ".join(f"`{s}`" for s in sids)


def show(res):
    src = res.get("sources", {})
    if res["intent"] == "out_of_scope":
        st.write(res["message"])
        return
    if res["intent"] == "ask_about_scheme":
        if not res["answerable"]:
            st.warning(res["message"])
        for c in res["claims"]:
            st.markdown(f"- {c['claim']}  {cites(c['sids'], src)}")
    else:
        tabs = st.tabs(["✅ Eligible", "❓ Need more info", "❌ Not eligible"])
        for tab, status in zip(tabs, ("eligible", "needs_info", "not_eligible")):
            with tab:
                items = [e for e in res["schemes"] if e["status"] == status]
                if not items:
                    st.caption("None")
                for e in items:
                    with st.expander(e["scheme_name"], expanded=(status == "eligible")):
                        if not e["rules_verified"]:
                            st.caption("⚠ rules for this scheme have not been human-verified")
                        for c in e["criteria"]:
                            icon = {"met": "✔", "unmet": "✘", "unknown": "?"}[c["status"]]
                            flag = "" if c["verified"] in (True, None) else "  ⚠ not confirmed by source"
                            st.markdown(f"{icon} {c['text']} {cites(c['sids'], src)}{flag}")
                        for c in e["claims"]:
                            st.markdown(f"💡 {c['claim']} {cites(c['sids'], src)}")
        if res["follow_up"]:
            st.info("To narrow this down: " + " ".join(res["follow_up"]))
    if src:
        with st.expander("Source sentences"):
            for sid, v in sorted(src.items()):
                st.markdown(f"**`{sid}`** (page {v['page']}, {v['section']})  \n> {v['text']}")
    with st.expander("Profile understood / latency"):
        st.json({"profile": res["profile"], "latency_s": res["latency"], "stats": res.get("stats")})


st.session_state.setdefault("messages", [])
st.session_state.setdefault("profile", {})
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"]) if m["role"] == "user" else show(m["result"])

if text := st.chat_input("e.g. I'm a 2nd year BE student, OBC, family income 2.8 lakh, 78% in PUC"):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.write(text)
    with st.chat_message("assistant"):
        with st.spinner("Searching scheme documents..."):
            result = engine(mode, verify).answer(text, st.session_state.profile)
        st.session_state.profile = result["profile"]
        show(result)
    st.session_state.messages.append({"role": "assistant", "result": result})
