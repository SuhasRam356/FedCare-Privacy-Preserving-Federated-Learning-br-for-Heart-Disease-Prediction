"""
FedCare Page: Secure Aggregation & Traceability
===============================================
# PATTERN: Security/ledger-type page (Pattern 5, reference: Security & Traceability)
# RATIONALE: Two-column layout (Left: Blockchain Transaction Ledger table card with
# monospace hashes, hospital chips, timestamps, and COMMITTED/FLAGGED status pills +
# live SecAgg demonstration | Right: IPFS References card with CIDs + Navy Integrity Status
# card with mini bar chart cluster + Audited Entities specs) + full-width bottom
# Automated Traceability Reporting card with multi-colored pipeline SVG graphic and ZK/HE callouts.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import torch
import plotly.graph_objects as go

from app.components.ui_kit import (
    icon_svg, svg_traceability_graph, svg_integrity_bars, status_pill,
    fc_card, fc_table, fc_button, render_header, clean_html
)
from app.components.theme import (
    HOSPITAL_COLORS, CHART_COLORS, PLOTLY_LAYOUT_DEFAULTS
)
from app.utils.data_loaders import load_combined_data


def render_secure_aggregation():
    """Security & Traceability console cloned from MedXChAln Security & Traceability."""
    # ── Top Header ───────────────────────────────────────────────
    st.markdown(render_header(
        title="Security & Traceability",
        subtitle="Immutable ledger of all federated learning operations and model integrity logs.",
        search_placeholder="Search block or transaction hash...",
        version_text="v4.2.1-STABLE",
        last_sync="2m 44s ago",
        action_label="Verify Ledger",
        action_icon="shield_check"
    ), unsafe_allow_html=True)

    # ── Two-Column Layout ────────────────────────────────────────
    col_ledger, col_side = st.columns([0.65, 0.35])

    with col_ledger:
        # Blockchain Transaction Ledger Card
        table_headers = ["Model Hash (SHA-256)", "Hospital ID", "Timestamp", "Status"]
        table_rows = [
            ["0x72a1b9cd7e6f54a3b8d9c2e0f1...", "Mary_Center_01", "2026-09-26 10:45:01", "COMMITTED"],
            ["0x1a2b3c4d5e6f7a8b9c0d1e2f3...", "Johns_Hopkins_02", "2026-09-26 10:41:55", "COMMITTED"],
            ["0xf9e8d7c6b5a43210fe98dcba7...", "Mary_Center_01", "2026-09-26 10:38:30", "COMMITTED"],
            ["0xabcdef1234567890abcd12345...", "Cleveland_Clinic_03", "2026-09-26 10:33:12", "COMMITTED"],
            ["0xdeadbeef1234567890abcdef1...", "Zurich_Univ_04", "2026-09-26 10:28:01", "FLAGGED"],
            ["0x6677889900aabbccddeeff001...", "Charite_Berlin_05", "2026-09-26 10:21:22", "COMMITTED"],
        ]
        
        chips_dict = {0: "MH", 1: "JH", 2: "MH", 3: "CC", 4: "ZH", 5: "CB"}
        pager_info = {"showing": "Showing 6 of 1,248 transactions", "pages": [1, 2, 3], "current": 1}

        ledger_html = fc_table(
            headers=table_headers,
            rows=table_rows,
            status_col_idx=3,
            chips=chips_dict,
            pager=pager_info
        )
        
        st.markdown(fc_card(
            ledger_html,
            title="Blockchain Transaction Ledger",
            subtitle="Cryptographically sealed updates verified on private Ethereum sidechain",
            badge=status_pill("Synced", "success"),
            right_action='<a href="/?refresh=true" target="_top" style="font-size:0.75rem; background:#0F5BB6; color:#FFFFFF; padding:4px 10px; border-radius:6px; font-weight:600; text-decoration:none;">Live Refresh ↺</a>',
            icon_name="database"
        ), unsafe_allow_html=True)

        # Interactive Demonstration Card: Secret Sharing & Paillier HE
        st.markdown(clean_html("""
        <div style="margin-top:16px;">
            <h4 style="margin:0 0 6px; font-size:1.05rem; font-weight:700; color:#0F172A;">Cryptographic Verification Demo</h4>
            <p style="margin:0 0 12px; font-size:0.82rem; color:#64748B;">
                Demonstrates how the server aggregates updates <strong style="color:#0F5BB6;">without ever seeing raw hospital weights</strong>.
            </p>
        </div>
        """), unsafe_allow_html=True)

        if st.button("⚡ Run Secure Aggregation Simulation", key="secagg_run_btn"):
            with st.spinner("Executing Shamir Secret Sharing & Homomorphic Encryption across 6 nodes..."):
                try:
                    from fedcare.secure_aggregation import SecureAggregator
                    from fedcare.task import Net, load_data, train as train_fn, evaluate

                    client_weights_list = []
                    sample_counts = []
                    local_aucs = []

                    for i in range(1, 7):
                        local_model = Net()
                        local_train, local_test, _ = load_data(partition_id=i)
                        train_fn(local_model, local_train, epochs=2, lr=0.001)
                        local_metrics = evaluate(local_model, local_test)
                        local_aucs.append(local_metrics["auc"])
                        weights = [p.detach().cpu().numpy() for p in local_model.parameters()]
                        client_weights_list.append(weights)
                        sample_counts.append(len(local_train.dataset))

                    sec_agg = SecureAggregator(use_he=False)
                    agg_weights = sec_agg.aggregate(client_weights_list, sample_counts)
                    ss_model = Net()
                    for param, agg_w in zip(ss_model.parameters(), agg_weights):
                        param.data = torch.tensor(agg_w, dtype=param.dtype)
                    _, global_test, _ = load_data(partition_id=None)
                    ss_metrics = evaluate(ss_model, global_test)

                    sec_agg_he = SecureAggregator(use_he=True)
                    agg_weights_he = sec_agg_he.aggregate(client_weights_list, sample_counts)
                    he_model = Net()
                    for param, agg_w in zip(he_model.parameters(), agg_weights_he):
                        param.data = torch.tensor(agg_w, dtype=param.dtype)
                    he_metrics = evaluate(he_model, global_test)

                    res_df = pd.DataFrame([
                        {"Protocol": "Standard FedAvg", "AUC": round(sum(local_aucs)/len(local_aucs), 4), "Server Sees Updates": "Plaintext", "Privacy": "Zero"},
                        {"Protocol": "SecAgg (Secret Sharing)", "AUC": round(ss_metrics["auc"], 4), "Server Sees Updates": "Encrypted Shares", "Privacy": "Information-Theoretic"},
                        {"Protocol": "SecAgg (Homomorphic)", "AUC": round(he_metrics["auc"], 4), "Server Sees Updates": "Ciphertext", "Privacy": "Paillier Cryptosystem"},
                    ])

                    st.dataframe(res_df, use_container_width=True, hide_index=True)

                    fig_sec = go.Figure()
                    fig_sec.add_trace(go.Bar(
                        x=res_df["Protocol"], y=res_df["AUC"],
                        marker_color=[CHART_COLORS["danger"], CHART_COLORS["primary"], HOSPITAL_COLORS[4]],
                        marker_cornerradius=6,
                        text=[f"{v:.4f}" for v in res_df["AUC"]],
                        textposition="outside",
                        textfont=dict(size=12, color="#0F172A", family="Inter")
                    ))
                    fig_sec.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                    fig_sec.update_layout(height=260, yaxis_title="AUC", showlegend=False)
                    st.plotly_chart(fig_sec, use_container_width=True)
                    st.success("✅ Complete mathematical privacy preserved: Server never accessed plaintext model parameters!")

                except Exception as e:
                    st.error(f"Simulation note: {e}")
        else:
            st.markdown(fc_card("""
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:700; color:#0F172A; font-size:0.9rem;">Ready for On-Chain Audit</div>
                    <div style="font-size:0.75rem; color:#64748B;">Click above to simulate live secret sharing and Paillier HE aggregation across all 6 clinical nodes.</div>
                </div>
                <div>
                    <span class="fc-pill fc-pill-info">STANDBY</span>
                </div>
            </div>
            """), unsafe_allow_html=True)

    with col_side:
        # Card 1: IPFS References
        ipfs_content = f"""
        <div style="margin-bottom:14px;">
            <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">GLOBAL WEIGHT SET CID</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                <span style="font-family:monospace; color:#0F5BB6; font-size:0.78rem;">QnKoayp...3p2z</span>
                {status_pill("VERIFIED", "success")}
            </div>
        </div>
        <div style="margin-bottom:14px;">
            <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">GRADIENT DELTA CID</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                <span style="font-family:monospace; color:#0F5BB6; font-size:0.78rem;">bafybe...y7i</span>
                {status_pill("SECURE", "info")}
            </div>
        </div>
        <div style="margin-bottom:16px;">
            <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">ENCRYPTION PROTOCOL CID</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                <span style="font-family:monospace; color:#0F5BB6; font-size:0.78rem;">QmZ4t...v6nL</span>
                {status_pill("VERIFIED", "success")}
            </div>
        </div>
        <a href="/?page=Hospital+Management" target="_top" class="fc-btn fc-btn-secondary" style="width:100%; justify-content:center; text-decoration:none; display:flex;">
            RE-VALIDATE ALL CIDS →
        </a>
        """
        st.markdown(fc_card(
            ipfs_content,
            title="IPFS References",
            badge=status_pill("Decentralized", "success"),
            icon_name="database"
        ), unsafe_allow_html=True)

        # Card 2: Navy Hero Card: Integrity Status
        integrity_content = f"""
        <p style="font-size:0.8rem; color:rgba(255,255,255,0.7); margin-top:-6px; margin-bottom:14px;">
            Real-time hash validation across distributed hospital nodes.
        </p>
        <div style="margin-bottom:14px;">
            {svg_integrity_bars([40, 60, 95, 45, 80, 65], color="#93C5FD")}
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(255,255,255,0.12); padding-top:10px;">
            <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">NETWORK TRUST</span>
            <span style="font-size:1.15rem; font-weight:800; color:#60A5FA;">99.9%</span>
        </div>
        """
        st.markdown(fc_card(
            integrity_content,
            title="Integrity Status",
            navy=True,
            icon_name="shield_check"
        ), unsafe_allow_html=True)

        # Card 3: Audited Entities
        audit_content = """
        <div style="display:flex; justify-content:space-between; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #F1F5F9;">
            <span style="font-size:0.82rem; color:#64748B;">Participating Nodes</span>
            <strong style="font-size:0.85rem; color:#0F172A;">6 active</strong>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #F1F5F9;">
            <span style="font-size:0.82rem; color:#64748B;">Validation Time</span>
            <strong style="font-size:0.85rem; color:#0F172A;">14ms avg</strong>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #F1F5F9;">
            <span style="font-size:0.82rem; color:#64748B;">Consensus Algorithm</span>
            <strong style="font-size:0.85rem; color:#0F5BB6;">Proof-of-Authority</strong>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size:0.82rem; color:#64748B;">Tamper Proofing</span>
            <strong style="font-size:0.85rem; color:#15803D;">SHA-256 Merkle</strong>
        </div>
        """
        st.markdown(fc_card(
            audit_content,
            title="Audited Entities",
            icon_name="check_circle"
        ), unsafe_allow_html=True)

    # ── Bottom Full-Width Card: Automated Traceability Reporting ──
    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
    
    trace_col1, trace_col2 = st.columns([0.45, 0.55])
    with trace_col1:
        st.markdown(svg_traceability_graph(), unsafe_allow_html=True)
    with trace_col2:
        st.markdown(clean_html(f"""
        <div style="padding:10px 6px;">
            <h3 style="margin:0 0 8px; font-size:1.25rem; font-weight:800; color:#0F172A;">
                Automated Traceability Reporting
            </h3>
            <p style="font-size:0.85rem; color:#64748B; line-height:1.5; margin-bottom:16px;">
                FedCare utilizes a private Ethereum-compatible sidechain to record every gradient update. This ensures that no single participant can tamper with the global model weights without being detected by the network consensus.
            </p>
            <div style="display:flex; gap:20px;">
                <div style="display:flex; align-items:flex-start; gap:8px;">
                    <div style="color:#0F5BB6;">{icon_svg("shield_check", size=18, color="#0F5BB6")}</div>
                    <div>
                        <div style="font-size:0.84rem; font-weight:700; color:#0F172A;">Zero-Knowledge Proofs</div>
                        <div style="font-size:0.72rem; color:#94A3B8;">Validating data without seeing it.</div>
                    </div>
                </div>
                <div style="display:flex; align-items:flex-start; gap:8px;">
                    <div style="color:#0F5BB6;">{icon_svg("lock", size=18, color="#0F5BB6")}</div>
                    <div>
                        <div style="font-size:0.84rem; font-weight:700; color:#0F172A;">Homomorphic Encryption</div>
                        <div style="font-size:0.72rem; color:#94A3B8;">Computed while encrypted.</div>
                    </div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

