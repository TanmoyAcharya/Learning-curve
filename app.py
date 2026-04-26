import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PV & Battery Learning Curve Explorer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.2rem; font-weight: 800; color: #1a1a2e;
    background: linear-gradient(90deg, #f7971e, #ffd200, #4facfe);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
  .kpi-box {
    background: #f0f4ff; border-radius: 12px; padding: 1rem 1.2rem;
    border-left: 4px solid #4facfe; margin-bottom: 0.6rem;
  }
  .kpi-label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-value { font-size: 1.7rem; font-weight: 700; color: #1a1a2e; }
  .kpi-unit  { font-size: 0.82rem; color: #888; }
  .insight-box {
    background: #fffbe6; border-radius: 10px; padding: 0.9rem 1.1rem;
    border-left: 4px solid #ffd200; font-size: 0.9rem; margin-top: 0.5rem;
  }
  section[data-testid="stSidebar"] { background: #1a1a2e; }
  section[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
  section[data-testid="stSidebar"] .stSlider > div > div > div { background: #4facfe; }
</style>
""", unsafe_allow_html=True)

# ── Historical data ─────────────────────────────────────────────────────────────
PV_HIST = pd.DataFrame({
    "year": [1976,1977,1978,1979,1980,1982,1984,1986,1988,1990,1992,1994,
             1996,1998,2000,2002,2004,2006,2007,2008,2009,2010,2011,2012,
             2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    "cum_mw": [0.002,0.004,0.007,0.012,0.021,0.05,0.12,0.28,0.58,1.1,
               2.0,3.5,6.0,10,20,40,80,150,250,400,750,1400,2800,5000,
               9000,14000,22000,35000,50000,70000,95000,130000,170000,
               220000,290000],
    "price_usd": [76,60,46,35,27,16,10,6.0,4.2,3.1,2.2,1.7,1.3,
                  1.1,1.0,0.95,0.90,3.5,3.8,4.2,2.5,1.8,1.1,0.70,
                  0.62,0.55,0.45,0.35,0.28,0.22,0.20,0.17,0.15,0.11,0.08]
})

BAT_HIST = pd.DataFrame({
    "year": [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],
    "cum_gwh": [0.1,0.2,0.4,0.8,1.5,3,6,12,22,40,70,130,210,350,550,800],
    "price_eur": [1050,900,750,620,500,380,280,215,175,145,130,120,110,105,105,110]
})

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Learning Curve Explorer")
    st.markdown("---")
    st.markdown("### Simulation Parameters")

    st.markdown("**Solar PV**")
    lr_pv = st.slider("PV Learning Rate (%)", 10, 50, 23, 1,
                      help="Historical range: 19–40%. Default is long-run average (23%)")
    ref_price_pv = st.number_input("PV Ref. Price today (€/Wp)", 0.01, 5.0, 0.08, 0.01)

    st.markdown("**Lithium Battery**")
    lr_bat = st.slider("Battery Learning Rate (%)", 5, 35, 15, 1,
                       help="Industry estimates: 12–20%. Default 15%.")
    ref_price_bat = st.number_input("Battery Ref. Price today (€/kWh)", 50, 500, 110, 10)

    st.markdown("---")
    cum_start_pv  = st.number_input("Current Cum. PV (GW)", 500, 2000, 600, 50)
    cum_start_bat = st.number_input("Current Cum. Battery (GWh)", 200, 2000, 800, 100)
    growth_pv  = st.slider("Annual PV deployment growth (%/yr)", 5, 40, 20, 1)
    growth_bat = st.slider("Annual Battery growth (%/yr)", 5, 50, 30, 1)
    years_ahead = st.slider("Forecast horizon (years)", 5, 30, 20, 1)

    st.markdown("---")
    st.markdown("**RET Assignment 2026**  \nTanmoy Acharya | LUT University")

# ── Helper functions ─────────────────────────────────────────────────────────────
def learning_curve_forecast(ref_price, lr, cum_now_gw, annual_growth_pct, n_years):
    """Project future prices using learning curve formula."""
    lr_frac = lr / 100
    gr = annual_growth_pct / 100
    cum = [cum_now_gw]
    prices = [ref_price]
    for _ in range(n_years):
        new_cum = cum[-1] * (1 + gr)
        new_price = ref_price * (new_cum / cum_now_gw) ** (np.log(1 - lr_frac) / np.log(2))
        cum.append(new_cum)
        prices.append(new_price)
    return np.array(cum), np.array(prices)


def pv_lcoe(module_price, irr_kwh=1800, pr=0.8, lifetime=25, wacc=0.06,
            bos_fraction=0.6):
    """Rough LCOE estimate: CAPEX = module + BoS, fixed O&M 1%/yr.
    
    Note: module_price is in €/Wp (typical utility-scale module prices).
    The *1000 factor converts capex from €/kWp for proper LCOE calculation.
    Returns LCOE in €ct/kWh.
    """
    # Total installed system cost per kWp (€/kWp)
    # module_price (€/Wp) / (1 - bos_fraction) = total system cost per Wp
    # *1000 converts to per kWp
    capex_per_kwp = module_price / (1 - bos_fraction) * 1000  # €/kWp
    aep_kwh = irr_kwh * pr                                       # kWh/yr per kWp
    crf = wacc * (1 + wacc)**lifetime / ((1 + wacc)**lifetime - 1)
    lcoe = (capex_per_kwp * crf + capex_per_kwp * 0.01) / aep_kwh  # €/kWh
    return round(lcoe * 100, 2)   # €ct/kWh


def bat_lcos(bat_price_eur, cycles_per_year=365, lifetime=20, wacc=0.06,
             eff=0.93, dod=0.85):
    """Rough LCOS estimate for battery storage.
    
    Parameters:
    - bat_price_eur: battery price per kWh of total capacity (€/kWh)
    - cycles_per_year: equivalent full cycles per year
    - lifetime: battery lifetime in years
    - wacc: weighted average cost of capital
    - eff: round-trip efficiency
    - dod: depth of discharge
    
    Returns: LCOS in €ct/kWh of delivered energy.
    Note: The DOD reduces the usable energy per cycle, which is accounted
    for in the annual discharged energy (cycles * dod * eff).
    """
    # Annualized capital cost per kWh installed capacity
    crf = wacc * (1 + wacc)**lifetime / ((1 + wacc)**lifetime - 1)
    annual_cost = bat_price_eur * crf + bat_price_eur * 0.01  # +1% fixed O&M
    # Annual delivered energy per kWh of installed capacity
    # (each cycle discharges 'dod' kWh, with 'eff' round-trip efficiency)
    annual_delivered = cycles_per_year * dod * eff  # kWh delivered per year per kWh installed
    # LCOS = annual cost / annual delivered (€/kWh) -> convert to €ct/kWh
    lcos = (annual_cost / annual_delivered) * 100
    return round(lcos, 2)   # €ct/kWh


# ── Main content ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">☀️🔋 PV & Battery Learning Curve Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive simulation of the experience curves of Solar PV and Lithium-ion Batteries · RET Assignment 2026 · Tanmoy Acharya · LUT University</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Experience Curves", "🔮 Cost Forecast", "⚡ LCOE / LCOS Calculator", "🌍 Sunbelt Opportunity"])

# ══════════════════ TAB 1 – HISTORICAL EXPERIENCE CURVES ══════════════════════
with tab1:
    st.subheader("Historical Experience Curves — log-log view")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Solar PV Module Prices (1976–2023)**")
        log_cum = np.log10(PV_HIST["cum_mw"])
        log_price = np.log10(PV_HIST["price_usd"])
        coeffs = np.polyfit(log_cum, log_price, 1)
        fitted = 10 ** np.polyval(coeffs, log_cum)
        lr_fit = round((1 - 2**coeffs[0]) * 100, 1)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=PV_HIST["cum_mw"], y=PV_HIST["price_usd"],
            mode="markers+text", name="Historic data",
            text=[str(y) if y in [1976,1980,1990,2000,2010,2020,2023] else "" for y in PV_HIST["year"]],
            textposition="top center",
            marker=dict(size=9, color="#f7971e", line=dict(color="white", width=1))
        ))
        fig1.add_trace(go.Scatter(
            x=PV_HIST["cum_mw"], y=fitted,
            mode="lines", name=f"Fit LR={lr_fit}%",
            line=dict(color="#4facfe", width=2, dash="dash")
        ))
        fig1.update_layout(
            xaxis=dict(type="log", title="Cumulative PV installed (MW)"),
            yaxis=dict(type="log", title="Module price (USD 2010/Wp)"),
            height=380, template="plotly_white",
            legend=dict(x=0.6, y=0.95)
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Fitted Learning Rate (1976–2023)</div><div class="kpi-value">{lr_fit}%</div></div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("**Lithium-ion Battery Pack Prices (2010–2025)**")
        log_cum_b = np.log10(BAT_HIST["cum_gwh"])
        log_price_b = np.log10(BAT_HIST["price_eur"])
        coeffs_b = np.polyfit(log_cum_b, log_price_b, 1)
        fitted_b = 10 ** np.polyval(coeffs_b, log_cum_b)
        lr_fit_b = round((1 - 2**coeffs_b[0]) * 100, 1)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=BAT_HIST["cum_gwh"], y=BAT_HIST["price_eur"],
            mode="markers+text", name="Historic data",
            text=[str(y) if y in [2010,2015,2020,2025] else "" for y in BAT_HIST["year"]],
            textposition="top center",
            marker=dict(size=9, color="#43e97b", line=dict(color="white", width=1))
        ))
        fig2.add_trace(go.Scatter(
            x=BAT_HIST["cum_gwh"], y=fitted_b,
            mode="lines", name=f"Fit LR={lr_fit_b}%",
            line=dict(color="#fa709a", width=2, dash="dash")
        ))
        fig2.update_layout(
            xaxis=dict(type="log", title="Cumulative battery installed (GWh)"),
            yaxis=dict(type="log", title="Pack price (€/kWh)"),
            height=380, template="plotly_white",
            legend=dict(x=0.55, y=0.95)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Fitted Learning Rate (2010–2025)</div><div class="kpi-value">{lr_fit_b}%</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="insight-box">💡 <b>Key observation:</b> Both curves show a straight line on a log-log plot — the hallmark of an experience curve. PV has maintained this trend for <b>five decades</b>; batteries are following almost the same path. The slope encodes the learning rate; steeper = faster learning.</div>', unsafe_allow_html=True)

# ══════════════════ TAB 2 – FORECAST ══════════════════════════════════════════
with tab2:
    st.subheader("Cost Forecast — adjust parameters in the sidebar ←")

    cum_pv, price_pv = learning_curve_forecast(ref_price_pv, lr_pv, cum_start_pv, growth_pv, years_ahead)
    cum_bat, price_bat = learning_curve_forecast(ref_price_bat, lr_bat, cum_start_bat, growth_bat, years_ahead)
    future_years = list(range(2025, 2025 + years_ahead + 1))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Solar PV Module Price Forecast**")
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=future_years, y=price_pv,
            mode="lines+markers", name="PV price forecast",
            line=dict(color="#f7971e", width=3),
            fill="tozeroy", fillcolor="rgba(247,151,30,0.1)"
        ))
        fig_pv.add_hline(y=0.05, line_dash="dot", line_color="red",
                         annotation_text="0.05 €/Wp milestone", annotation_position="bottom right")
        fig_pv.update_layout(
            xaxis_title="Year", yaxis_title="€/Wp",
            height=340, template="plotly_white"
        )
        st.plotly_chart(fig_pv, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Price in {2025+years_ahead}</div><div class="kpi-value">{price_pv[-1]:.3f}</div><div class="kpi-unit">€/Wp</div></div>', unsafe_allow_html=True)
        with c2:
            drop = (1 - price_pv[-1]/price_pv[0]) * 100
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Total price drop</div><div class="kpi-value">{drop:.0f}%</div></div>', unsafe_allow_html=True)
        with c3:
            cum_final = cum_pv[-1]
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Cum. PV in {2025+years_ahead}</div><div class="kpi-value">{cum_final/1000:.0f}</div><div class="kpi-unit">TW</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**Lithium-ion Battery Pack Price Forecast**")
        fig_bat = go.Figure()
        fig_bat.add_trace(go.Scatter(
            x=future_years, y=price_bat,
            mode="lines+markers", name="Battery price forecast",
            line=dict(color="#43e97b", width=3),
            fill="tozeroy", fillcolor="rgba(67,233,123,0.1)"
        ))
        fig_bat.add_hline(y=50, line_dash="dot", line_color="red",
                          annotation_text="50 €/kWh milestone", annotation_position="bottom right")
        fig_bat.update_layout(
            xaxis_title="Year", yaxis_title="€/kWh",
            height=340, template="plotly_white"
        )
        st.plotly_chart(fig_bat, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Price in {2025+years_ahead}</div><div class="kpi-value">{price_bat[-1]:.0f}</div><div class="kpi-unit">€/kWh</div></div>', unsafe_allow_html=True)
        with c2:
            drop_b = (1 - price_bat[-1]/price_bat[0]) * 100
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Total price drop</div><div class="kpi-value">{drop_b:.0f}%</div></div>', unsafe_allow_html=True)
        with c3:
            cum_bat_final = cum_bat[-1]
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Cum. Battery in {2025+years_ahead}</div><div class="kpi-value">{cum_bat_final:.0f}</div><div class="kpi-unit">GWh</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Sensitivity Analysis — PV price (€/Wp) in target year")
    lr_range   = [10, 15, 20, 23, 30, 40]
    grow_range = [10, 15, 20, 25, 30]
    rows = {}
    for lr_s in lr_range:
        row = {}
        for gr_s in grow_range:
            _, p = learning_curve_forecast(ref_price_pv, lr_s, cum_start_pv, gr_s, years_ahead)
            row[f"{gr_s}% growth"] = round(p[-1], 3)
        rows[f"LR={lr_s}%"] = row
    sens_df = pd.DataFrame(rows).T
    def _color_sens(val):
        try:
            vmin = sens_df.values.min()
            vmax = sens_df.values.max()
            ratio = (float(val) - vmin) / (vmax - vmin + 1e-9)
            r = int(220 * ratio + 60 * (1 - ratio))
            g = int(60 * ratio + 180 * (1 - ratio))
            b = 60
            return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold"
        except Exception:
            return ""
    st.dataframe(sens_df.style.map(_color_sens), use_container_width=True)
    st.caption(f"PV module price (€/Wp) in {2025 + years_ahead} for different learning rate and deployment growth combinations. Darker = cheaper.")

# ══════════════════ TAB 3 – LCOE / LCOS ══════════════════════════════════════
with tab3:
    st.subheader("⚡ LCOE & LCOS Calculator — the real-world cost metric")
    st.markdown("This calculator estimates how cheap electricity from solar and storage will become as technology costs continue to fall along the learning curve.")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### Solar PV LCOE")
        irr_slider = st.slider("Solar irradiation (kWh/m²/yr)", 800, 2500, 1800, 50,
                               help="Europe~1200, MENA/Sunbelt~1800-2500")
        wacc_pv = st.slider("WACC (%)", 2, 15, 6, 1)
        bos_frac = st.slider("BoS fraction of total system cost", 0.3, 0.8, 0.6, 0.05,
                             help="Balance-of-system (inverter, mounting, wiring, installation)")
        years_pv_sys = st.slider("System lifetime (years)", 20, 35, 25, 1)

        lcoe_now  = pv_lcoe(ref_price_pv, irr_slider, 0.8, years_pv_sys, wacc_pv/100, bos_frac)
        lcoe_fut  = pv_lcoe(price_pv[-1],  irr_slider, 0.8, years_pv_sys, wacc_pv/100, bos_frac)

        fig_lcoe = go.Figure()
        fig_lcoe.add_trace(go.Scatter(
            x=future_years,
            y=[pv_lcoe(p, irr_slider, 0.8, years_pv_sys, wacc_pv/100, bos_frac) for p in price_pv],
            mode="lines", name="LCOE",
            line=dict(color="#f7971e", width=3),
            fill="tozeroy", fillcolor="rgba(247,151,30,0.12)"
        ))
        fig_lcoe.add_hline(y=3, line_dash="dot", line_color="green",
                           annotation_text="3 €ct/kWh", annotation_position="bottom right")
        fig_lcoe.update_layout(xaxis_title="Year", yaxis_title="LCOE (€ct/kWh)",
                               height=300, template="plotly_white")
        st.plotly_chart(fig_lcoe, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">LCOE today</div><div class="kpi-value">{lcoe_now}</div><div class="kpi-unit">€ct/kWh</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">LCOE in {2025+years_ahead}</div><div class="kpi-value">{lcoe_fut}</div><div class="kpi-unit">€ct/kWh</div></div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("#### Battery LCOS (daily cycling)")
        cycles_yr = st.slider("Cycles per year", 200, 365, 365, 5)
        lifetime_bat_yr = st.slider("Battery lifetime (years)", 10, 25, 20, 1)
        wacc_bat = st.slider("WACC (%)", 2, 15, 6, 1)
        dod = st.slider("Depth of discharge", 0.7, 0.95, 0.85, 0.05)

        lcos_now  = bat_lcos(ref_price_bat, cycles_yr, lifetime_bat_yr, wacc_bat/100, 0.93, dod)
        lcos_fut  = bat_lcos(price_bat[-1],  cycles_yr, lifetime_bat_yr, wacc_bat/100, 0.93, dod)

        fig_lcos = go.Figure()
        fig_lcos.add_trace(go.Scatter(
            x=future_years,
            y=[bat_lcos(p, cycles_yr, lifetime_bat_yr, wacc_bat/100, 0.93, dod) for p in price_bat],
            mode="lines", name="LCOS",
            line=dict(color="#43e97b", width=3),
            fill="tozeroy", fillcolor="rgba(67,233,123,0.12)"
        ))
        fig_lcos.add_hline(y=2, line_dash="dot", line_color="green",
                           annotation_text="2 €ct/kWh", annotation_position="bottom right")
        fig_lcos.update_layout(xaxis_title="Year", yaxis_title="LCOS (€ct/kWh)",
                               height=300, template="plotly_white")
        st.plotly_chart(fig_lcos, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">LCOS today</div><div class="kpi-value">{lcos_now}</div><div class="kpi-unit">€ct/kWh</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">LCOS in {2025+years_ahead}</div><div class="kpi-value">{lcos_fut}</div><div class="kpi-unit">€ct/kWh</div></div>', unsafe_allow_html=True)

    combined_now = lcoe_now + lcos_now
    combined_fut = lcoe_fut + lcos_fut
    st.markdown("---")
    st.markdown(f"""
    <div class="insight-box">
    ⚡ <b>Combined PV+Battery cost today:</b> {combined_now:.1f} €ct/kWh — 
    drops to <b>{combined_fut:.1f} €ct/kWh</b> in {2025+years_ahead} under selected parameters.  
    Compare: European grid retail prices are typically 20–35 €ct/kWh. 
    The economics of self-supply are already compelling, and improving fast.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════ TAB 4 – SUNBELT ══════════════════════════════════════════
with tab4:
    st.subheader("🌍 Sunbelt Opportunity — Solar advantage across the globe")

    st.markdown("""
    The Sunbelt region (roughly ±40° latitude) has the world's highest solar irradiance.  
    Combine this with the learning-curve-driven cost decline, and the LCOE in Sunbelt countries 
    is dramatically lower than in northern regions — **creating an energy competitive advantage** 
    that did not exist in the fossil-fuel era.
    """)

    regions = pd.DataFrame({
        "Region": ["Saharan Africa", "Arabian Peninsula", "Northern India/Pakistan",
                   "Australia (outback)", "Chile Atacama", "South Brazil",
                   "Southeast Asia", "Southern Europe (Spain/Italy)",
                   "Northern Europe (Finland/UK)", "Central Europe (Germany)"],
        "Irradiance_kWh": [2400, 2300, 2000, 2200, 2500, 1900, 1800, 1700, 1000, 1100],
        "Type": ["Sunbelt","Sunbelt","Sunbelt","Sunbelt","Sunbelt","Sunbelt",
                 "Sunbelt","Transition","Northern","Northern"]
    })

    regions["LCOE_now"]    = regions["Irradiance_kWh"].apply(
        lambda x: pv_lcoe(ref_price_pv, x, 0.82, 25, 0.06, 0.60))
    regions["LCOE_future"] = regions["Irradiance_kWh"].apply(
        lambda x: pv_lcoe(price_pv[-1], x, 0.82, 25, 0.06, 0.60))

    color_map = {"Sunbelt": "#f7971e", "Transition": "#4facfe", "Northern": "#a0aec0"}

    fig_sun = go.Figure()
    for rtype, color in color_map.items():
        sub = regions[regions["Type"] == rtype]
        fig_sun.add_trace(go.Bar(
            x=sub["Region"], y=sub["LCOE_now"],
            name=f"{rtype} — now",
            marker_color=color, opacity=0.85
        ))
        fig_sun.add_trace(go.Bar(
            x=sub["Region"], y=sub["LCOE_future"],
            name=f"{rtype} — {2025+years_ahead}",
            marker_color=color, opacity=0.45,
            marker_line=dict(color=color, width=2)
        ))

    fig_sun.update_layout(
        barmode="group",
        yaxis_title="LCOE (€ct/kWh)",
        xaxis_tickangle=-30,
        height=440, template="plotly_white",
        legend=dict(x=0.6, y=0.95),
        title=f"PV LCOE by region — today vs {2025+years_ahead}"
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    display_df = (
        regions[["Region", "Type", "Irradiance_kWh", "LCOE_now", "LCOE_future"]]
        .rename(columns={
            "Irradiance_kWh": "Irradiance (kWh/m²/yr)",
            "LCOE_now": "LCOE now (€ct/kWh)",
            "LCOE_future": f"LCOE {2025+years_ahead} (€ct/kWh)"
        })
        .set_index("Region")
    )
    lcoe_cols = ["LCOE now (€ct/kWh)", f"LCOE {2025+years_ahead} (€ct/kWh)"]
    _vmin_l = display_df[lcoe_cols].values.min()
    _vmax_l = display_df[lcoe_cols].values.max()
    def _color_lcoe(val):
        try:
            ratio = (float(val) - _vmin_l) / (_vmax_l - _vmin_l + 1e-9)
            r = int(200 * ratio + 60 * (1 - ratio))
            g = int(60 * ratio + 180 * (1 - ratio))
            return f"background-color: rgb({r},{g},60); color: white; font-weight: bold"
        except Exception:
            return ""
    st.dataframe(display_df.style.map(_color_lcoe, subset=lcoe_cols), use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="insight-box">
    🌍 <b>Key insight:</b> Sunbelt countries already have LCOE values of 
    <b>{regions[regions['Type']=='Sunbelt']['LCOE_now'].min():.1f}–{regions[regions['Type']=='Sunbelt']['LCOE_now'].max():.1f} €ct/kWh</b> today, 
    falling to around <b>{regions[regions['Type']=='Sunbelt']['LCOE_future'].min():.1f}–{regions[regions['Type']=='Sunbelt']['LCOE_future'].max():.1f} €ct/kWh</b> by {2025+years_ahead}.  
    Northern Europe will see {regions[regions['Type']=='Northern']['LCOE_now'].min():.1f}–{regions[regions['Type']=='Northern']['LCOE_now'].max():.1f} €ct/kWh today, 
    dropping to {regions[regions['Type']=='Northern']['LCOE_future'].min():.1f}–{regions[regions['Type']=='Northern']['LCOE_future'].max():.1f} €ct/kWh.  
    The gap is structural and growing — <b>this is the energy geopolitics of the 21st century</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### What this means for Sunbelt countries
    - **Energy sovereignty**: sunlight is domestic — no import dependency, no price volatility
    - **Leapfrogging**: decentralised PV-battery systems can electrify rural areas faster and cheaper than grid extension
    - **Green hydrogen export**: cheap solar → cheap electrolysis → exportable clean fuel (H₂, NH₃, e-methanol)
    - **Industrial competitiveness**: energy-intensive industries (aluminium, steel, chemicals) gain a structural cost advantage
    - **Risk**: without investment in manufacturing capability and institutions, solar wealth may not translate to development — the 'resource curse' lesson applies
    """)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem;">
RET Assignment 2026 · Experience Curves of Solar PV and Lithium Batteries · 
Tanmoy Acharya · LUT University · 
Data sources: ITRPV (2019), Schmidt et al. (2017 — Nature Energy), Breyer (2026), Keiner & Breyer et al. (2026)
</div>
""", unsafe_allow_html=True)
