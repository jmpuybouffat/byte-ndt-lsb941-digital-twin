import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- LIAISON AVEC VOTRE LIBRAIRIE ---
from Library.byte_ndt_physics import ferrari2, discrete_windows

# --- 1. CONFIGURATION ET TRADUCTION ---
st.set_page_config(page_title="Byte NDT - Design 2D PA", layout="wide")
lang = st.sidebar.selectbox("Langue / Language", ["Français", "English"])

T = {
    "Français": {
        "title": "🛡️ Byte NDT : Design 2D PA (Jumeau Numérique)",
        "params": "Paramètres d'entrée",
        "probe": "Géométrie Sonde",
        "target": "Cible & Balayage",
        "physics": "Physique des Milieux",
        "viz_ray": "🔦 Tracé de Rayons 3D (Ferrari)",
        "viz_delay": "📈 Profil des Retards (µs)",
        "matrix": "📊 Matrice et Export CSV",
        "export_btn": "📥 Télécharger Loi Focale (CSV)",
        "info": "Moteur Physique : Snell-Descartes 3D / Méthode de Ferrari",
        "strat": "🎯 Stratégie : Gorge n°1 (Intrados/Extrados)",
        "depth": "Profondeur cible : -27.5 mm",
        "msg": "Le balayage en Skew (φ) permet de couvrir les rayons de raccordement des deux flancs de la dent."
    },
    "English": {
        "title": "🛡️ Byte NDT: 2D PA Design (Digital Twin)",
        "params": "Input Parameters",
        "probe": "Probe Geometry",
        "target": "Target & Steering",
        "physics": "Media Physics",
        "viz_ray": "🔦 3D Ray Plotting (Ferrari)",
        "viz_delay": "📈 Delay Profile (µs)",
        "matrix": "📊 Matrix and CSV Export",
        "export_btn": "📥 Download Focal Law (CSV)",
        "info": "Physics Engine: 3D Snell's Law / Ferrari Method",
        "strat": "🎯 Strategy: Groove #1 (Intrados/Extrados)",
        "depth": "Target Depth: -27.5 mm",
        "msg": "Skew (φ) steering allows coverage of fillet radii on both tooth faces."
    }
}

# --- 2. MOTEUR DE CALCUL DES RETARDS ---
def calculate_paut_delays(Mx, My, sx, sy, thetat, phi, theta2, DT0, DF, c1, c2):
    cr = c1 / c2
    ex = (np.arange(Mx) - (Mx - 1) / 2) * sx
    ey = (np.arange(My) - (My - 1) / 2) * sy
    Ex, Ey = np.meshgrid(ex, ey)
    ang1_rad = np.arcsin(np.clip(c1 * np.sin(np.radians(theta2)) / c2, -1, 1))
    DQ = DT0 * np.tan(ang1_rad) + DF * np.tan(np.radians(theta2))
    tx, ty = DQ * np.cos(np.radians(phi)), DQ * np.sin(np.radians(phi))
    Db = np.sqrt((tx - Ex * np.cos(np.radians(thetat)))**2 + (ty - Ey)**2)
    De = DT0 + Ex * np.sin(np.radians(thetat))
    xi = np.zeros((My, Mx))
    for i in range(My):
        for j in range(Mx):
            xi[i, j] = ferrari2(cr, DF, De[i, j], Db[i, j])
    t = 1000 * np.sqrt(xi**2 + De**2) / c1 + 1000 * np.sqrt(DF**2 + (Db - xi)**2) / c2
    td = np.max(t) - t
    return td, Ex, Ey, De, tx, ty, xi

# --- 3. INTERFACE UTILISATEUR (SIDEBAR) ---
st.title(T[lang]["title"])
st.sidebar.header(T[lang]["params"])

Mx = st.sidebar.number_input("Mx (Elements X)", 1, 32, 8)
My = st.sidebar.number_input("My (Elements Y)", 1, 32, 8)
sx = st.sidebar.number_input("Pitch X (mm)", 0.1, 2.0, 0.60)
sy = st.sidebar.number_input("Pitch Y (mm)", 0.1, 2.0, 0.60)
thetat = st.sidebar.slider("Angle Sonde/Wedge (°)", 0, 60, 0)

theta2 = st.sidebar.slider("Sector θ2 (°)", 35, 75, 60)
phi = st.sidebar.slider("Skew φ (°)", -30, 30, 0)
f_depth = st.sidebar.number_input("Focus Depth DF (mm)", 10.0, 300.0, 27.5)

c1 = st.sidebar.slider("Vitesse C1 (m/s) - Sabot", 900, 3000, 2340, step=20)
c2 = st.sidebar.slider("Vitesse C2 (m/s) - Pièce", 2000, 4000, 3240, step=20)
DT0 = st.sidebar.number_input("Height DT0 (mm)", value=30.0)

# --- 4. EXECUTION DU CALCUL ---
td, Ex, Ey, De, tx, ty, xi = calculate_paut_delays(Mx, My, sx, sy, thetat, phi, theta2, DT0, f_depth, c1, c2)

col1, col2 = st.columns(2)
with col1:
    st.subheader(T[lang]["viz_ray"])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(My):
        for j in range(Mx):
            x1, y1, z1 = Ex[i,j]*np.cos(np.radians(thetat)), Ey[i,j], De[i,j]
            dist_xy = np.sqrt((tx-x1)**2 + (ty-y1)**2)
            x_int = x1 + xi[i,j]*(tx-x1)/dist_xy if dist_xy > 0 else x1
            y_int = y1 + xi[i,j]*(ty-y1)/dist_xy if dist_xy > 0 else y1
            ax.plot([x1, x_int], [y1, y_int], [z1, 0], color='blue', alpha=0.2)
            ax.plot([x_int, tx], [y_int, ty], [0, -f_depth], color='red', alpha=0.2)
    ax.view_init(elev=20, azim=45)

with col2:
    st.subheader(T[lang]["viz_delay"])
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(range(len(td.flatten())), td.flatten(), color='royalblue')
    st.pyplot(fig_bar)

# --- 5. CIBLE LSB 941 ---
st.divider()
c_img, c_txt = st.columns([1, 2])
with c_img:
    st.image("Assets/LSB941_root.png", caption="Turbine LSB 941")
with c_txt:
    st.subheader(T[lang]["strat"])
    st.write(T[lang]["depth"])
    st.info(T[lang]["msg"])

z_g1 = -27.5
y_off = 12
ax.plot([-50, 50], [y_off, y_off], [z_g1, z_g1], color='gold', linestyle='--', linewidth=3, label="Intrados")
ax.plot([-50, 50], [-y_off, -y_off], [z_g1, z_g1], color='orange', linestyle='--', linewidth=3, label="Extrados")
ax.legend()

with col1:
    st.pyplot(fig)

st.subheader(T[lang]["matrix"])
st.dataframe(pd.DataFrame(td).style.format("{:.3f}"))
st.download_button(T[lang]["export_btn"], pd.DataFrame(td).to_csv(index=False).encode('utf-8'), "loi_focale.csv", "text/csv")
st.caption(f"© 2026 Byte NDT | {T[lang]['info']}")
