
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Byte NDT - Digital Twin LSB 941", layout="wide")

st.title("Byte NDT - Digital Twin LSB 941 - INTRADOS")
st.caption("Proof of Concept: EDM indications, PAUT position, focal law visualization and inspection follow-up.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Mode:",
    ["1. Preuve de Concept", "2. Phase 2 : PAUT"],
    index=0
)

st.sidebar.subheader("Position du Sabot")
sabot_pos = st.sidebar.slider("Position", 0, 99, 50)

st.sidebar.subheader("Ajustement CAO (STL)")
scale = st.sidebar.slider("Échelle / Scale", 0.10, 3.00, 1.00, 0.01)

with st.sidebar.expander("Rotations CAO / CAD Rotations", expanded=False):
    rx = st.slider("Rotation X (deg)", -180.0, 180.0, 0.0, 1.0)
    ry = st.slider("Rotation Y (deg)", -180.0, 180.0, 0.0, 1.0)
    rz = st.slider("Rotation Z (deg)", -180.0, 180.0, 0.0, 1.0)

with st.sidebar.expander("Offsets CAO / CAD Offsets", expanded=False):
    ox = st.slider("Offset X", -200.0, 200.0, 0.0, 1.0)
    oy = st.slider("Offset Y", -200.0, 200.0, 0.0, 1.0)
    oz = st.slider("Offset Z", -200.0, 200.0, 0.0, 0.5)

st.sidebar.subheader("Calage Courbes (Maple)")
with st.sidebar.expander("Offsets Courbes / Curves Offsets", expanded=False):
    cx = st.slider("Courbe Offset X", -200.0, 200.0, 0.0, 1.0)
    cy = st.slider("Courbe Offset Y", -200.0, 200.0, 0.0, 1.0)
    cz = st.slider("Courbe Offset Z", -200.0, 200.0, 0.0, 0.5)

st.sidebar.subheader("Lois focales")
angle_deg = st.sidebar.slider("Angle faisceau (deg)", 35, 70, 45)
skew_deg = st.sidebar.slider("Skew (deg)", -20, 20, 0)
focus_mm = st.sidebar.slider("Focus distance (mm)", 20, 200, 90)
active_elements = st.sidebar.slider("Éléments actifs", 8, 32, 32)

# -----------------------------
# Geometry helpers
# -----------------------------
def rot_matrix(rx, ry, rz):
    ax, ay, az = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    return Rz @ Ry @ Rx

def transform_points(P):
    R = rot_matrix(rx, ry, rz)
    return (P * scale) @ R.T + np.array([ox, oy, oz])

# Synthetic intrados blade-root surface: replace later by STL/STEP-derived mesh
u = np.linspace(-90, 130, 60)
v = np.linspace(-80, 120, 45)
U, V = np.meshgrid(u, v)
Z = 250 + 45*np.exp(-((U-20)/115)**2) + 18*np.cos((V+20)/80) - 0.002*(U-20)**2
surface_pts = np.column_stack([U.ravel(), V.ravel(), Z.ravel()])
surface_pts = transform_points(surface_pts)
X = surface_pts[:,0].reshape(U.shape)
Y = surface_pts[:,1].reshape(U.shape)
Zt = surface_pts[:,2].reshape(U.shape)

# EDM indications: 14 points along groove/intrados
n_edm = 14
t = np.linspace(0, 1, n_edm)
edm_x = -70 + 170*t
edm_y = -45 + 18*np.sin(np.pi*t)
edm_z = 292 + 18*np.sin(np.pi*t) + 2*np.cos(2*np.pi*t)
EDM = np.column_stack([edm_x + cx, edm_y + cy, edm_z + cz])
edm_detected = int(1 + (sabot_pos / 99) * (n_edm - 1))

# PAUT/sabot position
idx = max(0, min(n_edm-1, edm_detected-1))
pa = np.array([EDM[idx,0]-10, EDM[idx,1]-35, EDM[idx,2]-35])

# Beam direction from angle + skew
ang = np.deg2rad(angle_deg)
skw = np.deg2rad(skew_deg)
direction = np.array([np.sin(ang)*np.cos(skw), np.sin(skw), np.cos(ang)])
beam_end = pa + focus_mm * direction

# Focal law delays, simple 1D array
pitch = 0.6e-3
c_steel = 5900.0
els = np.arange(active_elements) - (active_elements-1)/2
pos = els * pitch
delays_us = (pos * np.sin(np.deg2rad(angle_deg)) / c_steel) * 1e6
delays_us -= delays_us.min()

df = pd.DataFrame({
    "Element": np.arange(1, active_elements+1),
    "Delay_us": delays_us
})

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([2.2, 1])

with left:
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zt,
        colorscale="Greys",
        opacity=0.35,
        showscale=False,
        name="Intrados CAD surface"
    ))

    fig.add_trace(go.Scatter3d(
        x=EDM[:,0], y=EDM[:,1], z=EDM[:,2],
        mode="markers",
        marker=dict(size=5, color="red"),
        name="EDM indications"
    ))

    fig.add_trace(go.Scatter3d(
        x=[EDM[idx,0]], y=[EDM[idx,1]], z=[EDM[idx,2]],
        mode="markers",
        marker=dict(size=10, color="lime"),
        name=f"EDM detected #{edm_detected}"
    ))

    fig.add_trace(go.Scatter3d(
        x=[pa[0]], y=[pa[1]], z=[pa[2]],
        mode="markers",
        marker=dict(size=9, color="orange", symbol="square"),
        name="PAUT / Sabot"
    ))

    fig.add_trace(go.Scatter3d(
        x=[pa[0], beam_end[0]], y=[pa[1], beam_end[1]], z=[pa[2], beam_end[2]],
        mode="lines",
        line=dict(color="yellow", width=7),
        name="Focal law beam"
    ))

    # Groove/reference curve
    curve_t = np.linspace(0, 1, 100)
    fig.add_trace(go.Scatter3d(
        x=-90 + 220*curve_t + cx,
        y=-55 + 35*np.sin(np.pi*curve_t) + cy,
        z=235 + 80*np.sin(np.pi*curve_t) + cz,
        mode="lines",
        line=dict(width=4, color="lightblue"),
        name="Maple curve / groove reference"
    ))

    fig.update_layout(
        height=680,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(x=0.72, y=0.98)
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.header("🔬 Suivi de l'Inspection")
    st.success(f"🎯 EDM DÉTECTÉ : N° {edm_detected} / {n_edm}")

    st.subheader("Paramètres faisceau")
    st.write(f"Angle : **{angle_deg}°**")
    st.write(f"Skew : **{skew_deg}°**")
    st.write(f"Focus : **{focus_mm} mm**")
    st.write(f"Éléments actifs : **{active_elements}**")

    st.subheader("Loi focale")
    st.dataframe(df, use_container_width=True, height=260)
    st.download_button(
        "Télécharger loi focale CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="byte_ndt_focal_law.csv",
        mime="text/csv"
    )

    st.subheader("Étapes Byte NDT")
    st.write("1. CAD / STL calé")
    st.write("2. Courbes Maple calées")
    st.write("3. EDM positionnées")
    st.write("4. Lois focales calculées")
    st.write("5. Réponse indication à intégrer")
    st.write("6. Analyse + rapport")
