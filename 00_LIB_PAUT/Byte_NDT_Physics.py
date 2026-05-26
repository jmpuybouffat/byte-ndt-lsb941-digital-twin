import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# Paramètres
sector_start = 35
sector_end = 70
skew_start = -30
skew_end = 30
pas = 1.0
focal_point = (-27.16, 0, 0)  # x, y, z

# Calcul du faisceau
angles_sector = np.arange(sector_start, sector_end + pas, pas)
angles_skew = np.arange(skew_start, skew_end + pas, pas)
X, Y = np.meshgrid(angles_sector, angles_skew)
Z = np.sin(np.sqrt(X**2 + Y**2))  # Remplacer par calcul réel

# Affichage Streamlit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Faisceau 3D (sectorial + skew)')
st.pyplot(fig)
plt.savefig('faisceau_3d.png')

# ============================================================
# BYTE NDT TWIN - Validated geometry focal laws
# ============================================================

st.markdown("---")
st.header("🧠 Byte NDT Twin - Validated PA curves / Courbes PA validées")

st.write(
    """
    This module uses the validated PA1/PA2 trajectories and C1/C2 indication curves
    to generate focal laws for the two inspection conditions:
    PA1 → C1 and PA2 → C2.
    """
)

probe_choice = st.selectbox(
    "Probe configuration / Configuration sonde",
    ["1D 32 elements", "1D 16 elements", "2D 8x8"]
)

if st.button("Generate Twin focal laws / Générer les lois focales Twin"):

    try:
        import twin_focal_laws as twin

        PA1 = twin.load_curve("PA1_ORIGINAL_METHOD_VALIDATED.csv")
        PA2 = twin.load_curve("PA2_ORIGINAL_METHOD_VALIDATED.csv")
        C1 = twin.load_curve("C1_INDICATION_VALIDATED.csv")
        C2 = twin.load_curve("C2_INDICATION_VALIDATED.csv")

        if probe_choice == "1D 32 elements":
            laws_A = twin.compute_focal_laws(
                PA1, C1, "A_PA1_to_C1",
                probe_type="1D",
                n_elements_1d=32
            )
            laws_B = twin.compute_focal_laws(
                PA2, C2, "B_PA2_to_C2",
                probe_type="1D",
                n_elements_1d=32
            )

        elif probe_choice == "1D 16 elements":
            laws_A = twin.compute_focal_laws(
                PA1, C1, "A_PA1_to_C1",
                probe_type="1D",
                n_elements_1d=16
            )
            laws_B = twin.compute_focal_laws(
                PA2, C2, "B_PA2_to_C2",
                probe_type="1D",
                n_elements_1d=16
            )

        else:
            laws_A = twin.compute_focal_laws(
                PA1, C1, "A_PA1_to_C1",
                probe_type="2D",
                nx=8,
                ny=8
            )
            laws_B = twin.compute_focal_laws(
                PA2, C2, "B_PA2_to_C2",
                probe_type="2D",
                nx=8,
                ny=8
            )

        st.success("Focal laws generated successfully / Lois focales générées avec succès")

        st.subheader("Condition A: PA1 → C1")
        st.dataframe(laws_A.head(100))

        st.download_button(
            label="Download focal_laws_A.csv",
            data=laws_A.to_csv(index=False).encode("utf-8"),
            file_name="focal_laws_A_PA1_to_C1.csv",
            mime="text/csv"
        )

        st.subheader("Condition B: PA2 → C2")
        st.dataframe(laws_B.head(100))

        st.download_button(
            label="Download focal_laws_B.csv",
            data=laws_B.to_csv(index=False).encode("utf-8"),
            file_name="focal_laws_B_PA2_to_C2.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error while generating focal laws: {e}")
