
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

DIR_INPUTS = BASE_DIR / "01_INPUTS_VALIDATED_GEOMETRY"
DIR_FOCAL = BASE_DIR / "02_FOCAL_LAWS"
DIR_IMAGES = BASE_DIR / "06_IMAGES_V62F"
DIR_REPORTS = BASE_DIR / "07_REPORTS_V62F"
DIR_STREAMLIT = BASE_DIR / "08_STREAMLIT"
DIR_VALIDATION = BASE_DIR / "09_VALIDATION_RESTART"
DIR_LIB = BASE_DIR / "00_LIB_PAUT"

st.set_page_config(
    page_title="Byte NDT - LSB941 Digital Twin",
    page_icon="🧠",
    layout="wide",
)

def files(folder, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    out = []
    if folder.exists():
        for p in patterns:
            out += list(folder.glob(p))
    return sorted(set(out))

def read_csv_safe(path):
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Cannot read {path.name}: {e}")
            return None

st.title("Byte NDT - LSB941 Digital Twin Demonstrator")
st.caption("Démonstrateur bilingue - AI-assisted NDT design, focal laws, scan workflow, machine learning preparation")

c1, c2, c3 = st.columns(3)
c1.metric("Current package", "V6.2f + restart validation")
c2.metric("Probe configurations", "1D16 / 1D32 / 2D8x8")
c3.metric("Reference EDM", "11 indications")

st.info("This Streamlit page is a publication demonstrator. It uses the consolidated final folder and does not recalculate the physics.")

tabs = st.tabs([
    "English",
    "Français",
    "Scans V6.2f",
    "Focal laws",
    "Validation",
    "Code ↔ Hardware",
    "Files",
])

with tabs[0]:
    st.header("English overview")
    st.write("Byte NDT is an AI-assisted digital twin workflow for ultrasonic inspection of complex turbine blade roots.")
    st.write("Phase 1: AI-assisted examination design: CAD geometry, EDM references, PA probe definition, focal-law generation and scan workflow.")
    st.write("Phase 2: machine learning: simulated and real acquisition datasets, detection, sizing, -6 dB length measurement and automatic reporting.")
    st.write("The bridge between the code and hardware is made through focal laws, acquisition-board export, ADC/DAC signals and the digital analysis chain.")

with tabs[1]:
    st.header("Vue française")
    st.write("Byte NDT est une chaîne de jumeau numérique assistée par IA pour le contrôle ultrasonore de racines d’aubes complexes.")
    st.write("Phase 1 : étude et design assistés par IA : géométrie CAO, EDM de référence, définition PA, génération des lois focales et workflow de scan.")
    st.write("Phase 2 : machine learning : données simulées et réelles, détection, dimensionnement, mesure à -6 dB et rapport automatique.")
    st.write("Le pont entre le code et le hardware est assuré par les lois focales, l’export carte, les signaux ADC/DAC et la chaîne d’analyse numérique.")

with tabs[2]:
    st.header("CIVA-like scan images / Images de scan type CIVA")
    image_files = files(DIR_IMAGES, ["*.png", "*.jpg", "*.jpeg"])
    if not image_files:
        st.warning("No image found in 06_IMAGES_V62F.")
    else:
        selected = st.multiselect(
            "Images to display",
            image_files,
            default=image_files[:min(6, len(image_files))],
            format_func=lambda p: p.name,
        )
        for img in selected:
            st.subheader(img.name)
            st.image(str(img), use_container_width=True)
    st.warning("Engineering note: V6.2f is the current visual demonstrator basis. Final hardware calibration remains pending.")

with tabs[3]:
    st.header("Focal laws / Lois focales")
    focal_files = files(DIR_FOCAL, "*.csv")
    for f in focal_files:
        st.write(f"📄 {f.name}")
    if focal_files:
        choice = st.selectbox("Preview focal-law file", focal_files, format_func=lambda p: p.name)
        df = read_csv_safe(choice)
        if df is not None:
            st.dataframe(df.head(300), use_container_width=True)

with tabs[4]:
    st.header("Restart validation / Validation de reprise")
    validation_files = files(DIR_VALIDATION, ["*.csv", "*.txt"])
    for f in validation_files:
        st.write(f"📄 {f.name}")
    decision = [f for f in validation_files if "DECISION" in f.name.upper()]
    for f in decision:
        st.subheader(f.name)
        st.code(f.read_text(encoding="utf-8", errors="ignore"))
    csvs = [f for f in validation_files if f.suffix.lower() == ".csv"]
    if csvs:
        choice = st.selectbox("Preview validation CSV", csvs, format_func=lambda p: p.name)
        df = read_csv_safe(choice)
        if df is not None:
            st.dataframe(df.head(300), use_container_width=True)
    st.success("Restart validation confirms that EDM references are inside the CAD groove zone and that realistic paths exist.")

with tabs[5]:
    st.header("Code ↔ Hardware bridge")
    st.write("Model → digital focal laws → acquisition board / DAC → PA probe → steel component → echoes → probe → ADC → digital signal → AI analysis → report")
    st.write("This bridge supports the transition from simulated focal laws to hardware implementation and future machine-learning datasets.")
    st.write("The focal-law files are preserved in the package for future import into acquisition boards.")

with tabs[6]:
    st.header("Final package files")
    folders = [
        ("Inputs", DIR_INPUTS),
        ("Focal laws", DIR_FOCAL),
        ("Images V6.2f", DIR_IMAGES),
        ("Reports V6.2f", DIR_REPORTS),
        ("Streamlit exports", DIR_STREAMLIT),
        ("Restart validation", DIR_VALIDATION),
        ("PAUT library", DIR_LIB),
    ]
    for title, folder in folders:
        with st.expander(title):
            if not folder.exists():
                st.warning(f"Missing folder: {folder}")
            else:
                all_files = sorted([p for p in folder.rglob("*") if p.is_file()])
                if not all_files:
                    st.info("Empty folder.")
                for f in all_files[:250]:
                    st.write(f"`{f.relative_to(BASE_DIR)}`")
                if len(all_files) > 250:
                    st.write(f"... {len(all_files)-250} more files")

st.divider()
st.caption("Byte NDT - LSB941 Digital Twin demonstrator. Status: V6.2f visual basis + restart path validation. Final field calibration and hardware validation pending.")
