import numpy as np
from scipy.optimize import brentq

# --- 1. MOTEUR FERRARI (RAYONS & GÉOMÉTRIE) ---
def ferrari2(cr, DF, DT, DX):
    if abs(cr - 1) < 1e-6: return DX * DT / (DF + DT)
    def f_int(x, cr, df, dp, dpf):
        return x/np.sqrt(x**2 + dp**2) - cr*(dpf-x)/np.sqrt((dpf-x)**2 + df**2)
    try:
        a, b = (0, DX) if DX >= 0 else (DX, 0)
        return brentq(f_int, a, b, args=(cr, DF, DT, DX))
    except: return DX * DT / (DF + DT)

# --- 2. TRANSMISSION ACOUSTIQUE (ZOEPPRITZ) ---
def T_fluid_solid(d1, cp1, d2, cp2, cs2, theta1_deg, wave_type='s'):
    iang = np.radians(theta1_deg)
    sin_i = np.sin(iang)
    sinp, sins = (cp2/cp1)*sin_i, (cs2/cp1)*sin_i
    # Gestion des angles critiques via nombres complexes
    cosp = np.where(sinp < 1, np.sqrt(1-sinp**2), 1j*np.sqrt(sinp**2-1))
    coss = np.where(sins < 1, np.sqrt(1-sins**2), 1j*np.sqrt(sins**2-1))
    cosi = np.sqrt(1-sin_i**2)
    denom = cosp + (d2/d1)*(cp2/cp1)*cosi*(4*((cs2/cp2)**2)*sins*coss*sinp*cosp + 1 - 4*(sins**2)*(coss**2))
    return -(4*cosp*sins*cosi)/denom if wave_type == 's' else (2*cosi*(1-2*sins**2))/denom

# --- 3. FENÊTRAGE (APODISATION) ---
def discrete_windows(M, window_type='rect'):
    m = np.arange(M)
    if window_type == 'Ham': return 0.54 - 0.46 * np.cos(2 * np.pi * m / (M - 1))
    if window_type == 'Han': return (np.sin(np.pi * m / (M - 1)))**2
    return np.ones(M)

# --- 4. MOTEUR SOMMERFELD (CHAMP DE PRESSION / CIVA-LIKE) ---
def calculate_beam_field(L1, L2, sx, sy, freq, mat, Dt0, theta2, phi, DF, ROI):
    """
    Calcule la tache focale sur la gorge LSB 941.
    ROI : grille de points {'xs': array, 'zs': array}
    """
    d1, cp1, d2, cp2, cs2, w_type = mat
    c2 = cs2 if w_type == 's' else cp2
    k2 = 2000 * np.pi * freq / c2
    
    # Création de la grille de calcul dans l'acier
    X, Z = np.meshgrid(ROI['xs'], ROI['zs'])
    V_total = np.zeros_like(X, dtype=complex)
    
    # Coordonnées des centres des éléments (Sonde matricielle)
    ex = (np.arange(L1) - (L1-1)/2) * sx
    ey = (np.arange(L2) - (L2-1)/2) * sy
    
    # Sommation de Rayleigh-Sommerfeld optimisée NumPy
    for i in range(L1):
        for j in range(L2):
            # Distance de l'élément au point de la grille (approchée pour le web)
            r = np.sqrt((X - ex[i])**2 + (Z + Dt0)**2)
            V_total += np.exp(1j * k2 * r) / r
            
    return np.abs(V_total)

# --- 5. ÉCHELLE dB & CONTOURS (MODE EXPERTISE) ---
def apply_db_mask(vmag):
    """Applique les paliers -3, -6, -9, -12 dB"""
    # Éviter log(0)
    vmag = vmag + 1e-10
    v_db = 20 * np.log10(vmag / np.max(vmag))
    
    mask = np.zeros_like(v_db)
    mask[v_db >= -3] = 1.0     # Zone rouge (Pic)
    mask[(v_db < -3) & (v_db >= -6)] = 0.75
    mask[(v_db < -6) & (v_db >= -12)] = 0.40
    mask[v_db < -12] = 0.05    # Fond bleu
    return mask
