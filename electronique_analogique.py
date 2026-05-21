__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as sp_signal
from scipy.optimize import fsolve, brentq, minimize, curve_fit
from scipy.integrate import quad
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Cache pour les calculs coûteux
@st.cache_data
def compute_filter_response(R, C, f_arr, ftype):
    omega = 2 * np.pi * f_arr
    tau = R * C
    if ftype == "LP":
        H = 1 / (1 + 1j * omega * tau)
    else:
        H = 1j * omega * tau / (1 + 1j * omega * tau)
    return H

@st.cache_data
def compute_aop_response(gain_DC, GBW, f_arr):
    f_c = GBW / abs(gain_DC) if gain_DC != 0 else GBW
    H = gain_DC / (1 + 1j * f_arr / f_c)
    return H

FORMULES = {
    "Loi d'Ohm":              r"U = R \cdot I,\quad P = UI = \frac{U^2}{R} = RI^2",
    "Diviseur de tension":    r"U_{out} = U_{in}\frac{R_2}{R_1+R_2}",
    "Diviseur de courant":    r"I_{R1} = I\frac{R_2}{R_1+R_2}",
    "Filtre RC passe-bas":    r"H(j\omega)=\frac{1}{1+j\omega RC},\quad f_c=\frac{1}{2\pi RC}",
    "Filtre RL passe-haut":   r"H(j\omega)=\frac{j\omega L/R}{1+j\omega L/R}",
    "RLC série (résonance)":  r"\omega_0=\frac{1}{\sqrt{LC}},\quad Q=\frac{\omega_0 L}{R}=\frac{1}{R}\sqrt{\frac{L}{C}}",
    "Gain AOP":               r"G=-\frac{R_f}{R_1}\text{ (inverseur)},\quad G=1+\frac{R_f}{R_1}\text{ (non-inv.)}",
    "Diode Shockley":         r"I = I_s\!\left(e^{V/(nV_T)}-1\right),\quad V_T=\frac{kT}{q}\approx26\text{mV}",
    "Transistor BJT (Ic)":    r"I_C=\beta I_B=I_s e^{V_{BE}/V_T}",
    "Bande passante AOP":     r"f_{-3dB}=\frac{GBW}{|A_v|}",
    "Bruit thermique":        r"V_n=\sqrt{4kTR\Delta f}\quad\text{(tension de bruit)}",
    "Énergie condensateur":   r"E=\frac{1}{2}CV^2",
    "Énergie inductance":     r"E=\frac{1}{2}LI^2",
    "Puissance réactive":     r"Q=UI\sin\phi,\quad S=UI,\quad P=UI\cos\phi",
    # --- Quadripôles ---
    "Matrice Z":              r"\begin{pmatrix}U_1\\U_2\end{pmatrix}=\begin{pmatrix}Z_{11}&Z_{12}\\Z_{21}&Z_{22}\end{pmatrix}\begin{pmatrix}I_1\\I_2\end{pmatrix}",
    "Matrice ABCD":           r"\begin{pmatrix}U_1\\I_1\end{pmatrix}=\begin{pmatrix}A&B\\C&D\end{pmatrix}\begin{pmatrix}U_2\\-I_2\end{pmatrix},\quad AD-BC=1",
    "Impédance entrée":       r"Z_{in}=\frac{A Z_L+B}{C Z_L+D}",
    "Gain en tension":        r"A_v=\frac{Z_L}{A Z_L+B}",
    # --- Filtres passifs avancés ---
    "Butterworth ordre n":    r"|H(j\omega)|^2=\frac{1}{1+(\omega/\omega_c)^{2n}}",
    "Chebyshev":              r"|H(j\omega)|^2=\frac{1}{1+\varepsilon^2 T_n^2(\omega/\omega_c)}",
    "Filtre LC passe-bas":    r"f_c=\frac{1}{2\pi\sqrt{LC}},\quad Z_0=\sqrt{L/C}",
    # --- Semi-conducteurs ---
    "Concentration intrinsèque": r"n_i^2=N_c N_v \exp\!\left(-\frac{E_g}{k_BT}\right)",
    "Jonction PN (V_bi)":     r"V_{bi}=\frac{k_BT}{q}\ln\!\frac{N_A N_D}{n_i^2}",
    "Courant de diffusion":   r"J_n=qD_n\frac{dn}{dx},\quad D_n=\mu_n\frac{k_BT}{q}",
    "Largeur de déplétion":   r"W=\sqrt{\frac{2\varepsilon(V_{bi}-V)}{q}\!\left(\frac{1}{N_A}+\frac{1}{N_D}\right)}",
    # --- Diodes applications ---
    "Redresseur monoalternance": r"V_{out}=V_m\sin\omega t - V_D\text{ (si }V>V_D)",
    "Régulation Zener":       r"V_{out}=V_Z,\quad I_Z=\frac{V_{in}-V_Z}{R_s}-I_L",
    "Diode varicap":          r"C_j(V)=\frac{C_{j0}}{(1-V/V_{bi})^m}",
    # --- Transistor BJT avancé ---
    "Early effect":           r"I_C=I_S e^{V_{BE}/V_T}\left(1+\frac{V_{CE}}{V_A}\right)",
    "Gain petits signaux gm": r"g_m=\frac{I_C}{V_T},\quad r_\pi=\frac{\beta}{g_m},\quad r_o=\frac{V_A}{I_C}",
    "Gain émetteur commun":   r"A_v=-g_m(R_C\|r_o),\quad R_{in}=r_\pi\|R_B",
    # --- AOP avancé ---
    "Intégrateur AOP":        r"V_{out}=-\frac{1}{RC}\int V_{in}\,dt",
    "Dérivateur AOP":         r"V_{out}=-RC\frac{dV_{in}}{dt}",
    "Filtre actif Sallen-Key":r"H(s)=\frac{\omega_0^2}{s^2+(\omega_0/Q)s+\omega_0^2},\quad\omega_0=\frac{1}{RC}",
    "Comparateur à hystérésis":r"V_{th\pm}=\pm V_{sat}\frac{R_1}{R_1+R_2}",
}

COMPOSANTS_STANDARDS = {
    "Résistances (E24)":  [10,11,12,13,15,16,18,20,22,24,27,30,33,36,39,43,47,51,56,62,68,75,82,91],
    "Condensateurs (pF)": [10,12,15,18,22,27,33,39,47,56,68,82,100,120,150,180,220,270,330,390,470],
    "Inductances (μH)":   [0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470,1000],
}

# Couleurs cyberpunk
COULEURS = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00','#ff4400','#88ccff','#cc88ff']

def style_layout(fig, title="", xlab="", ylab="", h=450):
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(5,0,20,0.8)',
        font=dict(color='#c0d0ff'),
        xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
        yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
        height=h,
    )
    return fig


# ============================================================
# MOTEUR ELECTRONIQUE (ORIGINAL)
# ============================================================
class ElecEngine:
    """Moteur de calcul en électronique analogique."""

    def __init__(self, fs: float = 1e6):
        self.fs = fs

    def Z_R(self, R: float, f: float) -> complex:
        return complex(R, 0)

    def Z_C(self, C: float, f: float) -> complex:
        omega = 2 * np.pi * f
        return complex(0, -1/(omega*C)) if f > 0 else complex(np.inf, 0)

    def Z_L(self, L: float, f: float) -> complex:
        omega = 2 * np.pi * f
        return complex(0, omega*L)

    def Z_serie(self, *impedances) -> complex:
        return sum(impedances)

    def Z_parallele(self, *impedances) -> complex:
        return 1 / sum(1/z for z in impedances if z != 0)

    def filtre_RC(self, R: float, C: float,
                  f_arr: np.ndarray, ftype: str = "LP") -> np.ndarray:
        return compute_filter_response(R, C, f_arr, ftype)

    def filtre_RLC(self, R: float, L: float, C: float,
                   f_arr: np.ndarray, ftype: str = "BP") -> np.ndarray:
        omega = 2 * np.pi * f_arr
        omega0 = 1 / np.sqrt(L * C)
        Q = omega0 * L / R
        if ftype == "BP":
            H = (1j * omega / (omega0 * Q)) / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        elif ftype == "LP":
            H = 1 / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        elif ftype == "HP":
            H = -(omega/omega0)**2 / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        else:
            H = (1 - (omega/omega0)**2) / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        return H

    def fc_RC(self, R: float, C: float) -> float:
        return 1 / (2 * np.pi * R * C)

    def freq_resonance(self, L: float, C: float) -> float:
        return 1 / (2 * np.pi * np.sqrt(L * C))

    def facteur_Q(self, R: float, L: float, C: float) -> float:
        f0 = self.freq_resonance(L, C)
        omega0 = 2 * np.pi * f0
        return omega0 * L / R

    def gain_inverseur(self, Rf: float, R1: float) -> float:
        return -Rf / R1

    def gain_non_inverseur(self, Rf: float, R1: float) -> float:
        return 1 + Rf / R1

    def gain_differentiel(self, R1: float, R2: float,
                           R3: float, R4: float) -> float:
        return (R4 / (R3 + R4)) * (1 + R2/R1) - R2/R1

    def bode_aop(self, gain_DC: float, GBW: float,
                 f_arr: np.ndarray) -> np.ndarray:
        return compute_aop_response(gain_DC, GBW, f_arr)

    def reponse_impulsionnelle_filtre(self, R: float, C: float,
                                       t: np.ndarray) -> np.ndarray:
        tau = R * C
        return 1 - np.exp(-t / tau)

    def courant_diode(self, V, Is: float = 1e-12,
                       n: float = 1.0, T: float = 300) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        VT = 1.3806e-23 * T / 1.6022e-19
        return Is * (np.exp(np.clip(V/(n*VT), -100, 100)) - 1)

    def point_fonctionnement_diode(self, Vcc: float, R: float,
                                    Is: float = 1e-12) -> dict:
        VT = 0.02585
        def equations(V_d):
            I_d = Is * (np.exp(V_d/(VT)) - 1)
            I_R = (Vcc - V_d) / R
            return I_d - I_R
        try:
            V_d = brentq(equations, 0, Vcc * 0.99)
            I_d = Is * (np.exp(V_d/VT) - 1)
        except:
            V_d, I_d = 0.6, (Vcc-0.6)/R
        return {"V_d": V_d, "I_d": I_d, "P_diode": V_d*I_d,
                "P_resistor": I_d**2 * R}

    def point_repos_BJT(self, Vcc: float, Rb: float, Rc: float,
                         beta: float = 100) -> dict:
        VBE = 0.7
        IB = (Vcc - VBE) / Rb if Rb > 0 else 0
        IC = beta * IB
        VCE = Vcc - IC * Rc
        if VCE < 0.2:
            VCE = 0.2
            IC = (Vcc - VCE) / Rc
            IB = IC / beta
        return {"IB_uA": IB*1e6, "IC_mA": IC*1e3, "VCE": VCE,
                "regime": "Saturation" if VCE < 0.3 else
                          "Actif" if VCE < Vcc else "Blocage"}

    def droite_charge(self, Vcc: float, Rc: float,
                       n: int = 200) -> tuple:
        V = np.linspace(0, Vcc, n)
        I = (Vcc - V) / Rc
        return V, I

    def bruit_thermique(self, R: float, T: float, BW: float) -> float:
        k = 1.3806e-23
        return np.sqrt(4 * k * T * R * BW)

    def SNR_db(self, V_signal: float, V_bruit: float) -> float:
        return 20 * np.log10(V_signal / (V_bruit + 1e-15))

    def oscillateur_colpitts(self, L: float,
                              C1: float, C2: float) -> dict:
        C_eq = C1 * C2 / (C1 + C2)
        f0 = 1 / (2 * np.pi * np.sqrt(L * C_eq))
        rapport = C1 / C2
        return {"f0_MHz": f0/1e6, "C_eq_nF": C_eq*1e9,
                "rapport_C1_C2": rapport}


# ============================================================
# NOUVEAU MOTEUR ÉTENDU
# ============================================================
class ElecEngineExt:
    """
    Moteur de calcul étendu — Quadripôles, Filtres passifs avancés,
    Semi-conducteurs, Diodes-Applications, Transistors petits signaux,
    Amplificateurs Opérationnels avancés.
    """

    # ----------------------------------------------------------------
    # QUADRIPÔLES
    # ----------------------------------------------------------------
    def matrice_ABCD_serie(self, Z: complex) -> np.ndarray:
        """Quadripôle série : impédance Z en série."""
        return np.array([[1, Z], [0, 1]], dtype=complex)

    def matrice_ABCD_shunt(self, Y: complex) -> np.ndarray:
        """Quadripôle shunt : admittance Y en parallèle."""
        return np.array([[1, 0], [Y, 1]], dtype=complex)

    def matrice_ABCD_LC_LP(self, L: float, C: float,
                            f: np.ndarray) -> np.ndarray:
        """Filtre LC passe-bas en échelle (cellule L-C)."""
        omega = 2 * np.pi * f
        Z_L = 1j * omega * L
        Y_C = 1j * omega * C
        # Cascade : série L + shunt C
        resultats = []
        for zl, yc in zip(Z_L, Y_C):
            M_L = np.array([[1, zl], [0, 1]])
            M_C = np.array([[1, 0], [yc, 1]])
            resultats.append(M_L @ M_C)
        return np.array(resultats)

    def gain_tension_ABCD(self, M: np.ndarray, Z_L: complex) -> complex:
        """Gain en tension Av = Z_L / (A·Z_L + B) depuis matrice ABCD."""
        A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
        return Z_L / (A * Z_L + B)

    def impedance_entree_ABCD(self, M: np.ndarray, Z_L: complex) -> complex:
        """Impédance d'entrée depuis matrice ABCD."""
        A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
        return (A * Z_L + B) / (C * Z_L + D)

    def cascade_ABCD(self, *matrices) -> np.ndarray:
        """Produit de matrices ABCD (cascade de quadripôles)."""
        M = np.eye(2, dtype=complex)
        for mat in matrices:
            M = M @ mat
        return M

    def convertir_ABCD_vers_Z(self, M: np.ndarray) -> np.ndarray:
        """Conversion ABCD → matrice Z."""
        A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
        det = A  # pour réseau réciproque AD-BC=1
        Z11 = A / C
        Z12 = (A*D - B*C) / C
        Z21 = 1 / C
        Z22 = D / C
        return np.array([[Z11, Z12], [Z21, Z22]])

    def adaptation_impedance_L(self, R_source: float,
                                R_charge: float, f: float) -> dict:
        """
        Réseau d'adaptation en L (matching).
        Calcule L et C pour adapter R_source à R_charge.
        """
        if R_source <= R_charge:
            Q = np.sqrt(R_charge / R_source - 1)
            X_s = Q * R_source
            X_p = R_charge / Q
        else:
            Q = np.sqrt(R_source / R_charge - 1)
            X_s = Q * R_charge
            X_p = R_source / Q
        omega = 2 * np.pi * f
        L_val = X_s / omega
        C_val = 1 / (omega * X_p)
        return {"Q": Q, "L_nH": L_val*1e9, "C_pF": C_val*1e12,
                "X_serie": X_s, "X_parallele": X_p,
                "BW_MHz": f / Q / 1e6}

    def parametre_S11(self, Z_in: complex, Z0: float = 50.0) -> complex:
        """Coefficient de réflexion S11."""
        return (Z_in - Z0) / (Z_in + Z0)

    # ----------------------------------------------------------------
    # FILTRES PASSIFS AVANCÉS
    # ----------------------------------------------------------------
    def butterworth_ordre(self, attenuation_dB: float,
                          rapport_freq: float) -> int:
        """Calcul de l'ordre minimal Butterworth."""
        n = np.log10(10**(attenuation_dB/10) - 1) / (2 * np.log10(rapport_freq))
        return int(np.ceil(n))

    def filtre_butterworth(self, f_arr: np.ndarray, fc: float,
                           n: int, ftype: str = "lowpass") -> np.ndarray:
        """Réponse en fréquence filtre Butterworth d'ordre n."""
        omega = f_arr / fc
        if ftype == "lowpass":
            H2 = 1 / (1 + omega**(2*n))
        else:  # highpass
            H2 = omega**(2*n) / (1 + omega**(2*n))
        return np.sqrt(H2)

    def filtre_chebyshev(self, f_arr: np.ndarray, fc: float,
                         n: int, ripple_dB: float = 1.0) -> np.ndarray:
        """Réponse amplitude Chebyshev type I."""
        eps = np.sqrt(10**(ripple_dB/10) - 1)
        omega = f_arr / fc
        # Polynôme de Chebyshev via cos
        Tn = np.where(omega <= 1,
                      np.cos(n * np.arccos(np.clip(omega, -1, 1))),
                      np.cosh(n * np.arccosh(np.clip(omega, 1, None))))
        return 1 / np.sqrt(1 + eps**2 * Tn**2)

    def filtre_bessel_approx(self, f_arr: np.ndarray, fc: float,
                              n: int = 2) -> np.ndarray:
        """Approximation amplitude filtre de Bessel (phase linéaire)."""
        omega = f_arr / fc
        # Approximation par retard de groupe constant
        if n == 2:
            H = 3 / (3 + 1j*3*omega*2*np.pi + (1j*omega*2*np.pi)**2)
        else:
            H = 1 / (1 + 1j*omega)**n
        return np.abs(H)

    def cellule_LC_resonance(self, L: float, C: float, R: float,
                              f_arr: np.ndarray) -> dict:
        """Réponse complète cellule RLC série."""
        omega = 2 * np.pi * f_arr
        omega0 = 1 / np.sqrt(L * C)
        f0 = omega0 / (2 * np.pi)
        Q = omega0 * L / R
        BW = f0 / Q
        Z = R + 1j*(omega*L - 1/(omega*C))
        return {"Z": Z, "f0_Hz": f0, "Q": Q, "BW_Hz": BW,
                "Z0_ohm": np.sqrt(L/C)}

    def reponse_indicielle_RLC(self, R: float, L: float, C: float,
                                t: np.ndarray) -> np.ndarray:
        """Réponse indicielle RLC série (cas sous-amorti, critique, sur-amorti)."""
        omega0 = 1 / np.sqrt(L * C)
        alpha = R / (2 * L)
        delta = alpha**2 - omega0**2
        if delta < 0:  # sous-amorti
            omega_d = np.sqrt(omega0**2 - alpha**2)
            y = 1 - np.exp(-alpha*t) * (np.cos(omega_d*t) + alpha/omega_d * np.sin(omega_d*t))
        elif delta == 0:  # critique
            y = 1 - np.exp(-alpha*t) * (1 + alpha*t)
        else:  # sur-amorti
            r1 = -alpha + np.sqrt(delta)
            r2 = -alpha - np.sqrt(delta)
            y = 1 + (r2*np.exp(r1*t) - r1*np.exp(r2*t)) / (r1 - r2)
        return np.clip(y, -5, 5)

    def groupe_retard(self, f_arr: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Calcul du retard de groupe τ_g = -dφ/dω."""
        phase = np.unwrap(np.angle(H))
        omega = 2 * np.pi * f_arr
        tau_g = -np.gradient(phase, omega)
        return tau_g

    # ----------------------------------------------------------------
    # SEMI-CONDUCTEURS
    # ----------------------------------------------------------------
    def concentration_intrinseque(self, T: float,
                                   Eg_eV: float = 1.12,
                                   mat: str = "Si") -> float:
        """
        Concentration intrinsèque ni(T) pour Si/Ge/GaAs.
        Retourne ni en cm^-3.
        """
        k_B = 1.3806e-23
        q = 1.6022e-19
        # Constantes matériaux (Nc·Nv)^0.5 à 300K
        params = {
            "Si":   {"ni300": 1.5e10, "Eg": 1.12},
            "Ge":   {"ni300": 2.4e13, "Eg": 0.66},
            "GaAs": {"ni300": 1.8e6,  "Eg": 1.42},
        }
        p = params.get(mat, params["Si"])
        Eg = p["Eg"]
        ni_300 = p["ni300"]
        # Approximation : ni ∝ T^1.5 · exp(-Eg/(2kT))
        ni = ni_300 * (T/300)**1.5 * np.exp(-q*Eg/(2*k_B) * (1/T - 1/300))
        return ni

    def tension_built_in(self, NA: float, ND: float,
                          T: float = 300.0, mat: str = "Si") -> float:
        """Tension de built-in V_bi (V) d'une jonction PN."""
        k_B = 1.3806e-23
        q = 1.6022e-19
        VT = k_B * T / q
        ni = self.concentration_intrinseque(T, mat=mat)
        return VT * np.log(NA * ND / ni**2)

    def largeur_depletion(self, NA: float, ND: float,
                           V_pol: float = 0.0, T: float = 300.0,
                           mat: str = "Si") -> dict:
        """Largeur totale de déplétion W et répartitions xp, xn."""
        q = 1.6022e-19
        eps_r = {"Si": 11.7, "Ge": 16.0, "GaAs": 12.9}
        eps0 = 8.8542e-12
        eps = eps_r.get(mat, 11.7) * eps0
        V_bi = self.tension_built_in(NA, ND, T, mat)
        V_eff = max(V_bi - V_pol, 0.01)
        W = np.sqrt(2 * eps * V_eff / q * (1/NA + 1/ND))
        xp = W * ND / (NA + ND)
        xn = W * NA / (NA + ND)
        return {"W_nm": W*1e9, "xp_nm": xp*1e9, "xn_nm": xn*1e9,
                "V_bi": V_bi, "E_max": q*NA*xp/eps}

    def mobilite_Si(self, T: float = 300.0,
                    N_dopage: float = 1e16) -> dict:
        """
        Mobilité des porteurs dans le Si (modèle Caughey-Thomas simplifié).
        Retourne mu_n et mu_p en cm²/(V·s).
        """
        # Mobilités à 300K pour Si intrinsèque
        mu_n0 = 1400 * (300/T)**2.4
        mu_p0 = 450  * (300/T)**2.2
        # Correction dopage
        N_ref_n = 1e17
        N_ref_p = 2e17
        mu_n = mu_n0 / (1 + (N_dopage/N_ref_n)**0.8)
        mu_p = mu_p0 / (1 + (N_dopage/N_ref_p)**0.8)
        return {"mu_n": mu_n, "mu_p": mu_p,
                "D_n": mu_n * 0.02585 * (T/300),
                "D_p": mu_p * 0.02585 * (T/300)}

    def capacite_jonction(self, NA: float, ND: float,
                           V_arr: np.ndarray, T: float = 300.0,
                           mat: str = "Si") -> np.ndarray:
        """Capacité de jonction Cj(V) en F/m²."""
        q = 1.6022e-19
        eps_r = {"Si": 11.7, "Ge": 16.0, "GaAs": 12.9}
        eps0 = 8.8542e-12
        eps = eps_r.get(mat, 11.7) * eps0
        V_bi = self.tension_built_in(NA, ND, T, mat)
        V_eff = np.maximum(V_bi - V_arr, 0.01)
        Cj0 = np.sqrt(q * eps / 2 * NA*ND/(NA+ND) / V_bi)
        return Cj0 / np.sqrt(V_eff / V_bi)

    def courant_saturation_Is(self, NA: float, ND: float,
                               W_p: float, W_n: float,
                               T: float = 300.0) -> float:
        """Courant de saturation Is de la jonction PN (A/cm²)."""
        q = 1.6022e-19
        ni = self.concentration_intrinseque(T)
        mob = self.mobilite_Si(T, (NA+ND)/2)
        Dn = mob["D_n"] * 1e-4  # cm²/s → m²/s
        Dp = mob["D_p"] * 1e-4
        Is = q * ni**2 * 1e6 * (np.sqrt(Dn)/(NA * W_n) + np.sqrt(Dp)/(ND * W_p))
        return Is

    # ----------------------------------------------------------------
    # DIODES — APPLICATIONS
    # ----------------------------------------------------------------
    def redresseur_monoalternance(self, Vm: float, f_res: float,
                                   R_ch: float, C_liss: float,
                                   V_D: float = 0.7,
                                   n_periodes: int = 10) -> dict:
        """
        Simulation redresseur monoalternance avec lissage par condensateur.
        Retourne les signaux temporels et les métriques.
        """
        T_period = 1 / f_res
        t = np.linspace(0, n_periodes * T_period, 5000)
        v_in = Vm * np.sin(2 * np.pi * f_res * t)
        v_out = np.zeros_like(t)
        dt = t[1] - t[0]
        tau = R_ch * C_liss

        # Simulation avec RC
        v_c = 0.0
        for i, vi in enumerate(v_in):
            if vi - V_D > v_c:
                v_c = vi - V_D
            else:
                v_c *= np.exp(-dt / tau)
            v_out[i] = max(v_c, 0)

        V_moy = np.mean(v_out[len(v_out)//2:])
        V_ondulation = np.max(v_out[len(v_out)//2:]) - np.min(v_out[len(v_out)//2:])
        taux_ondulation = V_ondulation / (V_moy + 1e-9) * 100
        return {"t": t, "v_in": v_in, "v_out": v_out,
                "V_moy": V_moy, "V_ondulation": V_ondulation,
                "taux_ondulation_pct": taux_ondulation,
                "V_pic": Vm - V_D}

    def redresseur_bialternance(self, Vm: float, f_res: float,
                                 R_ch: float, C_liss: float,
                                 V_D: float = 0.7,
                                 n_periodes: int = 10) -> dict:
        """Redresseur bialternance (pont de Graëtz) avec lissage."""
        T_period = 1 / f_res
        t = np.linspace(0, n_periodes * T_period, 5000)
        v_in = Vm * np.sin(2 * np.pi * f_res * t)
        v_rect = np.abs(v_in) - 2 * V_D  # 2 diodes en conduction
        v_out = np.zeros_like(t)
        dt = t[1] - t[0]
        tau = R_ch * C_liss
        v_c = 0.0
        for i, vr in enumerate(v_rect):
            if vr > v_c:
                v_c = vr
            else:
                v_c *= np.exp(-dt / tau)
            v_out[i] = max(v_c, 0)
        V_moy = np.mean(v_out[len(v_out)//2:])
        V_ondulation = np.max(v_out[len(v_out)//2:]) - np.min(v_out[len(v_out)//2:])
        return {"t": t, "v_in": v_in, "v_out": v_out,
                "V_moy": V_moy, "V_ondulation": V_ondulation,
                "taux_ondulation_pct": V_ondulation/(V_moy+1e-9)*100,
                "V_pic": Vm - 2*V_D}

    def regulateur_zener(self, V_in_arr: np.ndarray, V_Z: float,
                          R_s: float, I_L: float,
                          I_Z_min: float = 1e-3) -> dict:
        """
        Régulateur Zener : calcul tension de sortie, courant Zener,
        plage de régulation et dissipation.
        """
        V_out = np.minimum(V_in_arr, V_Z)
        I_Z = (V_in_arr - V_Z) / R_s - I_L
        regime = np.where(I_Z >= I_Z_min, "Régulation", "Hors régulation")
        P_Z = V_Z * np.maximum(I_Z, 0)
        V_in_min = V_Z + (I_L + I_Z_min) * R_s
        return {"V_out": V_out, "I_Z": I_Z, "P_Z": P_Z,
                "V_in_min": V_in_min, "regime": regime}

    def ecrêteur_diode(self, v_in: np.ndarray, V_ref: float,
                        mode: str = "haut") -> np.ndarray:
        """Écrêteur à diode (limiteur de tension)."""
        V_D = 0.7
        if mode == "haut":
            return np.minimum(v_in, V_ref + V_D)
        elif mode == "bas":
            return np.maximum(v_in, V_ref - V_D)
        else:
            return np.clip(v_in, -abs(V_ref) - V_D, abs(V_ref) + V_D)

    def multiplicateur_tension(self, Vm: float, f_res: float,
                                n_etages: int = 3) -> dict:
        """
        Multiplicateur de tension de Cockcroft-Walton.
        V_out théorique = 2·n·Vm.
        """
        V_D = 0.7
        V_out_ideal = 2 * n_etages * Vm
        V_out_reel = 2 * n_etages * (Vm - V_D)
        return {"V_out_ideal": V_out_ideal,
                "V_out_reel": V_out_reel,
                "n_etages": n_etages,
                "rendement_pct": V_out_reel/V_out_ideal*100}

    # ----------------------------------------------------------------
    # TRANSISTOR BJT — PETITS SIGNAUX ET CAS AVANCÉS
    # ----------------------------------------------------------------
    def parametres_petits_signaux_BJT(self, IC_mA: float,
                                       beta: float = 100,
                                       VA: float = 100.0,
                                       T: float = 300.0) -> dict:
        """
        Paramètres hybrides-π du transistor BJT en petits signaux.
        IC en mA, VA tension d'Early en V.
        """
        k_B = 1.3806e-23
        q = 1.6022e-19
        VT = k_B * T / q
        IC = IC_mA * 1e-3
        gm = IC / VT
        r_pi = beta / gm
        r_o = VA / IC
        r_e = VT / IC
        return {"gm_mS": gm*1e3, "r_pi_kOhm": r_pi/1e3,
                "r_o_kOhm": r_o/1e3, "r_e_Ohm": r_e,
                "beta": beta, "VT_mV": VT*1e3,
                "f_T_GHz": gm/(2*np.pi*1e-12)/1e9}  # approximation ft

    def gain_emetteur_commun(self, IC_mA: float, RC: float,
                              RE: float = 0.0, beta: float = 100,
                              VA: float = 100.0,
                              bypass_RE: bool = True) -> dict:
        """
        Gain en tension montage émetteur commun.
        RC, RE en Ohms.
        """
        ps = self.parametres_petits_signaux_BJT(IC_mA, beta, VA)
        gm = ps["gm_mS"] * 1e-3
        r_pi = ps["r_pi_kOhm"] * 1e3
        r_o = ps["r_o_kOhm"] * 1e3
        RC_eff = RC * r_o / (RC + r_o)  # RC ∥ ro
        if bypass_RE or RE == 0:
            Av = -gm * RC_eff
            Rin = r_pi
        else:
            Av = -gm * RC_eff / (1 + gm * RE)
            Rin = r_pi + (1 + beta) * RE
        return {"Av": Av, "Av_dB": 20*np.log10(abs(Av)),
                "Rin_kOhm": Rin/1e3, "Rout_kOhm": r_o/1e3}

    def gain_base_commune(self, IC_mA: float, RC: float,
                           beta: float = 100) -> dict:
        """Gain montage base commune."""
        ps = self.parametres_petits_signaux_BJT(IC_mA, beta)
        gm = ps["gm_mS"] * 1e-3
        alpha = beta / (beta + 1)
        Av = alpha * RC * gm / (1 + gm * ps["r_e_Ohm"])
        return {"Av": Av, "Av_dB": 20*np.log10(abs(Av)),
                "alpha": alpha, "Rin_Ohm": ps["r_e_Ohm"]}

    def polarisation_diviseur(self, Vcc: float, R1: float, R2: float,
                               RC: float, RE: float,
                               beta: float = 100) -> dict:
        """
        Point de repos BJT avec polarisation par diviseur de tension.
        Tous les R en Ohms.
        """
        VBE = 0.7
        VTH = Vcc * R2 / (R1 + R2)
        RTH = R1 * R2 / (R1 + R2)
        IB = (VTH - VBE) / (RTH + (1 + beta) * RE)
        IC = beta * IB
        IE = (1 + beta) * IB
        VCE = Vcc - IC * RC - IE * RE
        if VCE < 0.2:
            VCE = 0.2
        regime = "Saturation" if VCE < 0.3 else "Actif" if VCE < Vcc*0.95 else "Blocage"
        S_stabilite = (1 + beta) / (1 + beta * RE/(RTH + RE))
        return {"IB_uA": IB*1e6, "IC_mA": IC*1e3, "IE_mA": IE*1e3,
                "VCE_V": VCE, "VTH_V": VTH, "RTH_kOhm": RTH/1e3,
                "regime": regime, "S_stabilite": S_stabilite}

    def reponse_freq_BJT(self, IC_mA: float, RC: float, RS: float,
                          Cbe: float = 10e-12, Cbc: float = 2e-12,
                          beta: float = 100,
                          f_arr: np.ndarray = None) -> np.ndarray:
        """
        Réponse fréquentielle BJT (modèle hybride-π avec Cbe, Cbc).
        Retourne H(f) normalisée.
        """
        if f_arr is None:
            f_arr = np.logspace(4, 10, 500)
        ps = self.parametres_petits_signaux_BJT(IC_mA, beta)
        gm = ps["gm_mS"] * 1e-3
        r_pi = ps["r_pi_kOhm"] * 1e3
        omega = 2 * np.pi * f_arr
        # Fréquence de coupure haute β
        f_beta = 1 / (2*np.pi * r_pi * (Cbe + Cbc*(1 + gm*RC)))
        H = gm * RC / np.sqrt(1 + (f_arr/f_beta)**2)
        H_norm = H / (gm * RC)
        return H_norm, f_beta

    # ----------------------------------------------------------------
    # AMPLIFICATEURS OPÉRATIONNELS — AVANCÉS
    # ----------------------------------------------------------------
    def integrateur_AOP(self, R: float, C: float,
                         f_arr: np.ndarray) -> np.ndarray:
        """Réponse fréquentielle intégrateur AOP idéal."""
        omega = 2 * np.pi * f_arr
        return 1 / (1j * omega * R * C)

    def derivateur_AOP(self, R: float, C: float,
                        f_arr: np.ndarray) -> np.ndarray:
        """Réponse fréquentielle dérivateur AOP idéal."""
        omega = 2 * np.pi * f_arr
        return -1j * omega * R * C

    def filtre_actif_sallen_key_LP(self, R1: float, R2: float,
                                    C1: float, C2: float,
                                    f_arr: np.ndarray) -> np.ndarray:
        """
        Filtre actif passe-bas Sallen-Key 2ème ordre.
        H(s) = ω₀² / (s² + ω₀/Q·s + ω₀²)
        """
        omega = 2 * np.pi * f_arr
        omega0 = 1 / np.sqrt(R1 * R2 * C1 * C2)
        Q = np.sqrt(R1 * R2 * C1 * C2) / (C2 * (R1 + R2))
        s = 1j * omega
        H = omega0**2 / (s**2 + omega0/Q * s + omega0**2)
        return H, omega0/(2*np.pi), Q

    def oscillateur_wien(self, R: float, C: float) -> dict:
        """Oscillateur de Wien : fréquence d'oscillation."""
        f0 = 1 / (2 * np.pi * R * C)
        gain_min = 3.0  # Gain AOP minimal pour osciller
        return {"f0_Hz": f0, "f0_kHz": f0/1e3,
                "gain_min": gain_min,
                "Rf_sur_R1": gain_min - 1}

    def comparateur_hysteresis(self, V_sat: float,
                                R1: float, R2: float,
                                V_ref: float = 0.0) -> dict:
        """
        Trigger de Schmitt (comparateur à hystérésis).
        Seuils de basculement haut/bas.
        """
        rapport = R1 / (R1 + R2)
        V_th_haut = V_ref + V_sat * rapport
        V_th_bas  = V_ref - V_sat * rapport
        hysteresis = V_th_haut - V_th_bas
        return {"V_th_haut": V_th_haut, "V_th_bas": V_th_bas,
                "hysteresis": hysteresis, "rapport": rapport}

    def ampli_instrumentation(self, Rg: float, R: float = 10e3,
                               Vd: float = 0.0,
                               Vcm: float = 0.0) -> dict:
        """
        Amplificateur d'instrumentation (3 AOP).
        Gain G = 1 + 2R/Rg, CMRR idéal = ∞.
        """
        G = 1 + 2 * R / Rg
        Vout = G * Vd
        return {"G": G, "G_dB": 20*np.log10(G),
                "Vout": Vout, "CMRR_ideal": "∞ (théorique)"}

    def stabilite_AOP(self, gain_DC: float, GBW: float,
                       C_charge: float, R_out: float = 100.0) -> dict:
        """
        Analyse de stabilité AOP avec charge capacitive.
        Marge de phase approximative.
        """
        f_c = GBW / abs(gain_DC)
        f_pole_sortie = 1 / (2 * np.pi * R_out * C_charge)
        marge_phase = 90 - np.degrees(np.arctan(f_c / f_pole_sortie))
        stable = marge_phase > 45
        return {"f_c_kHz": f_c/1e3, "f_pole_kHz": f_pole_sortie/1e3,
                "marge_phase_deg": marge_phase,
                "stable": "✅ Stable" if stable else "⚠️ Risque instabilité"}

    def generateur_fonction_AOP(self, R: float, C: float,
                                 V_sat: float) -> dict:
        """
        Générateur de signal carré/triangulaire (intégrateur + comparateur).
        """
        f0 = 1 / (4 * R * C)
        V_triangle = V_sat
        return {"f0_Hz": f0, "V_carre_V": V_sat,
                "V_triangle_V": V_triangle,
                "periode_ms": 1/f0*1e3}


# ============================================================
# PAGE PRINCIPALE (ORIGINALE — CONSERVÉE INTÉGRALEMENT)
# ============================================================
def electronique_analogique_page():
    st.markdown("## 🔌 Électronique Analogique")
    st.markdown("*Choisissez le mode principal de l’électronique à explorer*")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Électronique", "Électronique Analogique Avancée"],
        index=0,
        horizontal=True,
        key="electronique_mode"
    )

    if mode == "Électronique Analogique Avancée":
        st.info("Accès au module avancé d'électronique analogique.")
        electronique_analogique_enrichie_page()
        return

    st.markdown("### ⚡ Mode Électronique principal")
    st.markdown("Explorez les fonctions de base et les outils de calcul rapides en électronique analogique.")

    engine = ElecEngine()

    section = st.selectbox(
        "Section",
        [
            "📡 Filtres RC/RLC",
            "🔧 AOP & Amplificateurs",
            "💡 Diodes",
            "🔦 Transistors BJT",
            "📻 Oscillateurs & Bruit",
            "📖 Théorie",
        ],
        key="section_electronique_analogique"
    )


    # ============================================================
    # TAB 1 : FILTRES
    # ============================================================
    if section == "📡 Filtres RC/RLC":
        st.markdown("### 📡 Filtres analogiques")
        col1, col2 = st.columns([1, 2])

        with col1:
            type_filtre = st.selectbox("Topologie", ["RC", "RLC"])
            if type_filtre == "RC":
                R_f = st.slider("R (Ω)", 1.0, 1e6, 1000.0, 1.0)
                C_f = st.slider("C (nF)", 0.1, 10000.0, 100.0, 0.1) * 1e-9
                fc = engine.fc_RC(R_f, C_f)
                st.metric("f_c (Hz)", f"{fc:.2f}")
                st.metric("τ = RC (ms)", f"{R_f*C_f*1e3:.4f}")
                ftype_RC = st.radio("Type", ["LP", "HP"], horizontal=True)
            else:
                R_f = st.slider("R (Ω)", 1.0, 1000.0, 10.0, 0.1)
                L_f = st.slider("L (μH)", 0.1, 10000.0, 100.0, 0.1) * 1e-6
                C_f = st.slider("C (nF)", 0.1, 10000.0, 100.0, 0.1) * 1e-9
                f0 = engine.freq_resonance(L_f, C_f)
                Q = engine.facteur_Q(R_f, L_f, C_f)
                st.metric("f₀ (kHz)", f"{f0/1e3:.3f}")
                st.metric("Q", f"{Q:.3f}")
                st.metric("BW (Hz)", f"{f0/Q:.1f}")
                ftype_RLC = st.radio("Type", ["LP","HP","BP","BR"], horizontal=True)

        with col2:
            f_arr = np.logspace(1, 7, 500)
            if type_filtre == "RC":
                H = engine.filtre_RC(R_f, C_f, f_arr, ftype_RC)
            else:
                H = engine.filtre_RLC(R_f, L_f, C_f, f_arr, ftype_RLC)

            mag_dB = 20 * np.log10(np.abs(H) + 1e-12)
            phase_deg = np.angle(H, deg=True)

            fig_filt = make_subplots(rows=2, cols=1,
                subplot_titles=["Gain (dB)", "Phase (°)"])
            fig_filt.add_trace(go.Scatter(x=f_arr, y=mag_dB, mode='lines',
                name='|H(f)|', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
            fig_filt.add_trace(go.Scatter(x=f_arr, y=phase_deg, mode='lines',
                name='∠H(f)', line=dict(color='#7700ff', width=2.5)), row=2, col=1)
            fig_filt.add_hline(y=-3, line_color='#ffcc00', line_dash='dash',
                               annotation_text="-3dB", row=1, col=1)
            fig_filt.add_hline(y=-45, line_color='#ffcc00', line_dash='dash',
                               annotation_text="-45°", row=2, col=1)

            fc_marker = engine.fc_RC(R_f, C_f) if type_filtre == "RC" \
                        else engine.freq_resonance(L_f, C_f)
            fig_filt.add_vline(x=fc_marker, line_color='#ff00cc', line_dash='dot',
                              annotation_text=f"f_c={fc_marker:.1f}Hz")

            fig_filt.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                                   color='#c0d0ff', title_text="f (Hz)")
            fig_filt.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_filt.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=480,
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                showlegend=False
            )
            st.plotly_chart(fig_filt, use_container_width=True)

            st.markdown("#### ⏱️ Réponse indicielle")
            t_ind = np.linspace(0, 5*R_f*C_f if type_filtre=="RC"
                                else 5/fc_marker*2*np.pi, 300)
            y_ind = engine.reponse_impulsionnelle_filtre(R_f, C_f, t_ind)
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(x=t_ind*1e3, y=y_ind, mode='lines',
                line=dict(color='#00ccff', width=2.5), name='Vout/Vin'))
            fig_ind.add_hline(y=0.632, line_color='#ffcc00', line_dash='dash',
                              annotation_text="63.2% (t=τ)")
            fig_ind.update_layout(
                title="Réponse indicielle RC", xaxis_title="t (ms)",
                yaxis_title="Vout/Vin",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=240, showlegend=False
            )
            st.plotly_chart(fig_ind, use_container_width=True)

    # ============================================================
    # TAB 2 : AOP
    # ============================================================
    elif section == "🔧 AOP & Amplificateurs":
        st.markdown("### 🔧 Amplificateurs Opérationnels")
        col1, col2 = st.columns([1, 2])

        with col1:
            config_aop = st.selectbox("Configuration", [
                "Inverseur", "Non-inverseur", "Différentiel",
                "Intégrateur", "Dérivateur", "Comparateur"
            ])
            R1_aop = st.slider("R₁ (kΩ)", 0.1, 1000.0, 10.0, 0.1) * 1e3
            Rf_aop = st.slider("Rf (kΩ)", 0.1, 1000.0, 100.0, 0.1) * 1e3
            GBW = st.slider("GBW (MHz)", 0.1, 100.0, 1.0, 0.1) * 1e6
            Vin_aop = st.slider("Vin (V)", -10.0, 10.0, 1.0, 0.1)
            Vsat = st.slider("Saturation ±Vsat (V)", 1.0, 15.0, 12.0, 0.5)

            if config_aop == "Inverseur":
                gain = engine.gain_inverseur(Rf_aop, R1_aop)
            elif config_aop == "Non-inverseur":
                gain = engine.gain_non_inverseur(Rf_aop, R1_aop)
            elif config_aop == "Différentiel":
                R3 = st.slider("R₃ (kΩ)", 0.1, 1000.0, 10.0, 0.1) * 1e3
                R4 = st.slider("R₄ (kΩ)", 0.1, 1000.0, 100.0, 0.1) * 1e3
                gain = engine.gain_differentiel(R1_aop, Rf_aop, R3, R4)
            else:
                gain = engine.gain_inverseur(Rf_aop, R1_aop)

            Vout_ideal = np.clip(gain * Vin_aop, -Vsat, Vsat)
            f_bande = GBW / max(abs(gain), 1)

            st.metric("Gain A_v", f"{gain:.3f}")
            st.metric("Gain (dB)", f"{20*np.log10(abs(gain)):.2f}" if gain != 0 else "0")
            st.metric("Vout (V)", f"{Vout_ideal:.3f}")
            st.metric("f_-3dB (kHz)", f"{f_bande/1e3:.2f}")
            st.metric("Saturation", "✅ NON" if abs(Vout_ideal) < Vsat else "⚠️ OUI")

        with col2:
            f_bode = np.logspace(1, 7, 400)
            H_aop = engine.bode_aop(gain, GBW, f_bode)
            mag_aop = 20 * np.log10(np.abs(H_aop) + 1e-12)
            phase_aop = np.angle(H_aop, deg=True)

            fig_aop = make_subplots(rows=2, cols=1,
                subplot_titles=["Gain (dB)", "Phase (°)"])
            fig_aop.add_trace(go.Scatter(x=f_bode, y=mag_aop, mode='lines',
                name='|A(f)|', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
            fig_aop.add_trace(go.Scatter(x=f_bode, y=phase_aop, mode='lines',
                name='∠A(f)', line=dict(color='#7700ff', width=2.5)), row=2, col=1)
            fig_aop.add_hline(y=20*np.log10(abs(gain))-3,
                              line_color='#ffcc00', line_dash='dash',
                              annotation_text="-3dB", row=1, col=1)
            fig_aop.add_vline(x=f_bande, line_color='#ff00cc', line_dash='dot',
                              annotation_text=f"f_c={f_bande:.0f}Hz")

            fig_aop.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                                  color='#c0d0ff', title_text="f (Hz)")
            fig_aop.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_aop.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=450,
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                showlegend=False
            )
            st.plotly_chart(fig_aop, use_container_width=True)

            Vin_sweep = np.linspace(-Vsat, Vsat, 200)
            Vout_sweep = np.clip(gain * Vin_sweep, -Vsat, Vsat)
            fig_vt = go.Figure()
            fig_vt.add_trace(go.Scatter(x=Vin_sweep, y=Vout_sweep, mode='lines',
                line=dict(color='#00ccff', width=3), name='Vout(Vin)'))
            fig_vt.add_hline(y=Vsat, line_color='#ff4444', line_dash='dot',
                             annotation_text=f"+Vsat={Vsat}V")
            fig_vt.add_hline(y=-Vsat, line_color='#ff4444', line_dash='dot',
                             annotation_text=f"-Vsat={-Vsat}V")
            fig_vt.update_layout(
                title="Transfert Vout = f(Vin)",
                xaxis_title="Vin (V)", yaxis_title="Vout (V)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=240, showlegend=False
            )
            st.plotly_chart(fig_vt, use_container_width=True)

    # ============================================================
    # TAB 3 : DIODES
    # ============================================================
    elif section == "💡 Diodes":
        st.markdown("### 💡 Diodes — Modèle de Shockley")
        col1, col2 = st.columns([1, 2])

        with col1:
            Is = st.slider("Is (pA)", 0.001, 100.0, 1.0, 0.01) * 1e-12
            n_id = st.slider("Idéalité n", 1.0, 2.0, 1.0, 0.01)
            T_diode = st.slider("T (K)", 250.0, 400.0, 300.0, 1.0)
            Vcc_d = st.slider("Vcc (V)", 0.5, 20.0, 5.0, 0.1)
            R_diode = st.slider("R série (Ω)", 1.0, 10000.0, 1000.0, 1.0)

            VT = 1.3806e-23 * T_diode / 1.6022e-19
            st.metric("VT (mV)", f"{VT*1000:.2f}")

            pf = engine.point_fonctionnement_diode(Vcc_d, R_diode, Is)
            st.metric("Vd (V)", f"{pf['V_d']:.4f}")
            st.metric("Id (mA)", f"{pf['I_d']*1000:.4f}")
            st.metric("P_diode (mW)", f"{pf['P_diode']*1000:.4f}")
            st.metric("P_résistance (mW)", f"{pf['P_resistor']*1000:.4f}")

        with col2:
            V_range = np.linspace(-1.0, 1.0, 1000)
            I_diode = engine.courant_diode(V_range, Is, n_id, T_diode)
            I_circuit = (Vcc_d - V_range) / R_diode

            fig_diode = go.Figure()
            fig_diode.add_trace(go.Scatter(x=V_range, y=I_diode*1000, mode='lines',
                name='I_diode (mA)', line=dict(color='#00ccff', width=3)))
            fig_diode.add_trace(go.Scatter(x=V_range, y=I_circuit*1000, mode='lines',
                name='Droite de charge', line=dict(color='#ffcc00', width=2.5,
                dash='dash')))
            fig_diode.add_trace(go.Scatter(x=[pf['V_d']], y=[pf['I_d']*1000],
                mode='markers', name='Point Q',
                marker=dict(color='#ff00cc', size=14, symbol='star')))

            fig_diode.update_layout(
                title=f"Caractéristique I-V diode + droite de charge",
                xaxis_title="V (V)", yaxis_title="I (mA)",
                yaxis=dict(range=[-1, Vcc_d/R_diode*1000*1.1]),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=430,
            )
            st.plotly_chart(fig_diode, use_container_width=True)

            st.markdown("#### 🌡️ Effet de la température")
            T_arr = [250, 300, 350, 400]
            V_range2 = np.linspace(0, 0.9, 500)
            fig_T = go.Figure()
            colors_T = ['#00ccff','#7700ff','#ff00cc','#ffcc00']
            for Ti, ci in zip(T_arr, colors_T):
                Ii = engine.courant_diode(V_range2, Is, n_id, Ti)
                fig_T.add_trace(go.Scatter(x=V_range2, y=np.log10(Ii+1e-15),
                    mode='lines', name=f'T={Ti}K',
                    line=dict(color=ci, width=2)))
            fig_T.update_layout(
                title="log₁₀(I) vs V pour différentes T",
                xaxis_title="V (V)", yaxis_title="log₁₀(I) (A)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320,
            )
            st.plotly_chart(fig_T, use_container_width=True)

    # ============================================================
    # TAB 4 : TRANSISTORS BJT
    # ============================================================
    elif section == "🔦 Transistors BJT":
        st.markdown("### 🔦 Transistor BJT")
        col1, col2 = st.columns([1, 2])

        with col1:
            Vcc_bjt = st.slider("Vcc (V)", 1.0, 30.0, 12.0, 0.5)
            Rb_bjt = st.slider("Rb (kΩ)", 1.0, 10000.0, 470.0, 1.0) * 1e3
            Rc_bjt = st.slider("Rc (kΩ)", 0.1, 10.0, 1.0, 0.1) * 1e3
            beta = st.slider("β (hFE)", 10, 500, 100)

            pq = engine.point_repos_BJT(Vcc_bjt, Rb_bjt, Rc_bjt, beta)
            st.metric("IB (μA)", f"{pq['IB_uA']:.3f}")
            st.metric("IC (mA)", f"{pq['IC_mA']:.3f}")
            st.metric("VCE (V)", f"{pq['VCE']:.3f}")
            st.metric("Régime", pq["regime"])
            st.metric("Gain Av ≈", f"{-beta*Rc_bjt/((1+beta)*26e-3):.1f}")

        with col2:
            VCE_arr = np.linspace(0, Vcc_bjt, 300)
            IB_list = np.linspace(1, 100, 6) * 1e-6
            V_dc, I_dc = engine.droite_charge(Vcc_bjt, Rc_bjt)

            fig_bjt = go.Figure()
            colors_b = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00','#ff4400']

            for i, IB in enumerate(IB_list):
                IC = beta * IB * np.ones_like(VCE_arr)
                IC_sat = np.minimum(IC, (Vcc_bjt - VCE_arr) / Rc_bjt)
                IC_sat = np.where(VCE_arr < 0.2, VCE_arr/0.2*beta*IB, IC_sat)
                fig_bjt.add_trace(go.Scatter(
                    x=VCE_arr, y=IC_sat*1000, mode='lines',
                    name=f'IB={IB*1e6:.0f}μA',
                    line=dict(color=colors_b[i], width=2)
                ))

            fig_bjt.add_trace(go.Scatter(x=V_dc, y=I_dc*1000, mode='lines',
                name='Droite de charge',
                line=dict(color='#ffffff', width=2.5, dash='dash')))
            fig_bjt.add_trace(go.Scatter(
                x=[pq['VCE']], y=[pq['IC_mA']], mode='markers',
                name='Point Q',
                marker=dict(color='#ff00cc', size=14, symbol='star')
            ))
            fig_bjt.update_layout(
                title="Famille de caractéristiques BJT",
                xaxis_title="VCE (V)", yaxis_title="IC (mA)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=430,
            )
            st.plotly_chart(fig_bjt, use_container_width=True)

    # ============================================================
    # TAB 5 : OSCILLATEURS & BRUIT
    # ============================================================
    elif section == "📻 Oscillateurs & Bruit":
        st.markdown("### 📻 Oscillateurs & Bruit thermique")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 🔁 Oscillateur Colpitts")
            L_col = st.slider("L (μH)", 0.1, 1000.0, 10.0, 0.1) * 1e-6
            C1_col = st.slider("C₁ (nF)", 0.1, 10000.0, 100.0, 0.1) * 1e-9
            C2_col = st.slider("C₂ (nF)", 0.1, 10000.0, 47.0, 0.1) * 1e-9

            col_res = engine.oscillateur_colpitts(L_col, C1_col, C2_col)
            st.metric("f₀ (MHz)", f"{col_res['f0_MHz']:.4f}")
            st.metric("C_eq (nF)", f"{col_res['C_eq_nF']:.4f}")
            st.metric("Rapport C₁/C₂", f"{col_res['rapport_C1_C2']:.3f}")

            st.markdown("#### 🔊 Bruit thermique")
            R_bruit = st.slider("R (kΩ)", 0.01, 1000.0, 10.0, 0.1) * 1e3
            T_bruit = st.slider("T (K)", 77.0, 500.0, 300.0, 1.0)
            BW_bruit = st.slider("BW (kHz)", 0.1, 10000.0, 100.0, 1.0) * 1e3

            Vn = engine.bruit_thermique(R_bruit, T_bruit, BW_bruit)
            V_sig = st.slider("V_signal (μV)", 0.1, 1000.0, 100.0, 0.1) * 1e-6
            SNR = engine.SNR_db(V_sig, Vn)

            st.metric("Vn_rms (nV/√Hz)", f"{engine.bruit_thermique(R_bruit, T_bruit, 1)*1e9:.3f}")
            st.metric("Vn total (μV)", f"{Vn*1e6:.3f}")
            st.metric("SNR (dB)", f"{SNR:.2f}")

        with col2:
            f_bruit = np.logspace(1, 8, 500)
            Sn_V = 4 * 1.3806e-23 * T_bruit * R_bruit * np.ones_like(f_bruit)

            fig_bruit = go.Figure()
            fig_bruit.add_trace(go.Scatter(
                x=f_bruit, y=np.sqrt(Sn_V)*1e9, mode='lines',
                name=f'DSP bruit ({R_bruit/1e3:.0f}kΩ)',
                line=dict(color='#00ccff', width=2.5),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'
            ))
            fig_bruit.add_vline(x=BW_bruit, line_color='#ffcc00', line_dash='dash',
                                annotation_text=f"BW={BW_bruit/1e3:.0f}kHz")
            fig_bruit.update_layout(
                title="Densité spectrale de bruit thermique",
                xaxis_title="f (Hz)", yaxis_title="Vn (nV/√Hz)",
                xaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=360,
            )
            st.plotly_chart(fig_bruit, use_container_width=True)

            t_osc = np.linspace(0, 5/col_res["f0_MHz"]/1e6, 2000)
            f0 = col_res["f0_MHz"] * 1e6
            y_osc = np.cos(2*np.pi*f0*t_osc)
            y_noisy = y_osc + np.random.normal(0, 0.05, len(t_osc))
            fig_osc = go.Figure()
            fig_osc.add_trace(go.Scatter(x=t_osc*1e6, y=y_osc, mode='lines',
                name='Signal pur', line=dict(color='#00ccff', width=2)))
            fig_osc.add_trace(go.Scatter(x=t_osc*1e6, y=y_noisy, mode='lines',
                name='Avec bruit', line=dict(color='rgba(119,0,255,0.5)', width=1)))
            fig_osc.update_layout(
                title=f"Oscillateur Colpitts — f₀={col_res['f0_MHz']:.4f} MHz",
                xaxis_title="t (μs)", yaxis_title="V (u.a.)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300,
            )
            st.plotly_chart(fig_osc, use_container_width=True)

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    elif section == "📖 Théorie":
        st.markdown("### 📖 Formulaire électronique analogique")
        cols = st.columns(2)
        col_idx = 0
        for nom, formule in FORMULES.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1
        st.markdown("---")
        st.markdown("### 🔬 Composants standards")
        df_comp = pd.DataFrame([
            {"Type": "Résistances (E24)", "Valeurs": "10, 11, 12, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91"},
            {"Type": "Condensateurs (pF)", "Valeurs": "10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100, 120, 150, 180, 220, 270, 330, 390, 470"},
            {"Type": "Inductances (μH)", "Valeurs": "0.1, 0.22, 0.47, 1, 2.2, 4.7, 10, 22, 47, 100, 220, 470, 1000"},
        ])
        st.dataframe(df_comp, use_container_width=True)
        st.markdown("---")
        for r in ["Razavi — *Design of Analog CMOS Integrated Circuits* (McGraw-Hill, 2017)",
                  "Sedra & Smith — *Microelectronic Circuits* (Oxford, 2020)",
                  "Horowitz & Hill — *The Art of Electronics* (Cambridge, 2015)"]:
            st.markdown(f"- {r}")


# ============================================================
# NOUVEAUX ONGLETS — CHAPITRES ENRICHIS
# ============================================================
def electronique_analogique_enrichie_page():
    """
    Page enrichie avec les 6 chapitres :
    Quadripôles | Filtres passifs | Semi-conducteurs |
    Diodes-Applications | Transistors | AOP avancés
    """
    st.markdown("## 🔌 Électronique Analogique — Options Avancées")
    st.markdown("*Quadripôles · Filtres passifs · Semi-conducteurs · Diodes · Transistors · AOP*")
    st.markdown("---")

    eng = ElecEngineExt()
    zeng = ElecEngine()
    couleurs = COULEURS

    # Sélecteur sidebar
    with st.sidebar.expander("📚 Options Électronique", expanded=True):
        chapitre = st.radio("Option", [
            "🔲 Quadripôles",
            "📡 Filtres passifs avancés",
            "🔬 Semi-conducteurs",
            "💡 Diodes — Applications",
            "🔦 Transistors avancés",
            "⚙️ AOP avancés",
        ])

    # ================================================================
    # CHAPITRE 1 : QUADRIPÔLES
    # ================================================================
    if chapitre == "🔲 Quadripôles":
        st.markdown("## 🔲 Section 1 — Quadripôles")
        st.markdown("Représentation matricielle des circuits à 2 ports (ABCD, Z, Y, S).")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Simulation ABCD",
            "📊 Adaptation d'impédance",
            "🔬 Paramètres S",
            "📖 Théorie & Formules"
        ])

        # ---- sub1 : Simulation ABCD ----
        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Paramètres ABCD")
                topo_quad = st.selectbox("Topologie", [
                    "Série Z", "Shunt Y", "Filtre LC passe-bas",
                    "Cascade L + C", "Ligne de transmission"
                ])
                R_quad = st.slider("R / Z série (Ω)", 1.0, 1000.0, 50.0, 1.0)
                L_quad = st.slider("L (μH)", 0.01, 1000.0, 10.0, 0.1) * 1e-6
                C_quad = st.slider("C (nF)", 0.1, 10000.0, 100.0, 0.1) * 1e-9
                ZL_quad = st.slider("Charge Z_L (Ω)", 1.0, 10000.0, 50.0, 1.0)

                f_quad_arr = np.logspace(3, 8, 400)

                # Calcul ABCD selon topologie
                Av_quad = np.zeros(len(f_quad_arr), dtype=complex)
                Zin_quad = np.zeros(len(f_quad_arr), dtype=complex)
                for idx, f in enumerate(f_quad_arr):
                    omega = 2*np.pi*f
                    if topo_quad == "Série Z":
                        M = eng.matrice_ABCD_serie(complex(R_quad, omega*L_quad))
                    elif topo_quad == "Shunt Y":
                        M = eng.matrice_ABCD_shunt(1/complex(R_quad, -1/(omega*C_quad+1e-30)))
                    elif topo_quad == "Filtre LC passe-bas":
                        M_L = eng.matrice_ABCD_serie(1j*omega*L_quad)
                        M_C = eng.matrice_ABCD_shunt(1j*omega*C_quad)
                        M = eng.cascade_ABCD(M_L, M_C)
                    elif topo_quad == "Cascade L + C":
                        M_L = eng.matrice_ABCD_serie(1j*omega*L_quad)
                        M_C = eng.matrice_ABCD_shunt(1j*omega*C_quad)
                        M_R = eng.matrice_ABCD_serie(R_quad)
                        M = eng.cascade_ABCD(M_R, M_L, M_C)
                    else:  # Ligne de transmission (approximation)
                        beta_l = omega * np.sqrt(L_quad*C_quad)
                        Z0 = np.sqrt(L_quad/C_quad)
                        M = np.array([[np.cos(beta_l), 1j*Z0*np.sin(beta_l)],
                                      [1j/Z0*np.sin(beta_l), np.cos(beta_l)]])
                    Av_quad[idx] = eng.gain_tension_ABCD(M, ZL_quad)
                    Zin_quad[idx] = eng.impedance_entree_ABCD(M, ZL_quad)

                # Métriques
                st.metric("Gain DC (dB)", f"{20*np.log10(abs(Av_quad[0])+1e-12):.2f}")
                st.metric("|Z_in| à 1kHz (Ω)", f"{abs(Zin_quad[0]):.2f}")
                idx_fc = np.argmin(np.abs(20*np.log10(np.abs(Av_quad)+1e-12) + 3))
                st.metric("f_-3dB (kHz)", f"{f_quad_arr[idx_fc]/1e3:.3f}")

            with col2:
                mag_quad = 20*np.log10(np.abs(Av_quad)+1e-12)
                phase_quad = np.angle(Av_quad, deg=True)

                fig_q = make_subplots(rows=2, cols=1,
                    subplot_titles=["Gain |Av| (dB)", "Phase ∠Av (°)"])
                fig_q.add_trace(go.Scatter(x=f_quad_arr, y=mag_quad, mode='lines',
                    name='|Av|', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                fig_q.add_trace(go.Scatter(x=f_quad_arr, y=phase_quad, mode='lines',
                    name='∠Av', line=dict(color='#7700ff', width=2.5)), row=2, col=1)
                fig_q.add_hline(y=-3, line_color='#ffcc00', line_dash='dash',
                    annotation_text="-3dB", row=1, col=1)
                fig_q.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                    color='#c0d0ff', title_text="f (Hz)")
                fig_q.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_q.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=460,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_q, use_container_width=True)

                # Impédance d'entrée
                fig_zin = go.Figure()
                fig_zin.add_trace(go.Scatter(x=f_quad_arr, y=np.abs(Zin_quad),
                    mode='lines', name='|Z_in| (Ω)',
                    line=dict(color='#00ff88', width=2.5)))
                fig_zin.add_trace(go.Scatter(x=f_quad_arr,
                    y=np.angle(Zin_quad, deg=True), mode='lines',
                    name='∠Z_in (°)', line=dict(color='#ff00cc', width=2,
                    dash='dash')))
                fig_zin.update_layout(
                    title="Impédance d'entrée Z_in(f)",
                    xaxis_title="f (Hz)", yaxis_title="Z (Ω) / Phase (°)",
                    xaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300)
                st.plotly_chart(fig_zin, use_container_width=True)

                # Exemple d'impédances élémentaires et conversion ABCD → Z
                f_test = f_quad_arr[0]
                Zr_test = zeng.Z_R(R_quad, f_test)
                Zc_test = zeng.Z_C(C_quad, f_test)
                Z_series = zeng.Z_serie(Zr_test, 1j * 2*np.pi*f_test * L_quad)
                Z_parallel = zeng.Z_parallele(Zr_test, 1/(1j * 2*np.pi*f_test * C_quad))
                M_LC = eng.matrice_ABCD_LC_LP(L_quad, C_quad, np.array([f_test]))[0]
                Z_from_M = eng.convertir_ABCD_vers_Z(M_LC)

                st.markdown("#### 🔧 Impédances et conversion de quadripôle")
                st.write(f"Z_R({R_quad:.1f}Ω,@{f_test:.0f}Hz) = {Zr_test.real:.2f} + {Zr_test.imag:.2f}j")
                st.write(f"Z série (R + jωL) = {Z_series.real:.2f} + {Z_series.imag:.2f}j")
                st.write(f"Z // C = {Z_parallel.real:.2f} + {Z_parallel.imag:.2f}j")
                st.write("Matrice Z extraite de la cellule LC passe-bas :")
                st.write(Z_from_M)

                # Export CSV
                df_quad = pd.DataFrame({
                    "f_Hz": f_quad_arr,
                    "Av_dB": mag_quad,
                    "Phase_deg": phase_quad,
                    "Zin_ohm": np.abs(Zin_quad),
                })
                st.download_button("📥 Export CSV quadripôle",
                    df_quad.to_csv(index=False).encode(),
                    "quadripole.csv", "text/csv")

        # ---- sub2 : Adaptation d'impédance ----
        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Réseau d'adaptation en L")
                Rs_adapt = st.slider("R_source (Ω)", 1.0, 10000.0, 50.0, 1.0)
                Rl_adapt = st.slider("R_charge (Ω)", 1.0, 10000.0, 200.0, 1.0)
                f_adapt = st.slider("Fréquence (MHz)", 0.1, 1000.0, 100.0, 0.1) * 1e6

                res = eng.adaptation_impedance_L(Rs_adapt, Rl_adapt, f_adapt)
                st.metric("Q", f"{res['Q']:.3f}")
                st.metric("L (nH)", f"{res['L_nH']:.2f}")
                st.metric("C (pF)", f"{res['C_pF']:.2f}")
                st.metric("BW (MHz)", f"{res['BW_MHz']:.3f}")

                st.markdown("---")
                st.markdown("#### Diagnostic")
                if Rs_adapt < Rl_adapt:
                    st.info("🔵 Rs < Rl : L en série côté source, C en shunt côté charge")
                else:
                    st.info("🟣 Rs > Rl : C en série côté source, L en shunt côté charge")

            with col2:
                # Cercle d'adaptation sur diagramme de Smith simplifié
                theta = np.linspace(0, 2*np.pi, 300)
                fig_smith = go.Figure()
                # Cercle unité
                fig_smith.add_trace(go.Scatter(
                    x=np.cos(theta), y=np.sin(theta),
                    mode='lines', name='|Γ|=1',
                    line=dict(color='rgba(255,255,255,0.2)', width=1)))
                # Points source et charge normalisés
                Z0 = 50
                Zs_norm = Rs_adapt / Z0
                Zl_norm = Rl_adapt / Z0
                S11_s = (Zs_norm - 1)/(Zs_norm + 1)
                S11_l = (Zl_norm - 1)/(Zl_norm + 1)
                fig_smith.add_trace(go.Scatter(
                    x=[np.real(S11_s)], y=[np.imag(S11_s)],
                    mode='markers+text', name='Source',
                    marker=dict(color='#00ccff', size=14, symbol='circle'),
                    text=['Source'], textposition='top center'))
                fig_smith.add_trace(go.Scatter(
                    x=[np.real(S11_l)], y=[np.imag(S11_l)],
                    mode='markers+text', name='Charge',
                    marker=dict(color='#ff00cc', size=14, symbol='square'),
                    text=['Charge'], textposition='top center'))
                fig_smith.add_trace(go.Scatter(x=[0], y=[0],
                    mode='markers', name='Centre (Z0)',
                    marker=dict(color='#ffcc00', size=10)))
                fig_smith.update_layout(
                    title=f"Diagramme de Smith simplifié (Z₀={Z0}Ω)",
                    xaxis_title="Γ réel", yaxis_title="Γ imaginaire",
                    xaxis=dict(range=[-1.1, 1.1], gridcolor='rgba(100,0,255,0.2)',
                               color='#c0d0ff'),
                    yaxis=dict(range=[-1.1, 1.1], gridcolor='rgba(100,0,255,0.2)',
                               color='#c0d0ff', scaleanchor='x'),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=420)
                st.plotly_chart(fig_smith, use_container_width=True)

                # Gain disponible vs fréquence
                f_scan = np.logspace(5, 10, 300)
                Gain_adapt = np.zeros(len(f_scan))
                for i, fi in enumerate(f_scan):
                    ri = eng.adaptation_impedance_L(Rs_adapt, Rl_adapt, fi)
                    Gain_adapt[i] = 4*Rs_adapt*Rl_adapt/(Rs_adapt+Rl_adapt)**2 * (1+ri['Q']**2)
                fig_ga = go.Figure()
                fig_ga.add_trace(go.Scatter(x=f_scan/1e6, y=Gain_adapt,
                    mode='lines', name='Gain adapté',
                    line=dict(color='#00ff88', width=2.5)))
                fig_ga.add_vline(x=f_adapt/1e6, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"f={f_adapt/1e6:.0f}MHz")
                fig_ga.update_layout(
                    title="Gain disponible avec adaptation L",
                    xaxis_title="f (MHz)", yaxis_title="Gain disponible",
                    xaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=280)
                st.plotly_chart(fig_ga, use_container_width=True)

        # ---- sub3 : Paramètres S ----
        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Paramètres S (S11, S21)")
                Z0_s = st.slider("Z₀ de référence (Ω)", 10.0, 300.0, 50.0, 1.0)
                ZL_s = st.slider("Z_charge (Ω)", 1.0, 5000.0, 100.0, 1.0)
                R_s_param = st.slider("R série (Ω)", 0.0, 500.0, 10.0, 1.0)

                f_s_arr = np.logspace(5, 9, 400)
                S11_arr = np.zeros(len(f_s_arr), dtype=complex)
                S21_arr = np.zeros(len(f_s_arr), dtype=complex)
                for idx, f in enumerate(f_s_arr):
                    omega = 2*np.pi*f
                    Zin = R_s_param + ZL_s  # simplification
                    S11_arr[idx] = eng.parametre_S11(Zin, Z0_s)
                    S21_arr[idx] = 1 - abs(S11_arr[idx])**2

                st.metric("|S11| à 1MHz (dB)", f"{20*np.log10(abs(S11_arr[0])+1e-12):.2f}")
                st.metric("TOS", f"{(1+abs(S11_arr[0]))/(1-abs(S11_arr[0])+1e-9):.3f}")

            with col2:
                fig_s = make_subplots(rows=2, cols=1,
                    subplot_titles=["|S11| (dB)", "|S21| (linéaire)"])
                fig_s.add_trace(go.Scatter(x=f_s_arr/1e6,
                    y=20*np.log10(np.abs(S11_arr)+1e-12),
                    mode='lines', name='S11',
                    line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                fig_s.add_trace(go.Scatter(x=f_s_arr/1e6, y=np.abs(S21_arr),
                    mode='lines', name='S21',
                    line=dict(color='#ff00cc', width=2.5)), row=2, col=1)
                fig_s.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                    color='#c0d0ff', title_text="f (MHz)")
                fig_s.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_s.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=440,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_s, use_container_width=True)

        # ---- sub4 : Théorie ----
        with sub4:
            st.markdown("### 📖 Théorie des Quadripôles")
            formules_quad = {
                "Matrice Z": r"\begin{pmatrix}U_1\\U_2\end{pmatrix}=\begin{pmatrix}Z_{11}&Z_{12}\\Z_{21}&Z_{22}\end{pmatrix}\begin{pmatrix}I_1\\I_2\end{pmatrix}",
                "Matrice Y": r"\begin{pmatrix}I_1\\I_2\end{pmatrix}=\begin{pmatrix}Y_{11}&Y_{12}\\Y_{21}&Y_{22}\end{pmatrix}\begin{pmatrix}U_1\\U_2\end{pmatrix}",
                "Matrice ABCD": r"\begin{pmatrix}U_1\\I_1\end{pmatrix}=\begin{pmatrix}A&B\\C&D\end{pmatrix}\begin{pmatrix}U_2\\-I_2\end{pmatrix}",
                "Réciprocité": r"AD-BC=1\text{ (réseau passif réciproque)}",
                "Gain tension": r"A_v=\frac{Z_L}{AZ_L+B}",
                "Impédance entrée": r"Z_{in}=\frac{AZ_L+B}{CZ_L+D}",
                "Paramètre S11": r"S_{11}=\frac{Z_{in}-Z_0}{Z_{in}+Z_0}",
                "TOS": r"\text{TOS}=\frac{1+|S_{11}|}{1-|S_{11}|}",
                "Adaptation en L": r"Q=\sqrt{\frac{R_{max}}{R_{min}}-1},\quad BW=\frac{f_0}{Q}",
            }
            cols = st.columns(2)
            col_idx = 0
            for nom, f in formules_quad.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f)
                col_idx += 1
            st.markdown("---")
            st.markdown("**Références :** Pozar — *Microwave Engineering* (Wiley, 2011) | Bowick — *RF Circuit Design* (Newnes, 2008)")

    # ================================================================
    # CHAPITRE 2 : FILTRES PASSIFS AVANCÉS
    # ================================================================
    elif chapitre == "📡 Filtres passifs avancés":
        st.markdown("## 📡 Section 2 — Filtres Passifs Avancés")
        st.markdown("Butterworth, Chebyshev, Bessel, RLC — synthèse et comparaison.")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Comparaison approx.",
            "📊 RLC — Régimes transitoires",
            "🔬 Synthèse filtre LC",
            "📖 Théorie & Formules"
        ])

        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Paramètres filtre")
                fc_filt = st.slider("f_c (kHz)", 0.1, 1000.0, 10.0, 0.1) * 1e3
                ordre_filt = st.slider("Ordre n", 1, 8, 3)
                ripple_cheb = st.slider("Ondulation Chebyshev (dB)", 0.1, 3.0, 1.0, 0.1)
                type_filt = st.radio("Type", ["Passe-bas", "Passe-haut"], horizontal=True)
                show_all = st.checkbox("Comparer les 3 approx.", True)

                f_cmp = np.logspace(np.log10(fc_filt*0.01), np.log10(fc_filt*100), 500)
                if type_filt == "Passe-bas":
                    H_butter = eng.filtre_butterworth(f_cmp, fc_filt, ordre_filt, "lowpass")
                    H_cheb   = eng.filtre_chebyshev(f_cmp, fc_filt, ordre_filt, ripple_cheb)
                    H_bessel = eng.filtre_bessel_approx(f_cmp, fc_filt, ordre_filt)
                else:
                    H_butter = eng.filtre_butterworth(f_cmp, fc_filt, ordre_filt, "highpass")
                    H_cheb   = eng.filtre_chebyshev(fc_filt**2/f_cmp, fc_filt, ordre_filt, ripple_cheb)
                    H_bessel = eng.filtre_bessel_approx(fc_filt**2/f_cmp, fc_filt, ordre_filt)

                # Ordre minimal Butterworth pour -40dB à 10fc
                n_min = eng.butterworth_ordre(40, 10)
                st.metric("Ordre min Butterworth (-40dB@10fc)", str(n_min))
                st.metric("Atténuation à 10·fc (dB)",
                          f"{-20*ordre_filt:.1f}")
                st.metric("f_c réelle (kHz)", f"{fc_filt/1e3:.3f}")

            with col2:
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(x=f_cmp/1e3,
                    y=20*np.log10(H_butter+1e-12), mode='lines',
                    name='Butterworth', line=dict(color='#00ccff', width=2.5)))
                if show_all:
                    fig_cmp.add_trace(go.Scatter(x=f_cmp/1e3,
                        y=20*np.log10(H_cheb+1e-12), mode='lines',
                        name='Chebyshev', line=dict(color='#ff00cc', width=2.5)))
                    fig_cmp.add_trace(go.Scatter(x=f_cmp/1e3,
                        y=20*np.log10(H_bessel+1e-12), mode='lines',
                        name='Bessel', line=dict(color='#00ff88', width=2.5)))
                fig_cmp.add_hline(y=-3, line_color='#ffcc00', line_dash='dash',
                    annotation_text="-3dB")
                fig_cmp.add_hline(y=-ripple_cheb, line_color='#ff4400',
                    line_dash='dot', annotation_text=f"-{ripple_cheb}dB (ripple)")
                fig_cmp.add_vline(x=fc_filt/1e3, line_color='#7700ff',
                    line_dash='dot', annotation_text="fc")
                fig_cmp.update_layout(
                    title=f"Comparaison — Ordre {ordre_filt}",
                    xaxis_title="f (kHz)", yaxis_title="Gain (dB)",
                    xaxis_type='log', yaxis=dict(range=[-80, 5]),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=420)
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Tableau comparatif
                df_cmp = pd.DataFrame({
                    "Caractéristique": ["Platitude bande passante", "Raideur coupure",
                                        "Linéarité phase", "Ondulation BP"],
                    "Butterworth": ["⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "Aucune"],
                    "Chebyshev":   ["⭐⭐", "⭐⭐⭐⭐⭐", "⭐", f"±{ripple_cheb}dB"],
                    "Bessel":      ["⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐⭐", "Aucune"],
                })
                st.dataframe(df_cmp, use_container_width=True)
                df_export = pd.DataFrame({
                    "f_kHz": f_cmp/1e3,
                    "Butterworth_dB": 20*np.log10(H_butter+1e-12),
                    "Chebyshev_dB": 20*np.log10(H_cheb+1e-12),
                    "Bessel_dB": 20*np.log10(H_bessel+1e-12),
                })
                st.download_button("📥 Export CSV filtres",
                    df_export.to_csv(index=False).encode(),
                    "filtres_comparaison.csv", "text/csv")

        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Régimes transitoires RLC")
                R_rlc = st.slider("R (Ω) — RLC", 0.1, 1000.0, 50.0, 0.1)
                L_rlc = st.slider("L (mH)", 0.1, 1000.0, 10.0, 0.1) * 1e-3
                C_rlc = st.slider("C (μF)", 0.01, 1000.0, 1.0, 0.01) * 1e-6

                omega0_rlc = 1/np.sqrt(L_rlc*C_rlc)
                alpha_rlc = R_rlc/(2*L_rlc)
                f0_rlc = omega0_rlc/(2*np.pi)
                Q_rlc = omega0_rlc*L_rlc/R_rlc
                BW_rlc = f0_rlc/Q_rlc

                regime_str = ("Sous-amorti (oscillant)" if alpha_rlc < omega0_rlc
                              else "Critique" if abs(alpha_rlc-omega0_rlc)<1
                              else "Sur-amorti (exponentiel)")
                st.metric("f₀ (Hz)", f"{f0_rlc:.2f}")
                st.metric("Q", f"{Q_rlc:.3f}")
                st.metric("BW (Hz)", f"{BW_rlc:.2f}")
                st.metric("Régime", regime_str)
                st.metric("α / ω₀", f"{alpha_rlc/omega0_rlc:.4f}")

            with col2:
                t_rlc = np.linspace(0, 5/alpha_rlc if alpha_rlc > 0 else 0.01, 600)
                y_rlc = eng.reponse_indicielle_RLC(R_rlc, L_rlc, C_rlc, t_rlc)
                t_ms = t_rlc * 1e3

                fig_rlc = go.Figure()
                fig_rlc.add_trace(go.Scatter(x=t_ms, y=y_rlc, mode='lines',
                    name='V_C(t)/E', line=dict(color='#00ccff', width=2.5)))
                fig_rlc.add_hline(y=1.0, line_color='#ffcc00', line_dash='dot',
                    annotation_text="Régime permanent")
                if alpha_rlc < omega0_rlc:
                    fig_rlc.add_hline(y=1+np.exp(-np.pi*alpha_rlc/np.sqrt(omega0_rlc**2-alpha_rlc**2)),
                        line_color='#ff4400', line_dash='dash',
                        annotation_text="Dépassement max")
                fig_rlc.update_layout(
                    title=f"Réponse indicielle RLC — {regime_str}",
                    xaxis_title="t (ms)", yaxis_title="V_C / E",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=350)
                st.plotly_chart(fig_rlc, use_container_width=True)

                # Réponse fréquentielle RLC
                f_rlc = np.logspace(np.log10(f0_rlc*0.01), np.log10(f0_rlc*100), 500)
                res_rlc = eng.cellule_LC_resonance(L_rlc, C_rlc, R_rlc, f_rlc)
                fig_rlc2 = make_subplots(rows=1, cols=2,
                    subplot_titles=["Impédance |Z| (Ω)", "Retard de groupe τ_g (ms)"])
                fig_rlc2.add_trace(go.Scatter(x=f_rlc, y=np.abs(res_rlc["Z"]),
                    mode='lines', name='|Z|',
                    line=dict(color='#7700ff', width=2.5)), row=1, col=1)
                H_rlc_freq = 1/res_rlc["Z"] * (1/(2*np.pi*f_rlc*C_rlc*1j+1e-30))
                tau_g = eng.groupe_retard(f_rlc, H_rlc_freq)
                fig_rlc2.add_trace(go.Scatter(x=f_rlc, y=tau_g*1e3,
                    mode='lines', name='τ_g',
                    line=dict(color='#ff00cc', width=2.5)), row=1, col=2)
                fig_rlc2.add_vline(x=f0_rlc, line_color='#ffcc00', line_dash='dash',
                    annotation_text="f₀")
                fig_rlc2.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                    color='#c0d0ff')
                fig_rlc2.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_rlc2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=320,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_rlc2, use_container_width=True)

        with sub3:
            st.markdown("#### 🔧 Synthèse filtre LC passe-bas (tableau normalisé)")
            n_synth = st.slider("Ordre n (synthèse)", 1, 7, 3)
            fc_synth = st.slider("f_c de synthèse (kHz)", 0.1, 10000.0, 100.0, 0.1) * 1e3
            Z0_synth = st.slider("Impédance caractéristique Z₀ (Ω)", 10.0, 600.0, 50.0, 1.0)

            # Valeurs normalisées Butterworth passe-bas (g_k)
            butter_g = {
                1: [1.0000],
                2: [1.4142, 1.4142],
                3: [1.0000, 2.0000, 1.0000],
                4: [0.7654, 1.8478, 1.8478, 0.7654],
                5: [0.6180, 1.6180, 2.0000, 1.6180, 0.6180],
                6: [0.5176, 1.4142, 1.9319, 1.9319, 1.4142, 0.5176],
                7: [0.4450, 1.2470, 1.8019, 2.0000, 1.8019, 1.2470, 0.4450],
            }
            g = butter_g.get(n_synth, butter_g[3])
            omega_c = 2*np.pi*fc_synth

            # Dénormalisation
            composants = []
            for k, gk in enumerate(g):
                if k % 2 == 0:  # inductance en série
                    L_val = gk * Z0_synth / omega_c
                    composants.append({"Élément": f"L{k+1}", "Valeur normalisée": f"{gk:.4f}",
                                        "Valeur réelle": f"{L_val*1e6:.3f} μH", "Type": "Série"})
                else:  # condensateur en shunt
                    C_val = gk / (Z0_synth * omega_c)
                    composants.append({"Élément": f"C{k+1}", "Valeur normalisée": f"{gk:.4f}",
                                        "Valeur réelle": f"{C_val*1e9:.3f} nF", "Type": "Shunt"})

            df_synth = pd.DataFrame(composants)
            st.dataframe(df_synth, use_container_width=True)
            st.download_button("📥 Export CSV synthèse",
                df_synth.to_csv(index=False).encode(),
                "synthese_LC.csv", "text/csv")

            # Vérification réponse synthétisée
            f_verif = np.logspace(np.log10(fc_synth*0.01), np.log10(fc_synth*100), 400)
            H_verif = eng.filtre_butterworth(f_verif, fc_synth, n_synth, "lowpass")
            fig_synth = go.Figure()
            fig_synth.add_trace(go.Scatter(x=f_verif/1e3,
                y=20*np.log10(H_verif+1e-12), mode='lines',
                name=f'Butterworth n={n_synth}',
                line=dict(color='#00ccff', width=2.5)))
            fig_synth.add_hline(y=-3, line_color='#ffcc00', line_dash='dash')
            fig_synth.add_vline(x=fc_synth/1e3, line_color='#ff00cc', line_dash='dot')
            fig_synth.update_layout(
                title=f"Réponse filtre synthétisé n={n_synth}, fc={fc_synth/1e3:.1f}kHz",
                xaxis_title="f (kHz)", yaxis_title="Gain (dB)", xaxis_type='log',
                yaxis=dict(range=[-80, 5]),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=350)
            st.plotly_chart(fig_synth, use_container_width=True)

        with sub4:
            st.markdown("### 📖 Théorie des Filtres Passifs")
            formules_filt = {
                "Butterworth |H|²": r"|H(j\omega)|^2=\frac{1}{1+\left(\frac{\omega}{\omega_c}\right)^{2n}}",
                "Chebyshev |H|²": r"|H(j\omega)|^2=\frac{1}{1+\varepsilon^2 T_n^2\!\left(\frac{\omega}{\omega_c}\right)}",
                "Ordre Butterworth": r"n\geq\frac{\log\!\left(\frac{10^{A_s/10}-1}{10^{A_p/10}-1}\right)}{2\log(\Omega_s/\Omega_p)}",
                "RLC résonance": r"\omega_0=\frac{1}{\sqrt{LC}},\quad Q=\frac{\omega_0 L}{R},\quad BW=\frac{\omega_0}{Q}",
                "Régime sous-amorti": r"v_C(t)=E\!\left[1-e^{-\alpha t}\!\left(\cos\omega_d t+\frac{\alpha}{\omega_d}\sin\omega_d t\right)\right]",
                "Pulsation amortie": r"\omega_d=\sqrt{\omega_0^2-\alpha^2},\quad\alpha=\frac{R}{2L}",
                "Retard de groupe": r"\tau_g(\omega)=-\frac{d\phi(\omega)}{d\omega}",
                "Synthèse Butterworth": r"g_k=2\sin\!\left(\frac{(2k-1)\pi}{2n}\right),\;k=1\ldots n",
            }
            cols = st.columns(2)
            col_idx = 0
            for nom, f in formules_filt.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f)
                col_idx += 1
            st.markdown("---")
            st.markdown("**Références :** Williams & Taylor — *Electronic Filter Design Handbook* (McGraw-Hill) | Zverev — *Handbook of Filter Synthesis* (Wiley)")

    # ================================================================
    # CHAPITRE 3 : SEMI-CONDUCTEURS
    # ================================================================
    elif chapitre == "🔬 Semi-conducteurs":
        st.markdown("## 🔬 Section 3 — Semi-conducteurs")
        st.markdown("Physique des porteurs, jonction PN, capacité de déplétion.")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Porteurs & Mobilité",
            "📊 Jonction PN",
            "🔬 Capacité de jonction",
            "📖 Théorie & Formules"
        ])

        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Matériau et dopage")
                materiau = st.selectbox("Matériau", ["Si", "Ge", "GaAs"])
                N_dop = st.slider("N dopage (cm⁻³)", 1e13, 1e18, 1e16,
                                  format="%.0e")
                T_sc = st.slider("T (K)", 200.0, 600.0, 300.0, 5.0)

                ni = eng.concentration_intrinseque(T_sc, mat=materiau)
                mob = eng.mobilite_Si(T_sc, N_dop)

                st.metric("ni (cm⁻³)", f"{ni:.3e}")
                st.metric("μ_n (cm²/V·s)", f"{mob['mu_n']:.1f}")
                st.metric("μ_p (cm²/V·s)", f"{mob['mu_p']:.1f}")
                st.metric("D_n (cm²/s)", f"{mob['D_n']:.3f}")
                st.metric("D_p (cm²/s)", f"{mob['D_p']:.3f}")
                st.metric("ρ_n (Ω·cm)", f"{1/(N_dop*mob['mu_n']*1.6022e-19):.3e}")

            with col2:
                # ni vs T
                T_arr = np.linspace(200, 600, 200)
                ni_arr = np.array([eng.concentration_intrinseque(T, mat=materiau)
                                   for T in T_arr])
                fig_ni = go.Figure()
                fig_ni.add_trace(go.Scatter(x=T_arr, y=ni_arr, mode='lines',
                    name=f'ni — {materiau}',
                    line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.08)'))
                fig_ni.add_vline(x=T_sc, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"T={T_sc}K")
                fig_ni.update_layout(
                    title=f"Concentration intrinsèque ni(T) — {materiau}",
                    xaxis_title="T (K)", yaxis_title="ni (cm⁻³)",
                    yaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320)
                st.plotly_chart(fig_ni, use_container_width=True)

                # Mobilité vs dopage
                N_arr = np.logspace(13, 19, 200)
                mu_n_arr = np.array([eng.mobilite_Si(T_sc, N)["mu_n"] for N in N_arr])
                mu_p_arr = np.array([eng.mobilite_Si(T_sc, N)["mu_p"] for N in N_arr])
                fig_mob = go.Figure()
                fig_mob.add_trace(go.Scatter(x=N_arr, y=mu_n_arr, mode='lines',
                    name='μ_n', line=dict(color='#00ccff', width=2.5)))
                fig_mob.add_trace(go.Scatter(x=N_arr, y=mu_p_arr, mode='lines',
                    name='μ_p', line=dict(color='#ff00cc', width=2.5)))
                fig_mob.add_vline(x=N_dop, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"N={N_dop:.0e}")
                fig_mob.update_layout(
                    title=f"Mobilité vs dopage à T={T_sc}K",
                    xaxis_title="N (cm⁻³)", yaxis_title="μ (cm²/V·s)",
                    xaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300)
                st.plotly_chart(fig_mob, use_container_width=True)

        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Jonction PN")
                NA_jn = st.slider("NA (cm⁻³)", 1e14, 1e18, 1e16, format="%.0e")
                ND_jn = st.slider("ND (cm⁻³)", 1e14, 1e18, 1e15, format="%.0e")
                V_pol = st.slider("Polarisation V (V)", -5.0, 1.0, 0.0, 0.05)
                mat_jn = st.selectbox("Matériau jonction", ["Si", "Ge", "GaAs"])
                T_jn = st.slider("T jonction (K)", 250.0, 400.0, 300.0, 5.0)

                dep = eng.largeur_depletion(NA_jn, ND_jn, V_pol, T_jn, mat_jn)
                st.metric("V_bi (V)", f"{dep['V_bi']:.4f}")
                st.metric("W total (nm)", f"{dep['W_nm']:.2f}")
                st.metric("xp (nm) — côté P", f"{dep['xp_nm']:.2f}")
                st.metric("xn (nm) — côté N", f"{dep['xn_nm']:.2f}")
                st.metric("E_max (V/m)", f"{dep['E_max']:.3e}")
                Is_sat = eng.courant_saturation_Is(NA_jn, ND_jn,
                                                  dep['xp_nm']*1e-9,
                                                  dep['xn_nm']*1e-9,
                                                  T_jn)
                st.metric("I_sat (A/cm²)", f"{Is_sat:.3e}")

            with col2:
                # Profil de charge et champ électrique
                xp_m = dep["xp_nm"] * 1e-9
                xn_m = dep["xn_nm"] * 1e-9
                x_prof = np.linspace(-xp_m*2, xn_m*2, 500)
                q = 1.6022e-19
                eps_r_dict = {"Si": 11.7, "Ge": 16.0, "GaAs": 12.9}
                eps = eps_r_dict.get(mat_jn, 11.7) * 8.8542e-12

                rho = np.where(x_prof < -xp_m, 0,
                      np.where(x_prof <= 0, -q*NA_jn*1e6,
                      np.where(x_prof <= xn_m, q*ND_jn*1e6, 0)))

                E_field = np.zeros_like(x_prof)
                for i in range(1, len(x_prof)):
                    dx = x_prof[i]-x_prof[i-1]
                    E_field[i] = E_field[i-1] - rho[i]*dx/eps

                fig_jn = make_subplots(rows=2, cols=1,
                    subplot_titles=["Densité de charge ρ (C/m³)",
                                    "Champ électrique E (V/m)"])
                fig_jn.add_trace(go.Scatter(x=x_prof*1e9, y=rho, mode='lines',
                    name='ρ(x)', line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'), row=1, col=1)
                fig_jn.add_trace(go.Scatter(x=x_prof*1e9, y=E_field, mode='lines',
                    name='E(x)', line=dict(color='#ff00cc', width=2.5)), row=2, col=1)
                fig_jn.add_vline(x=0, line_color='rgba(255,255,255,0.3)', line_dash='dot')
                fig_jn.add_vline(x=-dep['xp_nm'], line_color='#ffcc00', line_dash='dash',
                    annotation_text="-xp")
                fig_jn.add_vline(x=dep['xn_nm'], line_color='#ffcc00', line_dash='dash',
                    annotation_text="xn")
                fig_jn.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                    title_text="x (nm)")
                fig_jn.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_jn.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=460,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_jn, use_container_width=True)

                # Largeur de déplétion vs polarisation
                V_scan = np.linspace(-5, dep["V_bi"]*0.9, 200)
                W_scan = [eng.largeur_depletion(NA_jn, ND_jn, v, T_jn, mat_jn)["W_nm"]
                          for v in V_scan]
                fig_W = go.Figure()
                fig_W.add_trace(go.Scatter(x=V_scan, y=W_scan, mode='lines',
                    name='W(V)', line=dict(color='#00ff88', width=2.5)))
                fig_W.add_vline(x=V_pol, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"V={V_pol}V")
                fig_W.update_layout(
                    title="Largeur de déplétion W vs polarisation",
                    xaxis_title="V (V)", yaxis_title="W (nm)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=280)
                st.plotly_chart(fig_W, use_container_width=True)

        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Capacité de jonction Cj(V)")
                NA_cj = st.slider("NA (cm⁻³) Cj", 1e14, 1e18, 1e16, format="%.0e")
                ND_cj = st.slider("ND (cm⁻³) Cj", 1e14, 1e18, 1e15, format="%.0e")
                mat_cj = st.selectbox("Matériau Cj", ["Si", "Ge", "GaAs"])
                V_cj_op = st.slider("V opération (V)", -10.0, 0.8, -2.0, 0.1)

                V_cj_arr = np.linspace(-10, 0.8, 300)
                Cj_arr = eng.capacite_jonction(NA_cj, ND_cj, V_cj_arr, T=300, mat=mat_cj)
                Cj_op = eng.capacite_jonction(NA_cj, ND_cj,
                                               np.array([V_cj_op]), T=300, mat=mat_cj)[0]
                Vbi_cj = eng.tension_built_in(NA_cj, ND_cj, mat=mat_cj)

                st.metric("V_bi (V)", f"{Vbi_cj:.4f}")
                st.metric("Cj₀ (pF/m²)", f"{Cj_arr[np.argmin(np.abs(V_cj_arr))]*1e12:.3f}")
                st.metric(f"Cj à V={V_cj_op}V (pF/m²)", f"{Cj_op*1e12:.3f}")
                st.metric("Rapport Cj(V)/Cj₀",
                          f"{Cj_op/Cj_arr[np.argmin(np.abs(V_cj_arr))]:.3f}")

            with col2:
                fig_cj = go.Figure()
                fig_cj.add_trace(go.Scatter(x=V_cj_arr, y=Cj_arr*1e12, mode='lines',
                    name='Cj(V)', line=dict(color='#00ccff', width=2.5)))
                fig_cj.add_trace(go.Scatter(x=[V_cj_op], y=[Cj_op*1e12],
                    mode='markers', name='Point de travail',
                    marker=dict(color='#ff00cc', size=12, symbol='star')))
                fig_cj.add_vline(x=0, line_color='rgba(255,255,255,0.2)', line_dash='dot')
                fig_cj.update_layout(
                    title="Capacité de jonction Cj(V)",
                    xaxis_title="V (V)", yaxis_title="Cj (pF/m²)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=360)
                st.plotly_chart(fig_cj, use_container_width=True)

                # 1/Cj² vs V (linéarisation Mott-Schottky)
                fig_ms = go.Figure()
                fig_ms.add_trace(go.Scatter(x=V_cj_arr[V_cj_arr<0],
                    y=1/(Cj_arr[V_cj_arr<0]+1e-30)**2, mode='lines',
                    name='1/Cj² (m⁴/F²)',
                    line=dict(color='#7700ff', width=2.5)))
                fig_ms.update_layout(
                    title="Diagramme de Mott-Schottky (1/Cj² vs V)",
                    xaxis_title="V (V)", yaxis_title="1/Cj² (m⁴/F²)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=280)
                st.plotly_chart(fig_ms, use_container_width=True)

                df_cj = pd.DataFrame({"V_V": V_cj_arr, "Cj_pF_m2": Cj_arr*1e12})
                st.download_button("📥 Export CSV Cj(V)",
                    df_cj.to_csv(index=False).encode(), "capacite_jonction.csv", "text/csv")

        with sub4:
            st.markdown("### 📖 Théorie des Semi-conducteurs")
            formules_sc = {
                "Concentration ni": r"n_i^2=N_C N_V\exp\!\left(-\frac{E_g}{k_BT}\right)",
                "Loi d'action de masse": r"np=n_i^2",
                "Tension de built-in": r"V_{bi}=\frac{k_BT}{q}\ln\!\frac{N_A N_D}{n_i^2}",
                "Largeur déplétion": r"W=\sqrt{\frac{2\varepsilon(V_{bi}-V)}{q}\!\left(\frac{1}{N_A}+\frac{1}{N_D}\right)}",
                "Champ max": r"E_{max}=\frac{qN_D x_n}{\varepsilon}=\frac{2(V_{bi}-V)}{W}",
                "Capacité jonction": r"C_j(V)=\frac{C_{j0}}{\sqrt{1-V/V_{bi}}},\quad C_{j0}=\varepsilon A/W_0",
                "Mobilité (Caughey-Thomas)": r"\mu_n=\frac{\mu_{n0}}{1+(N/N_{ref})^{0.8}}",
                "Relation d'Einstein": r"D_n=\mu_n\frac{k_BT}{q}=\mu_n V_T",
                "Courant de drift": r"J_n^{drift}=qn\mu_n E,\quad J_p^{drift}=qp\mu_p E",
                "Courant de diffusion": r"J_n^{diff}=qD_n\frac{dn}{dx},\quad J_p^{diff}=-qD_p\frac{dp}{dx}",
            }
            cols = st.columns(2)
            col_idx = 0
            for nom, f in formules_sc.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f)
                col_idx += 1
            st.markdown("---")
            st.markdown("**Références :** Sze & Ng — *Physics of Semiconductor Devices* (Wiley, 2007) | Streetman & Banerjee — *Solid State Electronic Devices* (Pearson, 2015)")

    # ================================================================
    # CHAPITRE 4 : DIODES — APPLICATIONS
    # ================================================================
    elif chapitre == "💡 Diodes — Applications":
        st.markdown("## 💡 Section 4 — Diodes : Applications")
        st.markdown("Redresseurs, régulateurs Zener, écrêteurs, multiplicateurs de tension.")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Redresseurs",
            "📊 Régulateur Zener",
            "🔬 Écrêteurs & Multiplicateurs",
            "📖 Théorie & Formules"
        ])

        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Paramètres redresseur")
                type_rect = st.radio("Type", ["Monoalternance", "Bialternance (Graëtz)"],
                                     horizontal=True)
                Vm_rect = st.slider("Vm — tension pic (V)", 1.0, 340.0, 24.0, 0.5)
                f_rect = st.slider("f réseau (Hz)", 10.0, 400.0, 50.0, 1.0)
                R_ch_rect = st.slider("R charge (Ω)", 10.0, 10000.0, 100.0, 10.0)
                C_liss_rect = st.slider("C lissage (μF)", 1.0, 10000.0, 100.0, 1.0) * 1e-6
                V_D_rect = st.slider("V_D diode (V)", 0.3, 1.5, 0.7, 0.05)

                if type_rect == "Monoalternance":
                    res_rect = eng.redresseur_monoalternance(
                        Vm_rect, f_rect, R_ch_rect, C_liss_rect, V_D_rect)
                else:
                    res_rect = eng.redresseur_bialternance(
                        Vm_rect, f_rect, R_ch_rect, C_liss_rect, V_D_rect)

                st.metric("V_pic sortie (V)", f"{res_rect['V_pic']:.3f}")
                st.metric("V_moy (V)", f"{res_rect['V_moy']:.3f}")
                st.metric("V_ondulation crête-crête (V)", f"{res_rect['V_ondulation']:.3f}")
                st.metric("Taux d'ondulation (%)", f"{res_rect['taux_ondulation_pct']:.2f}")
                tau_rect = R_ch_rect * C_liss_rect
                st.metric("τ = RC (ms)", f"{tau_rect*1e3:.2f}")

            with col2:
                t_r = res_rect["t"]
                fig_rect = go.Figure()
                fig_rect.add_trace(go.Scatter(x=t_r*1e3, y=res_rect["v_in"],
                    mode='lines', name='V_in (V)',
                    line=dict(color='rgba(119,0,255,0.5)', width=1.5)))
                fig_rect.add_trace(go.Scatter(x=t_r*1e3, y=res_rect["v_out"],
                    mode='lines', name='V_out (V)',
                    line=dict(color='#00ccff', width=2.5)))
                fig_rect.add_hline(y=res_rect["V_moy"], line_color='#ffcc00',
                    line_dash='dash', annotation_text=f"V_moy={res_rect['V_moy']:.2f}V")
                fig_rect.update_layout(
                    title=f"Redresseur {type_rect} — Vm={Vm_rect}V, C={C_liss_rect*1e6:.0f}μF",
                    xaxis_title="t (ms)", yaxis_title="V (V)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380)
                st.plotly_chart(fig_rect, use_container_width=True)

                # Ondulation vs C
                C_scan = np.logspace(-5, -2, 100)
                if type_rect == "Monoalternance":
                    ond_scan = [eng.redresseur_monoalternance(
                        Vm_rect, f_rect, R_ch_rect, c, V_D_rect)["taux_ondulation_pct"]
                                for c in C_scan]
                else:
                    ond_scan = [eng.redresseur_bialternance(
                        Vm_rect, f_rect, R_ch_rect, c, V_D_rect)["taux_ondulation_pct"]
                                for c in C_scan]
                fig_ond = go.Figure()
                fig_ond.add_trace(go.Scatter(x=C_scan*1e6, y=ond_scan, mode='lines',
                    name='Taux ondulation (%)',
                    line=dict(color='#ff00cc', width=2.5)))
                fig_ond.add_vline(x=C_liss_rect*1e6, line_color='#ffcc00',
                    line_dash='dash', annotation_text=f"C={C_liss_rect*1e6:.0f}μF")
                fig_ond.update_layout(
                    title="Taux d'ondulation vs C lissage",
                    xaxis_title="C (μF)", yaxis_title="Taux ondulation (%)",
                    xaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=260)
                st.plotly_chart(fig_ond, use_container_width=True)

                df_rect = pd.DataFrame({
                    "t_ms": t_r*1e3,
                    "V_in_V": res_rect["v_in"],
                    "V_out_V": res_rect["v_out"],
                })
                st.download_button("📥 Export CSV redresseur",
                    df_rect.to_csv(index=False).encode(),
                    "redresseur.csv", "text/csv")

        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Régulateur Zener")
                V_Z_zen = st.slider("V_Z Zener (V)", 1.0, 30.0, 5.1, 0.1)
                R_s_zen = st.slider("R_s série (Ω)", 1.0, 10000.0, 470.0, 1.0)
                I_L_zen = st.slider("I_L charge (mA)", 0.0, 100.0, 10.0, 0.5) * 1e-3
                V_in_zen = st.slider("V_in (V)", V_Z_zen+0.1, V_Z_zen*5, V_Z_zen*2, 0.1)

                V_in_arr = np.linspace(0, V_Z_zen*4, 300)
                res_zen = eng.regulateur_zener(V_in_arr, V_Z_zen, R_s_zen, I_L_zen)

                # Valeurs au point de travail
                I_Z_op = (V_in_zen - V_Z_zen) / R_s_zen - I_L_zen
                P_Z_op = V_Z_zen * max(I_Z_op, 0)
                V_in_min = V_Z_zen + (I_L_zen + 1e-3) * R_s_zen

                st.metric("V_out (V)", f"{V_Z_zen:.2f}")
                st.metric("I_Z (mA)", f"{I_Z_op*1e3:.3f}")
                st.metric("P_Zener (mW)", f"{P_Z_op*1e3:.2f}")
                st.metric("V_in_min (V)", f"{V_in_min:.2f}")
                st.metric("Régulation", "✅ OK" if I_Z_op > 1e-3 else "⚠️ Hors plage")

            with col2:
                fig_zen = make_subplots(rows=2, cols=1,
                    subplot_titles=["V_out vs V_in", "I_Z et P_Z vs V_in"])
                fig_zen.add_trace(go.Scatter(x=V_in_arr, y=np.minimum(V_in_arr, V_Z_zen),
                    mode='lines', name='V_out',
                    line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                fig_zen.add_hline(y=V_Z_zen, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"V_Z={V_Z_zen}V", row=1, col=1)
                fig_zen.add_trace(go.Scatter(x=V_in_arr,
                    y=np.maximum(res_zen["I_Z"]*1e3, 0), mode='lines',
                    name='I_Z (mA)', line=dict(color='#ff00cc', width=2.5)), row=2, col=1)
                fig_zen.add_trace(go.Scatter(x=V_in_arr,
                    y=res_zen["P_Z"]*1e3, mode='lines',
                    name='P_Z (mW)', line=dict(color='#ffcc00', width=2,
                    dash='dash')), row=2, col=1)
                fig_zen.add_vline(x=V_in_zen, line_color='#00ff88', line_dash='dot',
                    annotation_text=f"V_in={V_in_zen}V")
                fig_zen.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                    title_text="V_in (V)")
                fig_zen.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_zen.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=450,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_zen, use_container_width=True)

        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Écrêteur & Multiplicateur")
                mode_ecr = st.selectbox("Mode écrêteur", ["haut", "bas", "double"])
                V_ref_ecr = st.slider("V_ref (V)", 0.0, 10.0, 3.0, 0.1)
                Vm_ecr = st.slider("Vm signal (V)", 1.0, 20.0, 5.0, 0.1)
                f_ecr = st.slider("f signal (Hz)", 1.0, 10000.0, 100.0, 1.0)

                n_mult = st.slider("Nb étages multiplicateur", 1, 6, 3)
                res_mult = eng.multiplicateur_tension(Vm_ecr, f_ecr, n_mult)
                st.metric("V_out multiplicateur (V)", f"{res_mult['V_out_reel']:.2f}")
                st.metric("Rendement (%)", f"{res_mult['rendement_pct']:.1f}")

            with col2:
                t_ecr = np.linspace(0, 3/f_ecr, 1000)
                v_sig = Vm_ecr * np.sin(2*np.pi*f_ecr*t_ecr)
                v_ecr = eng.ecrêteur_diode(v_sig, V_ref_ecr, mode_ecr)

                fig_ecr = go.Figure()
                fig_ecr.add_trace(go.Scatter(x=t_ecr*1e3, y=v_sig, mode='lines',
                    name='V_in', line=dict(color='rgba(119,0,255,0.6)', width=1.5)))
                fig_ecr.add_trace(go.Scatter(x=t_ecr*1e3, y=v_ecr, mode='lines',
                    name='V_out écrêté', line=dict(color='#00ccff', width=2.5)))
                if mode_ecr == "haut":
                    fig_ecr.add_hline(y=V_ref_ecr+0.7, line_color='#ffcc00',
                        line_dash='dash', annotation_text=f"Seuil={V_ref_ecr+0.7:.1f}V")
                elif mode_ecr == "bas":
                    fig_ecr.add_hline(y=V_ref_ecr-0.7, line_color='#ffcc00',
                        line_dash='dash', annotation_text=f"Seuil={V_ref_ecr-0.7:.1f}V")
                else:
                    fig_ecr.add_hline(y=V_ref_ecr+0.7, line_color='#ffcc00', line_dash='dash')
                    fig_ecr.add_hline(y=-V_ref_ecr-0.7, line_color='#ffcc00', line_dash='dash')
                fig_ecr.update_layout(
                    title=f"Écrêteur à diode — mode '{mode_ecr}'",
                    xaxis_title="t (ms)", yaxis_title="V (V)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400)
                st.plotly_chart(fig_ecr, use_container_width=True)

                df_ecr = pd.DataFrame({
                    "t_ms": t_ecr*1e3,
                    "V_in_V": v_sig,
                    "V_out_V": v_ecr,
                })
                st.download_button("📥 Export CSV écrêteur",
                    df_ecr.to_csv(index=False).encode(),
                    "ecreteur.csv", "text/csv")

        with sub4:
            st.markdown("### 📖 Théorie — Diodes Applications")
            formules_diodes_app = {
                "Redresseur monoalternance": r"V_{DC}=\frac{V_m-V_D}{\pi},\quad r=\frac{V_{ripple}}{V_{DC}}\approx\frac{1}{2fRC}",
                "Redresseur bialternance": r"V_{DC}=\frac{2(V_m-2V_D)}{\pi},\quad f_{ripple}=2f",
                "Ondulation avec C": r"\Delta V\approx\frac{I_L}{fC}=\frac{V_{DC}}{fRC}",
                "Diode Zener (régulation)": r"V_{out}=V_Z,\quad I_Z=\frac{V_{in}-V_Z}{R_s}-I_L",
                "Plage de régulation": r"V_{in,min}=V_Z+(I_{Z,min}+I_L)R_s",
                "Écrêteur haut": r"V_{out}=\min(V_{in},\;V_{ref}+V_D)",
                "Écrêteur bas": r"V_{out}=\max(V_{in},\;V_{ref}-V_D)",
                "Multiplicateur Cockcroft-Walton": r"V_{out}=2nV_m,\quad n=\text{nb étages}",
                "Diode varicap": r"C_j(V)=\frac{C_{j0}}{(1-V/V_{bi})^m},\quad m\approx0.5\text{ (abrupte)}",
            }
            cols = st.columns(2)
            col_idx = 0
            for nom, f in formules_diodes_app.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f)
                col_idx += 1
            st.markdown("---")
            st.markdown("**Références :** Boylestad & Nashelsky — *Electronic Devices and Circuit Theory* (Pearson, 2013)")

    # ================================================================
    # CHAPITRE 5 : TRANSISTORS AVANCÉS
    # ================================================================
    elif chapitre == "🔦 Transistors avancés":
        st.markdown("## 🔦 Section 5 — Transistors Avancés")
        st.markdown("Petits signaux, montages amplificateurs, polarisation, réponse fréquentielle.")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Petits signaux hybride-π",
            "📊 Montages amplificateurs",
            "🔬 Polarisation & Stabilité",
            "📖 Théorie & Formules"
        ])

        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Paramètres hybride-π")
                IC_ps = st.slider("IC (mA)", 0.1, 50.0, 1.0, 0.1)
                beta_ps = st.slider("β (hFE)", 10, 500, 100)
                VA_ps = st.slider("Tension d'Early VA (V)", 10.0, 300.0, 100.0, 5.0)
                RC_ps = st.slider("RC (kΩ)", 0.1, 20.0, 1.0, 0.1) * 1e3
                Cbe_ps = st.slider("Cbe (pF)", 1.0, 100.0, 10.0, 0.5) * 1e-12
                Cbc_ps = st.slider("Cbc (pF)", 0.1, 20.0, 2.0, 0.1) * 1e-12

                ps = eng.parametres_petits_signaux_BJT(IC_ps, beta_ps, VA_ps)
                st.metric("gm (mS)", f"{ps['gm_mS']:.3f}")
                st.metric("rπ (kΩ)", f"{ps['r_pi_kOhm']:.3f}")
                st.metric("ro (kΩ)", f"{ps['r_o_kOhm']:.2f}")
                st.metric("re (Ω)", f"{ps['r_e_Ohm']:.2f}")
                st.metric("VT (mV)", f"{ps['VT_mV']:.2f}")

            with col2:
                f_ps_arr = np.logspace(4, 11, 500)
                H_ps, f_beta = eng.reponse_freq_BJT(
                    IC_ps, RC_ps, 50, Cbe_ps, Cbc_ps, beta_ps, f_ps_arr)
                mag_ps = 20*np.log10(np.abs(H_ps)+1e-12)

                fig_ps = go.Figure()
                fig_ps.add_trace(go.Scatter(x=f_ps_arr, y=mag_ps, mode='lines',
                    name='|H(f)| normalisé',
                    line=dict(color='#00ccff', width=2.5)))
                fig_ps.add_vline(x=f_beta, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"fβ={f_beta/1e6:.1f}MHz")
                fig_ps.add_hline(y=-3, line_color='#ff00cc', line_dash='dot',
                    annotation_text="-3dB")
                fig_ps.update_layout(
                    title="Réponse fréquentielle BJT (modèle hybride-π)",
                    xaxis_title="f (Hz)", yaxis_title="Gain normalisé (dB)",
                    xaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=360)
                st.plotly_chart(fig_ps, use_container_width=True)

                # gm vs IC
                IC_scan = np.logspace(-3, 1, 200)  # 1μA à 10mA
                gm_scan = IC_scan*1e-3 / 0.02585
                fig_gm = go.Figure()
                fig_gm.add_trace(go.Scatter(x=IC_scan, y=gm_scan*1e3, mode='lines',
                    name='gm (mS)', line=dict(color='#00ff88', width=2.5)))
                fig_gm.add_vline(x=IC_ps, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"IC={IC_ps}mA")
                fig_gm.update_layout(
                    title="Transconductance gm vs IC",
                    xaxis_title="IC (mA)", yaxis_title="gm (mS)",
                    xaxis_type='log', yaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=270)
                st.plotly_chart(fig_gm, use_container_width=True)

        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Montage amplificateur")
                montage = st.selectbox("Montage", [
                    "Émetteur commun (sans RE bypass)",
                    "Émetteur commun (avec RE bypass)",
                    "Base commune",
                ])
                IC_amp = st.slider("IC (mA) montage", 0.1, 20.0, 2.0, 0.1)
                RC_amp = st.slider("RC (kΩ) montage", 0.1, 20.0, 2.0, 0.1) * 1e3
                RE_amp = st.slider("RE (Ω) montage", 0.0, 5000.0, 500.0, 10.0)
                beta_amp = st.slider("β montage", 10, 500, 100)
                VA_amp = st.slider("VA (V) montage", 10.0, 300.0, 100.0, 5.0)

                if "Base commune" in montage:
                    res_amp = eng.gain_base_commune(IC_amp, RC_amp, beta_amp)
                    st.metric("Av", f"{res_amp['Av']:.3f}")
                    st.metric("Av (dB)", f"{res_amp['Av_dB']:.2f}")
                    st.metric("α", f"{res_amp['alpha']:.4f}")
                    st.metric("Rin (Ω)", f"{res_amp['Rin_Ohm']:.2f}")
                else:
                    bypass = "bypass" in montage
                    res_amp = eng.gain_emetteur_commun(
                        IC_amp, RC_amp, RE_amp, beta_amp, VA_amp, bypass)
                    st.metric("Av", f"{res_amp['Av']:.3f}")
                    st.metric("Av (dB)", f"{res_amp['Av_dB']:.2f}")
                    st.metric("Rin (kΩ)", f"{res_amp['Rin_kOhm']:.3f}")
                    st.metric("Rout (kΩ)", f"{res_amp['Rout_kOhm']:.2f}")

            with col2:
                # Gain vs IC (courbe de polarisation optimale)
                IC_opt = np.logspace(-2, 1.5, 200)
                Av_opt = np.array([eng.gain_emetteur_commun(
                    ic, RC_amp, RE_amp, beta_amp, VA_amp,
                    "bypass" in montage)["Av"] for ic in IC_opt])

                fig_opt = go.Figure()
                fig_opt.add_trace(go.Scatter(x=IC_opt, y=np.abs(Av_opt), mode='lines',
                    name='|Av|', line=dict(color='#00ccff', width=2.5)))
                fig_opt.add_vline(x=IC_amp, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"IC={IC_amp}mA")
                fig_opt.update_layout(
                    title="Gain |Av| vs courant de polarisation IC",
                    xaxis_title="IC (mA)", yaxis_title="|Av|",
                    xaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=340)
                st.plotly_chart(fig_opt, use_container_width=True)

                # Simulation temporelle : signal amplifié
                t_sig = np.linspace(0, 5e-6, 1000)
                f_sig = 500e3
                v_in_sig = 0.01 * np.sin(2*np.pi*f_sig*t_sig)
                Av_val = res_amp["Av"] if "Av" in res_amp else res_amp["Av"]
                v_out_sig = np.clip(Av_val * v_in_sig, -12, 12)
                fig_sig = go.Figure()
                fig_sig.add_trace(go.Scatter(x=t_sig*1e6, y=v_in_sig*1e3, mode='lines',
                    name='V_in (mV)', line=dict(color='rgba(119,0,255,0.7)', width=1.5)))
                fig_sig.add_trace(go.Scatter(x=t_sig*1e6, y=v_out_sig, mode='lines',
                    name='V_out (V)', line=dict(color='#00ccff', width=2.5)))
                fig_sig.update_layout(
                    title=f"Signal amplifié — {montage}",
                    xaxis_title="t (μs)", yaxis_title="V",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=280)
                st.plotly_chart(fig_sig, use_container_width=True)

        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Polarisation diviseur de tension")
                Vcc_pol = st.slider("Vcc (V) pol", 5.0, 30.0, 12.0, 0.5)
                R1_pol = st.slider("R1 (kΩ)", 1.0, 1000.0, 100.0, 1.0) * 1e3
                R2_pol = st.slider("R2 (kΩ)", 1.0, 1000.0, 22.0, 1.0) * 1e3
                RC_pol = st.slider("RC (kΩ) pol", 0.1, 20.0, 2.2, 0.1) * 1e3
                RE_pol = st.slider("RE (kΩ) pol", 0.01, 5.0, 1.0, 0.01) * 1e3
                beta_pol = st.slider("β polarisation", 10, 500, 100)

                res_pol = eng.polarisation_diviseur(
                    Vcc_pol, R1_pol, R2_pol, RC_pol, RE_pol, beta_pol)

                st.metric("IC (mA)", f"{res_pol['IC_mA']:.3f}")
                st.metric("VCE (V)", f"{res_pol['VCE_V']:.3f}")
                st.metric("V_TH (V)", f"{res_pol['VTH_V']:.3f}")
                st.metric("R_TH (kΩ)", f"{res_pol['RTH_kOhm']:.2f}")
                st.metric("Régime", res_pol["regime"])
                st.metric("Facteur stabilité S", f"{res_pol['S_stabilite']:.2f}")

            with col2:
                # Sensibilité au β
                beta_arr = np.linspace(20, 400, 200)
                IC_beta = np.array([eng.polarisation_diviseur(
                    Vcc_pol, R1_pol, R2_pol, RC_pol, RE_pol, b)["IC_mA"]
                                    for b in beta_arr])
                VCE_beta = np.array([eng.polarisation_diviseur(
                    Vcc_pol, R1_pol, R2_pol, RC_pol, RE_pol, b)["VCE_V"]
                                     for b in beta_arr])

                fig_pol = make_subplots(rows=2, cols=1,
                    subplot_titles=["IC (mA) vs β", "VCE (V) vs β"])
                fig_pol.add_trace(go.Scatter(x=beta_arr, y=IC_beta, mode='lines',
                    name='IC (mA)', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                fig_pol.add_trace(go.Scatter(x=beta_arr, y=VCE_beta, mode='lines',
                    name='VCE (V)', line=dict(color='#ff00cc', width=2.5)), row=2, col=1)
                fig_pol.add_vline(x=beta_pol, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"β={beta_pol}")
                fig_pol.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                    title_text="β (hFE)")
                fig_pol.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_pol.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=440,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_pol, use_container_width=True)

                variation_IC = (IC_beta.max()-IC_beta.min())/IC_beta.mean()*100
                st.metric("Variation IC sur β=[20,400] (%)", f"{variation_IC:.1f}")
                df_pol = pd.DataFrame({
                    "beta": beta_arr,
                    "IC_mA": IC_beta,
                    "VCE_V": VCE_beta,
                })
                st.download_button("📥 Export CSV polarisation",
                    df_pol.to_csv(index=False).encode(),
                    "polarisation_beta.csv", "text/csv")

        with sub4:
            st.markdown("### 📖 Théorie — Transistor BJT Petits Signaux")
            formules_bjt = {
                "Transconductance": r"g_m=\frac{I_C}{V_T}=\frac{I_C q}{k_BT}",
                "Résistance rπ": r"r_\pi=\frac{\beta}{g_m}=\frac{V_T}{I_B}",
                "Résistance d'Early ro": r"r_o=\frac{V_A}{I_C}",
                "Résistance re": r"r_e=\frac{V_T}{I_C}=\frac{r_\pi}{1+\beta}",
                "Gain émetteur commun": r"A_v=-g_m(R_C\|r_o)\quad\text{(RE bypassé)}",
                "Gain sans bypass RE": r"A_v=-\frac{g_m R_C}{1+g_m R_E}",
                "Rin émetteur commun": r"R_{in}=r_\pi\|R_B\quad\text{(sans RE)}",
                "Gain base commune": r"A_v=\alpha\frac{R_C}{r_e}\approx g_m R_C,\quad\alpha=\frac{\beta}{\beta+1}",
                "Fréquence de coupure haute": r"f_\beta=\frac{1}{2\pi r_\pi(C_{be}+C_{bc}(1+g_m R_C))}",
                "Effet Early": r"I_C=I_S e^{V_{BE}/V_T}\left(1+\frac{V_{CE}}{V_A}\right)",
                "Facteur stabilité S": r"S=\frac{1+\beta}{1+\beta R_E/(R_{TH}+R_E)}",
            }
            cols = st.columns(2)
            col_idx = 0
            for nom, f in formules_bjt.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f)
                col_idx += 1
            st.markdown("---")
            st.markdown("**Références :** Razavi — *Fundamentals of Microelectronics* (Wiley, 2021) | Sedra & Smith — *Microelectronic Circuits* (Oxford, 2020)")

    # ================================================================
    # CHAPITRE 6 : AOP AVANCÉS
    # ================================================================
    elif chapitre == "⚙️ AOP avancés":
        st.markdown("## ⚙️ Section 6 — Amplificateurs Opérationnels Avancés")
        st.markdown("Intégrateur, dérivateur, Sallen-Key, oscillateur de Wien, Schmitt, ampli instrumentation.")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Filtres actifs",
            "📊 Oscillateurs AOP",
            "🔬 Cas avancés",
            "📖 Théorie & Formules"
        ])

        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Filtre actif")
                type_fa = st.selectbox("Type", [
                    "Intégrateur", "Dérivateur", "Sallen-Key LP"
                ])
                R_fa = st.slider("R (kΩ) FA", 0.1, 1000.0, 10.0, 0.1) * 1e3
                C_fa = st.slider("C (nF) FA", 0.1, 10000.0, 10.0, 0.1) * 1e-9
                R2_fa = st.slider("R2 (kΩ) SK", 0.1, 1000.0, 10.0, 0.1) * 1e3
                C2_fa = st.slider("C2 (nF) SK", 0.1, 10000.0, 10.0, 0.1) * 1e-9
                GBW = st.slider("GBW (MHz)", 0.1, 1000.0, 20.0, 0.1) * 1e6

                f_fa = np.logspace(1, 7, 500)
                if type_fa == "Intégrateur":
                    H_fa = eng.integrateur_AOP(R_fa, C_fa, f_fa)
                    fc_fa = 1/(2*np.pi*R_fa*C_fa)
                    st.metric("f_c intégrateur (Hz)", f"{fc_fa:.2f}")
                elif type_fa == "Dérivateur":
                    H_fa = eng.derivateur_AOP(R_fa, C_fa, f_fa)
                    fc_fa = 1/(2*np.pi*R_fa*C_fa)
                    st.metric("f_c dérivateur (Hz)", f"{fc_fa:.2f}")
                else:  # Sallen-Key
                    H_fa, f0_sk, Q_sk = eng.filtre_actif_sallen_key_LP(
                        R_fa, R2_fa, C_fa, C2_fa, f_fa)
                    st.metric("f₀ Sallen-Key (Hz)", f"{f0_sk:.2f}")
                    st.metric("Q", f"{Q_sk:.4f}")
                    st.metric("Amortissement ξ", f"{1/(2*Q_sk):.4f}")
                    A_v_DC = st.slider("Gain DC approximatif", 1.0, 1000.0, 20.0, 1.0)
                    C_charge = st.slider("C charge (pF)", 1.0, 1000.0, 100.0, 1.0) * 1e-12
                    R_out = st.slider("R sortie (Ω)", 10.0, 1000.0, 100.0, 1.0)
                    stab = eng.stabilite_AOP(A_v_DC, GBW, C_charge, R_out)
                    st.metric("Marge de phase (°)", f"{stab['marge_phase_deg']:.1f}")
                    st.metric("Stabilité AOP", stab['stable'])

            with col2:
                mag_fa = 20*np.log10(np.abs(H_fa)+1e-12)
                phase_fa = np.angle(H_fa, deg=True)

                fig_fa = make_subplots(rows=2, cols=1,
                    subplot_titles=["Gain (dB)", "Phase (°)"])
                fig_fa.add_trace(go.Scatter(x=f_fa, y=mag_fa, mode='lines',
                    name='|H|', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                fig_fa.add_trace(go.Scatter(x=f_fa, y=phase_fa, mode='lines',
                    name='∠H', line=dict(color='#7700ff', width=2.5)), row=2, col=1)
                fig_fa.add_hline(y=-3, line_color='#ffcc00', line_dash='dash',
                    annotation_text="-3dB", row=1, col=1)
                fig_fa.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                    color='#c0d0ff', title_text="f (Hz)")
                fig_fa.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_fa.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=440,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_fa, use_container_width=True)

                # Retard de groupe
                tau_g_fa = eng.groupe_retard(f_fa, H_fa)
                fig_tg = go.Figure()
                fig_tg.add_trace(go.Scatter(x=f_fa,
                    y=np.clip(tau_g_fa*1e6, -1000, 1000),
                    mode='lines', name='τ_g (μs)',
                    line=dict(color='#00ff88', width=2.5)))
                fig_tg.update_layout(
                    title="Retard de groupe τ_g(f)",
                    xaxis_title="f (Hz)", yaxis_title="τ_g (μs)", xaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=240)
                st.plotly_chart(fig_tg, use_container_width=True)

        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Oscillateurs AOP")
                osc_type = st.selectbox("Type oscillateur", [
                    "Wien", "Carré-Triangulaire"
                ])
                R_osc = st.slider("R (kΩ) osc", 0.1, 1000.0, 10.0, 0.1) * 1e3
                C_osc = st.slider("C (nF) osc", 0.1, 10000.0, 10.0, 0.1) * 1e-9
                V_sat_osc = st.slider("±V_sat (V)", 1.0, 15.0, 12.0, 0.5)

                if osc_type == "Wien":
                    res_osc = eng.oscillateur_wien(R_osc, C_osc)
                    st.metric("f₀ (kHz)", f"{res_osc['f0_kHz']:.4f}")
                    st.metric("Gain min AOP", f"{res_osc['gain_min']:.1f}")
                    st.metric("Rf/R1 requis", f"{res_osc['Rf_sur_R1']:.1f}")
                    f0_sim = res_osc["f0_Hz"]
                else:
                    res_osc = eng.generateur_fonction_AOP(R_osc, C_osc, V_sat_osc)
                    st.metric("f₀ (Hz)", f"{res_osc['f0_Hz']:.2f}")
                    st.metric("V carré (V)", f"{res_osc['V_carre_V']:.1f}")
                    st.metric("V triangle (V)", f"{res_osc['V_triangle_V']:.1f}")
                    st.metric("Période (ms)", f"{res_osc['periode_ms']:.3f}")
                    f0_sim = res_osc["f0_Hz"]

            with col2:
                # Simulation temporelle de l'oscillateur
                T_sim = 1/f0_sim if f0_sim > 0 else 1e-3
                t_sim = np.linspace(0, 5*T_sim, 2000)

                if osc_type == "Wien":
                    # Signal sinusoïdal avec envelope de démarrage
                    tau_env = 2*T_sim
                    y_sim = V_sat_osc * 0.7 * (1-np.exp(-t_sim/tau_env)) * np.sin(2*np.pi*f0_sim*t_sim)
                else:
                    # Carré + triangulaire
                    t_norm = t_sim * f0_sim
                    y_carre = V_sat_osc * np.sign(np.sin(2*np.pi*t_norm))
                    y_tri = V_sat_osc * 2/np.pi * np.arcsin(np.sin(2*np.pi*t_norm))

                fig_sim = go.Figure()
                if osc_type == "Wien":
                    fig_sim.add_trace(go.Scatter(x=t_sim*1e3, y=y_sim, mode='lines',
                        name='V_out sinusoïdal',
                        line=dict(color='#00ccff', width=2.5)))
                else:
                    fig_sim.add_trace(go.Scatter(x=t_sim*1e3, y=y_carre, mode='lines',
                        name='Carré', line=dict(color='#00ccff', width=2.5)))
                    fig_sim.add_trace(go.Scatter(x=t_sim*1e3, y=y_tri, mode='lines',
                        name='Triangulaire', line=dict(color='#ff00cc', width=2,
                        dash='dash')))
                fig_sim.update_layout(
                    title=f"Oscillateur {osc_type} — f₀={f0_sim:.2f} Hz",
                    xaxis_title="t (ms)", yaxis_title="V (V)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=350)
                st.plotly_chart(fig_sim, use_container_width=True)

        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Comparateur à hystérésis (Schmitt)")
                V_sat_sch = st.slider("±V_sat Schmitt (V)", 1.0, 15.0, 12.0, 0.5)
                R1_sch = st.slider("R1 Schmitt (kΩ)", 0.1, 100.0, 10.0, 0.1) * 1e3
                R2_sch = st.slider("R2 Schmitt (kΩ)", 0.1, 100.0, 100.0, 0.1) * 1e3
                V_ref_sch = st.slider("V_ref (V)", -5.0, 5.0, 0.0, 0.1)

                res_sch = eng.comparateur_hysteresis(V_sat_sch, R1_sch, R2_sch, V_ref_sch)
                st.metric("V_th haut (V)", f"{res_sch['V_th_haut']:.3f}")
                st.metric("V_th bas (V)", f"{res_sch['V_th_bas']:.3f}")
                st.metric("Hystérésis (V)", f"{res_sch['hysteresis']:.3f}")
                st.metric("Rapport R1/(R1+R2)", f"{res_sch['rapport']:.4f}")

                st.markdown("---")
                st.markdown("#### Ampli d'instrumentation")
                Rg_inst = st.slider("Rg (Ω)", 100.0, 100000.0, 1000.0, 100.0)
                R_inst = st.slider("R (kΩ) inst", 1.0, 100.0, 10.0, 0.5) * 1e3
                Vd_inst = st.slider("Vd_diff (mV)", -100.0, 100.0, 10.0, 0.5) * 1e-3

                res_inst = eng.ampli_instrumentation(Rg_inst, R_inst, Vd_inst)
                st.metric("Gain G", f"{res_inst['G']:.2f}")
                st.metric("G (dB)", f"{res_inst['G_dB']:.2f}")
                st.metric("V_out (V)", f"{res_inst['Vout']:.4f}")

            with col2:
                # Transfert Schmitt trigger
                Vin_sch = np.linspace(-V_sat_sch*1.5, V_sat_sch*1.5, 1000)
                Vout_sch = np.zeros_like(Vin_sch)
                state = -V_sat_sch
                for i, v in enumerate(Vin_sch):
                    if state == -V_sat_sch and v > res_sch["V_th_haut"]:
                        state = V_sat_sch
                    elif state == V_sat_sch and v < res_sch["V_th_bas"]:
                        state = -V_sat_sch
                    Vout_sch[i] = state

                fig_sch = go.Figure()
                fig_sch.add_trace(go.Scatter(x=Vin_sch, y=Vout_sch, mode='lines',
                    name='V_out Schmitt',
                    line=dict(color='#00ccff', width=2.5)))
                fig_sch.add_vline(x=res_sch["V_th_haut"], line_color='#ffcc00',
                    line_dash='dash', annotation_text=f"V+={res_sch['V_th_haut']:.2f}V")
                fig_sch.add_vline(x=res_sch["V_th_bas"], line_color='#ff00cc',
                    line_dash='dash', annotation_text=f"V-={res_sch['V_th_bas']:.2f}V")
                fig_sch.update_layout(
                    title="Transfert Trigger de Schmitt",
                    xaxis_title="V_in (V)", yaxis_title="V_out (V)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320)
                st.plotly_chart(fig_sch, use_container_width=True)

                # Gain ampli instrumentation vs Rg
                Rg_arr = np.logspace(1, 6, 300)
                G_arr = 1 + 2*R_inst/Rg_arr
                fig_inst = go.Figure()
                fig_inst.add_trace(go.Scatter(x=Rg_arr, y=G_arr, mode='lines',
                    name='G(Rg)', line=dict(color='#00ff88', width=2.5)))
                fig_inst.add_vline(x=Rg_inst, line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"Rg={Rg_inst:.0f}Ω")
                fig_inst.update_layout(
                    title="Gain ampli instrumentation vs Rg",
                    xaxis_title="Rg (Ω)", yaxis_title="G",
                    xaxis_type='log', yaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=280)
                st.plotly_chart(fig_inst, use_container_width=True)

        with sub4:
            st.markdown("### 📖 Théorie — AOP Avancés")
            formules_aop_adv = {
                "Intégrateur": r"V_{out}(t)=-\frac{1}{RC}\int_0^t V_{in}(\tau)d\tau,\quad H(j\omega)=\frac{-1}{j\omega RC}",
                "Dérivateur": r"V_{out}(t)=-RC\frac{dV_{in}}{dt},\quad H(j\omega)=-j\omega RC",
                "Sallen-Key LP": r"H(s)=\frac{\omega_0^2}{s^2+\frac{\omega_0}{Q}s+\omega_0^2},\quad\omega_0=\frac{1}{\sqrt{R_1R_2C_1C_2}}",
                "Facteur Q Sallen-Key": r"Q=\frac{\sqrt{R_1R_2C_1C_2}}{C_2(R_1+R_2)}",
                "Oscillateur Wien — condition": r"\frac{R_f}{R_1}=2,\quad f_0=\frac{1}{2\pi RC}",
                "Générateur carré-triangulaire": r"f_0=\frac{1}{4RC},\quad V_{tri}=\pm V_{sat}\frac{R_1}{R_2}",
                "Trigger de Schmitt": r"V_{th\pm}=V_{ref}\pm V_{sat}\frac{R_1}{R_1+R_2}",
                "Ampli instrumentation": r"G=1+\frac{2R}{R_G},\quad V_{out}=G(V_+-V_-)",
                "Stabilité — marge de phase": r"\phi_m=90°-\arctan\!\left(\frac{f_c}{f_{pole}}\right)>45°",
                "Produit gain-bande": r"GBW=|A_v|\cdot f_{-3dB}=\text{constante}",
            }
            cols = st.columns(2)
            col_idx = 0
            for nom, f in formules_aop_adv.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f)
                col_idx += 1
            st.markdown("---")
            st.markdown("**Références :** Franco — *Design with Operational Amplifiers* (McGraw-Hill, 2014) | Horowitz & Hill — *The Art of Electronics* (Cambridge, 2015)")
