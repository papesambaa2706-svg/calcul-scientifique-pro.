__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import numpy as np
import math

def safe_trapz(y, x=None, dx=1.0, axis=-1):
    if hasattr(np, 'trapz'):
        return np.trapz(y, x=x, dx=dx, axis=axis)
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    y = np.asanyarray(y)
    if x is None:
        x = dx * np.arange(y.shape[axis])
    if axis != -1 and axis != y.ndim - 1:
        y = np.moveaxis(y, axis, -1)
    return np.sum((y[..., 1:] + y[..., :-1]) * 0.5 * np.diff(x), axis=-1)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve, minimize_scalar
from scipy import stats, optimize, integrate, signal as sp_signal
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES PHYSIQUES
# ============================================================
CONSTANTES = {
    "h_Planck":     (6.626e-34,  "J·s",   "Constante de Planck"),
    "hbar":         (1.055e-34,  "J·s",   "Constante de Planck réduite"),
    "c_lumiere":    (2.998e8,    "m/s",   "Vitesse de la lumière"),
    "k_Boltzmann":  (1.381e-23,  "J/K",   "Constante de Boltzmann"),
    "epsilon_0":    (8.854e-12,  "F/m",   "Permittivité du vide"),
    "e_electron":   (1.602e-19,  "C",     "Charge élémentaire"),
}

# ============================================================
# FORMULAIRE SCIENTIFIQUE
# ============================================================
FORMULES = {
    # Laser
    "Décroissance laser":       r"I(t) = I_0 \cdot e^{-\gamma t}",
    "Demi-vie":                 r"t_{1/2} = \frac{\ln 2}{\gamma}",
    "Équations de taux (N₂)":  r"\frac{dN_2}{dt} = R_p - \frac{N_2}{\tau}",
    "Inversion de population":  r"\Delta N = N_2 - N_1",
    "Gain laser":               r"g(\nu) = \sigma(\nu)\,\Delta N",
    "Section efficace":         r"\sigma(\nu) = \frac{\lambda^2}{8\pi n^2 \tau}\,g(\nu)",
    "Équation de Frantz-Nodvik":r"I_{out} = I_{sat}\ln\left[1 + \left(e^{I_{in}/I_{sat}}-1\right)e^{g_0 L}\right]",
    "Puissance seuil":          r"P_{th} = \frac{h\nu \cdot V \cdot \Delta N_{th}}{\eta_p \tau_p}",
    "Longueur d'onde":          r"\lambda = \frac{c}{\nu} = \frac{hc}{E_2 - E_1}",
    "Fréquence modes cavité":   r"\nu_m = \frac{mc}{2nL}, \quad m \in \mathbb{N}",
    "Facteur Q cavité":         r"Q = \frac{2\pi\nu_0 \tau_p}{1} = \frac{\omega_0}{2\delta}",
    "Finesse":                  r"\mathcal{F} = \frac{\pi\sqrt{R}}{1-R}",
    "Équation Bloch optique":   r"\dot{\rho}_{12} = -\left(\frac{1}{T_2} + i\delta\right)\rho_{12} + \frac{i\Omega}{2}(\rho_{11}-\rho_{22})",

    # Cavité & pertes
    "Loi de Beer-Lambert":      r"I(x) = I_0\,e^{-\alpha x}",
    "Pertes totales":           r"\alpha_{tot} = \alpha_{abs} + \alpha_{dif} + \alpha_{mir}",
    "Pertes miroirs":           r"\alpha_{mir} = -\frac{\ln(R_1 R_2)}{2L}",
    "Temps de vie photon":      r"\tau_p = \frac{n L}{c\,\alpha_{tot} L} = \frac{1}{c\,\alpha_{tot}}",
    "Transmittance":            r"T(\lambda) = e^{-\alpha(\lambda) L}",
    "Distance demi-atténuation":r"L_{1/2} = \frac{\ln 2}{\alpha}",
    "Distance 1/e":             r"L_{1/e} = \frac{1}{\alpha}",
    "Bilan puissance":          r"P_{out} = P_{in}(1-T_1)(1-T_2)\,e^{-2\alpha L}",
    "Seuil laser":              r"\alpha_{tot} = g_0 \Rightarrow g_{th} = \alpha_{int} + \alpha_{mir}",

    # Gaussienne
    "Gaussienne 1D":          r"f(x) = A\,\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
    "Gaussienne normalisée":  r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
    "FWHM":                   r"\text{FWHM} = 2\sqrt{2\ln 2}\,\sigma \approx 2.3548\,\sigma",
    "Aire totale":            r"\int_{-\infty}^{+\infty}f(x)\,dx = A\sigma\sqrt{2\pi}",
    "Gaussienne 2D":          r"f(x,y)=A\exp\!\left(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}\right)",
    "Entropie":               r"H = \tfrac{1}{2}\ln(2\pi e\sigma^2)",
    "Fonction caractéristique":r"\phi(t)=\exp\!\left(i\mu t - \tfrac{\sigma^2 t^2}{2}\right)",
    "Moment centré d'ordre n":r"\langle(x-\mu)^n\rangle = \begin{cases}0 & n\text{ impair}\\(n-1)!!\,\sigma^n & n\text{ pair}\end{cases}",
    "Convolution":            r"(f_1*f_2)(x) = A_{12}\exp\!\left(-\frac{(x-\mu_{12})^2}{2(\sigma_1^2+\sigma_2^2)}\right)",
    "Transformée de Fourier": r"\hat{f}(\xi)=A\sigma\sqrt{2\pi}\exp(-2\pi^2\sigma^2\xi^2-2\pi i\xi\mu)",
}

TYPES_LASER = {
    "He-Ne":        {"λ": 632.8,  "type": "Gaz",       "η": 0.001, "τ": 10e-9,  "P_typ": "1-50 mW"},
    "CO₂":          {"λ": 10600,  "type": "Gaz",       "η": 0.20,  "τ": 1e-6,   "P_typ": "10 W - 100 kW"},
    "Nd:YAG":       {"λ": 1064,   "type": "Solide",    "η": 0.04,  "τ": 230e-6, "P_typ": "1 mW - 10 kW"},
    "Ti:Saphir":    {"λ": 800,    "type": "Solide",    "η": 0.10,  "τ": 3.2e-6, "P_typ": "femtoseconde"},
    "GaAs (diode)": {"λ": 870,    "type": "Semi-cond", "η": 0.50,  "τ": 1e-9,   "P_typ": "1 mW - 10 W"},
    "Excimère ArF": {"λ": 193,    "type": "Gaz",       "η": 0.02,  "τ": 5e-9,   "P_typ": "1-100 W"},
    "Fibre Yb":     {"λ": 1030,   "type": "Fibre",     "η": 0.80,  "τ": 1e-3,   "P_typ": "1 W - 10 kW"},
}

MATERIAUX_OPTIQUES = {
    "Silice (SiO₂) — 1550 nm": {"alpha": 0.0003, "n": 1.444, "unite": "cm⁻¹"},
    "Nd:YAG — 1064 nm":         {"alpha": 0.002,  "n": 1.820, "unite": "cm⁻¹"},
    "ZnSe — CO₂ (10.6 μm)":    {"alpha": 0.001,  "n": 2.403, "unite": "cm⁻¹"},
    "Germanium — IR":           {"alpha": 0.05,   "n": 4.000, "unite": "cm⁻¹"},
    "Air ambiant":              {"alpha": 0.0001, "n": 1.000, "unite": "cm⁻¹"},
    "GaAs — 870 nm":            {"alpha": 0.5,    "n": 3.600, "unite": "cm⁻¹"},
    "Personnalisé":             {"alpha": 0.1,    "n": 1.5,   "unite": "cm⁻¹"},
}

APPLICATIONS = {
    "Optique — profil laser TEM₀₀": "Intensité I(r) = I₀ exp(-2r²/w²)",
    "Statistiques — loi normale":    "Erreurs de mesure, TCL",
    "Mécanique quantique — paquet":  "ψ(x) = A exp(-x²/4σ²) exp(ik₀x)",
    "Traitement signal — filtre":    "h(t) = exp(-t²/2σ²) (filtre gaussien)",
    "Imagerie — PSF":                "Point Spread Function en microscopie",
    "Finance — VaR":                 "Distribution des rendements",
}


# ============================================================
# MOTEUR LASER
# ============================================================
class LaserEngine:
    """Moteur de simulation laser complet : taux, modes, cavité."""

    def __init__(self, I0: float, gamma: float):
        self.I0 = I0
        self.gamma = gamma

    # --- Modèle simple ---
    def intensite_simple(self, t: np.ndarray) -> np.ndarray:
        return self.I0 * np.exp(-self.gamma * t)

    @property
    def demi_vie(self) -> float:
        return np.log(2) / self.gamma if self.gamma > 0 else np.inf

    @property
    def temps_vie(self) -> float:
        return 1.0 / self.gamma if self.gamma > 0 else np.inf

    # --- Équations de taux à 4 niveaux ---
    def equations_taux_4niveaux(self, t: np.ndarray,
                                 Rp: float, tau21: float,
                                 tau32: float, tau10: float,
                                 N_total: float = 1e20) -> dict:
        """
        Système 4 niveaux : pompage → N3 → N2 → N1 → N0
        """
        def dydt(t, y):
            N0, N1, N2, N3 = y
            dN3 = Rp * N0 - N3 / tau32
            dN2 = N3 / tau32 - N2 / tau21
            dN1 = N2 / tau21 - N1 / tau10
            dN0 = N1 / tau10 - Rp * N0
            return [dN0, dN1, dN2, dN3]

        y0 = [N_total, 0, 0, 0]
        sol = solve_ivp(dydt, [t[0], t[-1]], y0, t_eval=t,
                        method='RK45', rtol=1e-6, atol=1e-10)
        return {
            "t": sol.t,
            "N0": sol.y[0], "N1": sol.y[1],
            "N2": sol.y[2], "N3": sol.y[3],
            "inversion": sol.y[2] - sol.y[1],
        }

    # --- Oscillations de relaxation ---
    def oscillations_relaxation(self, t: np.ndarray,
                                 Rp: float, tau_c: float,
                                 tau_sp: float, N_th: float,
                                 S0: float = 1e3) -> dict:
        """Équations couplées photons/population autour du seuil."""
        def dydt(t, y):
            N, S = y
            dN = Rp - N / tau_sp - N * S / tau_c
            dS = N * S / tau_c - S / tau_c + N / tau_sp
            return [dN, dS]

        y0 = [N_th * 0.9, S0]
        sol = solve_ivp(dydt, [t[0], t[-1]], y0, t_eval=t,
                        method='RK45', rtol=1e-8, atol=1e-12,
                        max_step=(t[-1]-t[0])/2000)
        return {"t": sol.t, "N": sol.y[0], "S": sol.y[1]}

    # --- Profil spectral ---
    def profil_spectral(self, lambda_center: float,
                        delta_lambda: float, n_modes: int,
                        gain: float) -> tuple:
        """Spectre laser avec modes de cavité et profil de gain."""
        lambda_arr = np.linspace(lambda_center - 5*delta_lambda,
                                  lambda_center + 5*delta_lambda, 2000)
        # Profil de gain (lorentzien)
        gamma_l = delta_lambda / 2
        gain_profile = gain / (1 + ((lambda_arr - lambda_center)/gamma_l)**2)

        # Modes de cavité
        modes_lambda = []
        modes_gain = []
        delta_mode = delta_lambda / max(n_modes, 1)
        for m in range(-n_modes//2, n_modes//2 + 1):
            lm = lambda_center + m * delta_mode
            gm = gain / (1 + ((lm - lambda_center)/gamma_l)**2)
            modes_lambda.append(lm)
            modes_gain.append(gm)

        return lambda_arr, gain_profile, np.array(modes_lambda), np.array(modes_gain)

    # --- Frantz-Nodvik (amplification) ---
    def frantz_nodvik(self, I_in: np.ndarray,
                       g0: float, L: float, I_sat: float) -> np.ndarray:
        """Équation de Frantz-Nodvik pour amplificateur laser."""
        return I_sat * np.log(1 + (np.exp(I_in / I_sat) - 1) * np.exp(g0 * L))

    # --- Modes TEMₘₙ gaussiens ---
    def mode_gaussien(self, x: np.ndarray, y: np.ndarray,
                       w0: float, z: float,
                       lambda_um: float = 1.064) -> np.ndarray:
        """Profil TEM₀₀ gaussien en champ lointain.

        x, y sont en mètres, w0 est en millimètres et lambda_um en micromètres.
        """
        w0_m = w0 * 1e-3
        lambda_m = lambda_um * 1e-6
        zR = np.pi * w0_m**2 / lambda_m
        wz = w0_m * np.sqrt(1 + (z / zR)**2)
        X, Y = np.meshgrid(x, y)
        return np.exp(-2 * (X**2 + Y**2) / wz**2)

    # --- Énergie & puissance ---
    def energie_impulsion(self, I0: float, tau_p: float,
                           w0: float, forme: str = "gaussien") -> float:
        """Énergie d'une impulsion laser."""
        A_eff = np.pi * w0**2 / 2
        if forme == "gaussien":
            return I0 * A_eff * tau_p * np.sqrt(np.pi / (4 * np.log(2)))
        elif forme == "sech2":
            return I0 * A_eff * tau_p * 1.7627
        return I0 * A_eff * tau_p

    # --- Diagnostics ---
    def diagnostiquer(self, I0: float, gamma: float,
                       t_max: float) -> list:
        """Analyse automatique des paramètres laser."""
        diag = []
        t12 = np.log(2) / gamma if gamma > 0 else np.inf
        tau = 1 / gamma if gamma > 0 else np.inf

        diag.append({
            "Paramètre": "Demi-vie t₁/₂",
            "Valeur": f"{t12:.4f} s",
            "Statut": "✅ OK" if t12 < t_max else "⚠️ Hors fenêtre",
            "Note": f"{'Visible' if t12 < t_max else 'Augmenter t_max'}"
        })
        diag.append({
            "Paramètre": "Temps de vie τ",
            "Valeur": f"{tau:.4f} s",
            "Statut": "✅ OK",
            "Note": f"I(τ) = I₀/e = {I0/np.e:.3f}"
        })
        diag.append({
            "Paramètre": "Rapport I(t_max)/I₀",
            "Valeur": f"{np.exp(-gamma*t_max)*100:.2f}%",
            "Statut": "✅ OK" if np.exp(-gamma*t_max) > 1e-6 else "⚠️ Signal nul",
            "Note": "Augmenter I₀ ou réduire γ si signal nul"
        })
        diag.append({
            "Paramètre": "Régime",
            "Valeur": f"γ = {gamma:.3f} s⁻¹",
            "Statut": "🔴 Rapide" if gamma > 1 else "🟡 Moyen" if gamma > 0.1 else "🟢 Lent",
            "Note": f"{'Sur-amorti' if gamma > 1 else 'Normal'}"
        })
        return diag


# ============================================================
# MOTEUR PERTES DE CAVITÉ
# ============================================================
class CavityLossEngine:
    """Moteur d'analyse des pertes optiques et de cavité laser."""

    def __init__(self, I0: float, alpha: float):
        if alpha < 0:
            raise ValueError("α doit être ≥ 0")
        self.I0 = I0
        self.alpha = alpha

    # --- Propagation ---
    def intensite(self, x: np.ndarray) -> np.ndarray:
        """Loi de Beer-Lambert I(x) = I₀ exp(-αx)"""
        return self.I0 * np.exp(-self.alpha * x)

    @property
    def distance_1_e(self) -> float:
        """Distance où I = I₀/e"""
        return 1.0 / self.alpha if self.alpha > 0 else np.inf

    @property
    def distance_demi(self) -> float:
        """Distance où I = I₀/2"""
        return np.log(2) / self.alpha if self.alpha > 0 else np.inf

    def transmittance(self, L: float) -> float:
        """Transmittance T = I(L)/I₀"""
        return np.exp(-self.alpha * L)

    def absorbance(self, L: float) -> float:
        """Absorbance A = -log₁₀(T)"""
        return -np.log10(self.transmittance(L))

    # --- Cavité laser ---
    def pertes_miroirs(self, R1: float, R2: float, L: float) -> float:
        """Pertes dues aux miroirs α_mir = -ln(R₁R₂)/(2L)"""
        return -np.log(R1 * R2) / (2 * L)

    def pertes_totales(self, R1: float, R2: float, L: float) -> float:
        """Pertes totales α_tot = α_int + α_mir"""
        return self.alpha + self.pertes_miroirs(R1, R2, L)

    def finesse(self, R1: float, R2: float, alpha_int: float, L: float) -> float:
        """Finesse F = π√(R)/(1-R) ≈ π/(T+δ)"""
        R_eff = np.sqrt(R1 * R2)
        return np.pi * np.sqrt(R_eff) / (1 - R_eff)

    def temps_vie_photon(self, R1: float, R2: float, L: float, n: float = 1.5) -> float:
        """Temps de vie photon τ_p = nL/(c α_tot)"""
        c = 3e10  # cm/s
        alpha_tot = self.pertes_totales(R1, R2, L)
        return n * L / (c * alpha_tot)

    # --- Optimisation ---
    def optimiser_R2(self, R1: float, L: float, g0: float, alpha_int: float) -> dict:
        """Optimisation de R₂ pour puissance max"""
        def puissance(R2):
            alpha_tot = alpha_int + self.pertes_miroirs(R1, R2, L)
            if alpha_tot >= g0:
                return 0  # Au-dessous du seuil
            T1 = 1 - R1
            T2 = 1 - R2
            return T1 * T2 * np.exp(-2 * alpha_tot * L) / (1 - R1 * R2 * np.exp(-2 * alpha_tot * L))

        # Optimisation
        res = minimize_scalar(lambda r: -puissance(r), bounds=(0.1, 0.999), method='bounded')
        R2_opt = res.x
        P_max = puissance(R2_opt)

        return {
            "R2_opt": R2_opt,
            "P_max": P_max,
            "R2_range": np.linspace(0.1, 0.999, 100),
            "P_range": [puissance(r) for r in np.linspace(0.1, 0.999, 100)]
        }

    # --- Diagnostic ---
    def diagnostiquer(self, R1: float, R2: float, L: float) -> list:
        """Analyse automatique des pertes de cavité"""
        diag = []
        alpha_mir = self.pertes_miroirs(R1, R2, L)
        alpha_tot = self.pertes_totales(R1, R2, L)
        F = self.finesse(R1, R2, self.alpha, L)
        tau_p = self.temps_vie_photon(R1, R2, L)

        diag.append({
            "Paramètre": "Pertes miroirs",
            "Valeur": f"{alpha_mir:.4f} cm⁻¹",
            "Statut": "✅ OK" if alpha_mir < 0.1 else "⚠️ Élevées",
            "Note": f"R₁={R1:.3f}, R₂={R2:.3f}"
        })
        diag.append({
            "Paramètre": "Pertes totales",
            "Valeur": f"{alpha_tot:.4f} cm⁻¹",
            "Statut": "✅ OK" if alpha_tot < 0.5 else "⚠️ Critiques",
            "Note": "Doit être < g₀ pour oscillation"
        })
        diag.append({
            "Paramètre": "Finesse",
            "Valeur": f"{F:.1f}",
            "Statut": "✅ Bonne" if F > 10 else "⚠️ Faible",
            "Note": f"{'Sélectivité OK' if F > 10 else 'Large bande'}"
        })
        diag.append({
            "Paramètre": "Temps vie photon",
            "Valeur": f"{tau_p*1e9:.2f} ns",
            "Statut": "✅ OK" if tau_p > 1e-9 else "⚠️ Court",
            "Note": f"{'Stockage OK' if tau_p > 1e-9 else 'Dégénérescence rapide'}"
        })
        return diag


# ============================================================
# MOTEUR GAUSSIEN
# ============================================================
class GaussianEngine:
    """Moteur scientifique complet pour distributions gaussiennes."""

    def __init__(self, amplitude: float, sigma: float, mu: float = 0.0):
        if sigma <= 0:
            raise ValueError(f"σ doit être > 0, reçu : {sigma}")
        if amplitude <= 0:
            raise ValueError(f"Amplitude doit être > 0, reçu : {amplitude}")
        self.A = amplitude
        self.sigma = sigma
        self.mu = mu

    # --- Propriétés analytiques ---
    @property
    def fwhm(self) -> float:
        """Full Width at Half Maximum"""
        return 2 * np.sqrt(2 * np.log(2)) * self.sigma

    @property
    def aire(self) -> float:
        """Aire totale ∫ f(x) dx"""
        return self.A * self.sigma * np.sqrt(2 * np.pi)

    @property
    def entropie(self) -> float:
        """Entropie différentielle H = ½ ln(2πe σ²)"""
        return 0.5 * np.log(2 * np.pi * np.e * self.sigma**2)

    # --- Évaluation ---
    def eval_1d(self, x: np.ndarray) -> np.ndarray:
        """Gaussienne 1D f(x) = A exp(-(x-μ)²/(2σ²))"""
        return self.A * np.exp(-0.5 * ((x - self.mu) / self.sigma)**2)

    def eval_1d_normalise(self, x: np.ndarray) -> np.ndarray:
        """Gaussienne normalisée φ(x) = (1/(σ√(2π))) exp(-(x-μ)²/(2σ²))"""
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mu) / self.sigma)**2)

    def eval_2d(self, X: np.ndarray, Y: np.ndarray,
                mu_y: float = 0.0, sigma_y: float = None) -> np.ndarray:
        """Gaussienne 2D f(x,y) = A exp(-[(x-μ_x)²/(2σ_x²) + (y-μ_y)²/(2σ_y²)])"""
        if sigma_y is None:
            sigma_y = self.sigma
        return self.A * np.exp(-0.5 * (((X - self.mu) / self.sigma)**2 + ((Y - mu_y) / sigma_y)**2))

    # --- Moments statistiques ---
    def moments(self, ordre_max: int) -> dict:
        """Moments centrés d'ordre 0 à ordre_max"""
        moments = {}
        for n in range(ordre_max + 1):
            if n % 2 == 0:  # Moments pairs non nuls
                val = math.factorial(n) // (2**(n//2) * math.factorial(n//2)) * self.sigma**n
                moments[f"μ_{n}"] = val
            else:
                moments[f"μ_{n}"] = 0.0
        return moments

    def intervalle_confiance(self, confiance: float = 0.95) -> tuple:
        """Intervalle de confiance autour de μ"""
        z = stats.norm.ppf((1 + confiance) / 2)
        return self.mu - z * self.sigma, self.mu + z * self.sigma

    # --- Convolution ---
    def convolution(self, sigma2: float) -> 'GaussianEngine':
        """Convolution avec une autre gaussienne de paramètre σ₂"""
        sigma_conv = np.sqrt(self.sigma**2 + sigma2**2)
        return GaussianEngine(self.A, sigma_conv, self.mu)

    # --- Transformée de Fourier ---
    def fft_analytique(self, xi: np.ndarray) -> np.ndarray:
        """FFT analytique ĝ(ξ) = A σ √(2π) exp(-2π² σ² ξ² - 2π i ξ μ)"""
        return self.A * self.sigma * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi**2 * self.sigma**2 * xi**2 - 2j * np.pi * xi * self.mu)

    # --- Ajustement de données ---
    def fit_data(self, x_data: np.ndarray, y_data: np.ndarray) -> dict:
        """Ajustement gaussien sur données expérimentales"""
        try:
            def gaussienne(x, A, mu, sigma):
                return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

            popt, pcov = optimize.curve_fit(gaussienne, x_data, y_data,
                                          p0=[self.A, self.mu, self.sigma],
                                          bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
            A_fit, mu_fit, sigma_fit = popt
            perr = np.sqrt(np.diag(pcov))

            # Métriques
            y_fit = gaussienne(x_data, *popt)
            ss_res = np.sum((y_data - y_fit)**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)

            fwhm_fit = 2 * np.sqrt(2 * np.log(2)) * sigma_fit

            return {
                "A": A_fit, "μ": mu_fit, "σ": sigma_fit,
                "σ_A": perr[0], "σ_μ": perr[1], "σ_σ": perr[2],
                "R²": r_squared, "FWHM_fit": fwhm_fit
            }
        except Exception as e:
            return {"erreur": str(e)}

    # --- Diagnostic ---
    def diagnostiquer(self) -> list:
        """Analyse automatique de la gaussienne"""
        diag = []
        diag.append({
            "Paramètre": "Amplitude",
            "Valeur": f"{self.A:.4f}",
            "Statut": "✅ OK",
            "Note": "Amplitude positive"
        })
        diag.append({
            "Paramètre": "Écart-type σ",
            "Valeur": f"{self.sigma:.4f}",
            "Statut": "✅ OK" if self.sigma > 0 else "❌ Nul",
            "Note": f"{'Distribution étroite' if self.sigma < 1 else 'Large'}"
        })
        diag.append({
            "Paramètre": "FWHM",
            "Valeur": f"{self.fwhm:.4f}",
            "Statut": "✅ OK",
            "Note": f"Largeur à mi-hauteur"
        })
        diag.append({
            "Paramètre": "Entropie",
            "Valeur": f"{self.entropie:.4f}",
            "Statut": "✅ OK",
            "Note": "Mesure d'incertitude"
        })
        return diag


@st.cache_data(show_spinner=False)
def compute_decay(I0: float, gamma: float, t_max: float, n_points: int):
    t = np.linspace(0, t_max, n_points)
    return t, I0 * np.exp(-gamma * t)


@st.cache_data(show_spinner=False)
def solve_4niveaux(Rp: float, tau21: float, tau32: float,
                  tau10: float, t_max: float, n_points: int):
    t = np.linspace(0, t_max, n_points)

    def dydt(t, y):
        N0, N1, N2, N3 = y
        dN3 = Rp * N0 - N3 / tau32
        dN2 = N3 / tau32 - N2 / tau21
        dN1 = N2 / tau21 - N1 / tau10
        dN0 = N1 / tau10 - Rp * N0
        return [dN0, dN1, dN2, dN3]

    sol = solve_ivp(dydt, [t[0], t[-1]], [1e20, 0, 0, 0], t_eval=t,
                    method='RK45', rtol=1e-6, atol=1e-10)
    return {
        "t": sol.t,
        "N0": sol.y[0], "N1": sol.y[1],
        "N2": sol.y[2], "N3": sol.y[3],
        "inversion": sol.y[2] - sol.y[1],
    }


@st.cache_data(show_spinner=False)
def solve_relaxation(Rp_abs: float, tau_c: float, tau_sp: float,
                     N_th: float, t_max_osc: float, n_points: int = 2000):
    t = np.linspace(0, t_max_osc, n_points)

    def dydt(t, y):
        N, S = y
        dN = Rp_abs - N / tau_sp - N * S / tau_c
        dS = N * S / tau_c - S / tau_c + N / tau_sp
        return [dN, dS]

    sol = solve_ivp(dydt, [t[0], t[-1]], [N_th * 0.9, 1e3], t_eval=t,
                    method='RK45', rtol=1e-8, atol=1e-12,
                    max_step=(t[-1]-t[0]) / 2000)
    return {"t": sol.t, "N": sol.y[0], "S": sol.y[1]}


@st.cache_data(show_spinner=False)
def compute_spectral(lambda_center: float, delta_lambda: float,
                     n_modes: int, gain: float):
    lambda_arr = np.linspace(lambda_center - 5*delta_lambda,
                              lambda_center + 5*delta_lambda, 2000)
    gamma_l = delta_lambda / 2
    gain_profile = gain / (1 + ((lambda_arr - lambda_center)/gamma_l)**2)
    delta_mode = delta_lambda / max(n_modes, 1)
    modes_lambda = np.array([lambda_center + m * delta_mode
                             for m in range(-n_modes//2, n_modes//2 + 1)])
    modes_gain = gain / (1 + ((modes_lambda - lambda_center)/gamma_l)**2)
    return lambda_arr, gain_profile, modes_lambda, modes_gain


@st.cache_data(show_spinner=False)
def compute_beam_profile(w0: float, z: float, lambda_um: float):
    w0_m = w0 * 1e-3
    lambda_m = lambda_um * 1e-6
    zR = np.pi * w0_m**2 / lambda_m
    wz = w0_m * np.sqrt(1 + (z / zR)**2)
    x = np.linspace(-5*wz, 5*wz, 100)
    y = np.linspace(-5*wz, 5*wz, 100)
    X, Y = np.meshgrid(x, y)
    return x, y, np.exp(-2 * (X**2 + Y**2) / wz**2), zR, wz


@st.cache_data(show_spinner=False)
def compute_frantz_nodvik(g0: float, L: float, I_sat: float, I_max: float):
    I_in = np.linspace(0.01, I_max, 500)
    I_out = I_sat * np.log(1 + (np.exp(I_in / I_sat) - 1) * np.exp(g0 * L))
    return I_in, I_out, I_out / I_in


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def laser_page():
    st.markdown("## 💡 Simulation Laser & Optique Avancée")
    st.markdown("*Dynamique laser, pertes de cavité, profils gaussiens*")
    st.markdown("---")

    option = st.radio(
        "Options",
        [
            "💡 Simulation Laser",
            "🔲 Pertes de Cavité",
            "📊 Profil Gaussien",
        ],
        horizontal=True,
        key="laser_main_option"
    )

    # ============================================================
    # OPTION 1 : SIMULATION LASER
    # ============================================================
    if option == "💡 Simulation Laser":
        st.markdown("### 💡 Simulation Laser Avancée")
        st.markdown("*Dynamique laser, équations de taux, modes de cavité, amplification*")

        section = st.radio(
            "Section",
            [
                "📉 Décroissance & Taux",
                "🌊 Oscillations de relaxation",
                "📡 Spectre & Modes",
                "🔬 Profil TEM gaussien",
                "⚡ Amplification (F-N)",
                "📖 Théorie & Références",
            ],
            horizontal=True,
            key="laser_section"
        )

        # Original laser simulation content here
        if section == "📉 Décroissance & Taux":
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### ⚙️ Paramètres")

                laser_type = st.selectbox("Type de laser", ["Personnalisé"] + list(TYPES_LASER.keys()), key="laser_type_decay")
                if laser_type != "Personnalisé":
                    info_l = TYPES_LASER[laser_type]
                    st.info(f"λ = {info_l['λ']} nm | η = {info_l['η']*100:.0f}% | {info_l['P_typ']}")

                I0 = st.slider("Intensité initiale I₀", 0.1, 200.0, 10.0, 0.5, key="I0_decay")
                gamma = st.slider("Coefficient γ (s⁻¹)", 0.01, 5.0, 0.1, 0.01, key="gamma_decay")
                t_max = st.slider("Temps max (s)", 1.0, 100.0, 30.0, 1.0, key="tmax_decay")
                n_points = st.slider("Résolution", 200, 5000, 1000, 100, key="npts_decay")

                mode_affichage = st.radio("Affichage", ["Linéaire", "Semi-log"], horizontal=True, key="mode_decay")

                engine = LaserEngine(I0, gamma)

                st.markdown("### 📐 Résultats analytiques")
                st.metric("Demi-vie t₁/₂ (s)", f"{engine.demi_vie:.4f}")
                st.metric("Temps de vie τ (s)", f"{engine.temps_vie:.4f}")
                st.metric("I(t_max)", f"{I0 * np.exp(-gamma * t_max):.4e}")
                st.metric("I(τ) = I₀/e", f"{I0/np.e:.4f}")

                # Paramètres 4 niveaux
                st.markdown("### 🔬 Système 4 niveaux")
                show_4n = st.checkbox("Simuler système 4 niveaux", True, key="show_4n")
                if show_4n:
                    Rp = st.slider("Taux pompage Rp (s⁻¹)", 1e5, 1e8, 1e6,
                                   format="%.0e", step=1e5, key="Rp_4n")
                    tau21 = st.slider("τ₂₁ (μs)", 0.1, 500.0, 230.0, 1.0, key="tau21_4n")
                    tau32 = st.slider("τ₃₂ (ns)", 0.1, 100.0, 1.0, 0.1, key="tau32_4n")

            with col2:
                t, I = compute_decay(I0, gamma, t_max, n_points)

                fig = go.Figure()

                if mode_affichage == "Linéaire":
                    fig.add_trace(go.Scatter(
                        x=t, y=I, mode='lines', name='I(t)',
                        line=dict(color='#00ccff', width=3)
                    ))
                    fig.add_hline(y=I0/np.e, line_dash='dash', line_color='#ffcc00',
                                  annotation_text=f"I₀/e = {I0/np.e:.3f}")
                    fig.add_hline(y=I0/2, line_dash='dot', line_color='#7700ff',
                                  annotation_text=f"I₀/2 = {I0/2:.3f}")
                    fig.add_vline(x=engine.demi_vie, line_dash='dash', line_color='#ff00cc',
                                  annotation_text=f"t₁/₂={engine.demi_vie:.2f}s")
                    fig.add_vline(x=engine.temps_vie, line_dash='dot', line_color='#00ff88',
                                  annotation_text=f"τ={engine.temps_vie:.2f}s")
                else:
                    I_log = np.where(I > 0, I, 1e-12)
                    fig.add_trace(go.Scatter(
                        x=t, y=np.log10(I_log), mode='lines',
                        name='log₁₀(I(t))', line=dict(color='#00ccff', width=3)
                    ))
                    # Droite théorique
                    fig.add_trace(go.Scatter(
                        x=t, y=np.log10(I0) - gamma * t / np.log(10),
                        mode='lines', name='Droite théorique',
                        line=dict(color='#ffcc00', width=2, dash='dash')
                    ))

                fig.update_layout(
                    title="Décroissance laser I(t) = I₀·e^{-γt}",
                    xaxis_title="Temps (s)",
                    yaxis_title="Intensité" if mode_affichage == "Linéaire" else "log₁₀(I)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Système 4 niveaux
                if show_4n:
                    sol4 = solve_4niveaux(
                        Rp=Rp,
                        tau21=tau21*1e-6,
                        tau32=tau32*1e-9,
                        tau10=tau32*1e-9*0.1,
                        t_max=t_max * 1e-6,
                        n_points=2000,
                    )
                    fig4 = go.Figure()
                    colors4 = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88']
                    for i, (k, v) in enumerate([("N₀", sol4["N0"]), ("N₁", sol4["N1"]),
                                                 ("N₂", sol4["N2"]), ("N₃", sol4["N3"])]):
                        fig4.add_trace(go.Scatter(
                            x=sol4["t"]*1e6, y=v, mode='lines', name=k,
                            line=dict(color=colors4[i], width=2)
                        ))
                    fig4.add_trace(go.Scatter(
                        x=sol4["t"]*1e6, y=sol4["inversion"],
                        mode='lines', name='ΔN (inversion)',
                        line=dict(color='#ffffff', width=3, dash='dash')
                    ))
                    fig4.update_layout(
                        title="Populations système 4 niveaux",
                        xaxis_title="Temps (μs)", yaxis_title="Population (m⁻³)",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.92)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                        yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                        height=380,
                    )
                    st.plotly_chart(fig4, use_container_width=True)

                # Export
                df_exp = pd.DataFrame({"temps_s": t, "intensite": I})
                st.download_button("💾 Export CSV",
                                   df_exp.to_csv(index=False).encode(),
                                   "laser_simulation.csv", "text/csv", key="export_decay")

        elif section == "🌊 Oscillations de relaxation":
            st.markdown("### 🌊 Oscillations de relaxation laser")
            st.markdown("*Dynamique couplée photons ↔ population autour du seuil*")

            col1, col2 = st.columns([1, 2])
            with col1:
                Rp_osc = st.slider("Taux de pompe Rp (×Rp_th)", 1.01, 5.0, 1.5, 0.01, key="Rp_osc")
                tau_c_osc = st.slider("Durée de vie photon τ_c (ns)", 0.1, 100.0, 10.0, 0.1, key="tau_c_osc")
                tau_sp_osc = st.slider("Durée de vie spontanée τ_sp (μs)", 0.1, 1000.0, 230.0, 1.0, key="tau_sp_osc")
                t_max_osc = st.slider("Durée simulation (μs)", 1.0, 500.0, 50.0, 1.0, key="tmax_osc")

                tau_c = tau_c_osc * 1e-9
                tau_sp = tau_sp_osc * 1e-6
                N_th = 1.0 / (tau_c)
                Rp_abs = Rp_osc * N_th / tau_sp

                # Fréquence théorique
                omega_r = 0.0
                if Rp_abs * tau_sp > 1:
                    omega_r = np.sqrt((Rp_abs * tau_sp - 1) / (tau_c * tau_sp))
                f_osc = omega_r / (2 * np.pi) if omega_r > 0 else 0.0

                st.metric("Fréquence oscillations (MHz)",
                          f"{f_osc/1e6:.3f}" if f_osc > 0 else "N/A")
                st.metric("Période (ns)",
                          f"{1/f_osc*1e9:.1f}" if f_osc > 0 else "N/A")

            with col2:
                sol_osc = solve_relaxation(
                    Rp_abs=Rp_abs,
                    tau_c=tau_c,
                    tau_sp=tau_sp,
                    N_th=N_th,
                    t_max_osc=t_max_osc * 1e-6,
                    n_points=2000,
                )

                fig_osc = make_subplots(rows=2, cols=1,
                    subplot_titles=["Densité photons S(t)", "Population N(t)"])

                fig_osc.add_trace(go.Scatter(
                    x=sol_osc["t"]*1e6, y=sol_osc["S"], mode='lines',
                    name='S(t)', line=dict(color='#00ccff', width=2.5)
                ), row=1, col=1)
                fig_osc.add_trace(go.Scatter(
                    x=sol_osc["t"]*1e6, y=sol_osc["N"]/N_th, mode='lines',
                    name='N(t)/N_th', line=dict(color='#7700ff', width=2.5)
                ), row=2, col=1)

                fig_osc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    height=480,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_osc.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                      title_text="Temps (μs)")
                fig_osc.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                st.plotly_chart(fig_osc, use_container_width=True)

        elif section == "📡 Spectre & Modes":
            st.markdown("### 📡 Spectre d'émission et modes de cavité")

            col1, col2 = st.columns([1, 2])
            with col1:
                laser_sel = st.selectbox("Laser", list(TYPES_LASER.keys()), key="laser_spec")
                info_spec = TYPES_LASER[laser_sel]
                lambda_c = info_spec["λ"]

                delta_lambda = st.slider("Largeur spectrale Δλ (nm)", 0.001, 50.0,
                                          0.1 if "Nd" in laser_sel else 1.0, 0.001, key="delta_lambda")
                n_modes = st.slider("Nombre de modes longitudinaux", 1, 20, 5, key="n_modes")
                gain_val = st.slider("Gain g₀", 0.1, 10.0, 1.0, 0.1, key="gain_val")

                L_cav = st.slider("Longueur de cavité L (cm)", 1, 200, 30, key="L_cav")
                n_ref = st.slider("Indice de réfraction n", 1.0, 3.5, 1.0, 0.01, key="n_ref")

                delta_nu = 3e8 / (2 * n_ref * L_cav * 1e-2)
                st.metric("Espacement modes (MHz)", f"{delta_nu/1e6:.1f}")
                st.metric("Finesse théorique", f"{np.pi*0.99/(1-0.99):.0f}")

            with col2:
                lambda_arr, gain_prof, modes_l, modes_g = compute_spectral(
                    lambda_center=lambda_c,
                    delta_lambda=delta_lambda,
                    n_modes=n_modes,
                    gain=gain_val,
                )

                fig_spec = go.Figure()
                fig_spec.add_trace(go.Scatter(
                    x=lambda_arr, y=gain_prof, mode='lines',
                    name='Profil de gain', line=dict(color='rgba(0,204,255,0.5)', width=2)
                ))

                colors_modes = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88',
                                '#ffcc00', '#ff4444', '#44ff88']
                for i, (lm, gm) in enumerate(zip(modes_l, modes_g)):
                    fig_spec.add_trace(go.Scatter(
                        x=[lm, lm], y=[0, gm], mode='lines',
                        name=f'Mode {i+1}',
                        line=dict(color=colors_modes[i % len(colors_modes)], width=3)
                    ))

                fig_spec.update_layout(
                    title=f"Spectre laser — {laser_sel} (λ={lambda_c} nm)",
                    xaxis_title="Longueur d'onde (nm)", yaxis_title="Gain (u.a.)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                              range=[0, gain_val * 1.1]),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=450,
                )
                st.plotly_chart(fig_spec, use_container_width=True)

        elif section == "🔬 Profil TEM gaussien":
            st.markdown("### 🔬 Profil de faisceau TEM₀₀ gaussien")

            col1, col2 = st.columns([1, 2])
            with col1:
                w0_beam = st.slider("Waist w₀ (mm)", 0.1, 10.0, 1.0, 0.1, key="w0_beam")
                z_beam = st.slider("Distance z (m)", 0.0, 10.0, 0.0, 0.1, key="z_beam")
                lambda_beam = st.slider("Longueur d'onde (μm)", 0.4, 11.0, 1.064, 0.001, key="lambda_beam")

                zR = np.pi * (w0_beam*1e-3)**2 / (lambda_beam*1e-6)
                wz = w0_beam * np.sqrt(1 + (z_beam/zR)**2)
                Rz = z_beam * (1 + (zR/z_beam)**2) if z_beam > 0 else np.inf

                st.metric("Longueur Rayleigh z_R (m)", f"{zR:.3f}")
                st.metric("Waist à z: w(z) (mm)", f"{wz:.3f}")
                st.metric("Divergence θ (mrad)", f"{lambda_beam*1e-6/(np.pi*w0_beam*1e-3)*1e3:.3f}")

            with col2:
                x_b, y_b, Z_beam, zR, wz = compute_beam_profile(
                    w0=w0_beam,
                    z=z_beam,
                    lambda_um=lambda_beam,
                )

                fig_beam = go.Figure(data=[go.Surface(
                    z=Z_beam, x=x_b*1e3, y=y_b*1e3,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                    lighting=dict(ambient=0.5, diffuse=0.8),
                )])
                fig_beam.update_layout(
                    title=f"Profil TEM₀₀ — z={z_beam} m, w(z)={wz:.2f} mm",
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.8)',
                        xaxis=dict(color='#c0d0ff', title='x (mm)'),
                        yaxis=dict(color='#c0d0ff', title='y (mm)'),
                        zaxis=dict(color='#c0d0ff', title='I (u.a.)'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c0d0ff'),
                    height=480,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_beam, use_container_width=True)

                # Propagation du waist
                z_prop = np.linspace(0, 10*zR, 500)
                wz_prop = w0_beam * np.sqrt(1 + (z_prop/zR)**2)
                fig_prop = go.Figure()
                fig_prop.add_trace(go.Scatter(
                    x=z_prop, y=wz_prop, mode='lines', name='+w(z)',
                    line=dict(color='#00ccff', width=2.5)
                ))
                fig_prop.add_trace(go.Scatter(
                    x=z_prop, y=-wz_prop, mode='lines', name='-w(z)',
                    line=dict(color='#00ccff', width=2.5)
                ))
                fig_prop.add_vline(x=zR, line_color='#ffcc00', line_dash='dash',
                                   annotation_text=f"z_R={zR:.2f}m")
                fig_prop.update_layout(
                    title="Propagation du faisceau gaussien",
                    xaxis_title="z (m)", yaxis_title="w(z) (mm)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=300,
                )
                st.plotly_chart(fig_prop, use_container_width=True)

        elif section == "⚡ Amplification (F-N)":
            st.markdown("### ⚡ Amplificateur laser — Frantz-Nodvik")

            col1, col2 = st.columns([1, 2])
            with col1:
                g0_fn = st.slider("Gain petit signal g₀ (cm⁻¹)", 0.01, 2.0, 0.5, 0.01, key="g0_fn")
                L_fn = st.slider("Longueur amplificateur L (cm)", 1.0, 50.0, 10.0, 0.5, key="L_fn")
                I_sat_fn = st.slider("Intensité saturation I_sat (W/cm²)", 0.1, 1000.0, 100.0, 1.0, key="I_sat_fn")
                I_max = st.slider("I_in max (W/cm²)", 1.0, 5000.0, 500.0, 10.0, key="I_max_fn")

                G_lin = np.exp(g0_fn * L_fn)
                st.metric("Gain linéaire G = e^(g₀L)", f"{G_lin:.2f}")
                st.metric("Gain (dB)", f"{10*np.log10(G_lin):.2f}")

            with col2:
                I_in, I_out, G_eff = compute_frantz_nodvik(
                    g0=g0_fn,
                    L=L_fn,
                    I_sat=I_sat_fn,
                    I_max=I_max,
                )

                fig_fn = make_subplots(rows=2, cols=1,
                    subplot_titles=["I_out vs I_in", "Gain effectif G(I_in)"])

                fig_fn.add_trace(go.Scatter(
                    x=I_in, y=I_out, mode='lines', name='I_out (F-N)',
                    line=dict(color='#00ccff', width=3)
                ), row=1, col=1)
                fig_fn.add_trace(go.Scatter(
                    x=I_in, y=G_lin * I_in, mode='lines', name='I_out (linéaire)',
                    line=dict(color='rgba(255,200,0,0.5)', width=2, dash='dash')
                ), row=1, col=1)
                fig_fn.add_vline(x=I_sat_fn, line_color='#ff00cc', line_dash='dot',
                                 annotation_text="I_sat", row=1, col=1)

                fig_fn.add_trace(go.Scatter(
                    x=I_in, y=G_eff, mode='lines', name='G effectif',
                    line=dict(color='#7700ff', width=2.5)
                ), row=2, col=1)
                fig_fn.add_hline(y=G_lin, line_color='#ffcc00', line_dash='dash',
                                 annotation_text=f"G₀={G_lin:.1f}", row=2, col=1)

                fig_fn.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_fn.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                     title_text="I_in (W/cm²)")
                fig_fn.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                st.plotly_chart(fig_fn, use_container_width=True)

        elif section == "📖 Théorie & Références":
            st.markdown("### 📖 Formulaire scientifique laser")
            cols = st.columns(2)
            col_idx = 0

            for nom, formule in FORMULES.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(formule)
                col_idx += 1

            st.markdown("---")
            st.markdown("### 🔬 Types de lasers")
            df_lasers = pd.DataFrame([
                {"Laser": k, "λ (nm)": v["λ"], "Type": v["type"],
                 "η (%)": f"{v['η']*100:.0f}", "τ": str(v["τ"]), "Puissance": v["P_typ"]}
                for k, v in TYPES_LASER.items()
            ])
            st.dataframe(df_lasers, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📐 Diagnostic laser")
            engine_diag = LaserEngine(10.0, 0.1)
            diag = engine_diag.diagnostiquer(10.0, 0.1, 30.0)
            st.dataframe(pd.DataFrame(diag), use_container_width=True)

            st.markdown("---")
            st.markdown("### 📚 Références")
            refs = [
                "Saleh & Teich — *Fundamentals of Photonics* (Wiley, 2007)",
                "Svelto — *Principles of Lasers* (Springer, 2010)",
                "Siegman — *Lasers* (University Science Books, 1986)",
                "Yariv — *Quantum Electronics* (Wiley, 1989)",
            ]
            for r in refs:
                st.markdown(f"- {r}")

    # ============================================================
    # OPTION 2 : PERTES DE CAVITÉ
    # ============================================================
    elif option == "🔲 Pertes de Cavité":
        st.markdown("## 🔲 Pertes de Cavité Optique Avancées")
        st.markdown("*Beer-Lambert, cavité laser, finesse, optimisation, spectre*")
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📉 Propagation",
            "⚖️ Cavité Laser",
            "🌡️ Carte de pertes",
            "🎯 Optimisation R₂",
            "⚗️ Diagnostic",
            "📖 Théorie"
        ])

        # Config partagée
        with st.sidebar.expander("⚙️ Matériau & paramètres", expanded=True):
            mat = st.selectbox("Matériau optique", list(MATERIAUX_OPTIQUES.keys()), key="mat_cavity")
            mat_info = MATERIAUX_OPTIQUES[mat]

            if mat == "Personnalisé":
                alpha_base = st.slider("α interne (cm⁻¹)", 0.001, 2.0, 0.1, 0.001, key="alpha_base_cav")
                n_ref = st.slider("Indice n", 1.0, 5.0, 1.5, 0.01, key="n_ref_cav")
            else:
                alpha_base = mat_info["alpha"]
                n_ref = mat_info["n"]
                st.info(f"α = {alpha_base} cm⁻¹ | n = {n_ref}")

            I0 = st.slider("Intensité initiale I₀", 0.1, 200.0, 10.0, 0.5, key="I0_cav")
            L_max = st.slider("Distance max (cm)", 1.0, 200.0, 50.0, 1.0, key="L_max_cav")

        engine = CavityLossEngine(I0, alpha_base)

        # TAB 1 : PROPAGATION
        with tab1:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### ⚙️ Paramètres")
                alpha_var = st.slider("α (cm⁻¹)", 0.001, 2.0, alpha_base, 0.001, key="alpha_var")
                eng1 = CavityLossEngine(I0, alpha_var)

                mode_y = st.radio("Échelle", ["Linéaire", "Logarithmique"], horizontal=True, key="mode_y_cav")
                show_markers = st.checkbox("Marqueurs caractéristiques", True, key="show_markers_cav")

                st.markdown("### 📐 Distances caractéristiques")
                st.metric("Distance 1/e (cm)", f"{eng1.distance_1_e:.3f}")
                st.metric("Distance 1/2 (cm)", f"{eng1.distance_demi:.3f}")
                st.metric("I à L_max", f"{eng1.intensite(np.array([L_max]))[0]:.4f}")
                perte_pct = (1 - eng1.transmittance(L_max)) * 100
                st.metric("Perte totale (%)", f"{perte_pct:.2f}")
                st.metric("Transmittance", f"{eng1.transmittance(L_max):.4f}")
                st.metric("Absorbance", f"{eng1.absorbance(L_max):.4f}")

            with col2:
                x = np.linspace(0, L_max, 2000)
                I = eng1.intensite(x)

                fig = go.Figure()

                if mode_y == "Logarithmique":
                    y_plot = np.log10(np.maximum(I, 1e-12))
                    y_theo = np.log10(I0) - alpha_var * x / np.log(10)
                    fig.add_trace(go.Scatter(
                        x=x, y=y_theo, mode='lines', name='Droite théorique',
                        line=dict(color='#ffcc00', width=2, dash='dash')
                    ))
                else:
                    y_plot = I

                fig.add_trace(go.Scatter(
                    x=x, y=y_plot, mode='lines',
                    name=f'I(x) α={alpha_var:.3f}',
                    line=dict(color='#00ccff', width=3),
                    fill='tozeroy' if mode_y == "Linéaire" else 'none',
                    fillcolor='rgba(0,204,255,0.1)'
                ))

                if show_markers and mode_y == "Linéaire":
                    # 1/e
                    L_1e = eng1.distance_1_e
                    if L_1e < L_max:
                        fig.add_vline(x=L_1e, line_color='#00ff88', line_dash='dash',
                                      annotation_text=f"L_1/e={L_1e:.2f}cm")
                        fig.add_hline(y=I0/np.e, line_color='rgba(0,255,136,0.4)',
                                      line_dash='dot', annotation_text="I₀/e")
                    # 1/2
                    L_12 = eng1.distance_demi
                    if L_12 < L_max:
                        fig.add_vline(x=L_12, line_color='#7700ff', line_dash='dash',
                                      annotation_text=f"L_1/2={L_12:.2f}cm")
                        fig.add_hline(y=I0/2, line_color='rgba(119,0,255,0.4)',
                                      line_dash='dot', annotation_text="I₀/2")

                fig.update_layout(
                    title=f"Propagation Beer-Lambert — {mat}",
                    xaxis_title="Distance x (cm)",
                    yaxis_title="Intensité I(x)" if mode_y == "Linéaire" else "log₁₀(I(x))",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Comparaison multi-alpha
                st.markdown("#### 📊 Comparaison de coefficients α")
                alphas_comp = st.multiselect("Coefficients à comparer",
                    [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                    default=[0.05, 0.1, 0.2], key="alphas_comp")

                if alphas_comp:
                    colors_c = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00','#ff4400','#ffffff']
                    fig_cmp = go.Figure()
                    for i, ac in enumerate(sorted(alphas_comp)):
                        ec = CavityLossEngine(I0, ac)
                        Ic = ec.intensite(x)
                        y_c = Ic if mode_y == "Linéaire" else np.log10(np.maximum(Ic, 1e-12))
                        fig_cmp.add_trace(go.Scatter(
                            x=x, y=y_c, mode='lines', name=f'α={ac}',
                            line=dict(color=colors_c[i % len(colors_c)], width=2)
                        ))
                    fig_cmp.update_layout(
                        title="Comparaison multi-α",
                        xaxis_title="Distance x (cm)",
                        yaxis_title="I(x)" if mode_y == "Linéaire" else "log₁₀(I)",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.92)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                        yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                        height=360,
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                df_exp = pd.DataFrame({"x_cm": x, "I": I, "T": I/I0})
                st.download_button("💾 Export CSV",
                                   df_exp.to_csv(index=False).encode(),
                                   "cavity_losses.csv", "text/csv", key="export_cavity")

        # TAB 2 : CAVITÉ LASER
        with tab2:
            st.markdown("### ⚖️ Analyse de cavité laser")
            col1, col2 = st.columns([1, 2])

            with col1:
                R1 = st.slider("Réflectivité R₁ (%)", 50.0, 99.9, 99.5, 0.1, key="R1_cav") / 100
                R2 = st.slider("Réflectivité R₂ (%)", 10.0, 99.9, 70.0, 0.1, key="R2_cav") / 100
                L_cav = st.slider("Longueur cavité L (cm)", 1.0, 100.0, 20.0, 0.5, key="L_cav_cav")
                alpha_int = st.slider("Pertes internes α_int (cm⁻¹)", 0.001, 0.5, alpha_base, 0.001, key="alpha_int_cav")
                nu0 = st.slider("Fréquence ν₀ (THz)", 100.0, 600.0, 282.0, 1.0, key="nu0_cav")

                eng_cav = CavityLossEngine(I0, alpha_int)

                alpha_mir = eng_cav.pertes_miroirs(R1, R2, L_cav)
                alpha_tot = eng_cav.pertes_totales(R1, R2, L_cav)
                fin = eng_cav.finesse(R1, R2, alpha_int, L_cav)

                st.markdown("### 📐 Résultats")
                st.metric("Pertes miroirs (cm⁻¹)", f"{alpha_mir:.4f}")
                st.metric("Pertes totales (cm⁻¹)", f"{alpha_tot:.4f}")
                st.metric("Finesse", f"{fin:.1f}")
                st.metric("Gain seuil g_th (cm⁻¹)", f"{alpha_tot:.4f}")

                c = 3e10
                tau_p = n_ref * L_cav / (c * alpha_tot * L_cav)
                st.metric("Temps vie photon τ_p (ns)", f"{tau_p*1e9:.3f}")
                delta_nu = 1 / (2 * np.pi * tau_p)
                st.metric("Largeur de raie Δν (MHz)", f"{delta_nu/1e6:.2f}")

            with col2:
                # Profil R1 et R2 vs pertes
                R1_arr = np.linspace(0.5, 0.999, 200)
                R2_arr = np.linspace(0.5, 0.999, 200)
                R1g, R2g = np.meshgrid(R1_arr, R2_arr)
                alpha_m_map = -np.log(R1g * R2g) / (2 * L_cav)
                alpha_tot_map = alpha_m_map + alpha_int

                fig_cav = go.Figure(data=go.Heatmap(
                    z=alpha_tot_map,
                    x=R1_arr * 100, y=R2_arr * 100,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    colorbar=dict(title='α_tot (cm⁻¹)', tickfont=dict(color='#c0d0ff'))
                ))
                fig_cav.add_trace(go.Scatter(
                    x=[R1*100], y=[R2*100], mode='markers', name='Opération',
                    marker=dict(color='#ff0000', size=14, symbol='star',
                               line=dict(width=2, color='#ffffff'))
                ))
                fig_cav.update_layout(
                    title="Pertes totales α_tot(R₁, R₂)",
                    xaxis_title="R₁ (%)", yaxis_title="R₂ (%)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff'),
                    yaxis=dict(color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=430,
                )
                st.plotly_chart(fig_cav, use_container_width=True)

                # Finesse vs R
                R_arr = np.linspace(0.1, 0.999, 500)
                fin_arr = np.pi * np.sqrt(R_arr) / (1 - R_arr)
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Scatter(
                    x=R_arr*100, y=fin_arr, mode='lines', name='Finesse',
                    line=dict(color='#00ccff', width=3)
                ))
                fig_fin.add_vline(x=R2*100, line_color='#ff00cc', line_dash='dash',
                                  annotation_text=f"R₂={R2*100:.1f}%")
                fig_fin.update_layout(
                    title="Finesse vs Réflectivité R",
                    xaxis_title="R (%)", yaxis_title="Finesse",
                    yaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=320,
                )
                st.plotly_chart(fig_fin, use_container_width=True)

        # TAB 3 : CARTE DE PERTES
        with tab3:
            st.markdown("### 🌡️ Cartographie des pertes")
            col1, col2 = st.columns([1, 2])

            with col1:
                L_carte = st.slider("Longueur cavité L (cm)", 1.0, 100.0, 20.0, key="L_carte")
                alpha_c = st.slider("α_int (cm⁻¹)", 0.001, 0.5, alpha_base, 0.001, key="alpha_c")
                vue = st.radio("Vue", ["R₁ × R₂", "α × L"], horizontal=True, key="vue_carte")

            with col2:
                if vue == "R₁ × R₂":
                    r1a = np.linspace(0.5, 0.999, 60)
                    r2a = np.linspace(0.5, 0.999, 60)
                    Rg1, Rg2 = np.meshgrid(r1a, r2a)
                    Z = -np.log(Rg1 * Rg2) / (2 * L_carte) + alpha_c

                    fig_c = go.Figure(data=[go.Surface(
                        z=Z, x=r1a*100, y=r2a*100,
                        colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                        showscale=True,
                    )])
                    fig_c.update_layout(
                        scene=dict(
                            bgcolor='rgba(5,0,20,0.9)',
                            xaxis=dict(color='#c0d0ff', title='R₁ (%)'),
                            yaxis=dict(color='#c0d0ff', title='R₂ (%)'),
                            zaxis=dict(color='#c0d0ff', title='α_tot (cm⁻¹)'),
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#c0d0ff'),
                        height=480, margin=dict(l=0,r=0,t=40,b=0)
                    )
                else:
                    al_arr = np.linspace(0.001, 1.0, 60)
                    Lar = np.linspace(1, 100, 60)
                    Ag, Lg = np.meshgrid(al_arr, Lar)
                    T_map = np.exp(-Ag * Lg)

                    fig_c = go.Figure(data=[go.Surface(
                        z=T_map, x=al_arr, y=Lar,
                        colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                        showscale=True,
                    )])
                    fig_c.update_layout(
                        scene=dict(
                            bgcolor='rgba(5,0,20,0.9)',
                            xaxis=dict(color='#c0d0ff', title='α (cm⁻¹)'),
                            yaxis=dict(color='#c0d0ff', title='L (cm)'),
                            zaxis=dict(color='#c0d0ff', title='Transmittance T'),
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#c0d0ff'),
                        height=480, margin=dict(l=0,r=0,t=40,b=0)
                    )
                st.plotly_chart(fig_c, use_container_width=True)

        # TAB 4 : OPTIMISATION R₂
        with tab4:
            st.markdown("### 🎯 Optimisation de R₂ pour puissance max")
            col1, col2 = st.columns([1, 2])

            with col1:
                R1_opt = st.slider("R₁ (%)", 50.0, 99.9, 99.0, 0.1, key="R1_opt") / 100
                L_opt  = st.slider("L (cm)", 1.0, 100.0, 20.0, 0.5, key="L_opt")
                g0_opt = st.slider("Gain g₀ (cm⁻¹)", 0.01, 2.0, 0.5, 0.01, key="g0_opt")
                ai_opt = st.slider("α_int (cm⁻¹)", 0.001, 0.5, alpha_base, 0.001, key="ai_opt")

                eng_opt = CavityLossEngine(I0, ai_opt)
                res_opt = eng_opt.optimiser_R2(R1_opt, L_opt, g0_opt, ai_opt)

                st.metric("R₂ optimal (%)", f"{res_opt['R2_opt']*100:.2f}")
                st.metric("Puissance max (u.a.)", f"{res_opt['P_max']:.4f}")
                st.metric("g₀/α_tot (seuil)",
                          f"{g0_opt / eng_opt.pertes_totales(R1_opt, res_opt['R2_opt'], L_opt):.3f}")

            with col2:
                fig_opt = go.Figure()
                fig_opt.add_trace(go.Scatter(
                    x=res_opt['R2_range']*100, y=res_opt['P_range'],
                    mode='lines', name='P_out(R₂)',
                    line=dict(color='#00ccff', width=3)
                ))
                fig_opt.add_vline(x=res_opt['R2_opt']*100,
                                  line_color='#ffcc00', line_dash='dash',
                                  annotation_text=f"R₂*={res_opt['R2_opt']*100:.1f}%")
                fig_opt.add_trace(go.Scatter(
                    x=[res_opt['R2_opt']*100], y=[res_opt['P_max']],
                    mode='markers', name='Optimum',
                    marker=dict(color='#00ff88', size=14, symbol='star')
                ))
                fig_opt.update_layout(
                    title="Puissance de sortie vs R₂",
                    xaxis_title="R₂ (%)", yaxis_title="P_out (u.a.)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=430,
                )
                st.plotly_chart(fig_opt, use_container_width=True)

        # TAB 5 : DIAGNOSTIC
        with tab5:
            st.markdown("### ⚗️ Diagnostic automatique")

            R1_d = st.slider("R₁ (%)", 50.0, 99.9, 99.0, 0.1, key="R1_d") / 100
            R2_d = st.slider("R₂ (%)", 10.0, 99.9, 70.0, 0.1, key="R2_d") / 100
            L_d  = st.slider("L (cm)", 1.0, 100.0, 20.0, 0.5, key="L_d")

            eng_d = CavityLossEngine(I0, alpha_base)
            diag = eng_d.diagnostiquer(R1_d, R2_d, L_d)
            st.dataframe(pd.DataFrame(diag), use_container_width=True)

            st.markdown("#### 📋 Tableau d'erreurs optiques")
            erreurs = {
                "Problème": ["α trop élevé", "Finesse faible", "R₂ non optimisé",
                             "Instabilité transverse", "Pertes diffusion"],
                "Cause": ["Absorption matériau", "Réflectivité < 50%",
                          "R₂ ≠ R₂_opt", "Rayon de courbure", "Rugosité surface"],
                "Symptôme": ["I décroît vite", "Mauvaise sélectivité",
                             "Puissance sous-optimale", "Faisceau déformé", "Pertes excess"],
                "Solution": ["Changer matériau", "Miroirs HR", "Optimiser R₂",
                             "Ajuster géométrie", "Polir les miroirs"]
            }
            st.dataframe(pd.DataFrame(erreurs), use_container_width=True)

        # TAB 6 : THÉORIE
        with tab6:
            st.markdown("### 📖 Formulaire scientifique")
            cols = st.columns(2)
            col_idx = 0

            for nom, formule in FORMULES.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(formule)
                col_idx += 1

            st.markdown("---")
            st.markdown("### 🔬 Matériaux optiques")
            df_mat = pd.DataFrame([
                {"Matériau": k, "α (cm⁻¹)": v["alpha"], "n": v["n"]}
                for k, v in MATERIAUX_OPTIQUES.items() if k != "Personnalisé"
            ])
            st.dataframe(df_mat, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📚 Références")
            for r in [
                "Saleh & Teich — *Fundamentals of Photonics* (Wiley, 2007)",
                "Svelto — *Principles of Lasers* (Springer, 2010)",
                "Yariv — *Optical Electronics in Modern Communications* (Oxford, 2006)",
            ]:
                st.markdown(f"- {r}")

    # ============================================================
    # OPTION 3 : PROFIL GAUSSIEN
    # ============================================================
    elif option == "📊 Profil Gaussien":
        st.markdown("## 📊 Profil Gaussien Avancé")
        st.markdown("*Modélisation, analyse multi-gaussienne, FFT, convolution, ajustement*")
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Gaussienne 1D",
            "🌐 Gaussienne 2D",
            "🔄 Multi-gaussiennes",
            "📡 Spectre & Convolution",
            "🔬 Ajustement données",
            "📖 Théorie"
        ])

        # TAB 1 : GAUSSIENNE 1D
        with tab1:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### ⚙️ Paramètres")
                amplitude = st.slider("Amplitude A", 0.1, 10.0, 1.0, 0.1, key="amplitude_gauss")
                sigma     = st.slider("Écart-type σ", 0.1, 5.0, 1.0, 0.05, key="sigma_gauss")
                mu        = st.slider("Centre μ", -10.0, 10.0, 0.0, 0.1, key="mu_gauss")
                x_range   = st.slider("Plage [-R, R]", 5.0, 30.0, 10.0, 1.0, key="x_range_gauss")
                show_fill = st.checkbox("Remplissage", True, key="show_fill_gauss")
                show_norm = st.checkbox("Superposer normalisée", False, key="show_norm_gauss")
                show_ic   = st.checkbox("Intervalle de confiance 95%", True, key="show_ic_gauss")
                mode_y    = st.radio("Échelle Y", ["Linéaire", "Log"], horizontal=True, key="mode_y_gauss")

                try:
                    engine = GaussianEngine(amplitude, sigma, mu)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                st.markdown("### 📐 Propriétés analytiques")
                st.metric("FWHM",  f"{engine.fwhm:.4f}")
                st.metric("Aire",  f"{engine.aire:.4f}")
                st.metric("Entropie", f"{engine.entropie:.4f}")

                ic_lo, ic_hi = engine.intervalle_confiance(0.95)
                st.metric("IC 95%", f"[{ic_lo:.3f}, {ic_hi:.3f}]")

            with col2:
                x = np.linspace(-x_range + mu, x_range + mu, 2000)
                y = engine.eval_1d(x)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y if mode_y == "Linéaire" else np.log10(np.maximum(y, 1e-12)),
                    mode='lines', name='f(x)',
                    line=dict(color='#00ccff', width=3),
                    fill='tozeroy' if show_fill else 'none',
                    fillcolor='rgba(0,204,255,0.15)'
                ))

                if show_norm:
                    y_norm = engine.eval_1d_normalise(x)
                    fig.add_trace(go.Scatter(
                        x=x, y=y_norm if mode_y == "Linéaire" else np.log10(np.maximum(y_norm, 1e-12)),
                        mode='lines', name='Normalisée',
                        line=dict(color='#7700ff', width=2, dash='dash')
                    ))

                if show_ic:
                    fig.add_vrect(x0=ic_lo, x1=ic_hi,
                                  fillcolor='rgba(0,255,136,0.08)',
                                  line_color='rgba(0,255,136,0.4)',
                                  annotation_text="IC 95%")

                # FWHM
                y_half = amplitude / 2
                x_fwhm = sigma * np.sqrt(2 * np.log(2))
                fig.add_shape(type='line',
                    x0=mu - x_fwhm, x1=mu + x_fwhm,
                    y0=y_half, y1=y_half,
                    line=dict(color='#ffcc00', width=2, dash='dot'))
                fig.add_annotation(x=mu, y=y_half,
                    text=f"FWHM={engine.fwhm:.3f}",
                    font=dict(color='#ffcc00'), showarrow=False, yshift=10)

                # Centre μ
                fig.add_vline(x=mu, line_color='rgba(255,255,255,0.4)',
                              line_dash='dash', annotation_text=f"μ={mu:.2f}")

                fig.update_layout(
                    title=f"Gaussienne — A={amplitude}, σ={sigma}, μ={mu}",
                    xaxis_title="x",
                    yaxis_title="f(x)" if mode_y == "Linéaire" else "log₁₀(f(x))",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=480,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Moments
                st.markdown("#### 📊 Moments statistiques")
                moments = engine.moments(6)
                cols_m = st.columns(3)
                for i, (k, v) in enumerate(moments.items()):
                    with cols_m[i % 3]:
                        st.metric(k, f"{v:.4e}" if abs(v) > 1000 else f"{v:.4f}")

                # Export
                df_exp = pd.DataFrame({"x": x, "f(x)": y})
                st.download_button("💾 Export CSV",
                                   df_exp.to_csv(index=False).encode(),
                                   "gaussienne_1d.csv", "text/csv", key="export_gauss")

        # TAB 2 : GAUSSIENNE 2D
        with tab2:
            st.markdown("### 🌐 Gaussienne 2D")
            col1, col2 = st.columns([1, 2])

            with col1:
                amp_2d   = st.slider("Amplitude A", 0.1, 10.0, 1.0, 0.1, key="amp_2d")
                sigma_x  = st.slider("σ_x", 0.1, 5.0, 1.0, 0.1, key="sigma_x")
                sigma_y  = st.slider("σ_y", 0.1, 5.0, 1.5, 0.1, key="sigma_y")
                mu_x     = st.slider("μ_x", -5.0, 5.0, 0.0, 0.1, key="mu_x")
                mu_y     = st.slider("μ_y", -5.0, 5.0, 0.0, 0.1, key="mu_y")
                mode_2d  = st.radio("Vue", ["Surface 3D", "Contour 2D", "Les deux"], horizontal=True, key="mode_2d")

            with col2:
                x2 = np.linspace(-8, 8, 120)
                y2 = np.linspace(-8, 8, 120)
                X2, Y2 = np.meshgrid(x2, y2)
                eng2 = GaussianEngine(amp_2d, sigma_x, mu_x)
                Z2 = eng2.eval_2d(X2, Y2, mu_y=mu_y, sigma_y=sigma_y)

                if mode_2d in ["Surface 3D", "Les deux"]:
                    fig2 = go.Figure(data=[go.Surface(
                        z=Z2, x=x2, y=y2,
                        colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                        showscale=True,
                        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.6),
                    )])
                    fig2.update_layout(
                        title="Gaussienne 2D — Surface",
                        scene=dict(
                            bgcolor='rgba(5,0,20,0.8)',
                            xaxis=dict(color='#c0d0ff', title='x'),
                            yaxis=dict(color='#c0d0ff', title='y'),
                            zaxis=dict(color='#c0d0ff', title='f(x,y)'),
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#c0d0ff'),
                        height=480,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                if mode_2d in ["Contour 2D", "Les deux"]:
                    fig_ct = go.Figure(data=[go.Contour(
                        z=Z2, x=x2, y=y2,
                        colorscale=[[0,'#020817'],[0.4,'#7700ff'],[0.7,'#00ccff'],[1,'#ffffff']],
                        contours=dict(coloring='heatmap', showlabels=True,
                                     labelfont=dict(color='white', size=9)),
                        colorbar=dict(tickfont=dict(color='#c0d0ff'), title='f(x,y)')
                    )])
                    # Ellipse FWHM
                    theta = np.linspace(0, 2*np.pi, 200)
                    kfwhm = np.sqrt(2 * np.log(2))
                    fig_ct.add_trace(go.Scatter(
                        x=mu_x + sigma_x * kfwhm * np.cos(theta),
                        y=mu_y + sigma_y * kfwhm * np.sin(theta),
                        mode='lines', name='FWHM ellipse',
                        line=dict(color='#ffcc00', width=2.5, dash='dash')
                    ))
                    fig_ct.add_trace(go.Scatter(
                        x=[mu_x], y=[mu_y], mode='markers', name='Centre',
                        marker=dict(color='#ff00cc', size=12, symbol='cross')
                    ))
                    fig_ct.update_layout(
                        title="Contour 2D + Ellipse FWHM",
                        xaxis_title='x', yaxis_title='y',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.92)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                        height=430,
                    )
                    st.plotly_chart(fig_ct, use_container_width=True)

        # TAB 3 : MULTI-GAUSSIENNES
        with tab3:
            st.markdown("### 🔄 Superposition de gaussiennes")
            col1, col2 = st.columns([1, 2])

            with col1:
                n_gauss = st.slider("Nombre de gaussiennes", 1, 6, 3, key="n_gauss")
                x_mg = np.linspace(-15, 15, 2000)
                gauss_params = []

                for i in range(n_gauss):
                    st.markdown(f"**G{i+1}**")
                    c1, c2_c, c3 = st.columns(3)
                    with c1:
                        A_i = st.slider(f"A{i+1}", 0.1, 5.0, 1.0, 0.1, key=f"A{i}_mg")
                    with c2_c:
                        mu_i = st.slider(f"μ{i+1}", -10.0, 10.0, float((i-n_gauss//2)*3), 0.1, key=f"mu{i}_mg")
                    with c3:
                        s_i = st.slider(f"σ{i+1}", 0.1, 4.0, 1.0, 0.1, key=f"sig{i}_mg")
                    gauss_params.append((A_i, mu_i, s_i))

                show_total = st.checkbox("Afficher somme totale", True, key="show_total_mg")
                show_indiv = st.checkbox("Afficher individuelles", True, key="show_indiv_mg")

            with col2:
                colors_g = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88', '#ffcc00', '#ff4400']
                fig_mg = go.Figure()
                y_total = np.zeros_like(x_mg)

                for i, (Ai, mui, si) in enumerate(gauss_params):
                    eng_i = GaussianEngine(Ai, si, mui)
                    yi = eng_i.eval_1d(x_mg)
                    y_total += yi
                    if show_indiv:
                        fig_mg.add_trace(go.Scatter(
                            x=x_mg, y=yi, mode='lines',
                            name=f'G{i+1} (A={Ai}, μ={mui:.1f}, σ={si})',
                            line=dict(color=colors_g[i % len(colors_g)], width=2, dash='dot'),
                            opacity=0.7
                        ))

                if show_total:
                    fig_mg.add_trace(go.Scatter(
                        x=x_mg, y=y_total, mode='lines', name='Somme',
                        line=dict(color='#ffffff', width=3),
                        fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'
                    ))

                fig_mg.update_layout(
                    title=f"Superposition de {n_gauss} gaussiennes",
                    xaxis_title='x', yaxis_title='f(x)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=480,
                )
                st.plotly_chart(fig_mg, use_container_width=True)

                # Métriques globales
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Max total", f"{y_total.max():.4f}")
                with c2: st.metric("Aire totale", f"{safe_trapz(y_total, x_mg):.4f}")
                with c3: st.metric("x (max)", f"{x_mg[np.argmax(y_total)]:.3f}")

        # TAB 4 : SPECTRE & CONVOLUTION
        with tab4:
            st.markdown("### 📡 Transformée de Fourier & Convolution")
            col1, col2 = st.columns([1, 2])

            with col1:
                amp_f  = st.slider("Amplitude A", 0.1, 5.0, 1.0, 0.1, key="amp_f")
                sig_f  = st.slider("σ (gaussienne)", 0.1, 3.0, 1.0, 0.05, key="sig_f")
                mu_f   = st.slider("Centre μ", -5.0, 5.0, 0.0, 0.1, key="mu_f")
                sig2_f = st.slider("σ₂ (convolution)", 0.1, 3.0, 0.8, 0.05, key="sig2_f")

                eng_f = GaussianEngine(amp_f, sig_f, mu_f)
                st.metric("σ convolution √(σ₁²+σ₂²)",
                          f"{np.sqrt(sig_f**2 + sig2_f**2):.4f}")

            with col2:
                x_f = np.linspace(-15, 15, 4096)
                y_f = eng_f.eval_1d(x_f)

                # FFT numérique
                dt = x_f[1] - x_f[0]
                fft_num = np.fft.rfft(y_f)
                freqs_num = np.fft.rfftfreq(len(y_f), dt)
                mag_num = np.abs(fft_num) * dt

                # FFT analytique
                fft_ana = np.abs(eng_f.fft_analytique(freqs_num))

                # Convolution analytique
                eng_conv = eng_f.convolution(sig2_f)
                y_conv = eng_conv.eval_1d(x_f)

                fig_fft = make_subplots(rows=2, cols=1,
                    subplot_titles=["Signal + Convolution", "Spectre |FFT(f)|"])

                fig_fft.add_trace(go.Scatter(
                    x=x_f, y=y_f, mode='lines', name='f(x)',
                    line=dict(color='#00ccff', width=2.5)
                ), row=1, col=1)
                fig_fft.add_trace(go.Scatter(
                    x=x_f, y=y_conv, mode='lines', name=f'f*g (σ_conv={eng_conv.sigma:.2f})',
                    line=dict(color='#ff00cc', width=2.5, dash='dash')
                ), row=1, col=1)

                fig_fft.add_trace(go.Scatter(
                    x=freqs_num, y=mag_num, mode='lines', name='|FFT| numérique',
                    line=dict(color='#7700ff', width=2),
                    fill='tozeroy', fillcolor='rgba(119,0,255,0.15)'
                ), row=2, col=1)
                fig_fft.add_trace(go.Scatter(
                    x=freqs_num, y=fft_ana, mode='lines', name='|FFT| analytique',
                    line=dict(color='#ffcc00', width=2, dash='dash')
                ), row=2, col=1)

                fig_fft.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    height=540,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_fft.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_fft.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_fft.update_xaxes(title_text="Fréquence ξ", row=2, col=1,
                                      range=[0, min(3/sig_f, freqs_num[-1])])
                st.plotly_chart(fig_fft, use_container_width=True)

        # TAB 5 : AJUSTEMENT DONNÉES
        with tab5:
            st.markdown("### 🔬 Ajustement gaussien sur données expérimentales")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Saisir ou générer des données")
                mode_data = st.radio("Source", ["Données synthétiques", "Saisie manuelle"], horizontal=True, key="mode_data_fit")

                if mode_data == "Données synthétiques":
                    A_true  = st.slider("A vrai", 0.5, 5.0, 2.0, 0.1, key="A_true")
                    mu_true = st.slider("μ vrai", -5.0, 5.0, 1.0, 0.1, key="mu_true")
                    s_true  = st.slider("σ vrai", 0.1, 3.0, 1.0, 0.05, key="s_true")
                    bruit   = st.slider("Bruit σ_bruit", 0.0, 1.0, 0.15, 0.01, key="bruit")
                    n_pts   = st.slider("Points", 10, 200, 40, key="n_pts")
                    np.random.seed(42)
                    x_data = np.linspace(mu_true - 4*s_true, mu_true + 4*s_true, n_pts)
                    y_data = (A_true * np.exp(-0.5*((x_data-mu_true)/s_true)**2) +
                              np.random.normal(0, bruit, n_pts))
                else:
                    x_str = st.text_input("x (virgule)", "-3,-2,-1,0,1,2,3", key="x_str")
                    y_str = st.text_input("y (virgule)", "0.1,0.4,0.8,1.0,0.8,0.4,0.1", key="y_str")
                    try:
                        x_data = np.array([float(v.strip()) for v in x_str.split(',')])
                        y_data = np.array([float(v.strip()) for v in y_str.split(',')])
                    except:
                        st.error("Format invalide")
                        st.stop()

            with col2:
                eng_fit = GaussianEngine(1.0, 1.0, 0.0)
                fit = eng_fit.fit_data(x_data, y_data)

                if "erreur" not in fit:
                    x_fine = np.linspace(x_data.min()-1, x_data.max()+1, 500)
                    y_fit = fit["A"] * np.exp(-0.5 * ((x_fine - fit["μ"]) / fit["σ"])**2)

                    fig_fit = go.Figure()
                    fig_fit.add_trace(go.Scatter(
                        x=x_data, y=y_data, mode='markers', name='Données',
                        marker=dict(color='#ff00cc', size=10, symbol='circle',
                                   line=dict(width=2, color='#ffffff'))
                    ))
                    fig_fit.add_trace(go.Scatter(
                        x=x_fine, y=y_fit, mode='lines', name='Ajustement',
                        line=dict(color='#00ccff', width=3)
                    ))
                    fig_fit.update_layout(
                        title=f"Ajustement — R²={fit['R²']:.4f}",
                        xaxis_title='x', yaxis_title='y',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.92)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                        yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                        height=400,
                    )
                    st.plotly_chart(fig_fit, use_container_width=True)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("A ajusté", f"{fit['A']:.4f} ± {fit['σ_A']:.4f}")
                    with c2: st.metric("μ ajusté", f"{fit['μ']:.4f} ± {fit['σ_μ']:.4f}")
                    with c3: st.metric("σ ajusté", f"{fit['σ']:.4f} ± {fit['σ_σ']:.4f}")
                    with c4: st.metric("R²", f"{fit['R²']:.6f}")

                    st.metric("FWHM ajusté", f"{fit['FWHM_fit']:.4f}")

                    if mode_data == "Données synthétiques":
                        st.markdown("#### 🎯 Comparaison vrai vs ajusté")
                        comp = pd.DataFrame({
                            "Paramètre": ["A", "μ", "σ", "FWHM"],
                            "Vrai": [A_true, mu_true, s_true, 2.3548*s_true],
                            "Ajusté": [fit["A"], fit["μ"], fit["σ"], fit["FWHM_fit"]],
                            "Erreur (%)": [
                                abs(fit["A"]-A_true)/A_true*100,
                                abs(fit["μ"]-mu_true)/(abs(mu_true)+1e-10)*100,
                                abs(fit["σ"]-s_true)/s_true*100,
                                abs(fit["FWHM_fit"]-2.3548*s_true)/(2.3548*s_true)*100
                            ]
                        })
                        st.dataframe(comp.round(4), use_container_width=True)
                else:
                    st.error(f"Ajustement échoué : {fit['erreur']}")

        # TAB 6 : THÉORIE
        with tab6:
            st.markdown("### 📖 Formulaire scientifique")
            cols = st.columns(2)
            col_idx = 0

            for nom, formule in FORMULES.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(formule)
                col_idx += 1

            st.markdown("---")
            st.markdown("### 🔬 Applications")
            for app, desc in APPLICATIONS.items():
                st.markdown(f"- **{app}** : {desc}")

            st.markdown("---")
            st.markdown("### ⚗️ Diagnostic")
            try:
                eng_diag = GaussianEngine(1.0, 1.0, 0.0)
                diag = eng_diag.diagnostiquer()
                st.dataframe(pd.DataFrame(diag), use_container_width=True)
            except:
                pass

            st.markdown("---")
            st.markdown("### 📚 Références")
            for r in [
                "Abramowitz & Stegun — *Handbook of Mathematical Functions* (NIST, 1964)",
                "Goodman — *Statistical Optics* (Wiley, 2015)",
                "Papoulis — *Probability, Random Variables, and Stochastic Processes* (McGraw-Hill, 2002)",
            ]:
                st.markdown(f"- {r}")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Paramètres")

            laser_type = st.selectbox("Type de laser", ["Personnalisé"] + list(TYPES_LASER.keys()))
            if laser_type != "Personnalisé":
                info_l = TYPES_LASER[laser_type]
                st.info(f"λ = {info_l['λ']} nm | η = {info_l['η']*100:.0f}% | {info_l['P_typ']}")

            I0 = st.slider("Intensité initiale I₀", 0.1, 200.0, 10.0, 0.5)
            gamma = st.slider("Coefficient γ (s⁻¹)", 0.01, 5.0, 0.1, 0.01)
            t_max = st.slider("Temps max (s)", 1.0, 100.0, 30.0, 1.0)
            n_points = st.slider("Résolution", 200, 5000, 1000, 100)

            mode_affichage = st.radio("Affichage", ["Linéaire", "Semi-log"], horizontal=True)

            engine = LaserEngine(I0, gamma)

            st.markdown("### 📐 Résultats analytiques")
            st.metric("Demi-vie t₁/₂ (s)", f"{engine.demi_vie:.4f}")
            st.metric("Temps de vie τ (s)", f"{engine.temps_vie:.4f}")
            st.metric("I(t_max)", f"{I0 * np.exp(-gamma * t_max):.4e}")
            st.metric("I(τ) = I₀/e", f"{I0/np.e:.4f}")

            # Paramètres 4 niveaux
            st.markdown("### 🔬 Système 4 niveaux")
            show_4n = st.checkbox("Simuler système 4 niveaux", True)
            if show_4n:
                Rp = st.slider("Taux pompage Rp (s⁻¹)", 1e5, 1e8, 1e6,
                               format="%.0e", step=1e5)
                tau21 = st.slider("τ₂₁ (μs)", 0.1, 500.0, 230.0, 1.0)
                tau32 = st.slider("τ₃₂ (ns)", 0.1, 100.0, 1.0, 0.1)

        with col2:
            t, I = compute_decay(I0, gamma, t_max, n_points)

            fig = go.Figure()

            if mode_affichage == "Linéaire":
                fig.add_trace(go.Scatter(
                    x=t, y=I, mode='lines', name='I(t)',
                    line=dict(color='#00ccff', width=3)
                ))
                fig.add_hline(y=I0/np.e, line_dash='dash', line_color='#ffcc00',
                              annotation_text=f"I₀/e = {I0/np.e:.3f}")
                fig.add_hline(y=I0/2, line_dash='dot', line_color='#7700ff',
                              annotation_text=f"I₀/2 = {I0/2:.3f}")
                fig.add_vline(x=engine.demi_vie, line_dash='dash', line_color='#ff00cc',
                              annotation_text=f"t₁/₂={engine.demi_vie:.2f}s")
                fig.add_vline(x=engine.temps_vie, line_dash='dot', line_color='#00ff88',
                              annotation_text=f"τ={engine.temps_vie:.2f}s")
            else:
                I_log = np.where(I > 0, I, 1e-12)
                fig.add_trace(go.Scatter(
                    x=t, y=np.log10(I_log), mode='lines',
                    name='log₁₀(I(t))', line=dict(color='#00ccff', width=3)
                ))
                # Droite théorique
                fig.add_trace(go.Scatter(
                    x=t, y=np.log10(I0) - gamma * t / np.log(10),
                    mode='lines', name='Droite théorique',
                    line=dict(color='#ffcc00', width=2, dash='dash')
                ))

            fig.update_layout(
                title="Décroissance laser I(t) = I₀·e^{-γt}",
                xaxis_title="Temps (s)",
                yaxis_title="Intensité" if mode_affichage == "Linéaire" else "log₁₀(I)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Système 4 niveaux
            if show_4n:
                sol4 = solve_4niveaux(
                    Rp=Rp,
                    tau21=tau21*1e-6,
                    tau32=tau32*1e-9,
                    tau10=tau32*1e-9*0.1,
                    t_max=t_max * 1e-6,
                    n_points=2000,
                )
                fig4 = go.Figure()
                colors4 = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88']
                for i, (k, v) in enumerate([("N₀", sol4["N0"]), ("N₁", sol4["N1"]),
                                             ("N₂", sol4["N2"]), ("N₃", sol4["N3"])]):
                    fig4.add_trace(go.Scatter(
                        x=sol4["t"]*1e6, y=v, mode='lines', name=k,
                        line=dict(color=colors4[i], width=2)
                    ))
                fig4.add_trace(go.Scatter(
                    x=sol4["t"]*1e6, y=sol4["inversion"],
                    mode='lines', name='ΔN (inversion)',
                    line=dict(color='#ffffff', width=3, dash='dash')
                ))
                fig4.update_layout(
                    title="Populations système 4 niveaux",
                    xaxis_title="Temps (μs)", yaxis_title="Population (m⁻³)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=380,
                )
                st.plotly_chart(fig4, use_container_width=True)

            # Export
            df_exp = pd.DataFrame({"temps_s": t, "intensite": I})
            st.download_button("💾 Export CSV",
                               df_exp.to_csv(index=False).encode(),
                               "laser_simulation.csv", "text/csv")

    # ============================================================
    # SECTION 2 : OSCILLATIONS DE RELAXATION
    # ============================================================
    elif section == "🌊 Oscillations de relaxation":
        st.markdown("### 🌊 Oscillations de relaxation laser")
        st.markdown("*Dynamique couplée photons ↔ population autour du seuil*")

        col1, col2 = st.columns([1, 2])
        with col1:
            Rp_osc = st.slider("Taux de pompe Rp (×Rp_th)", 1.01, 5.0, 1.5, 0.01)
            tau_c_osc = st.slider("Durée de vie photon τ_c (ns)", 0.1, 100.0, 10.0, 0.1)
            tau_sp_osc = st.slider("Durée de vie spontanée τ_sp (μs)", 0.1, 1000.0, 230.0, 1.0)
            t_max_osc = st.slider("Durée simulation (μs)", 1.0, 500.0, 50.0, 1.0)

            tau_c = tau_c_osc * 1e-9
            tau_sp = tau_sp_osc * 1e-6
            N_th = 1.0 / (tau_c)
            Rp_abs = Rp_osc * N_th / tau_sp

            # Fréquence théorique
            omega_r = 0.0
            if Rp_abs * tau_sp > 1:
                omega_r = np.sqrt((Rp_abs * tau_sp - 1) / (tau_c * tau_sp))
            f_osc = omega_r / (2 * np.pi) if omega_r > 0 else 0.0

            st.metric("Fréquence oscillations (MHz)",
                      f"{f_osc/1e6:.3f}" if f_osc > 0 else "N/A")
            st.metric("Période (ns)",
                      f"{1/f_osc*1e9:.1f}" if f_osc > 0 else "N/A")

        with col2:
            sol_osc = solve_relaxation(
                Rp_abs=Rp_abs,
                tau_c=tau_c,
                tau_sp=tau_sp,
                N_th=N_th,
                t_max_osc=t_max_osc * 1e-6,
                n_points=2000,
            )

            fig_osc = make_subplots(rows=2, cols=1,
                subplot_titles=["Densité photons S(t)", "Population N(t)"])

            fig_osc.add_trace(go.Scatter(
                x=sol_osc["t"]*1e6, y=sol_osc["S"], mode='lines',
                name='S(t)', line=dict(color='#00ccff', width=2)
            ), row=1, col=1)
            fig_osc.add_trace(go.Scatter(
                x=sol_osc["t"]*1e6, y=sol_osc["N"]/N_th, mode='lines',
                name='N(t)/N_th', line=dict(color='#7700ff', width=2)
            ), row=2, col=1)

            fig_osc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                height=480,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_osc.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                  title_text="Temps (μs)")
            fig_osc.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_osc, use_container_width=True)

    # ============================================================
    # SECTION 3 : SPECTRE & MODES
    # ============================================================
    elif section == "📡 Spectre & Modes":
        st.markdown("### 📡 Spectre d'émission et modes de cavité")

        col1, col2 = st.columns([1, 2])
        with col1:
            laser_sel = st.selectbox("Laser", list(TYPES_LASER.keys()), key="spec_laser")
            info_spec = TYPES_LASER[laser_sel]
            lambda_c = info_spec["λ"]

            delta_lambda = st.slider("Largeur spectrale Δλ (nm)", 0.001, 50.0,
                                      0.1 if "Nd" in laser_sel else 1.0, 0.001)
            n_modes = st.slider("Nombre de modes longitudinaux", 1, 20, 5)
            gain_val = st.slider("Gain g₀", 0.1, 10.0, 1.0, 0.1)

            L_cav = st.slider("Longueur de cavité L (cm)", 1, 200, 30)
            n_ref = st.slider("Indice de réfraction n", 1.0, 3.5, 1.0, 0.01)

            delta_nu = 3e8 / (2 * n_ref * L_cav * 1e-2)
            st.metric("Espacement modes (MHz)", f"{delta_nu/1e6:.1f}")
            st.metric("Finesse théorique", f"{np.pi*0.99/(1-0.99):.0f}")

        with col2:
            lambda_arr, gain_prof, modes_l, modes_g = compute_spectral(
                lambda_center=lambda_c,
                delta_lambda=delta_lambda,
                n_modes=n_modes,
                gain=gain_val,
            )

            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(
                x=lambda_arr, y=gain_prof, mode='lines',
                name='Profil de gain', line=dict(color='rgba(0,204,255,0.5)', width=2)
            ))

            colors_modes = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88',
                            '#ffcc00', '#ff4444', '#44ff88']
            for i, (lm, gm) in enumerate(zip(modes_l, modes_g)):
                fig_spec.add_trace(go.Scatter(
                    x=[lm, lm], y=[0, gm], mode='lines',
                    name=f'Mode {i+1}',
                    line=dict(color=colors_modes[i % len(colors_modes)], width=3)
                ))

            fig_spec.update_layout(
                title=f"Spectre laser — {laser_sel} (λ={lambda_c} nm)",
                xaxis_title="Longueur d'onde (nm)", yaxis_title="Gain (u.a.)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                          range=[0, gain_val * 1.1]),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=450,
            )
            st.plotly_chart(fig_spec, use_container_width=True)

    # ============================================================
    # SECTION 4 : PROFIL TEM GAUSSIEN
    # ============================================================
    elif section == "🔬 Profil TEM gaussien":
        st.markdown("### 🔬 Profil de faisceau TEM₀₀ gaussien")

        col1, col2 = st.columns([1, 2])
        with col1:
            w0_beam = st.slider("Waist w₀ (mm)", 0.1, 10.0, 1.0, 0.1)
            z_beam = st.slider("Distance z (m)", 0.0, 10.0, 0.0, 0.1)
            lambda_beam = st.slider("Longueur d'onde (μm)", 0.4, 11.0, 1.064, 0.001)

            zR = np.pi * (w0_beam*1e-3)**2 / (lambda_beam*1e-6)
            wz = w0_beam * np.sqrt(1 + (z_beam/zR)**2)
            Rz = z_beam * (1 + (zR/z_beam)**2) if z_beam > 0 else np.inf

            st.metric("Longueur Rayleigh z_R (m)", f"{zR:.3f}")
            st.metric("Waist à z: w(z) (mm)", f"{wz:.3f}")
            st.metric("Divergence θ (mrad)", f"{lambda_beam*1e-6/(np.pi*w0_beam*1e-3)*1e3:.3f}")

        with col2:
            x_b, y_b, Z_beam, zR, wz = compute_beam_profile(
                w0=w0_beam,
                z=z_beam,
                lambda_um=lambda_beam,
            )

            fig_beam = go.Figure(data=[go.Surface(
                z=Z_beam, x=x_b*1e3, y=y_b*1e3,
                colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#ff4400'],[1,'#ffffff']],
                showscale=True,
                lighting=dict(ambient=0.5, diffuse=0.8),
            )])
            fig_beam.update_layout(
                title=f"Profil TEM₀₀ — z={z_beam} m, w(z)={wz:.2f} mm",
                scene=dict(
                    bgcolor='rgba(5,0,20,0.8)',
                    xaxis=dict(color='#c0d0ff', title='x (mm)'),
                    yaxis=dict(color='#c0d0ff', title='y (mm)'),
                    zaxis=dict(color='#c0d0ff', title='I (u.a.)'),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0d0ff'),
                height=480,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_beam, use_container_width=True)

            # Propagation du waist
            z_prop = np.linspace(0, 10*zR, 500)
            wz_prop = w0_beam * np.sqrt(1 + (z_prop/zR)**2)
            fig_prop = go.Figure()
            fig_prop.add_trace(go.Scatter(
                x=z_prop, y=wz_prop, mode='lines', name='+w(z)',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_prop.add_trace(go.Scatter(
                x=z_prop, y=-wz_prop, mode='lines', name='-w(z)',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_prop.add_vline(x=zR, line_color='#ffcc00', line_dash='dash',
                               annotation_text=f"z_R={zR:.2f}m")
            fig_prop.update_layout(
                title="Propagation du faisceau gaussien",
                xaxis_title="z (m)", yaxis_title="w(z) (mm)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                height=300,
            )
            st.plotly_chart(fig_prop, use_container_width=True)

    # ============================================================
    # SECTION 5 : AMPLIFICATION FRANTZ-NODVIK
    # ============================================================
    elif section == "⚡ Amplification (F-N)":
        st.markdown("### ⚡ Amplificateur laser — Frantz-Nodvik")

        col1, col2 = st.columns([1, 2])
        with col1:
            g0_fn = st.slider("Gain petit signal g₀ (cm⁻¹)", 0.01, 2.0, 0.5, 0.01)
            L_fn = st.slider("Longueur amplificateur L (cm)", 1.0, 50.0, 10.0, 0.5)
            I_sat_fn = st.slider("Intensité saturation I_sat (W/cm²)", 0.1, 1000.0, 100.0, 1.0)
            I_max = st.slider("I_in max (W/cm²)", 1.0, 5000.0, 500.0, 10.0)

            G_lin = np.exp(g0_fn * L_fn)
            st.metric("Gain linéaire G = e^(g₀L)", f"{G_lin:.2f}")
            st.metric("Gain (dB)", f"{10*np.log10(G_lin):.2f}")

        with col2:
            I_in, I_out, G_eff = compute_frantz_nodvik(
                g0=g0_fn,
                L=L_fn,
                I_sat=I_sat_fn,
                I_max=I_max,
            )

            fig_fn = make_subplots(rows=2, cols=1,
                subplot_titles=["I_out vs I_in", "Gain effectif G(I_in)"])

            fig_fn.add_trace(go.Scatter(
                x=I_in, y=I_out, mode='lines', name='I_out (F-N)',
                line=dict(color='#00ccff', width=3)
            ), row=1, col=1)
            fig_fn.add_trace(go.Scatter(
                x=I_in, y=G_lin * I_in, mode='lines', name='I_out (linéaire)',
                line=dict(color='rgba(255,200,0,0.5)', width=2, dash='dash')
            ), row=1, col=1)
            fig_fn.add_vline(x=I_sat_fn, line_color='#ff00cc', line_dash='dot',
                             annotation_text="I_sat", row=1, col=1)

            fig_fn.add_trace(go.Scatter(
                x=I_in, y=G_eff, mode='lines', name='G effectif',
                line=dict(color='#7700ff', width=2.5)
            ), row=2, col=1)
            fig_fn.add_hline(y=G_lin, line_color='#ffcc00', line_dash='dash',
                             annotation_text=f"G₀={G_lin:.1f}", row=2, col=1)

            fig_fn.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                height=520,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_fn.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                 title_text="I_in (W/cm²)")
            fig_fn.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_fn, use_container_width=True)

    # ============================================================
    # SECTION 6 : THÉORIE
    # ============================================================
    elif section == "📖 Théorie & Références":
        st.markdown("### 📖 Formulaire scientifique laser")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 🔬 Types de lasers")
        df_lasers = pd.DataFrame([
            {"Laser": k, "λ (nm)": v["λ"], "Type": v["type"],
             "η (%)": f"{v['η']*100:.0f}", "τ": str(v["τ"]), "Puissance": v["P_typ"]}
            for k, v in TYPES_LASER.items()
        ])
        st.dataframe(df_lasers, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📐 Diagnostic laser")
        engine_diag = LaserEngine(10.0, 0.1)
        diag = engine_diag.diagnostiquer(10.0, 0.1, 30.0)
        st.dataframe(pd.DataFrame(diag), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        refs = [
            "Saleh & Teich — *Fundamentals of Photonics* (Wiley, 2007)",
            "Svelto — *Principles of Lasers* (Springer, 2010)",
            "Siegman — *Lasers* (University Science Books, 1986)",
            "Yariv — *Quantum Electronics* (Wiley, 1989)",
        ]
        for r in refs:
            st.markdown(f"- {r}")
