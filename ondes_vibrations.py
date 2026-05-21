__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import re
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import find_peaks, spectrogram
from scipy.optimize import brentq, fsolve
from scipy.linalg import eigh
import pandas as pd
import importlib
import warnings
warnings.filterwarnings('ignore')


def safe_latex(f_latex: str):
    """Affiche une formule LaTeX en essayant un rendu Markdown, puis `st.latex` en fallback.

    Certains clients front-end peuvent lever des erreurs DOM (NotFoundError)
    lors du rendu MathJax ; on essaie un rendu plus robuste pour éviter de
    casser les pages de formules.
    """
    if f_latex is None:
        return
    # Convert Sympy expressions or other non-str objects to a LaTeX string
    try:
        if HAS_SYMPY and isinstance(f_latex, getattr(sp, 'Basic', (object,))):
            tex = sp.latex(f_latex)
        else:
            tex = str(f_latex)
    except Exception:
        tex = str(f_latex)

    latex_block = "$$" + tex + "$$"
    try:
        st.markdown(latex_block)
    except Exception:
        try:
            st.latex(tex)
        except Exception:
            st.write(tex)

try:
    sp = importlib.import_module('sympy')
    _sympy_parser = importlib.import_module('sympy.parsing.sympy_parser')
    standard_transformations = getattr(_sympy_parser, 'standard_transformations', None)
    implicit_multiplication_application = getattr(_sympy_parser, 'implicit_multiplication_application', None)
    parse_expr = getattr(_sympy_parser, 'parse_expr', None)
    HAS_SYMPY = True
except Exception:
    sp = None
    standard_transformations = None
    implicit_multiplication_application = None
    parse_expr = None
    HAS_SYMPY = False

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
CONSTANTES = {
    "c_son_air (m/s)":    343.0,
    "c_lumiere (m/s)":    2.998e8,
    "ρ_air (kg/m³)":      1.225,
    "g (m/s²)":           9.81,
    "Z_air (Pa·s/m)":     415.0,
    "f_audible_min (Hz)": 20.0,
    "f_audible_max (Hz)": 20000.0,
}

FORMULES_VIBRATIONS = {
    "Équation oscillateur libre":      r"\ddot{x} + \omega_0^2 x = 0",
    "Solution non amortie":            r"x(t) = A\cos(\omega_0 t + \phi)",
    "Pulsation propre":                r"\omega_0 = \sqrt{\frac{k}{m}} = \frac{2\pi}{T_0}",
    "Pendule simple":                  r"\omega_0 = \sqrt{\frac{g}{L}},\quad T_0 = 2\pi\sqrt{\frac{L}{g}}",
    "Ressort-masse":                   r"T_0 = 2\pi\sqrt{\frac{m}{k}}",
    "Équation amortie":                r"\ddot{x} + 2\xi\omega_0\dot{x} + \omega_0^2 x = 0",
    "Amortissement critique":          r"\xi = \frac{c}{2\sqrt{km}} = \frac{c}{c_c},\quad c_c = 2\sqrt{km}",
    "Solution sous-amortie (ξ<1)":    r"x(t) = Ae^{-\xi\omega_0 t}\cos(\omega_d t + \phi),\quad \omega_d = \omega_0\sqrt{1-\xi^2}",
    "Solution sur-amortie (ξ>1)":     r"x(t) = e^{-\xi\omega_0 t}(C_1 e^{\omega_0\sqrt{\xi^2-1}\,t}+C_2 e^{-\omega_0\sqrt{\xi^2-1}\,t})",
    "Solution critique (ξ=1)":        r"x(t) = (A + Bt)e^{-\omega_0 t}",
    "Décrément logarithmique":         r"\delta = \ln\frac{x(t)}{x(t+T_d)} = \frac{2\pi\xi}{\sqrt{1-\xi^2}}",
    "Composition harmonique":          r"x(t) = \sum_{i=1}^n A_i\cos(\omega_i t + \phi_i)",
    "Battements":                      r"x(t) = 2A\cos\!\left(\frac{\omega_2-\omega_1}{2}t\right)\cos\!\left(\frac{\omega_1+\omega_2}{2}t\right)",
    "Equation d'onde 1D":             r"\frac{\partial^2 u}{\partial t^2} = c^2\frac{\partial^2 u}{\partial x^2}",
    "Solution onde progressive":       r"u(x,t) = A\cos(\omega t - kx + \phi),\quad c = \frac{\omega}{k} = \lambda f",
    "Impédance acoustique":            r"Z = \rho c",
    "Intensité acoustique":            r"I = \frac{P^2}{2\rho c} = \frac{1}{2}\rho c A^2\omega^2",
    "Niveau sonore (dB)":              r"L = 10\log_{10}\!\left(\frac{I}{I_0}\right),\quad I_0=10^{-12}\text{ W/m}^2",
    "Effet Doppler":                   r"f_{obs} = f_s\frac{c \pm v_{obs}}{c \mp v_s}",
    "Onde stationnaire":               r"u(x,t) = 2A\cos(kx)\cos(\omega t)",
    "Modes propres corde":             r"f_n = \frac{n}{2L}\sqrt{\frac{T}{\mu}},\quad n = 1,2,3\ldots",
    "Énergie oscillateur":             r"E = \frac{1}{2}kA^2 = \frac{1}{2}m\omega_0^2 A^2",
}

CHAPITRES_VIBRATIONS = {
    "1 — Introduction à la vibration des systèmes": "intro_vibration",
    "2 — Oscillations libres non amorties (1 DDL)": "osc_libre_non_amorti",
    "3 — Composition des grandeurs harmoniques":     "composition_harmonique",
    "4 — Oscillations libres amorties (1 DDL)":     "osc_libre_amorti",
    "5 — Propagation dans les milieux illimités":    "propagation_milieux",
}


# ============================================================
# MOTEUR VIBRATIONS
# ============================================================
class VibrationEngine:
    """Moteur scientifique complet pour ondes et vibrations."""

    def __init__(self, m: float = 1.0, k: float = 1.0, c: float = 0.0):
        self.m = m
        self.k = k
        self.c = c
        self.omega0 = np.sqrt(k / m) if m > 0 else 0
        self.cc = 2 * np.sqrt(k * m)
        self.xi = c / self.cc if self.cc > 0 else 0
        self.omegad = self.omega0 * np.sqrt(max(1 - self.xi**2, 0))

    # -------------------------------------------------------
    # CHAPITRE 1 — INTRODUCTION
    # -------------------------------------------------------
    def systemes_vibratoires(self) -> dict:
        """Paramètres caractéristiques du système."""
        T0 = 2 * np.pi / self.omega0 if self.omega0 > 0 else np.inf
        f0 = self.omega0 / (2 * np.pi)
        return {
            "ω₀ (rad/s)": self.omega0,
            "f₀ (Hz)": f0,
            "T₀ (s)": T0,
            "ξ": self.xi,
            "c_c (N·s/m)": self.cc,
            "E₀ = ½kA²": None,
        }

    def pendule_simple(self, L: float, theta0: float,
                       t: np.ndarray) -> tuple:
        """Oscillation d'un pendule simple (petits angles)."""
        omega_p = np.sqrt(9.81 / L)
        T_p = 2 * np.pi / omega_p
        theta = theta0 * np.cos(omega_p * t)
        return theta, omega_p, T_p

    def pendule_nonlineaire(self, L: float, theta0_deg: float,
                             t: np.ndarray) -> np.ndarray:
        """Pendule non-linéaire sin(θ) sans approximation."""
        g = 9.81
        def dydt(y, t):
            return [y[1], -(g/L)*np.sin(y[0])]
        theta0 = np.radians(theta0_deg)
        sol = odeint(dydt, [theta0, 0], t)
        return sol[:, 0]

    def ressort_masse(self, x0: float, v0: float,
                       t: np.ndarray) -> np.ndarray:
        """Solution analytique ressort-masse sans amortissement."""
        A = np.sqrt(x0**2 + (v0/self.omega0)**2)
        phi = np.arctan2(-v0/self.omega0, x0)
        return A * np.cos(self.omega0 * t + phi)

    def energie_mecanique(self, x: np.ndarray,
                           v: np.ndarray) -> tuple:
        """Énergie cinétique, potentielle et totale."""
        Ec = 0.5 * self.m * v**2
        Ep = 0.5 * self.k * x**2
        Et = Ec + Ep
        return Ec, Ep, Et

    # -------------------------------------------------------
    # CHAPITRE 2 — OSCILLATIONS NON AMORTIES
    # -------------------------------------------------------
    def solution_non_amortie(self, A: float, phi: float,
                              t: np.ndarray) -> np.ndarray:
        """x(t) = A·cos(ω₀t + φ)"""
        return A * np.cos(self.omega0 * t + phi)

    def decomposition_initiale(self, x0: float,
                                v0: float) -> tuple:
        """Amplitude et phase depuis CI."""
        A = np.sqrt(x0**2 + (v0/self.omega0)**2)
        phi = np.arctan2(-v0, self.omega0 * x0)
        return A, phi

    def portrait_phase(self, A: float,
                        t: np.ndarray) -> tuple:
        """Portrait de phase (x, ẋ) — ellipse."""
        x = A * np.cos(self.omega0 * t)
        v = -A * self.omega0 * np.sin(self.omega0 * t)
        return x, v

    def diagramme_energie(self, A: float, n: int = 200) -> tuple:
        """Transfert E_c ↔ E_p sur une période."""
        x = np.linspace(-A, A, n)
        Ep = 0.5 * self.k * x**2
        Ec = 0.5 * self.k * A**2 - Ep
        return x, Ec, Ep

    def modes_propres_corde(self, L: float, T_tens: float,
                             mu: float, n_max: int = 5) -> pd.DataFrame:
        """Fréquences propres d'une corde vibrante."""
        c_corde = np.sqrt(T_tens / mu)
        modes = []
        for n in range(1, n_max + 1):
            fn = n * c_corde / (2 * L)
            lambdan = 2 * L / n
            modes.append({"n": n, "f_n (Hz)": round(fn, 4),
                          "λ_n (m)": round(lambdan, 4),
                          "T_n (ms)": round(1000/fn, 4)})
        return pd.DataFrame(modes)

    def oscillateur_2DDL(self, m1: float, m2: float,
                          k1: float, k2: float,
                          k12: float) -> tuple:
        """Valeurs propres d'un système 2 DDL couplé."""
        M = np.array([[m1, 0], [0, m2]])
        K = np.array([[k1+k12, -k12], [-k12, k2+k12]])
        vals, vecs = eigh(K, M)
        omega_n = np.sqrt(np.maximum(vals, 0))
        return omega_n, vecs

    # -------------------------------------------------------
    # CHAPITRE 3 — COMPOSITION HARMONIQUE
    # -------------------------------------------------------
    def composition_harmoniques(self, amplitudes: list,
                                  pulsations: list,
                                  phases: list,
                                  t: np.ndarray) -> tuple:
        """Superposition de N harmoniques."""
        x_total = np.zeros_like(t)
        composantes = []
        for A, omega, phi in zip(amplitudes, pulsations, phases):
            xi = A * np.cos(omega * t + phi)
            x_total += xi
            composantes.append(xi)
        return x_total, composantes

    def battements(self, A: float, omega1: float,
                   omega2: float, t: np.ndarray) -> tuple:
        """Battements entre deux oscillations proches."""
        x1 = A * np.cos(omega1 * t)
        x2 = A * np.cos(omega2 * t)
        x_total = x1 + x2
        # Enveloppe analytique
        delta_omega = abs(omega2 - omega1)
        omega_moy = (omega1 + omega2) / 2
        enveloppe_pos = 2 * A * np.abs(np.cos(delta_omega/2 * t))
        f_bat = delta_omega / (2 * np.pi)
        return x_total, x1, x2, enveloppe_pos, f_bat

    def representation_fresnel(self, amplitudes: list,
                                phases: list) -> tuple:
        """Représentation de Fresnel des vecteurs tournants."""
        Vx = [A * np.cos(phi) for A, phi in zip(amplitudes, phases)]
        Vy = [A * np.sin(phi) for A, phi in zip(amplitudes, phases)]
        # Résultante
        Ax_tot = sum(Vx)
        Ay_tot = sum(Vy)
        A_res = np.sqrt(Ax_tot**2 + Ay_tot**2)
        phi_res = np.arctan2(Ay_tot, Ax_tot)
        return Vx, Vy, A_res, phi_res

    def decomposition_fourier(self, signal_arr: np.ndarray,
                               dt: float) -> tuple:
        """Analyse spectrale par FFT."""
        N = len(signal_arr)
        freqs = rfftfreq(N, dt)
        F = rfft(signal_arr)
        magnitude = (2/N) * np.abs(F)
        phase_arr = np.angle(F, deg=True)
        return freqs, magnitude, phase_arr

    def signal_periodique(self, forme: str, f: float,
                           A: float, t: np.ndarray) -> np.ndarray:
        """Signaux périodiques standards."""
        from scipy.signal import square, sawtooth
        if forme == "Sinus":
            return A * np.sin(2*np.pi*f*t)
        elif forme == "Carré":
            return A * square(2*np.pi*f*t)
        elif forme == "Triangle":
            return A * sawtooth(2*np.pi*f*t, 0.5)
        elif forme == "Dent de scie":
            return A * sawtooth(2*np.pi*f*t)
        elif forme == "Impulsion":
            return A * (np.abs(np.mod(t*f, 1)) < 0.1).astype(float)
        return A * np.sin(2*np.pi*f*t)

    def serie_fourier_carre(self, f: float, A: float,
                             n_harm: int, t: np.ndarray) -> tuple:
        """Série de Fourier d'un signal carré."""
        x_approx = np.zeros_like(t)
        harmoniques = []
        for k in range(n_harm):
            n = 2*k + 1  # harmoniques impaires
            coeff = (4*A)/(np.pi*n)
            harm = coeff * np.sin(2*np.pi*n*f*t)
            x_approx += harm
            harmoniques.append((n, coeff, harm))
        return x_approx, harmoniques

    def sympy_parse_function(self, expr: str, var_str: object = "t"):
        """Parse une fonction utilisateur en expression Sympy.

        `var_str` peut être le nom du symbole (str) ou un objet `sympy.Symbol`.
        On s'assure d'utiliser le même symbole pour les opérations ultérieures.
        """
        if not HAS_SYMPY:
            raise RuntimeError("Sympy n'est pas disponible")
        expr = expr.strip()
        var_name = var_str if isinstance(var_str, str) else str(var_str)
        if expr in {"sin", "cos", "tan", "exp", "sqrt", "log", "sinh", "cosh", "tanh"}:
            expr = f"{expr}({var_name})"
        if hasattr(var_str, 'is_Symbol'):
            t = var_str
        else:
            t = sp.symbols(var_name, real=True)
        transformations = (
            standard_transformations +
            (implicit_multiplication_application,)
        )
        local_dict = {
            var_name: t,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "E": sp.E,
            "Abs": sp.Abs,
            "log": sp.log,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
        }
        return parse_expr(expr, transformations=transformations,
                          local_dict=local_dict, evaluate=True)

    def latexify_math_string(self, expr: str) -> str:
        if expr is None:
            return expr
        replacements = {
            'sin(': r'\\sin(',
            'cos(': r'\\cos(',
            'tan(': r'\\tan(',
            'exp(': r'\\exp(',
            'log(': r'\\log(',
            'sqrt(': r'\\sqrt(',
        }
        for plain, latex in replacements.items():
            expr = re.sub(rf'(?<!\\){re.escape(plain)}', latex, expr)
        return expr

    def fourier_series_symbolic(self, func_str: str,
                                var_str: str = "t",
                                period: float = 2*np.pi,
                                terms: int = 11,
                                interval: str = "symmetric"):
        """Calcule la série de Fourier symbolique d'une fonction périodique."""
        if not HAS_SYMPY:
            raise RuntimeError("Sympy n'est pas disponible")
        t = sp.symbols(var_str, real=True)
        func = self.sympy_parse_function(func_str, t)
        period_sym = sp.nsimplify(period, [sp.pi])
        if interval == "symmetric":
            interval_range = (t, -period_sym/2, period_sym/2)
        else:
            interval_range = (t, 0, period_sym)
        try:
            # If parse returned a callable-like object, make it an expression
            if callable(func):
                func = func(t)
        except Exception:
            pass

        series = sp.fourier_series(func, interval_range)
        if not hasattr(series, 'truncate'):
            raise RuntimeError(f"sp.fourier_series returned unexpected type: {type(series)}")
        expr = series.truncate(terms)
        # Try to convert floating numeric coefficients to exact symbolic
        # forms (e.g. rationals, multiples of pi) for nicer LaTeX output.
        try:
            expr = sp.nsimplify(expr, [sp.pi])
            expr = sp.simplify(expr)
        except Exception:
            try:
                expr = sp.simplify(expr)
            except Exception:
                pass
        # Compute general Fourier coefficients a_n (cos) and b_n (sin)
        try:
            n = sp.symbols('n', integer=True, positive=True)
            w0 = 2*sp.pi/period_sym
            a_mean = sp.simplify((1/period_sym) * sp.integrate(func, (t, interval_range[1], interval_range[2])))
            a_n = sp.simplify((2/period_sym) * sp.integrate(func * sp.cos(n*w0*t), (t, interval_range[1], interval_range[2])))
            b_n = sp.simplify((2/period_sym) * sp.integrate(func * sp.sin(n*w0*t), (t, interval_range[1], interval_range[2])))
            # Try to nsimplify coefficients for nicer latex
            try:
                a_mean = sp.nsimplify(a_mean, [sp.pi])
                a_n = sp.nsimplify(a_n, [sp.pi])
                b_n = sp.nsimplify(b_n, [sp.pi])
            except Exception:
                pass
        except Exception:
            n = sp.symbols('n', integer=True, positive=True)
            a_mean = None
            a_n = None
            b_n = None

        return func, series, expr, a_n, b_n, a_mean

    def fourier_transform_symbolic(self, func_str: str,
                                   time_var: str = "t",
                                   freq_var: str = "w"):
        """Calcule la transformée de Fourier symbolique d'une fonction."""
        if not HAS_SYMPY:
            raise RuntimeError("Sympy n'est pas disponible")
        t = sp.symbols(time_var, real=True)
        w = sp.symbols(freq_var, real=True)
        func = self.sympy_parse_function(func_str, time_var)
        X = sp.fourier_transform(func, t, w)
        return func, w, X

    def inverse_fourier_transform_symbolic(self, expr_w, w=None,
                                           time_var: str = "t"):
        """Calcule la transformée de Fourier inverse symbolique."""
        if not HAS_SYMPY:
            raise RuntimeError("Sympy n'est pas disponible")
        t = sp.symbols(time_var, real=True)
        if w is None:
            w = sp.symbols("w", real=True)
        x = sp.inverse_fourier_transform(expr_w, w, t)
        return t, x

    def fourier_series_expression(self, forme: str, A: float,
                                   f: float, n_terms: int) -> dict:
        """Retourne une expression LaTeX et un aperçu des termes de la série de Fourier."""
        omega = 2 * np.pi * f
        if forme == "Carré":
            formula = r"x(t) = \frac{4A}{\pi} \sum_{m=0}^{N-1} \frac{1}{2m+1} \sin\left((2m+1)\omega t\right)"
            term_list = []
            for k in range(n_terms):
                n = 2*k + 1
                coeff = 4*A/(np.pi*n)
                term_list.append(rf"{coeff:.3f}\sin({n}\omega t)")
        elif forme == "Triangle":
            formula = r"x(t) = \frac{8A}{\pi^2} \sum_{m=0}^{N-1} \frac{(-1)^m}{(2m+1)^2} \sin\left((2m+1)\omega t\right)"
            term_list = [rf"{8*A/(np.pi**2*(2*k+1)**2):.3f}\sin({2*k+1}\omega t)" for k in range(n_terms)]
        elif forme == "Dent de scie":
            formula = r"x(t) = \frac{2A}{\pi} \sum_{m=1}^{N} \frac{(-1)^{m+1}}{m} \sin(m\omega t)"
            term_list = [rf"{2*A/(np.pi*m)*((-1)**(m+1)):.3f}\sin({m}\omega t)" for m in range(1, n_terms+1)]
        elif forme == "Impulsion":
            formula = r"x(t) = \frac{A}{T} \sum_{m=-\infty}^{\infty} e^{i m \omega t}"
            term_list = [rf"\frac{{A}}{{T}} e^{{i {m} \omega t}}" for m in range(-n_terms//2, n_terms//2 + 1)]
        else:
            formula = r"x(t) = A \sin(\omega t)"
            term_list = [rf"A\sin(\omega t)"]
        return {
            "forme": forme,
            "formula": self.latexify_math_string(formula),
            "terms": [self.latexify_math_string(term) for term in term_list],
            "omega": omega,
        }

    # -------------------------------------------------------
    # CHAPITRE 4 — OSCILLATIONS AMORTIES
    # -------------------------------------------------------
    def solution_amortie(self, x0: float, v0: float,
                          t: np.ndarray) -> np.ndarray:
        """Solution exacte selon le régime d'amortissement."""
        xi, w0, wd = self.xi, self.omega0, self.omegad
        if xi < 1 - 1e-6:          # Sous-amorti
            A = np.sqrt(x0**2 + ((v0 + xi*w0*x0)/wd)**2)
            phi = np.arctan2(-(v0 + xi*w0*x0), wd*x0)
            return A * np.exp(-xi*w0*t) * np.cos(wd*t + phi)
        elif xi > 1 + 1e-6:        # Sur-amorti
            r1 = w0*(-xi + np.sqrt(xi**2 - 1))
            r2 = w0*(-xi - np.sqrt(xi**2 - 1))
            C2 = (v0 - r1*x0)/(r2 - r1)
            C1 = x0 - C2
            return C1*np.exp(r1*t) + C2*np.exp(r2*t)
        else:                       # Critique
            A = x0
            B = v0 + w0*x0
            return (A + B*t) * np.exp(-w0*t)

    def decrement_logarithmique(self) -> float:
        """δ = ln(x(t)/x(t+Td))"""
        if self.xi >= 1:
            return np.inf
        return 2*np.pi*self.xi / np.sqrt(1 - self.xi**2)

    def xi_depuis_decrement(self, delta: float) -> float:
        """Calcule ξ depuis le décrément logarithmique mesuré."""
        return delta / np.sqrt(4*np.pi**2 + delta**2)

    def enveloppe_amortie(self, A: float,
                           t: np.ndarray) -> tuple:
        """Enveloppe d'atténuation ±A·e^(-ξω₀t)."""
        env = A * np.exp(-self.xi * self.omega0 * t)
        return env, -env

    def temps_amortissement(self, A: float, fraction: float) -> float:
        """Temps pour que l'amplitude = fraction·A."""
        if self.xi <= 0:
            return np.inf
        return -np.log(fraction) / (self.xi * self.omega0)

    def bande_passante_Q(self) -> tuple:
        """Facteur de qualité Q et bande passante."""
        if self.xi <= 0:
            return np.inf, 0
        Q = 1 / (2 * self.xi)
        BW = self.omega0 / Q
        return Q, BW

    def reponse_forcee(self, F0: float, Omega: float,
                        t: np.ndarray) -> tuple:
        """Réponse forcée harmonique x = X·cos(Ωt - φ)."""
        r = Omega / self.omega0  # Rapport de fréquences
        denom = np.sqrt((1 - r**2)**2 + (2*self.xi*r)**2)
        X = (F0/self.k) / denom if denom > 0 else np.inf
        phi_resp = np.arctan2(2*self.xi*r, 1-r**2)
        x = X * np.cos(Omega*t - phi_resp)
        return x, X, phi_resp, r

    def courbe_resonance(self, r_range: np.ndarray,
                          xi_vals: list) -> dict:
        """Courbes de résonance pour différents ξ."""
        courbes = {}
        for xi in xi_vals:
            denom = np.sqrt((1 - r_range**2)**2 + (2*xi*r_range)**2)
            D = 1 / np.maximum(denom, 1e-10)
            courbes[xi] = D
        return courbes

    def comparaison_regimes(self, x0: float, v0: float,
                             t: np.ndarray) -> dict:
        """Compare sous-amorti, critique et sur-amorti."""
        resultats = {}
        for xi_val, label in [(0.1, "Sous-amorti ξ=0.1"),
                               (0.3, "Sous-amorti ξ=0.3"),
                               (1.0, "Critique ξ=1"),
                               (2.0, "Sur-amorti ξ=2")]:
            eng_tmp = VibrationEngine(self.m, self.k,
                                       xi_val * 2*np.sqrt(self.k*self.m))
            sol = eng_tmp.solution_amortie(x0, v0, t)
            resultats[label] = sol
        return resultats

    # -------------------------------------------------------
    # CHAPITRE 5 — PROPAGATION MILIEUX ILLIMITÉS
    # -------------------------------------------------------
    def onde_progressive(self, A: float, f: float,
                          c_onde: float, x: np.ndarray,
                          t_val: float, direction: str = "+") -> np.ndarray:
        """u(x,t) = A·cos(ωt ∓ kx)"""
        omega = 2*np.pi*f
        k = omega / c_onde
        signe = -1 if direction == "+" else +1
        return A * np.cos(omega*t_val + signe*k*x)

    def onde_stationnaire(self, A: float, f: float,
                           c_onde: float, x: np.ndarray,
                           t_vals: np.ndarray) -> np.ndarray:
        """Onde stationnaire = superposition des deux sens."""
        omega = 2*np.pi*f
        k = omega / c_onde
        result = np.zeros((len(t_vals), len(x)))
        for i, t_val in enumerate(t_vals):
            result[i] = 2*A*np.cos(k*x)*np.cos(omega*t_val)
        return result

    def dispersion(self, k_arr: np.ndarray,
                   type_milieu: str = "Non dispersif") -> np.ndarray:
        """Relation de dispersion ω(k)."""
        if type_milieu == "Non dispersif":
            c = CONSTANTES["c_son_air (m/s)"]
            return c * k_arr
        elif type_milieu == "Dispersif (ondes capillaires)":
            gamma = 0.072
            rho = 1000
            return np.sqrt(gamma/rho * k_arr**3)
        elif type_milieu == "Plasmon":
            omega_p = 1e14
            return np.sqrt(omega_p**2 + k_arr**2 * (3e8)**2)
        elif type_milieu == "Corde avec raideur":
            c = 340.0
            beta = 1e-4
            return c * k_arr * np.sqrt(1 + beta*k_arr**2)
        return k_arr * CONSTANTES["c_son_air (m/s)"]

    def vitesse_groupe(self, k_arr: np.ndarray,
                        omega_arr: np.ndarray) -> np.ndarray:
        """Vitesse de groupe vg = dω/dk."""
        return np.gradient(omega_arr, k_arr)

    def effet_doppler(self, f_source: float, c: float,
                       v_obs: float, v_source: float,
                       angle_deg: float = 0) -> float:
        """Fréquence observée — effet Doppler général."""
        angle = np.radians(angle_deg)
        f_obs = f_source * (c + v_obs*np.cos(angle)) / \
                (c - v_source*np.cos(angle))
        return f_obs

    def reflexion_transmission(self, Z1: float,
                                Z2: float) -> tuple:
        """Coefficients de réflexion et transmission (amplitude)."""
        r = (Z2 - Z1) / (Z2 + Z1)   # Réflexion
        t = 2*Z2 / (Z2 + Z1)         # Transmission
        R = r**2                       # Énergie réfléchie
        T = 1 - R                      # Énergie transmise
        return r, t, R, T

    def intensite_acoustique(self, A: float, f: float,
                              rho: float, c: float) -> float:
        """I = ½ρcω²A²"""
        omega = 2*np.pi*f
        return 0.5 * rho * c * omega**2 * A**2

    def niveau_sonore(self, I: float) -> float:
        """L = 10·log₁₀(I/I₀) en dB"""
        I0 = 1e-12
        return 10 * np.log10(max(I/I0, 1e-20))

    def paquet_onde_gaussien(self, x: np.ndarray,
                              k0: float, sigma_k: float,
                              t: float, c: float = 343.0) -> np.ndarray:
        """Paquet d'ondes gaussien avec dispersion."""
        sigma_x0 = 1 / (2*sigma_k)
        x_c = c * t
        sigma_t = np.sqrt(sigma_x0**2 + (sigma_k * c * t)**2 / sigma_x0**2)
        envelope = np.exp(-(x - x_c)**2 / (4*sigma_t**2))
        carrier = np.cos(k0*(x - c*t))
        return envelope * carrier


# ============================================================
# PAGE PRINCIPALE — ONDES & VIBRATIONS
# ============================================================
def ondes_vibrations_page():
    st.markdown("## 🌊 Ondes & Vibrations — Vue d'ensemble")
    st.markdown("*5 sections fondamentales traitées de façon interactive*")
    st.markdown("---")

    colors = ['#00ccff','#7700ff','#ff00cc','#00ff88',
               '#ffcc00','#ff4400','#88ccff','#cc88ff']

    PLOT_LAYOUT = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.92)',
        font=dict(color='#1f2937'),
    )
    AXIS_STYLE = dict(gridcolor='rgba(148,163,184,0.18)', color='#1f2937')

    def layout(fig, title="", xt="", yt="", h=420):
        fig.update_layout(**PLOT_LAYOUT, title=title,
                          xaxis_title=xt, yaxis_title=yt, height=h,
                          xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
                          legend=dict(bgcolor='rgba(0,0,0,0.5)'))
        return fig

    light_mode = st.checkbox(
        "Mode léger — accélérer le rendu des graphiques",
        value=True,
        help="Réduit légèrement la résolution des courbes pour un chargement plus rapide."
    )

    def resolution(n, min_n=200):
        n = int(n * (0.5 if light_mode else 1.0))
        return max(min_n, n)

    # ============================================================
    # SÉLECTION DE L'OPTION
    # ============================================================
    st.markdown("### 📚 Options — Vibrations")
    chapitre = st.radio(
        "Sélectionnez une option",
        list(CHAPITRES_VIBRATIONS.keys()),
        index=0,
        key="chapitre_vibrations"
    )
    ch_key = CHAPITRES_VIBRATIONS[chapitre]

    # ============================================================
    # CHAPITRE 1 — INTRODUCTION À LA VIBRATION
    # ============================================================
    if ch_key == "intro_vibration":
        st.markdown("## 📖 Section 1 — Introduction à la Vibration des Systèmes")
        st.markdown("""
        Un système vibratoire est caractérisé par :
        - Une **masse** *m* (énergie cinétique)
        - Une **raideur** *k* (énergie potentielle)
        - Un **amortisseur** *c* (dissipation)
        - Une **excitation** *F(t)* (source d'énergie)
        """)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Systèmes élémentaires",
            "🔬 Pendule",
            "⚡ Énergie",
            "📖 Formules"
        ])

        # ---- TAB 1 : SYSTÈMES ----
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### ⚙️ Paramètres du système")
                sys_type = st.selectbox("Type de système",
                    ["Ressort-masse", "Pendule simple",
                     "Système 2 DDL couplé"])
                m_val = st.slider("Masse m (kg)", 0.1, 50.0, 1.0, 0.1)
                k_val = st.slider("Raideur k (N/m)", 0.1, 500.0, 10.0, 0.1)
                x0_val = st.slider("Position initiale x₀ (m)", -5.0, 5.0, 1.0, 0.05)
                v0_val = st.slider("Vitesse initiale v₀ (m/s)", -10.0, 10.0, 0.0, 0.1)
                t_max = st.slider("Durée (s)", 1.0, 60.0, 10.0, 0.5)

                eng = VibrationEngine(m_val, k_val, 0)
                params = eng.systemes_vibratoires()

                st.markdown("### 📐 Caractéristiques")
                for k_p, v_p in params.items():
                    if v_p is not None:
                        st.metric(k_p, f"{v_p:.4f}")

            with col2:
                t = np.linspace(0, t_max, resolution(3000))

                if sys_type == "Ressort-masse":
                    x = eng.ressort_masse(x0_val, v0_val, t)
                    v = np.gradient(x, t)
                    titre = f"Ressort-Masse — ω₀={eng.omega0:.3f} rad/s"

                elif sys_type == "Pendule simple":
                    L_pend = st.slider("Longueur L (m)",
                                       0.1, 5.0, 1.0, 0.1)
                    theta_lin, omega_p, T_p = eng.pendule_simple(
                        L_pend, np.radians(x0_val*10), t)
                    theta_nl = eng.pendule_nonlineaire(
                        L_pend, x0_val*10, t)
                    x = np.degrees(theta_lin)
                    v = np.gradient(x, t)

                    fig_pend = go.Figure()
                    fig_pend.add_trace(go.Scatter(x=t, y=x, mode='lines',
                        name='Linéaire (petits angles)',
                        line=dict(color='#00ccff', width=2.5)))
                    fig_pend.add_trace(go.Scatter(x=t, y=np.degrees(theta_nl),
                        mode='lines', name='Non-linéaire (exact)',
                        line=dict(color='#ff00cc', width=2, dash='dash')))
                    layout(fig_pend, f"Pendule L={L_pend}m",
                           "t (s)", "θ (°)")
                    st.plotly_chart(fig_pend, use_container_width=True)
                    st.metric("T₀ (linéaire, s)", f"{T_p:.4f}")
                    st.stop()

                else:  # 2 DDL
                    m2 = st.slider("m₂ (kg)", 0.1, 10.0, 1.0, 0.1)
                    k12 = st.slider("k₁₂ (N/m)", 0.1, 100.0, 5.0, 0.1)
                    omegas, modes = eng.oscillateur_2DDL(
                        m_val, m2, k_val, k_val, k12)
                    st.metric("ω₁ (rad/s)", f"{omegas[0]:.4f}")
                    st.metric("ω₂ (rad/s)", f"{omegas[1]:.4f}")
                    st.markdown("**Modes propres :**")
                    st.dataframe(pd.DataFrame(modes,
                        columns=["Mode 1","Mode 2"],
                        index=["m₁","m₂"]).round(4),
                        use_container_width=True)
                    x = eng.ressort_masse(x0_val, v0_val, t)
                    v = np.gradient(x, t)
                    titre = "Système 2 DDL"

                fig = make_subplots(rows=2, cols=1,
                    subplot_titles=["x(t)", "ẋ(t)"])
                fig.add_trace(go.Scatter(x=t, y=x, mode='lines',
                    name='x(t)', line=dict(color='#00ccff', width=2.5)),
                    row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=v, mode='lines',
                    name='ẋ(t)', line=dict(color='#7700ff', width=2)),
                    row=2, col=1)
                fig.update_layout(**PLOT_LAYOUT, height=480,
                    title=titre, legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig.update_xaxes(**AXIS_STYLE, title_text="t (s)")
                fig.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig, use_container_width=True)

        # ---- TAB 2 : PENDULE DÉTAILLÉ ----
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                L_p = st.slider("L (m)", 0.1, 10.0, 1.0, 0.1, key="L_pend2")
                theta0_deg = st.slider("θ₀ (°)", 1.0, 170.0, 30.0, 1.0)
                t_sim = np.linspace(0, 4*2*np.pi*np.sqrt(L_p/9.81), resolution(1000))
                omega_lin = np.sqrt(9.81/L_p)
                T_lin = 2*np.pi/omega_lin

                # Correction non-linéaire de la période
                theta0_rad = np.radians(theta0_deg)
                T_nl = T_lin * (1 + (1/16)*theta0_rad**2 +
                                (11/3072)*theta0_rad**4)

                st.metric("T linéaire (s)", f"{T_lin:.4f}")
                st.metric("T non-linéaire (s)", f"{T_nl:.4f}")
                st.metric("Correction (%)",
                          f"{(T_nl-T_lin)/T_lin*100:.4f}")
                st.metric("f₀ (Hz)", f"{1/T_lin:.4f}")

            with col2:
                eng_p = VibrationEngine(1.0, 9.81/L_p, 0)
                theta_lin_2 = eng_p.ressort_masse(
                    np.radians(theta0_deg), 0, t_sim)
                theta_nl_2 = eng_p.pendule_nonlineaire(L_p, theta0_deg, t_sim)

                fig_p2 = go.Figure()
                fig_p2.add_trace(go.Scatter(x=t_sim,
                    y=np.degrees(theta_lin_2), mode='lines',
                    name='Linéaire', line=dict(color='#00ccff', width=2)))
                fig_p2.add_trace(go.Scatter(x=t_sim,
                    y=np.degrees(theta_nl_2), mode='lines',
                    name='Non-linéaire', line=dict(color='#ff00cc',
                    width=2, dash='dash')))
                layout(fig_p2, f"Pendule θ₀={theta0_deg}°", "t (s)", "θ (°)")
                st.plotly_chart(fig_p2, use_container_width=True)

        # ---- TAB 3 : ÉNERGIE ----
        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                m_e = st.slider("m (kg)", 0.1, 10.0, 1.0, 0.1, key="me")
                k_e = st.slider("k (N/m)", 1.0, 100.0, 10.0, 0.5, key="ke")
                A_e = st.slider("Amplitude A (m)", 0.1, 5.0, 1.0, 0.1)

                eng_e = VibrationEngine(m_e, k_e, 0)
                E_tot = 0.5 * k_e * A_e**2
                st.metric("E_tot = ½kA² (J)", f"{E_tot:.4f}")
                st.metric("E_c_max = ½mω₀²A² (J)",
                          f"{0.5*m_e*eng_e.omega0**2*A_e**2:.4f}")

            with col2:
                t_e = np.linspace(0, 4*np.pi/eng_e.omega0, resolution(500))
                x_e = eng_e.ressort_masse(A_e, 0, t_e)
                v_e = np.gradient(x_e, t_e)
                Ec, Ep, Et = eng_e.energie_mecanique(x_e, v_e)

                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=t_e, y=Ec, name='E cinétique',
                    line=dict(color='#00ccff', width=2.5)))
                fig_e.add_trace(go.Scatter(x=t_e, y=Ep, name='E potentielle',
                    line=dict(color='#7700ff', width=2.5)))
                fig_e.add_trace(go.Scatter(x=t_e, y=Et, name='E totale',
                    line=dict(color='#ffffff', width=2, dash='dash')))
                layout(fig_e, "Bilan énergétique", "t (s)", "E (J)")
                st.plotly_chart(fig_e, use_container_width=True)

                # Portrait de phase
                x_ph, v_ph = eng_e.portrait_phase(A_e, t_e)
                fig_ph = go.Figure()
                fig_ph.add_trace(go.Scatter(x=x_ph, y=v_ph, mode='lines',
                    name='Portrait de phase',
                    line=dict(color='#00ccff', width=2.5)))
                layout(fig_ph, "Portrait de phase (ellipse)", "x (m)", "ẋ (m/s)")
                st.plotly_chart(fig_ph, use_container_width=True)

        # ---- TAB 4 : FORMULES ----
        with tab4:
            st.markdown("### 📐 Formulaire — Vibrations libres non amorties")
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_VIBRATIONS.items():
                if any(k in nom for k in ["propre","Pendule","Ressort",
                                            "Pulsation","Énergie"]):
                    with cols[col_idx % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{nom}**")
                            safe_latex(f_latex)
                    col_idx += 1

    # ============================================================
    # CHAPITRE 2 — OSCILLATIONS LIBRES NON AMORTIES
    # ============================================================
    elif ch_key == "osc_libre_non_amorti":
        st.markdown("## 📖 Section 2 — Oscillations Libres Non Amorties (1 DDL)")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Solution & Paramètres",
            "🔄 Portrait de phase",
            "🎸 Modes propres (corde)",
            "🔗 Système 2 DDL",
            "📖 Formules"
        ])

        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                m2 = st.slider("m (kg)", 0.1, 50.0, 1.0, 0.1, key="m_ch2")
                k2 = st.slider("k (N/m)", 0.5, 500.0, 10.0, 0.5, key="k_ch2")
                x0_2 = st.slider("x₀ (m)", -5.0, 5.0, 2.0, 0.1, key="x0_ch2")
                v0_2 = st.slider("v₀ (m/s)", -20.0, 20.0, 0.0, 0.5, key="v0_ch2")
                t2 = st.slider("t_max (s)", 1.0, 50.0, 10.0, 0.5, key="t_ch2")

                eng2 = VibrationEngine(m2, k2, 0)
                A2, phi2 = eng2.decomposition_initiale(x0_2, v0_2)
                T0 = 2*np.pi/eng2.omega0

                st.metric("ω₀ (rad/s)", f"{eng2.omega0:.4f}")
                st.metric("T₀ (s)", f"{T0:.4f}")
                st.metric("f₀ (Hz)", f"{eng2.omega0/(2*np.pi):.4f}")
                st.metric("A (m)", f"{A2:.4f}")
                st.metric("φ (rad)", f"{phi2:.4f}")
                st.metric("φ (°)", f"{np.degrees(phi2):.2f}")

            with col2:
                t_arr2 = np.linspace(0, t2, resolution(3000))
                x_arr2 = eng2.solution_non_amortie(A2, phi2, t_arr2)
                Ec2, Ep2, Et2 = eng2.energie_mecanique(
                    x_arr2, np.gradient(x_arr2, t_arr2))

                fig2 = make_subplots(rows=2, cols=1,
                    subplot_titles=["x(t) = A·cos(ω₀t+φ)", "Énergie"])
                fig2.add_trace(go.Scatter(x=t_arr2, y=x_arr2, mode='lines',
                    name='x(t)', line=dict(color='#00ccff', width=2.5)),
                    row=1, col=1)
                fig2.add_hline(y=A2, line_color='#ffcc00', line_dash='dash',
                               annotation_text=f"A={A2:.2f}m", row=1, col=1)
                fig2.add_hline(y=-A2, line_color='#ffcc00', line_dash='dash',
                               row=1, col=1)
                fig2.add_trace(go.Scatter(x=t_arr2, y=Ec2, name='E_c',
                    line=dict(color='#00ccff', width=2)), row=2, col=1)
                fig2.add_trace(go.Scatter(x=t_arr2, y=Ep2, name='E_p',
                    line=dict(color='#7700ff', width=2)), row=2, col=1)
                fig2.add_trace(go.Scatter(x=t_arr2, y=Et2, name='E_tot',
                    line=dict(color='#ffffff', width=1.5, dash='dash')),
                    row=2, col=1)
                fig2.update_layout(**PLOT_LAYOUT, height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig2.update_xaxes(**AXIS_STYLE, title_text="t (s)")
                fig2.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig2, use_container_width=True)

                st.download_button("💾 Export CSV",
                    pd.DataFrame({"t":t_arr2,"x":x_arr2,"Ec":Ec2,
                                  "Ep":Ep2}).to_csv(index=False).encode(),
                    "oscil_non_amorti.csv", "text/csv")

        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                A_ph = st.slider("A (m)", 0.1, 5.0, 1.0, 0.1, key="Aph")
                m_ph = st.slider("m (kg)", 0.1, 10.0, 1.0, 0.1, key="mph")
                k_ph = st.slider("k (N/m)", 1.0, 100.0, 10.0, 1.0, key="kph")
                eng_ph = VibrationEngine(m_ph, k_ph, 0)
                st.metric("ω₀", f"{eng_ph.omega0:.4f} rad/s")
                st.info("""
                **Portrait de phase** d'un oscillateur non amorti :
                - Trajectoire **elliptique** fermée
                - Semi-axe horizontal : A (amplitude)
                - Semi-axe vertical : Aω₀ (vitesse max)
                - Sens **horaire** dans le plan (x, ẋ)
                """)
            with col2:
                t_ph2 = np.linspace(0, 4*np.pi/eng_ph.omega0, resolution(500))
                x_ph2, v_ph2 = eng_ph.portrait_phase(A_ph, t_ph2)

                # Plusieurs amplitudes
                fig_ph2 = go.Figure()
                for i, A_i in enumerate([A_ph*0.3, A_ph*0.6, A_ph]):
                    x_i, v_i = eng_ph.portrait_phase(A_i, t_ph2)
                    fig_ph2.add_trace(go.Scatter(x=x_i, y=v_i,
                        mode='lines', name=f'A={A_i:.2f}m',
                        line=dict(color=colors[i], width=2)))
                fig_ph2.add_trace(go.Scatter(x=[x_ph2[0]], y=[v_ph2[0]],
                    mode='markers', name='CI',
                    marker=dict(color='#ffcc00', size=12, symbol='star')))
                layout(fig_ph2, "Portrait de phase — ellipses",
                       "x (m)", "ẋ (m/s)")
                st.plotly_chart(fig_ph2, use_container_width=True)

        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                L_corde = st.slider("Longueur L (m)", 0.1, 5.0, 1.0, 0.1)
                T_tens = st.slider("Tension T (N)", 0.1, 1000.0, 100.0, 1.0)
                mu_lin = st.slider("μ linéique (g/m)", 0.1, 100.0, 5.0, 0.5) / 1000
                n_max_m = st.slider("Harmoniques max", 1, 10, 5)

                df_modes = VibrationEngine().modes_propres_corde(
                    L_corde, T_tens, mu_lin, n_max_m)
                st.dataframe(df_modes, use_container_width=True)
                c_corde = np.sqrt(T_tens/mu_lin)
                st.metric("Vitesse de l'onde (m/s)", f"{c_corde:.2f}")

            with col2:
                x_corde = np.linspace(0, L_corde, 300)
                fig_corde = go.Figure()
                for n in range(1, min(n_max_m+1, 7)):
                    y_mode = np.sin(n*np.pi*x_corde/L_corde)
                    fig_corde.add_trace(go.Scatter(x=x_corde,
                        y=y_mode + n*2.5, mode='lines',
                        name=f'n={n} | f={df_modes.iloc[n-1]["f_n (Hz)"]:.1f}Hz',
                        line=dict(color=colors[n-1], width=2.5)))
                layout(fig_corde, "Modes propres de la corde",
                       "x (m)", "ψ_n(x) + décalage")
                st.plotly_chart(fig_corde, use_container_width=True)

        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                m1_2ddl = st.slider("m₁ (kg)", 0.1, 10.0, 1.0, 0.1, key="m1_2")
                m2_2ddl = st.slider("m₂ (kg)", 0.1, 10.0, 1.5, 0.1, key="m2_2")
                k1_2ddl = st.slider("k₁ (N/m)", 0.1, 100.0, 10.0, 0.1, key="k1_2")
                k2_2ddl = st.slider("k₂ (N/m)", 0.1, 100.0, 10.0, 0.1, key="k2_2")
                k12_2ddl = st.slider("k₁₂ (N/m)", 0.0, 50.0, 5.0, 0.1, key="k12_2")

                eng_2ddl = VibrationEngine(m1_2ddl, k1_2ddl, 0)
                omegas, modes = eng_2ddl.oscillateur_2DDL(
                    m1_2ddl, m2_2ddl, k1_2ddl, k2_2ddl, k12_2ddl)

                for i, w in enumerate(omegas):
                    st.metric(f"ω_{i+1} (rad/s)", f"{w:.4f}")
                    st.metric(f"f_{i+1} (Hz)", f"{w/(2*np.pi):.4f}")

            with col2:
                t_2ddl = np.linspace(0, 20, resolution(2000))
                x1 = modes[0,0]*np.cos(omegas[0]*t_2ddl) + \
                     modes[0,1]*np.cos(omegas[1]*t_2ddl)
                x2 = modes[1,0]*np.cos(omegas[0]*t_2ddl) + \
                     modes[1,1]*np.cos(omegas[1]*t_2ddl)

                fig_2ddl = go.Figure()
                fig_2ddl.add_trace(go.Scatter(x=t_2ddl, y=x1/max(abs(x1)+1e-10),
                    mode='lines', name='m₁',
                    line=dict(color='#00ccff', width=2.5)))
                fig_2ddl.add_trace(go.Scatter(x=t_2ddl, y=x2/max(abs(x2)+1e-10),
                    mode='lines', name='m₂',
                    line=dict(color='#7700ff', width=2, dash='dash')))
                layout(fig_2ddl, "Réponse du système 2 DDL",
                       "t (s)", "x_i (normalisé)")
                st.plotly_chart(fig_2ddl, use_container_width=True)

                # Modes propres (diagramme barres)
                fig_modes = go.Figure()
                for i in range(2):
                    fig_modes.add_trace(go.Bar(
                        x=["m₁", "m₂"], y=modes[:, i],
                        name=f"Mode {i+1} (ω={omegas[i]:.2f})",
                        marker_color=colors[i]))
                layout(fig_modes, "Formes modales", "", "Amplitude (u.a.)")
                st.plotly_chart(fig_modes, use_container_width=True)

        with tab5:
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_VIBRATIONS.items():
                if any(k in nom for k in ["non amort","Solution non",
                                            "Pulsation","propre corde"]):
                    with cols[col_idx % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{nom}**")
                            safe_latex(f_latex)
                    col_idx += 1

    # ============================================================
    # CHAPITRE 3 — COMPOSITION HARMONIQUE
    # ============================================================
    elif ch_key == "composition_harmonique":
        st.markdown("## 📖 Section 3 — Composition des Grandeurs Harmoniques")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "➕ Superposition",
            "🥊 Battements",
            "🔄 Fresnel",
            "🎵 Fourier",
            "📖 Formules"
        ])

        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                n_harm = st.slider("Nombre d'harmoniques", 1, 6, 3,
                                   key="n_harm_sup")
                t_sup = np.linspace(0, 4*np.pi, resolution(2000))
                amps, omegas, phases = [], [], []
                for i in range(n_harm):
                    st.markdown(f"**Harmonique {i+1}**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        A_i = st.slider(f"A{i+1}", 0.1, 5.0, 1.0, 0.1,
                                        key=f"A_sup_{i}")
                    with c2:
                        w_i = st.slider(f"ω{i+1}", 0.5, 10.0, float(i+1), 0.1,
                                        key=f"w_sup_{i}")
                    with c3:
                        p_i = st.slider(f"φ{i+1} (°)", -180, 180, 0, 5,
                                        key=f"p_sup_{i}")
                    amps.append(A_i)
                    omegas.append(w_i)
                    phases.append(np.radians(p_i))

            with col2:
                eng_sup = VibrationEngine()
                x_tot, composantes = eng_sup.composition_harmoniques(
                    amps, omegas, phases, t_sup)

                fig_sup = go.Figure()
                for i, xi in enumerate(composantes):
                    fig_sup.add_trace(go.Scatter(x=t_sup, y=xi, mode='lines',
                        name=f'H{i+1} (A={amps[i]:.1f}, ω={omegas[i]:.1f})',
                        line=dict(color=colors[i], width=1.5, dash='dot'),
                        opacity=0.6))
                fig_sup.add_trace(go.Scatter(x=t_sup, y=x_tot, mode='lines',
                    name='Somme', line=dict(color='#ffffff', width=3)))
                layout(fig_sup, "Superposition d'harmoniques",
                       "t (s)", "x(t)")
                st.plotly_chart(fig_sup, use_container_width=True)

                # FFT
                dt_sup = t_sup[1]-t_sup[0]
                freqs_s, mag_s, _ = eng_sup.decomposition_fourier(x_tot, dt_sup)
                fig_fft_s = go.Figure()
                fig_fft_s.add_trace(go.Scatter(x=freqs_s, y=mag_s, mode='lines',
                    fill='tozeroy', fillcolor='rgba(119,0,255,0.2)',
                    line=dict(color='#7700ff', width=2), name='|FFT|'))
                layout(fig_fft_s, "Spectre FFT de la somme",
                       "f (rad/s → ×2π Hz)", "|FFT|")
                fig_fft_s.update_xaxes(range=[0, max(omegas)+2])
                st.plotly_chart(fig_fft_s, use_container_width=True)

        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                A_bat = st.slider("Amplitude A", 0.1, 5.0, 1.0, 0.1, key="A_bat")
                f1_bat = st.slider("f₁ (Hz)", 0.5, 20.0, 5.0, 0.1, key="f1_bat")
                f2_bat = st.slider("f₂ (Hz)", 0.5, 20.0, 5.5, 0.1, key="f2_bat")

                omega1 = 2*np.pi*f1_bat
                omega2 = 2*np.pi*f2_bat
                f_bat = abs(f2_bat - f1_bat)
                f_moy = (f1_bat + f2_bat) / 2

                st.metric("f battements (Hz)", f"{f_bat:.3f}")
                st.metric("T battements (s)", f"{1/f_bat:.3f}" if f_bat>0 else "∞")
                st.metric("f porteuse (Hz)", f"{f_moy:.3f}")
                st.info("""
                **Battements** : interférence de 2 sons proches.
                La fréquence de battement est |f₂ - f₁|.
                Le son perçu semble "pulsé" à cette fréquence.
                """)

            with col2:
                t_dur = max(5/f_bat, 2) if f_bat > 0 else 2
                t_bat = np.linspace(0, t_dur, resolution(8000))
                eng_bat = VibrationEngine()
                x_bat, x1_b, x2_b, env_bat, _ = eng_bat.battements(
                    A_bat, omega1, omega2, t_bat)

                fig_bat = make_subplots(rows=2, cols=1,
                    subplot_titles=["Signaux individuels",
                                    "Battements (somme) + Enveloppe"])
                fig_bat.add_trace(go.Scatter(x=t_bat, y=x1_b, mode='lines',
                    name=f'f₁={f1_bat}Hz',
                    line=dict(color='#00ccff', width=1.5)), row=1, col=1)
                fig_bat.add_trace(go.Scatter(x=t_bat, y=x2_b, mode='lines',
                    name=f'f₂={f2_bat}Hz',
                    line=dict(color='#7700ff', width=1.5)), row=1, col=1)
                fig_bat.add_trace(go.Scatter(x=t_bat, y=x_bat, mode='lines',
                    name='Battements', line=dict(color='#00ccff', width=2)),
                    row=2, col=1)
                fig_bat.add_trace(go.Scatter(x=t_bat, y=env_bat, mode='lines',
                    name='+Enveloppe',
                    line=dict(color='#ffcc00', width=2, dash='dash')),
                    row=2, col=1)
                fig_bat.add_trace(go.Scatter(x=t_bat, y=-env_bat, mode='lines',
                    name='-Enveloppe',
                    line=dict(color='#ffcc00', width=2, dash='dash'),
                    showlegend=False), row=2, col=1)
                fig_bat.update_layout(**PLOT_LAYOUT, height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_bat.update_xaxes(**AXIS_STYLE, title_text="t (s)")
                fig_bat.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig_bat, use_container_width=True)

        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                n_fr = st.slider("Nombre de vecteurs", 1, 5, 3, key="n_fr")
                amps_fr, phases_fr = [], []
                for i in range(n_fr):
                    c1, c2 = st.columns(2)
                    with c1:
                        A_f = st.slider(f"A{i+1}", 0.1, 5.0, 1.0, 0.1,
                                        key=f"Af_{i}")
                    with c2:
                        p_f = st.slider(f"φ{i+1} (°)", -180, 180, i*30, 5,
                                        key=f"pf_{i}")
                    amps_fr.append(A_f)
                    phases_fr.append(np.radians(p_f))

                eng_fr = VibrationEngine()
                Vx, Vy, A_res, phi_res = eng_fr.representation_fresnel(
                    amps_fr, phases_fr)

                st.metric("Amplitude résultante", f"{A_res:.4f}")
                st.metric("Phase résultante (°)", f"{np.degrees(phi_res):.2f}")

            with col2:
                fig_fr = go.Figure()
                x_cum, y_cum = 0, 0
                for i, (vx, vy) in enumerate(zip(Vx, Vy)):
                    fig_fr.add_annotation(
                        x=x_cum+vx, y=y_cum+vy,
                        ax=x_cum, ay=y_cum,
                        xref='x', yref='y', axref='x', ayref='y',
                        arrowhead=3, arrowsize=1.5, arrowwidth=3,
                        arrowcolor=colors[i]
                    )
                    fig_fr.add_trace(go.Scatter(
                        x=[x_cum, x_cum+vx], y=[y_cum, y_cum+vy],
                        mode='lines', name=f'A{i+1}={amps_fr[i]:.1f}',
                        line=dict(color=colors[i], width=3)))
                    x_cum += vx
                    y_cum += vy

                # Résultante
                fig_fr.add_annotation(x=A_res*np.cos(phi_res),
                    y=A_res*np.sin(phi_res), ax=0, ay=0,
                    xref='x', yref='y', axref='x', ayref='y',
                    arrowhead=3, arrowwidth=4, arrowcolor='#ffffff')

                max_r = sum(amps_fr) * 1.1
                fig_fr.update_layout(**PLOT_LAYOUT,
                    title="Représentation de Fresnel",
                    xaxis=dict(**AXIS_STYLE, range=[-max_r, max_r],
                               zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
                    yaxis=dict(**AXIS_STYLE, range=[-max_r, max_r],
                               zeroline=True, zerolinecolor='rgba(255,255,255,0.3)',
                               scaleanchor='x'),
                    height=500, legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_fr, use_container_width=True)

        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                forme = st.selectbox("Signal", ["Sinus","Carré","Triangle",
                                                "Dent de scie","Impulsion"])
                f_sig = st.slider("f (Hz)", 0.5, 20.0, 2.0, 0.1, key="f_sig")
                A_sig = st.slider("A", 0.1, 5.0, 1.0, 0.1, key="A_sig")
                n_four = st.slider("Harmoniques Fourier", 1, 20, 5, key="n_f")
                bruit_f = st.slider("Bruit σ", 0.0, 1.0, 0.0, 0.05)
                n_expr = st.slider("Termes de la série à afficher", 1, 11, 5, 2, key="n_expr")

                eng_f = VibrationEngine()
                expr_info = eng_f.fourier_series_expression(forme, A_sig, f_sig, n_expr)

                st.markdown("### Série de Fourier de la fonction choisie")
                safe_latex(expr_info["formula"])
                st.markdown("**Termes principaux :**")
                for term in expr_info["terms"]:
                    safe_latex(term)

                custom_expr = st.text_input(
                    "Fonction personnalisée f(t)",
                    "sin(t)",
                    key="fourier_custom_function"
                )
                custom_period = st.number_input(
                    "Période T", 0.1, 20.0, 2.0, 0.1,
                    key="fourier_custom_period"
                )
                custom_terms = st.slider(
                    "Termes calculés (sympy)", 1, 15, 7, 1,
                    key="fourier_custom_terms"
                )
                if HAS_SYMPY:
                    st.markdown("### 🧮 Outil Série de Fourier et Transformées")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        fn = st.text_input(
                            "Fonction périodique x(t)",
                            value=custom_expr,
                            help="Entrez une expression en t, par ex. sin(t)+t**2",
                            key="fourier_series_function_vib"
                        )
                        period = st.number_input(
                            "Période T", 0.1, 20.0, custom_period, 0.1,
                            help="Période du signal périodique",
                            key="fourier_series_period_vib"
                        )
                        interval_type = st.selectbox(
                            "Intervalle de définition",
                            ["Symétrique (-T/2, T/2)", "Positif (0, T)"],
                            key="fourier_series_interval_vib"
                        )
                        n_terms = st.slider(
                            "Nombre de termes", 3, 31, custom_terms, 2,
                            help="Nombre de termes renvoyés dans la série approchée",
                            key="fourier_series_terms_vib"
                        )
                        search_term = st.text_input(
                            "Recherche (ex: transformée, inverse, série)",
                            value="",
                            key="fourier_series_search_vib"
                        )
                        auto_tf = "transform" in search_term.lower() or "transformée" in search_term.lower()
                        auto_if = "inverse" in search_term.lower()
                        show_tf = st.checkbox(
                            "Afficher la transformée de Fourier symbolique",
                            value=auto_tf,
                            key="show_fourier_transform_vib"
                        )
                        show_if = st.checkbox(
                            "Afficher la transformée inverse de Fourier",
                            value=auto_if,
                            key="show_inverse_fourier_transform_vib"
                        )

                    with col2:
                        if fn.strip():
                            interval = "symmetric" if interval_type.startswith("Symétrique") else "positive"
                            try:
                                func, series_obj, series_expr, a_n_sym, b_n_sym, mean_sym = eng_f.fourier_series_symbolic(
                                    fn, "t", period, n_terms, interval
                                )
                                st.markdown("**Série de Fourier (approximation)**")
                                safe_latex(series_expr)
                                if mean_sym is not None:
                                    st.markdown("**Valeur moyenne**")
                                    safe_latex(mean_sym)
                                st.markdown("**Coefficient général a_n (cosinus)**")
                                if a_n_sym is not None:
                                    safe_latex(a_n_sym)
                                st.markdown("**Coefficient général b_n (sinus)**")
                                if b_n_sym is not None:
                                    safe_latex(b_n_sym)
                                # Option: afficher valeurs numériques de a_n/b_n
                                show_numeric = st.checkbox("Afficher valeurs numériques pour a_n/b_n", key="tf_show_numeric_vib")
                                if show_numeric:
                                    max_n = st.slider("Nombre de n à afficher", 1, 20, 5, key="tf_numeric_n_vib")
                                    n_sym = sp.symbols('n', integer=True, positive=True)
                                    if a_n_sym is not None:
                                        st.markdown("**Exemples numériques a_n**")
                                        for k in range(1, max_n+1):
                                            val = sp.N(a_n_sym.subs(n_sym, k))
                                            st.write(f"n={k}: {val}")
                                    if b_n_sym is not None:
                                        st.markdown("**Exemples numériques b_n**")
                                        for k in range(1, max_n+1):
                                            val = sp.N(b_n_sym.subs(n_sym, k))
                                            st.write(f"n={k}: {val}")
                            except Exception as err:
                                st.error(f"Impossible de calculer la série de Fourier : {err}")

                            if show_tf:
                                try:
                                    func, w, X = eng_f.fourier_transform_symbolic(fn, "t", "w")
                                    st.markdown("**Transformée de Fourier**")
                                    safe_latex(X)
                                except Exception as err:
                                    st.error(f"Impossible de calculer la transformée de Fourier : {err}")

                                if show_if:
                                    try:
                                        t_var, x_rec = eng_f.inverse_fourier_transform_symbolic(X, w, "t")
                                        st.markdown("**Transformée inverse de Fourier**")
                                        safe_latex(x_rec)
                                    except Exception as err:
                                        st.error(f"Impossible de calculer la TF inverse : {err}")
                else:
                    st.warning(
                        "Sympy n'est pas installé : installez 'sympy' pour activer le calcul symbolique."
                    )
                    st.markdown("`pip install sympy`")

                t_f = np.linspace(0, 4/f_sig, resolution(4096))
                s_f = eng_f.signal_periodique(forme, f_sig, A_sig, t_f)
                s_f += np.random.normal(0, bruit_f*A_sig, len(t_f))

            with col2:
                dt_f = t_f[1]-t_f[0]
                freqs_f, mag_f, phase_f = eng_f.decomposition_fourier(s_f, dt_f)

                fig_four = make_subplots(rows=3, cols=1,
                    subplot_titles=["Signal s(t)",
                                    "Spectre d'amplitude |FFT|",
                                    "Spectre de phase ∠FFT"])
                fig_four.add_trace(go.Scatter(x=t_f, y=s_f, mode='lines',
                    name=forme, line=dict(color='#00ccff', width=2)),
                    row=1, col=1)

                if forme == "Carré":
                    x_approx, harms = eng_f.serie_fourier_carre(
                        f_sig, A_sig, n_four, t_f)
                    fig_four.add_trace(go.Scatter(x=t_f, y=x_approx,
                        mode='lines', name=f'Fourier N={n_four}',
                        line=dict(color='#ffcc00', width=2, dash='dash')),
                        row=1, col=1)

                fig_four.add_trace(go.Bar(x=freqs_f[:len(freqs_f)//4],
                    y=mag_f[:len(freqs_f)//4],
                    marker_color='#7700ff', name='|FFT|'), row=2, col=1)
                fig_four.add_trace(go.Scatter(x=freqs_f[:len(freqs_f)//4],
                    y=phase_f[:len(freqs_f)//4], mode='markers',
                    marker=dict(color='#ff00cc', size=4), name='Phase'),
                    row=3, col=1)

                fig_four.update_layout(**PLOT_LAYOUT, height=650,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_four.update_xaxes(**AXIS_STYLE)
                fig_four.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig_four, use_container_width=True)

        with tab5:
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_VIBRATIONS.items():
                if any(k in nom for k in ["Composition","Battements","Fresnel"]):
                    with cols[col_idx % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{nom}**")
                            safe_latex(f_latex)
                    col_idx += 1

    # ============================================================
    # CHAPITRE 4 — OSCILLATIONS AMORTIES
    # ============================================================
    elif ch_key == "osc_libre_amorti":
        st.markdown("## 📖 Section 4 — Oscillations Libres Amorties (1 DDL)")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📉 Régimes d'amortissement",
            "🔄 Portrait de phase amorti",
            "📊 Décrément logarithmique",
            "📡 Réponse forcée",
            "📖 Formules"
        ])

        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                m_am = st.slider("m (kg)", 0.1, 10.0, 1.0, 0.1, key="m_am")
                k_am = st.slider("k (N/m)", 0.1, 100.0, 10.0, 0.5, key="k_am")
                xi_am = st.slider("ξ (amortissement)", 0.0, 3.0, 0.2, 0.01,
                                  key="xi_am")
                x0_am = st.slider("x₀ (m)", -5.0, 5.0, 2.0, 0.1, key="x0_am")
                v0_am = st.slider("v₀ (m/s)", -10.0, 10.0, 0.0, 0.1, key="v0_am")
                t_am = st.slider("t_max (s)", 1.0, 60.0, 20.0, 0.5, key="t_am")

                c_am = xi_am * 2 * np.sqrt(k_am * m_am)
                eng_am = VibrationEngine(m_am, k_am, c_am)

                regime = ("🟢 Sous-amorti" if xi_am < 1 else
                          "🟡 Critique" if abs(xi_am-1) < 0.01 else
                          "🔴 Sur-amorti")
                st.metric("Régime", regime)
                st.metric("ω₀ (rad/s)", f"{eng_am.omega0:.4f}")
                st.metric("ω_d (rad/s)", f"{eng_am.omegad:.4f}")
                if xi_am < 1:
                    delta = eng_am.decrement_logarithmique()
                    Q, BW = eng_am.bande_passante_Q()
                    st.metric("δ (décrément log.)", f"{delta:.4f}")
                    st.metric("Q (facteur qualité)", f"{Q:.3f}")
                    t_50 = eng_am.temps_amortissement(x0_am, 0.5)
                    st.metric("t pour A→50% (s)", f"{t_50:.3f}")

            with col2:
                t_arr_am = np.linspace(0, t_am, resolution(4000))
                x_am = eng_am.solution_amortie(x0_am, v0_am, t_arr_am)

                fig_am = go.Figure()
                fig_am.add_trace(go.Scatter(x=t_arr_am, y=x_am, mode='lines',
                    name=f'x(t) ξ={xi_am}',
                    line=dict(color='#00ccff', width=3)))

                if xi_am < 1:
                    env_p, env_m = eng_am.enveloppe_amortie(
                        abs(x0_am), t_arr_am)
                    fig_am.add_trace(go.Scatter(x=t_arr_am, y=env_p,
                        mode='lines', name='+Enveloppe',
                        line=dict(color='#ffcc00', width=2, dash='dash')))
                    fig_am.add_trace(go.Scatter(x=t_arr_am, y=env_m,
                        mode='lines', name='-Enveloppe',
                        line=dict(color='#ffcc00', width=2, dash='dash'),
                        showlegend=False))

                fig_am.add_hline(y=0, line_color='rgba(255,255,255,0.3)')
                layout(fig_am, f"Oscillateur amorti — {regime}",
                       "t (s)", "x(t)")
                st.plotly_chart(fig_am, use_container_width=True)

                # Comparaison tous les régimes
                fig_comp = go.Figure()
                xi_list = [0.1, 0.3, 1.0, 2.0]
                labels_xi = ["ξ=0.1 (sous)", "ξ=0.3 (sous)",
                             "ξ=1 (critique)", "ξ=2 (sur)"]
                for xi_i, lab_i, col_i in zip(xi_list, labels_xi, colors):
                    c_i = xi_i * 2*np.sqrt(k_am*m_am)
                    e_i = VibrationEngine(m_am, k_am, c_i)
                    x_i = e_i.solution_amortie(x0_am, v0_am, t_arr_am)
                    fig_comp.add_trace(go.Scatter(x=t_arr_am, y=x_i,
                        mode='lines', name=lab_i,
                        line=dict(color=col_i, width=2)))
                layout(fig_comp, "Comparaison des régimes d'amortissement",
                       "t (s)", "x(t)")
                st.plotly_chart(fig_comp, use_container_width=True)

        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                xi_ph2 = st.slider("ξ", 0.0, 0.99, 0.2, 0.01, key="xi_ph2")
                x0_ph2 = st.slider("x₀", -5.0, 5.0, 2.0, 0.1, key="x0_ph2")
                v0_ph2 = st.slider("v₀", -10.0, 10.0, 0.0, 0.5, key="v0_ph2")
                t_ph3 = st.slider("t_max", 1.0, 30.0, 10.0, 0.5, key="t_ph3")

                c_ph2 = xi_ph2 * 2*np.sqrt(10.0*1.0)
                eng_ph3 = VibrationEngine(1.0, 10.0, c_ph2)

                st.info("""
                **Portrait de phase amorti** :
                - Spirale **convergente** vers (0,0)
                - Vitesse de convergence ∝ ξ
                - Aucune spirale si ξ = 0 (ellipse)
                """)

            with col2:
                t_ph3_arr = np.linspace(0, t_ph3, resolution(3000))
                x_ph3 = eng_ph3.solution_amortie(x0_ph2, v0_ph2, t_ph3_arr)
                v_ph3 = np.gradient(x_ph3, t_ph3_arr)

                fig_ph3 = go.Figure()
                N = len(t_ph3_arr)
                for j in range(0, N-1, N//50):
                    fig_ph3.add_annotation(
                        x=x_ph3[min(j+5,N-1)], y=v_ph3[min(j+5,N-1)],
                        ax=x_ph3[j], ay=v_ph3[j],
                        xref='x', yref='y', axref='x', ayref='y',
                        arrowhead=2, arrowsize=0.8, arrowwidth=1,
                        arrowcolor=f'rgba(0,204,255,{0.3+0.7*j/N:.2f})'
                    )
                fig_ph3.add_trace(go.Scatter(x=x_ph3, y=v_ph3, mode='lines',
                    name=f'ξ={xi_ph2}',
                    line=dict(color='#00ccff', width=2.5)))
                fig_ph3.add_trace(go.Scatter(x=[x0_ph2], y=[v0_ph2],
                    mode='markers', name='Départ',
                    marker=dict(color='#ffcc00', size=12, symbol='star')))
                fig_ph3.add_trace(go.Scatter(x=[0], y=[0],
                    mode='markers', name='Équilibre',
                    marker=dict(color='#ff0000', size=10, symbol='x')))
                layout(fig_ph3, f"Portrait de phase amorti (ξ={xi_ph2})",
                       "x (m)", "ẋ (m/s)")
                st.plotly_chart(fig_ph3, use_container_width=True)

        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 📊 Mesure expérimentale du décrément")
                xi_dec = st.slider("ξ réel", 0.01, 0.9, 0.15, 0.01, key="xi_dec")
                n_periodes = st.slider("Nombre de périodes", 1, 20, 5, key="n_perd")
                bruit_dec = st.slider("Bruit de mesure σ", 0.0, 0.5, 0.05, 0.01)

                eng_dec = VibrationEngine(1.0, 10.0, xi_dec*2*np.sqrt(10))
                t_dec = np.linspace(0, n_periodes*2*np.pi/eng_dec.omegad, resolution(1000))
                x_dec = eng_dec.solution_amortie(2.0, 0.0, t_dec)
                x_dec_mesure = x_dec + np.random.normal(0, bruit_dec, len(t_dec))

                # Pics successifs
                peaks_idx, _ = find_peaks(x_dec_mesure, height=0.01)
                if len(peaks_idx) > 1:
                    xi_mesure = eng_dec.xi_depuis_decrement(
                        np.mean(np.diff(np.log(x_dec_mesure[peaks_idx]))))
                    st.metric("ξ mesuré", f"{abs(xi_mesure):.4f}")
                    st.metric("ξ réel", f"{xi_dec:.4f}")
                    st.metric("Erreur (%)", f"{abs(abs(xi_mesure)-xi_dec)/xi_dec*100:.2f}")
                delta_th = eng_dec.decrement_logarithmique()
                st.metric("δ théorique", f"{delta_th:.4f}")

            with col2:
                fig_dec = go.Figure()
                fig_dec.add_trace(go.Scatter(x=t_dec, y=x_dec_mesure,
                    mode='lines', name='Signal mesuré',
                    line=dict(color='rgba(0,204,255,0.5)', width=1.5)))
                fig_dec.add_trace(go.Scatter(x=t_dec, y=x_dec, mode='lines',
                    name='Signal théorique',
                    line=dict(color='#00ccff', width=2.5)))
                if len(peaks_idx) > 0:
                    fig_dec.add_trace(go.Scatter(
                        x=t_dec[peaks_idx], y=x_dec_mesure[peaks_idx],
                        mode='markers', name='Pics',
                        marker=dict(color='#ffcc00', size=10, symbol='star')))
                # Enveloppe
                env_dec, _ = eng_dec.enveloppe_amortie(2.0, t_dec)
                fig_dec.add_trace(go.Scatter(x=t_dec, y=env_dec, mode='lines',
                    name='Enveloppe',
                    line=dict(color='#ff00cc', width=2, dash='dash')))
                layout(fig_dec, f"Décrément logarithmique — ξ={xi_dec}",
                       "t (s)", "x(t)")
                st.plotly_chart(fig_dec, use_container_width=True)

                if len(peaks_idx) > 1:
                    df_pics = pd.DataFrame({
                        "n": range(len(peaks_idx)),
                        "t_n (s)": t_dec[peaks_idx].round(4),
                        "x_n (m)": x_dec_mesure[peaks_idx].round(4),
                    })
                    if len(df_pics) > 1:
                        df_pics["ln(x_n/x_{n+1})"] = [
                            round(np.log(df_pics.iloc[i]["x_n (m)"] /
                                  max(df_pics.iloc[i+1]["x_n (m)"], 1e-10)), 4)
                            for i in range(len(df_pics)-1)
                        ] + [None]
                    st.dataframe(df_pics, use_container_width=True)

        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 📡 Résonance forcée")
                xi_res = st.slider("ξ", 0.01, 2.0, 0.2, 0.01, key="xi_res")
                F0_res = st.slider("F₀ (N)", 0.1, 50.0, 1.0, 0.1, key="F0_res")
                Omega_res = st.slider("Ω/ω₀", 0.1, 3.0, 1.0, 0.01, key="Om_res")

                eng_res = VibrationEngine(1.0, 10.0, xi_res*2*np.sqrt(10))
                Omega = Omega_res * eng_res.omega0
                t_res = np.linspace(0, 20, resolution(3000))
                x_res, X_res, phi_r, r_res = eng_res.reponse_forcee(
                    F0_res, Omega, t_res)

                st.metric("Amplitude statique x_st", f"{F0_res/10:.4f} m")
                st.metric("Amplitude réponse X", f"{X_res:.4f} m")
                st.metric("Amplification D", f"{X_res/(F0_res/10):.4f}")
                st.metric("Phase φ (°)", f"{np.degrees(phi_r):.2f}")
                st.metric("r = Ω/ω₀", f"{r_res:.3f}")

                if xi_res < 1/np.sqrt(2):
                    r_res_opt = np.sqrt(1 - 2*xi_res**2)
                    D_max = 1/(2*xi_res*np.sqrt(1-xi_res**2))
                    st.metric("r_résonance", f"{r_res_opt:.4f}")
                    st.metric("D_max", f"{D_max:.4f}")

            with col2:
                r_range = np.linspace(0.01, 3.0, resolution(500))
                xi_vals_r = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
                courbes = eng_res.courbe_resonance(r_range, xi_vals_r)

                fig_res = go.Figure()
                for xi_i, col_i in zip(xi_vals_r, colors):
                    fig_res.add_trace(go.Scatter(x=r_range,
                        y=courbes[xi_i], mode='lines',
                        name=f'ξ={xi_i}',
                        line=dict(color=col_i, width=2)))
                fig_res.add_vline(x=1, line_color='rgba(255,255,255,0.3)',
                                  line_dash='dash',
                                  annotation_text="Résonance")
                fig_res.add_vline(x=Omega_res, line_color='#ffcc00',
                                  line_dash='dot',
                                  annotation_text=f"Ω/ω₀={Omega_res:.2f}")
                layout(fig_res, "Courbes de résonance D(r)",
                       "r = Ω/ω₀", "D = X/(F₀/k)")
                st.plotly_chart(fig_res, use_container_width=True)

                # Réponse temporelle
                fig_t_res = go.Figure()
                fig_t_res.add_trace(go.Scatter(x=t_res, y=x_res, mode='lines',
                    name='x(t) = X·cos(Ωt-φ)',
                    line=dict(color='#00ccff', width=2.5)))
                fig_t_res.add_hline(y=X_res, line_color='#ffcc00',
                                    line_dash='dash',
                                    annotation_text=f"X={X_res:.3f}")
                fig_t_res.add_hline(y=-X_res, line_color='#ffcc00',
                                    line_dash='dash')
                layout(fig_t_res, "Réponse forcée harmonique",
                       "t (s)", "x(t)")
                st.plotly_chart(fig_t_res, use_container_width=True)

        with tab5:
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_VIBRATIONS.items():
                if any(k in nom for k in ["amort","Amort","décr",
                                            "Décrément","bande"]):
                    with cols[col_idx % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{nom}**")
                            safe_latex(f_latex)
                    col_idx += 1

    # ============================================================
    # CHAPITRE 5 — PROPAGATION MILIEUX ILLIMITÉS
    # ============================================================
    elif ch_key == "propagation_milieux":
        st.markdown("## 📖 Section 5 — Propagation dans les Milieux Illimités")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🌊 Onde progressive",
            "📊 Onde stationnaire",
            "📡 Dispersion",
            "🔊 Acoustique",
            "🚀 Doppler",
            "📖 Formules"
        ])

        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                A_op = st.slider("Amplitude A (m)", 0.01, 5.0, 1.0, 0.05)
                f_op = st.slider("Fréquence f (Hz)", 0.1, 50.0, 5.0, 0.1)
                c_op = st.slider("Célérité c (m/s)", 1.0, 1000.0, 343.0, 1.0)
                direction = st.radio("Direction", ["+ (droite)", "- (gauche)"],
                                     horizontal=True)
                t_op = st.slider("Temps t (s)", 0.0, 2.0, 0.0, 0.01)

                omega_op = 2*np.pi*f_op
                k_op = omega_op / c_op
                lam_op = c_op / f_op

                st.metric("λ (m)", f"{lam_op:.4f}")
                st.metric("k (rad/m)", f"{k_op:.4f}")
                st.metric("ω (rad/s)", f"{omega_op:.4f}")
                st.metric("T (s)", f"{1/f_op:.4f}")

                sigma = st.slider("σ paquet d'onde (m)", 0.1, 5.0, 1.0, 0.1)
                show_paquet = st.checkbox("Superposer paquet d'onde", False)

            with col2:
                x_op = np.linspace(-3*lam_op, 3*lam_op, resolution(2000))
                dir_str = "+" if "+" in direction else "-"
                eng_op = VibrationEngine()
                u_op = eng_op.onde_progressive(A_op, f_op, c_op, x_op,
                                               t_op, dir_str)
                u_op2 = eng_op.onde_progressive(A_op, f_op, c_op, x_op,
                                                t_op + 1/f_op/4, dir_str)

                fig_op = go.Figure()
                fig_op.add_trace(go.Scatter(x=x_op, y=u_op, mode='lines',
                    name=f't={t_op:.2f}s',
                    line=dict(color='#00ccff', width=3)))
                fig_op.add_trace(go.Scatter(x=x_op, y=u_op2, mode='lines',
                    name=f't+T/4',
                    line=dict(color='rgba(119,0,255,0.5)', width=2,
                              dash='dash')))

                if show_paquet:
                    paq = eng_op.paquet_onde_gaussien(
                        x_op, k_op, 1/sigma, t_op, c_op)
                    fig_op.add_trace(go.Scatter(x=x_op, y=np.real(paq),
                        mode='lines', name='Paquet d\'onde',
                        line=dict(color='#ffcc00', width=2)))

                layout(fig_op, f"Onde progressive c={c_op} m/s",
                       "x (m)", "u(x,t)")
                st.plotly_chart(fig_op, use_container_width=True)

                # Évolution spatio-temporelle
                st.markdown("#### 🕐 Diagramme espace-temps (x,t)")
                x_xt = np.linspace(-3*lam_op, 3*lam_op, 150)
                t_xt = np.linspace(0, 3/f_op, 150)
                X_xt, T_xt = np.meshgrid(x_xt, t_xt)
                k_xt = omega_op / c_op
                signe_xt = -1 if "+" in direction else +1
                U_xt = A_op * np.cos(omega_op*T_xt + signe_xt*k_xt*X_xt)

                fig_xt = go.Figure(data=go.Heatmap(
                    z=U_xt, x=x_xt, y=t_xt,
                    colorscale=[[0,'#020817'],[0.5,'#7700ff'],[1,'#ffffff']],
                    colorbar=dict(title='u(x,t)',
                                  tickfont=dict(color='#c0d0ff'))
                ))
                fig_xt.update_layout(**PLOT_LAYOUT,
                    title="Diagramme espace-temps",
                    xaxis=dict(**AXIS_STYLE, title="x (m)"),
                    yaxis=dict(**AXIS_STYLE, title="t (s)"),
                    height=380)
                st.plotly_chart(fig_xt, use_container_width=True)

        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                A_stat = st.slider("A", 0.1, 3.0, 1.0, 0.1, key="A_stat")
                f_stat = st.slider("f (Hz)", 0.5, 20.0, 5.0, 0.5, key="f_stat")
                c_stat = st.slider("c (m/s)", 10.0, 500.0, 100.0, 10.0, key="c_stat")
                L_stat = st.slider("L (m)", 0.1, 10.0, 1.0, 0.1, key="L_stat")

                omega_s = 2*np.pi*f_stat
                k_s = omega_s / c_stat
                lam_s = 2*np.pi/k_s
                n_noeuds = int(L_stat / (lam_s/2))

                st.metric("λ (m)", f"{lam_s:.4f}")
                st.metric("Nœuds dans L", n_noeuds)
                st.metric("Ventres dans L", n_noeuds+1)

                st.info("""
                **Onde stationnaire** = superposition de deux ondes
                progressives de même amplitude et fréquence,
                se propageant en sens opposés.
                - **Nœuds** : u = 0 en tout temps
                - **Ventres** : amplitude maximale
                """)

            with col2:
                x_stat = np.linspace(0, L_stat, 300)
                t_stat = np.linspace(0, 2/f_stat, 16)
                eng_stat = VibrationEngine()
                U_stat = eng_stat.onde_stationnaire(
                    A_stat, f_stat, c_stat, x_stat, t_stat)

                fig_stat = go.Figure()
                for i, t_i in enumerate(t_stat):
                    alpha = 0.3 + 0.7*(i/len(t_stat))
                    fig_stat.add_trace(go.Scatter(x=x_stat, y=U_stat[i],
                        mode='lines', name=f't={t_i:.3f}s',
                        line=dict(color=f'rgba(0,204,255,{alpha:.2f})',
                                  width=1.5)))

                # Enveloppe
                env_stat = 2*A_stat*np.abs(np.cos(k_s*x_stat))
                fig_stat.add_trace(go.Scatter(x=x_stat, y=env_stat,
                    mode='lines', name='Enveloppe',
                    line=dict(color='#ffcc00', width=2.5, dash='dash')))
                fig_stat.add_trace(go.Scatter(x=x_stat, y=-env_stat,
                    mode='lines', name='-Enveloppe',
                    line=dict(color='#ffcc00', width=2.5, dash='dash'),
                    showlegend=False))

                # Nœuds
                x_noeuds = [(2*n+1)*np.pi/(2*k_s) for n in range(10)
                             if (2*n+1)*np.pi/(2*k_s) <= L_stat]
                if x_noeuds:
                    fig_stat.add_trace(go.Scatter(x=x_noeuds,
                        y=[0]*len(x_noeuds), mode='markers',
                        name='Nœuds',
                        marker=dict(color='#ff0000', size=10, symbol='x')))

                layout(fig_stat, "Onde stationnaire — snapshots",
                       "x (m)", "u(x,t)")
                st.plotly_chart(fig_stat, use_container_width=True)

        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                milieu = st.selectbox("Milieu", [
                    "Non dispersif",
                    "Dispersif (ondes capillaires)",
                    "Plasmon",
                    "Corde avec raideur"
                ])
                k_max = st.slider("k_max (rad/m)", 0.1, 100.0, 10.0, 0.1)

            with col2:
                k_arr = np.linspace(0.01, k_max, resolution(500))
                eng_disp = VibrationEngine()
                omega_arr = eng_disp.dispersion(k_arr, milieu)
                vph = omega_arr / k_arr
                vg = eng_disp.vitesse_groupe(k_arr, omega_arr)

                fig_disp = make_subplots(rows=2, cols=2,
                    subplot_titles=["ω(k)", "v_φ(k)", "v_g(k)",
                                    "v_φ vs v_g"])
                fig_disp.add_trace(go.Scatter(x=k_arr, y=omega_arr,
                    line=dict(color='#00ccff', width=2.5),
                    name='ω(k)'), row=1, col=1)
                fig_disp.add_trace(go.Scatter(x=k_arr, y=vph,
                    line=dict(color='#7700ff', width=2.5),
                    name='v_φ'), row=1, col=2)
                fig_disp.add_trace(go.Scatter(x=k_arr, y=vg,
                    line=dict(color='#ff00cc', width=2.5),
                    name='v_g'), row=2, col=1)
                fig_disp.add_trace(go.Scatter(x=vph, y=vg, mode='lines',
                    line=dict(color='#00ff88', width=2.5),
                    name='v_g vs v_φ'), row=2, col=2)

                fig_disp.update_layout(**PLOT_LAYOUT, height=500,
                    title=f"Dispersion — {milieu}",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_disp.update_xaxes(**AXIS_STYLE)
                fig_disp.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig_disp, use_container_width=True)

        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔊 Acoustique")
                A_ac = st.slider("Amplitude A (μm)", 0.001, 100.0, 1.0, 0.01)
                f_ac = st.slider("f (Hz)", 20.0, 20000.0, 1000.0, 10.0)
                rho_ac = st.slider("ρ (kg/m³)", 0.1, 2000.0, 1.225, 0.1)
                c_ac = st.slider("c (m/s)", 100.0, 5000.0, 343.0, 1.0)

                A_m = A_ac * 1e-6
                eng_ac = VibrationEngine()
                I = eng_ac.intensite_acoustique(A_m, f_ac, rho_ac, c_ac)
                L_dB = eng_ac.niveau_sonore(I)
                Z_ac = rho_ac * c_ac
                P_max = rho_ac * c_ac * (2*np.pi*f_ac) * A_m

                st.metric("Intensité I (W/m²)", f"{I:.4e}")
                st.metric("Niveau L (dB)", f"{L_dB:.2f}")
                st.metric("Z = ρc (Pa·s/m)", f"{Z_ac:.1f}")
                st.metric("Pression max (Pa)", f"{P_max:.4f}")

                st.markdown("### 🔀 Réflexion / Transmission")
                Z1 = st.slider("Z₁ (air, Pa·s/m)", 100.0, 5000.0, 415.0, 10.0)
                Z2 = st.slider("Z₂ (eau, Pa·s/m)", 100.0, 2e6, 1.5e6, 1000.0,
                               format="%.0f")
                r, t_coef, R, T = eng_ac.reflexion_transmission(Z1, Z2)
                st.metric("r (réflexion ampl.)", f"{r:.4f}")
                st.metric("t (transmission ampl.)", f"{t_coef:.4f}")
                st.metric("R (énergie réfl.)", f"{R*100:.2f}%")
                st.metric("T (énergie trans.)", f"{T*100:.2f}%")

            with col2:
                # Niveaux sonores de référence
                refs_dB = {
                    "Seuil audition":  0,
                    "Feuilles bruissent": 20,
                    "Conversation": 60,
                    "Voiture":     70,
                    "Concert rock": 110,
                    "Moteur à réaction": 140,
                    "Rupture du tympan": 160,
                }
                fig_dB = go.Figure(go.Bar(
                    x=list(refs_dB.keys()), y=list(refs_dB.values()),
                    marker=dict(
                        color=list(refs_dB.values()),
                        colorscale=[[0,'#00ff88'],[0.5,'#ffcc00'],[1,'#ff0000']],
                        showscale=True,
                        colorbar=dict(title='dB', tickfont=dict(color='#c0d0ff'))
                    )
                ))
                fig_dB.add_hline(y=L_dB, line_color='#00ccff', line_dash='dash',
                                 annotation_text=f"Notre source: {L_dB:.1f} dB")
                fig_dB.update_layout(**PLOT_LAYOUT,
                    title="Niveaux sonores de référence",
                    xaxis=dict(**AXIS_STYLE),
                    yaxis=dict(**AXIS_STYLE, title="Niveau (dB)"),
                    height=380)
                st.plotly_chart(fig_dB, use_container_width=True)

                # Propagation sphérique
                r_sph = np.logspace(-1, 3, 300)
                I_sph = I * 1.0 / (4*np.pi*r_sph**2 + 1e-30)
                L_sph = 10*np.log10(np.maximum(I_sph/1e-12, 1e-30))
                fig_sph = go.Figure()
                fig_sph.add_trace(go.Scatter(x=r_sph, y=L_sph, mode='lines',
                    name='L(r) = L₀ - 20log(r)',
                    line=dict(color='#00ccff', width=2.5)))
                fig_sph.update_layout(**PLOT_LAYOUT,
                    title="Propagation sphérique — L(r)",
                    xaxis=dict(**AXIS_STYLE, type='log', title="r (m)"),
                    yaxis=dict(**AXIS_STYLE, title="L (dB)"),
                    height=300)
                st.plotly_chart(fig_sph, use_container_width=True)

        with tab5:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🚀 Effet Doppler")
                f_src = st.slider("f source (Hz)", 100.0, 5000.0, 1000.0, 10.0)
                c_milieu = st.slider("c (m/s)", 100.0, 1000.0, 343.0, 1.0)
                v_obs = st.slider("v observateur (m/s)", -100.0, 100.0, 0.0, 1.0)
                v_src = st.slider("v source (m/s)", -c_milieu*0.9,
                                   c_milieu*0.9, 0.0, 1.0)
                angle = st.slider("Angle (°)", 0, 180, 0, 5)

                f_obs = VibrationEngine().effet_doppler(
                    f_src, c_milieu, v_obs, v_src, angle)

                st.metric("f observée (Hz)", f"{f_obs:.2f}")
                st.metric("Δf (Hz)", f"{f_obs - f_src:.2f}")
                st.metric("Décalage (%)", f"{(f_obs-f_src)/f_src*100:.2f}")
                signe = "🔵 Blueshift" if f_obs > f_src else "🔴 Redshift"
                st.metric("Effet", signe)

                Ma_src = abs(v_src) / c_milieu
                if Ma_src > 0.9:
                    st.warning(f"⚠️ Mach ≈ {Ma_src:.2f} — régime sonique!")

            with col2:
                v_src_arr = np.linspace(-c_milieu*0.9, c_milieu*0.9, resolution(400))
                f_obs_arr = np.array([
                    VibrationEngine().effet_doppler(f_src, c_milieu, 0, vs)
                    for vs in v_src_arr
                ])

                fig_dop = go.Figure()
                fig_dop.add_trace(go.Scatter(x=v_src_arr, y=f_obs_arr,
                    mode='lines', name='f_obs(v_s)',
                    line=dict(color='#00ccff', width=3)))
                fig_dop.add_hline(y=f_src, line_color='#ffcc00',
                                   line_dash='dash',
                                   annotation_text=f"f_s={f_src}Hz")
                fig_dop.add_vline(x=v_src, line_color='#ff00cc',
                                   line_dash='dot',
                                   annotation_text=f"v_s={v_src}m/s")
                layout(fig_dop, "Effet Doppler — f_obs vs v_source",
                       "v_source (m/s)", "f observée (Hz)")
                st.plotly_chart(fig_dop, use_container_width=True)

                # Cônes de Mach
                st.markdown("#### 🔺 Onde de choc (Mach > 1)")
                if Ma_src > 1:
                    angle_mach = np.degrees(np.arcsin(1/Ma_src))
                    st.metric("Angle de Mach (°)", f"{angle_mach:.2f}")
                    st.metric("sin(α) = 1/Ma", f"{1/Ma_src:.4f}")

        with tab6:
            st.markdown("### 📖 Formulaire — Propagation")
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_VIBRATIONS.items():
                if any(k in nom for k in ["onde","Onde","intensit","Doppler",
                                           "stationnaire","Mode","Impédance",
                                           "Intensité","Niveau"]):
                    with cols[col_idx % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{nom}**")
                            safe_latex(f_latex)
                    col_idx += 1

    # Export global
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Formulaire complet")
    if st.sidebar.button("📖 Afficher toutes les formules"):
        for nom, f_latex in FORMULES_VIBRATIONS.items():
            safe_latex(f_latex)
