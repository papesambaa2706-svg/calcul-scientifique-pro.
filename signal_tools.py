__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import numpy as np

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
from scipy import signal as sp_signal
from scipy import stats, integrate
from scipy.fft import fft, fftfreq, ifft, rfft, rfftfreq, fftshift
from scipy.signal import find_peaks, welch, spectrogram, butter, filtfilt
import pandas as pd
import importlib
try:
    sp = importlib.import_module('sympy')
    # import submodule dynamically to avoid static analyzer failures
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FORMULAIRE — SIGNAL CLASSIQUE (existant)
# ============================================================
FORMULES_SIGNAL = {
    "Transformée de Fourier":  r"\hat{f}(\nu) = \int_{-\infty}^{+\infty} f(t)\,e^{-2\pi i \nu t}\,dt",
    "TFD (FFT)":               r"X_k = \sum_{n=0}^{N-1} x_n\,e^{-2\pi i kn/N}",
    "Théorème de Shannon":     r"f_s \geq 2 f_{max}",
    "Énergie de Parseval":     r"\sum_n |x_n|^2 = \frac{1}{N}\sum_k |X_k|^2",
    "Convolution":             r"(f * g)(t) = \int_{-\infty}^{+\infty} f(\tau)g(t-\tau)\,d\tau",
    "SNR":                     r"\text{SNR} = 10\log_{10}\frac{P_{signal}}{P_{bruit}}",
}

# ============================================================
# FORMULAIRE — NOUVEAUX CHAPITRES
# ============================================================
FORMULES_ENERGIE_PUISSANCE = {
    "Énergie signal continu":       r"E = \int_{-\infty}^{+\infty} |x(t)|^2\,dt",
    "Énergie signal discret":       r"E = \sum_{n=-\infty}^{+\infty} |x[n]|^2",
    "Puissance moyenne":            r"P = \lim_{T\to\infty}\frac{1}{2T}\int_{-T}^{T}|x(t)|^2\,dt",
    "Puissance signal discret":     r"P = \lim_{N\to\infty}\frac{1}{2N+1}\sum_{n=-N}^{N}|x[n]|^2",
    "Puissance signal périodique":  r"P = \frac{1}{T}\int_0^T |x(t)|^2\,dt",
    "Signal énergie finie":         r"E < \infty \Rightarrow P = 0",
    "Signal puissance finie":       r"0 < P < \infty \Rightarrow E = \infty",
    "Énergie spectr. (Parseval)":   r"E = \int_{-\infty}^{+\infty}|X(f)|^2\,df = \int_{-\infty}^{+\infty}|x(t)|^2\,dt",
    "DSP (bilatérale)":             r"S_{xx}(f) = \lim_{T\to\infty}\frac{1}{T}|X_T(f)|^2",
    "DSE (spectr. énergie)":        r"\Psi_{xx}(f) = |X(f)|^2",
    "Valeur efficace (RMS)":        r"x_{rms} = \sqrt{P} = \sqrt{\frac{1}{T}\int_0^T x^2(t)\,dt}",
    "Facteur de crête":             r"CF = \frac{|x|_{max}}{x_{rms}}",
}

FORMULES_TF = {
    "Définition TF":               r"X(f) = \mathcal{F}\{x(t)\} = \int_{-\infty}^{+\infty}x(t)e^{-j2\pi ft}\,dt",
    "TF inverse":                  r"x(t) = \mathcal{F}^{-1}\{X(f)\} = \int_{-\infty}^{+\infty}X(f)e^{j2\pi ft}\,df",
    "Linéarité":                   r"\mathcal{F}\{ax(t)+by(t)\} = aX(f)+bY(f)",
    "Décalage temporel":           r"\mathcal{F}\{x(t-t_0)\} = X(f)e^{-j2\pi ft_0}",
    "Décalage fréquentiel":        r"\mathcal{F}\{x(t)e^{j2\pi f_0 t}\} = X(f-f_0)",
    "Mise à l'échelle":            r"\mathcal{F}\{x(at)\} = \frac{1}{|a|}X\!\left(\frac{f}{a}\right)",
    "Convolution TF":              r"\mathcal{F}\{x*h\} = X(f)\cdot H(f)",
    "Multiplication TF":           r"\mathcal{F}\{x\cdot y\} = X*Y",
    "Dérivation":                  r"\mathcal{F}\{x'(t)\} = j2\pi f\,X(f)",
    "Intégration":                 r"\mathcal{F}\!\left\{\int x\,dt\right\} = \frac{X(f)}{j2\pi f}",
    "Symétrie hermitienne":        r"X(-f) = X^*(f) \quad \text{si } x(t) \text{ réel}",
    "TF du rect":                  r"\mathcal{F}\{\text{rect}(t/\tau)\} = \tau\,\text{sinc}(\tau f)",
    "TF de Dirac":                 r"\mathcal{F}\{\delta(t)\} = 1",
    "TF du peigne":                r"\mathcal{F}\!\left\{\sum_n \delta(t-nT)\right\} = \frac{1}{T}\sum_k\delta\!\left(f-\frac{k}{T}\right)",
    "TFD (DFT)":                   r"X[k] = \sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N},\quad k=0,\ldots,N-1",
    "TFD inverse (IDFT)":          r"x[n] = \frac{1}{N}\sum_{k=0}^{N-1}X[k]e^{j2\pi kn/N}",
}

FORMULES_ECHANTILLONNAGE = {
    "Critère de Nyquist-Shannon":  r"f_e \geq 2f_{max}",
    "Fréquence de Nyquist":        r"f_N = \frac{f_e}{2}",
    "Signal échantillonné":        r"x_s(t) = x(t)\cdot p(t) = \sum_n x(nT_e)\delta(t-nT_e)",
    "Spectre du signal échant.":   r"X_s(f) = f_e\sum_k X(f - kf_e)",
    "Repliement spectral (aliasing)": r"f_{alias} = |f - k\cdot f_e|,\quad k\in\mathbb{Z}",
    "Reconstruction (Shannon)":    r"x(t) = \sum_n x(nT_e)\,\text{sinc}\!\left(\frac{t-nT_e}{T_e}\right)",
    "Filtre anti-repliement":      r"H_{AA}(f) = \text{rect}\!\left(\frac{f}{f_e}\right)",
    "Quantification (pas)":        r"\Delta = \frac{x_{max}-x_{min}}{2^N}",
    "SNR quantification":          r"\text{SNR}_Q = 6.02N + 1.76 \text{ dB}",
    "Erreur de quantification":    r"|e_q| \leq \frac{\Delta}{2},\quad P_e = \frac{\Delta^2}{12}",
    "DTFT":                        r"X(e^{j\omega}) = \sum_{n=-\infty}^{\infty}x[n]e^{-j\omega n}",
    "Relation fréq. analogique-numérique": r"\omega = 2\pi\frac{f}{f_e} = \frac{2\pi f}{f_e}",
}


# ============================================================
# CLASSE MOTEUR — SIGNAL CLASSIQUE (existante)
# ============================================================
class SignalEngine:
    """Moteur de traitement du signal avancé."""

    def __init__(self, fs: float = 2000.0):
        self.fs = fs

    def generer(self, sig_type: str, freq: float, amp: float,
                duration: float, freq2: float = None,
                noise_level: float = 0.0, phase: float = 0.0) -> tuple:
        N = int(self.fs * duration)
        t = np.linspace(0, duration, N, endpoint=False)
        if sig_type == "Sinus":
            s = amp * np.sin(2*np.pi*freq*t + phase)
        elif sig_type == "Carré":
            s = amp * sp_signal.square(2*np.pi*freq*t + phase)
        elif sig_type == "Triangle":
            s = amp * sp_signal.sawtooth(2*np.pi*freq*t + phase, 0.5)
        elif sig_type == "Dent de scie":
            s = amp * sp_signal.sawtooth(2*np.pi*freq*t + phase)
        elif sig_type == "Chirp":
            f2 = freq2 or freq*5
            s = amp * sp_signal.chirp(t, f0=freq, f1=f2, t1=duration)
        elif sig_type == "Bruit blanc":
            s = amp * np.random.normal(0, 1, N)
        elif sig_type == "Bruit rose":
            white = np.random.normal(0, 1, N)
            f_n = rfftfreq(N, 1/self.fs); f_n[0] = 1
            pink_filter = 1/np.sqrt(f_n); pink_filter[0] = 0
            s = amp * np.real(np.fft.irfft(rfft(white)*pink_filter, n=N))
        elif sig_type == "Multi-sinusoïdal":
            harmoniques = [1,2,3,5,7]
            s = sum(amp/h * np.sin(2*np.pi*freq*h*t) for h in harmoniques)
        elif sig_type == "Impulsion gaussienne":
            t0, sigma = duration/2, duration/10
            s = amp * np.exp(-0.5*((t-t0)/sigma)**2) * np.cos(2*np.pi*freq*t)
        else:
            s = amp * np.sin(2*np.pi*freq*t)
        if noise_level > 0:
            s += np.random.normal(0, noise_level*amp, N)
        return t, s

    def compute_fft(self, s: np.ndarray) -> tuple:
        N = len(s)
        win = np.hanning(N)
        s_win = s * win
        yf = rfft(s_win)
        xf = rfftfreq(N, 1/self.fs)
        magnitude = (2.0/N) * np.abs(yf)
        phase = np.angle(yf, deg=True)
        power_db = 20*np.log10(magnitude + 1e-12)
        return xf, magnitude, phase, power_db

    def metriques(self, s: np.ndarray) -> dict:
        rms = np.sqrt(np.mean(s**2))
        crest = np.max(np.abs(s)) / (rms + 1e-12)
        energy = np.sum(s**2) / self.fs
        xf, mag, _, _ = self.compute_fft(s)
        f_dom = xf[np.argmax(mag)] if len(mag)>0 else 0
        return {"RMS": rms, "Crête": np.max(np.abs(s)),
                "Facteur de crête": crest, "Énergie (J/Ω)": energy,
                "Fréquence dominante (Hz)": f_dom,
                "Moyenne": np.mean(s), "Écart-type": np.std(s)}

    def filtrer(self, s: np.ndarray, ftype: str, btype: str,
                cutoff, order: int = 4) -> np.ndarray:
        nyq = self.fs/2
        if isinstance(cutoff, (list, tuple)):
            Wn = [c/nyq for c in cutoff]
        else:
            Wn = cutoff/nyq
        Wn = np.clip(Wn, 1e-4, 0.9999)
        try:
            if ftype == "Butterworth":
                b, a = sp_signal.butter(order, Wn, btype=btype)
            elif ftype == "Chebyshev I":
                b, a = sp_signal.cheby1(order, 1, Wn, btype=btype)
            elif ftype == "Elliptique":
                b, a = sp_signal.ellip(order, 1, 40, Wn, btype=btype)
            else:
                b, a = sp_signal.butter(order, Wn, btype=btype)
            return sp_signal.filtfilt(b, a, s), b, a
        except:
            return s, None, None

    def enveloppe(self, s):
        analytic = sp_signal.hilbert(s)
        envelope = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase)/(2*np.pi)*self.fs
        return envelope, inst_phase, inst_freq

    def dsp(self, s):
        f, Pxx = sp_signal.welch(s, self.fs, nperseg=min(256, len(s)//4))
        return f, Pxx

    def autocorrelation(self, s):
        N = len(s)
        acf = np.correlate(s-s.mean(), s-s.mean(), mode='full')
        acf = acf/(acf[N-1])
        lags = np.arange(-(N-1), N)/self.fs
        return lags, acf


# ============================================================
# NOUVELLE CLASSE — ÉNERGIE & PUISSANCE
# ============================================================
class EnergiePuissanceEngine:
    """Moteur d'analyse énergie et puissance des signaux."""

    def __init__(self, fs: float = 1000.0):
        self.fs = fs

    def energie_signal(self, s: np.ndarray) -> float:
        """E = Σ|x[n]|² / fs (signal discret)."""
        return float(np.sum(np.abs(s)**2) / self.fs)

    def puissance_moyenne(self, s: np.ndarray) -> float:
        """P = (1/N) Σ|x[n]|²."""
        return float(np.mean(np.abs(s)**2))

    def puissance_rms(self, s: np.ndarray) -> float:
        """x_rms = √P."""
        return float(np.sqrt(self.puissance_moyenne(s)))

    def facteur_crete(self, s: np.ndarray) -> float:
        """CF = |x|_max / x_rms."""
        rms = self.puissance_rms(s)
        return float(np.max(np.abs(s)) / (rms + 1e-15))

    def type_signal(self, s: np.ndarray, t_max: float = 1.0) -> str:
        """Classifie : signal énergie finie ou puissance finie."""
        E = self.energie_signal(s)
        P = self.puissance_moyenne(s)
        if E < 1e6 and E > 1e-15:
            return "Énergie finie (E<∞, P→0)"
        elif 1e-10 < P < 1e10:
            return "Puissance finie (P<∞, E→∞)"
        return "Ni énergie ni puissance finie"

    def densite_spectrale_energie(self, s: np.ndarray) -> tuple:
        """Ψxx(f) = |X(f)|² — DSE."""
        N = len(s)
        X = rfft(s)
        freqs = rfftfreq(N, 1/self.fs)
        dse = np.abs(X)**2 / self.fs
        return freqs, dse

    def densite_spectrale_puissance(self, s: np.ndarray) -> tuple:
        """Sxx(f) via Welch."""
        f, Pxx = welch(s, self.fs, nperseg=min(256, len(s)//4))
        return f, Pxx

    def verification_parseval(self, s: np.ndarray) -> dict:
        """Vérifie Σ|x[n]|² = (1/N)Σ|X[k]|²."""
        E_temps = np.sum(np.abs(s)**2)
        X = fft(s)
        E_freq = np.sum(np.abs(X)**2) / len(s)
        return {"E_temps": E_temps, "E_freq": E_freq,
                "Erreur relative": abs(E_temps-E_freq)/(E_temps+1e-15),
                "Parseval vérifié": abs(E_temps-E_freq)/(E_temps+1e-15) < 1e-6}

    def energie_bande(self, s: np.ndarray,
                       f_lo: float, f_hi: float) -> float:
        """Énergie contenue dans la bande [f_lo, f_hi]."""
        freqs, dse = self.densite_spectrale_energie(s)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        return float(safe_trapz(dse[mask], freqs[mask])) * 2  # bilatéral

    def snr(self, s_signal: np.ndarray, s_bruit: np.ndarray) -> float:
        """SNR = 10·log₁₀(P_signal/P_bruit) en dB."""
        Ps = self.puissance_moyenne(s_signal)
        Pb = self.puissance_moyenne(s_bruit)
        return float(10*np.log10(Ps/(Pb+1e-15)))

    def puissance_instantanee(self, s: np.ndarray) -> np.ndarray:
        """p(t) = x²(t)."""
        return s**2

    def energie_cumulative(self, s: np.ndarray) -> np.ndarray:
        """E(t) = ∫₀ᵗ x²(τ)dτ (cumulée)."""
        return np.cumsum(s**2) / self.fs

    def catalogue_signaux_energie_puissance(self) -> dict:
        """Signaux types avec leur classification."""
        fs = self.fs
        t = np.linspace(0, 2, int(2*fs), endpoint=False)
        return {
            "Impulsion gaussienne":   (np.exp(-t**2)*np.cos(10*t), "Énergie"),
            "Exponentielle amortie":  (np.exp(-t)*np.cos(10*np.pi*t), "Énergie"),
            "Sinusoïde pure":         (np.sin(2*np.pi*5*t), "Puissance"),
            "Bruit blanc":            (np.random.randn(len(t))*0.5, "Puissance"),
            "Rampe tronquée":         (np.where(t<1, t, 0), "Énergie"),
            "Signal DC":              (np.ones_like(t)*2, "Puissance"),
        }


# ============================================================
# NOUVELLE CLASSE — TRANSFORMÉE DE FOURIER
# ============================================================
class TransformeeFourierEngine:
    """Moteur d'analyse complète via la TF et la TFD."""

    def __init__(self, fs: float = 1000.0):
        self.fs = fs

    def tf_analytique_gauche(self, type_signal: str,
                              t: np.ndarray, **kwargs) -> np.ndarray:
        """Signaux avec TF analytique connue."""
        if type_signal == "Rect":
            tau = kwargs.get("tau", 1.0)
            return np.where(np.abs(t) <= tau/2, 1.0, 0.0)
        elif type_signal == "Gaussienne":
            sigma = kwargs.get("sigma", 1.0)
            return np.exp(-np.pi*t**2/sigma**2)
        elif type_signal == "Exponentielle":
            a = kwargs.get("a", 1.0)
            return np.where(t >= 0, np.exp(-a*t), 0.0)
        elif type_signal == "Sinus":
            f0 = kwargs.get("f0", 5.0)
            return np.sin(2*np.pi*f0*t)
        elif type_signal == "Dirac approché":
            return np.where(np.abs(t) < 0.01, 100.0, 0.0)
        return np.zeros_like(t)

    def tf_analytique_droite(self, type_signal: str,
                              f: np.ndarray, **kwargs) -> np.ndarray:
        """TF analytique correspondante."""
        if type_signal == "Rect":
            tau = kwargs.get("tau", 1.0)
            return tau * np.sinc(tau*f)
        elif type_signal == "Gaussienne":
            sigma = kwargs.get("sigma", 1.0)
            return sigma * np.exp(-np.pi*sigma**2*f**2)
        elif type_signal == "Exponentielle":
            a = kwargs.get("a", 1.0)
            return 1/(a + 2j*np.pi*f)
        elif type_signal == "Sinus":
            f0 = kwargs.get("f0", 5.0)
            return (np.where(np.abs(f-f0)<0.5, 0.5j, 0) -
                    np.where(np.abs(f+f0)<0.5, 0.5j, 0))
        elif type_signal == "Dirac approché":
            return np.ones_like(f, dtype=complex)
        return np.zeros_like(f, dtype=complex)

    def tfd_complete(self, s: np.ndarray) -> tuple:
        """TFD bilatérale centrée."""
        N = len(s)
        X = fftshift(fft(s))
        freqs = fftshift(fftfreq(N, 1/self.fs))
        mag = np.abs(X)/N
        phase = np.angle(X, deg=True)
        return freqs, X, mag, phase

    def sympy_parse_function(self, expr: str, var_str: object = "t"):
        """Parse une fonction utilisateur en expression Sympy.

        `var_str` peut être le nom du symbole (str) ou un objet `sympy.Symbol`.
        On s'assure d'utiliser le même symbole pour les opérations ultérieures.
        """
        if not HAS_SYMPY:
            raise RuntimeError("Sympy n'est pas disponible")
        expr = expr.strip()
        # Normalize simple names like 'sin' -> 'sin(t)'
        var_name = var_str if isinstance(var_str, str) else str(var_str)
        if expr in {"sin", "cos", "tan", "exp", "sqrt", "log", "sinh", "cosh", "tanh"}:
            expr = f"{expr}({var_name})"
        # Use provided Symbol if given, else create one
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
        # If parsing returned a callable (e.g. the bare sin/cos function),
        # convert it to a proper Sympy expression in the symbol `t`.
        try:
            if callable(func):
                func = func(t)
        except Exception:
            pass

        # Compute Fourier series. Ensure `func` is an expression first;
        # if `sp.fourier_series` returns an unexpected type, raise.
        series = sp.fourier_series(func, interval_range)
        if not hasattr(series, 'truncate'):
            raise RuntimeError(f"sp.fourier_series returned unexpected type: {type(series)}")

        expr = series.truncate(terms)
        # Compute a_n, b_n and mean value symbolically
        try:
            n = sp.symbols('n', integer=True, positive=True)
            w0 = 2*sp.pi/period_sym
            a_mean = sp.simplify((1/period_sym) * sp.integrate(func, (t, interval_range[1], interval_range[2])))
            a_n = sp.simplify((2/period_sym) * sp.integrate(func * sp.cos(n*w0*t), (t, interval_range[1], interval_range[2])))
            b_n = sp.simplify((2/period_sym) * sp.integrate(func * sp.sin(n*w0*t), (t, interval_range[1], interval_range[2])))
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
        func = self.sympy_parse_function(func_str, t)
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

    def proprietes_tf(self, s: np.ndarray, t: np.ndarray) -> dict:
        """Vérifie numériquement les propriétés de la TF."""
        N = len(s)
        dt = t[1]-t[0]
        f = fftfreq(N, dt)

        X = fft(s)
        # Décalage temporel : x(t-t0) → e^(-j2πft0)·X(f)
        t0 = dt * 10
        s_shifted = np.interp(t-t0, t, s, left=0, right=0)
        X_shifted = fft(s_shifted)
        X_shifted_theo = X * np.exp(-2j*np.pi*f*t0)

        err_decalage = np.mean(np.abs(X_shifted-X_shifted_theo)**2) / \
                       (np.mean(np.abs(X)**2)+1e-15)

        # Parseval
        E_t = np.sum(np.abs(s)**2)*dt
        E_f = np.sum(np.abs(X)**2)/(N**2/dt)/N
        err_parseval = abs(E_t-E_f)/(E_t+1e-15)

        # Symétrie hermitienne (signal réel)
        sym_err = np.mean(np.abs(X - np.conj(np.roll(X[::-1], 1)))**2)
        sym_ok = bool(sym_err < 1e-10 * np.mean(np.abs(X)**2))

        return {
            "Décalage temporel (erreur rel.)": err_decalage,
            "Parseval (erreur rel.)": err_parseval,
            "Symétrie hermitienne": "✅" if sym_ok else "❌",
            "E temporelle": E_t,
            "E fréquentielle": E_f,
        }

    def effet_fenetre(self, s: np.ndarray,
                      type_fenetre: str) -> tuple:
        """Effet du fenêtrage sur le spectre."""
        fenetres = {
            "Rectangulaire": np.ones(len(s)),
            "Hanning":        np.hanning(len(s)),
            "Hamming":        np.hamming(len(s)),
            "Blackman":       np.blackman(len(s)),
            "Flattop":        sp_signal.windows.flattop(len(s)),
            "Kaiser (β=8)":   np.kaiser(len(s), 8),
        }
        win = fenetres.get(type_fenetre, np.ones(len(s)))
        s_win = s * win
        N = len(s)
        X = rfft(s_win)
        freqs = rfftfreq(N, 1/self.fs)
        mag = (2/win.sum()) * np.abs(X)
        return freqs, mag, win

    def zero_padding(self, s: np.ndarray, N_pad: int) -> tuple:
        """Zero-padding pour interpoler le spectre."""
        s_pad = np.zeros(N_pad)
        s_pad[:len(s)] = s
        X_pad = rfft(s_pad)
        freqs_pad = rfftfreq(N_pad, 1/self.fs)
        return freqs_pad, (2/N_pad)*np.abs(X_pad)

    def spectre_phase_complet(self, s: np.ndarray) -> tuple:
        """Spectre bilatéral complet avec phase déroulée."""
        freqs, X, mag, phase = self.tfd_complete(s)
        phase_unwrapped = np.unwrap(np.angle(X)) * 180/np.pi
        return freqs, mag, phase, phase_unwrapped

    def convolution_tf(self, s1: np.ndarray,
                        s2: np.ndarray) -> tuple:
        """Convolution via TF : x1*x2 ↔ X1·X2."""
        N = len(s1)+len(s2)-1
        X1 = fft(s1, N)
        X2 = fft(s2, N)
        conv_direct = np.convolve(s1, s2)
        conv_tf = np.real(ifft(X1*X2))
        return conv_direct, conv_tf[:N]


# ============================================================
# NOUVELLE CLASSE — ÉCHANTILLONNAGE
# ============================================================
class EchantillonnageEngine:
    """Moteur d'analyse de l'échantillonnage et de la reconstruction."""

    def __init__(self):
        pass

    def signal_echantillonne(self, f_signal: float,
                              f_e: float, A: float = 1.0,
                              t_max: float = 0.5,
                              n_cont: int = 5000) -> tuple:
        """Signal continu + échantillons discrets."""
        t_cont = np.linspace(0, t_max, n_cont)
        x_cont = A * np.sin(2*np.pi*f_signal*t_cont)
        t_samp = np.arange(0, t_max, 1/f_e)
        x_samp = A * np.sin(2*np.pi*f_signal*t_samp)
        return t_cont, x_cont, t_samp, x_samp

    def spectre_echantillonne(self, f_signal: float,
                               f_e: float, n_rep: int = 5) -> tuple:
        """Spectre du signal échantillonné (répétitions périodiques)."""
        f = np.linspace(-3*f_e, 3*f_e, 5000)
        X_s = np.zeros_like(f)
        for k in range(-n_rep, n_rep+1):
            f_center = f_signal + k*f_e
            X_s += np.exp(-(f-f_center)**2/(2*(f_e/100)**2))
            f_center_neg = -f_signal + k*f_e
            X_s += np.exp(-(f-f_center_neg)**2/(2*(f_e/100)**2))
        return f, X_s * f_e

    def frequences_aliasees(self, f_signal: float,
                             f_e: float,
                             n_max: int = 5) -> list:
        """Calcule les fréquences aliasées."""
        aliases = []
        for k in range(-n_max, n_max+1):
            f_alias = abs(f_signal - k*f_e)
            if 0 <= f_alias <= f_e/2:
                aliases.append({"k": k, "f_alias": f_alias,
                                 "Aliasing": k != 0})
        return sorted(aliases, key=lambda x: x["f_alias"])

    def reconstruction_shannon(self, t_samp: np.ndarray,
                                x_samp: np.ndarray,
                                t_rec: np.ndarray) -> np.ndarray:
        """Reconstruction par interpolation sinc de Shannon."""
        T = t_samp[1]-t_samp[0] if len(t_samp) > 1 else 1
        x_rec = np.zeros_like(t_rec)
        for n, tn in enumerate(t_samp):
            x_rec += x_samp[n] * np.sinc((t_rec-tn)/T)
        return x_rec

    def quantification(self, x: np.ndarray,
                        n_bits: int,
                        x_min: float = None,
                        x_max: float = None) -> tuple:
        """Quantification uniforme sur N bits."""
        if x_min is None: x_min = x.min()
        if x_max is None: x_max = x.max()
        n_levels = 2**n_bits
        delta = (x_max - x_min) / n_levels
        x_q = np.clip(x, x_min, x_max)
        levels = np.round((x_q - x_min) / delta).astype(int)
        levels = np.clip(levels, 0, n_levels-1)
        x_quantifie = x_min + levels*delta + delta/2
        erreur = x_quantifie - x
        return x_quantifie, erreur, delta

    def snr_quantification_theorique(self, n_bits: int) -> float:
        """SNR_Q = 6.02N + 1.76 dB."""
        return 6.02*n_bits + 1.76

    def snr_quantification_numerique(self, x: np.ndarray,
                                      x_q: np.ndarray) -> float:
        """SNR numérique = 10·log₁₀(P_signal/P_erreur)."""
        P_sig = np.mean(x**2)
        P_err = np.mean((x-x_q)**2)
        return float(10*np.log10(P_sig/(P_err+1e-15)))

    def critere_nyquist_check(self, f_signal: float,
                               f_e: float) -> dict:
        """Vérifie le critère de Shannon-Nyquist."""
        f_N = f_e/2
        ratio = f_e/(2*f_signal) if f_signal > 0 else np.inf
        shannon_ok = f_e >= 2*f_signal
        return {"f_e": f_e, "f_max": f_signal,
                "f_Nyquist": f_N, "f_e/2f_max": ratio,
                "Shannon": "✅ OK" if shannon_ok else "❌ Aliasing!",
                "Marge (dB)": 20*np.log10(ratio) if ratio > 0 else -np.inf}

    def analyse_aliasing_2D(self, f_max: float,
                             f_e_range: np.ndarray) -> pd.DataFrame:
        """Analyse du rapport fe/2fmax pour différentes fe."""
        rows = []
        for fe in f_e_range:
            fN = fe/2
            f_alias = abs(f_max - fe) if f_max > fN else f_max
            rows.append({
                "fe (Hz)": fe,
                "fN (Hz)": fN,
                "f_alias (Hz)": f_alias,
                "Shannon": "✅" if fe >= 2*f_max else "❌",
                "fe/2fmax": round(fe/(2*f_max), 3)
            })
        return pd.DataFrame(rows)


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def signal_page():
    st.markdown("## 〰️ Traitement du Signal — Vue d'ensemble")
    st.markdown("*DSP avancée · Énergie/Puissance · Transformée de Fourier · Échantillonnage*")
    st.markdown("---")

    section = st.selectbox(
        "Section",
        [
            "🎛️ DSP Classique",
            "⚡ Énergie & Puissance",
            "🔄 Transformée de Fourier",
            "📡 Échantillonnage",
        ],
        key="section_signal_tools"
    )

    PLOT_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(5,0,20,0.8)',
                       font=dict(color='#c0d0ff'))
    AXIS = dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
    colors = ['#00ccff','#7700ff','#ff00cc','#00ff88',
               '#ffcc00','#ff4400','#88ccff','#cc88ff']

    def layout(fig, title="", xt="", yt="", h=420):
        fig.update_layout(**PLOT_LAYOUT, title=title,
                          xaxis_title=xt, yaxis_title=yt, height=h,
                          xaxis=AXIS, yaxis=AXIS,
                          legend=dict(bgcolor='rgba(0,0,0,0.5)'))
        return fig

    # Sidebar config globale
    with st.sidebar.expander("⚙️ Config Signal", expanded=True):
        fs_global = st.slider("Fe (Hz)", 500, 20000, 2000, 500, key="fs_glob")

    engine = SignalEngine(fs=fs_global)
    ep_engine = EnergiePuissanceEngine(fs=fs_global)
    tf_engine = TransformeeFourierEngine(fs=fs_global)
    ech_engine = EchantillonnageEngine()

    # ============================================================
    # ONGLETS PRINCIPAUX
    # ============================================================

    # ============================================================
    # TAB 1 — DSP CLASSIQUE (code existant préservé)
    # ============================================================
    if section == "🎛️ DSP Classique":
        sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs([
            "🎛️ Génération & Analyse",
            "🔊 Filtrage",
            "📡 Analyses avancées",
            "📊 Métriques",
            "⚗️ Diagnostic",
            "📖 Théorie"
        ])

        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🎛️ Paramètres")
                sig_type = st.selectbox("Signal", ["Sinus","Carré","Triangle",
                    "Dent de scie","Chirp","Bruit blanc","Bruit rose",
                    "Multi-sinusoïdal","Impulsion gaussienne"])
                freq = st.slider("f₁ (Hz)", 1.0, 500.0, 10.0, 1.0)
                amp = st.slider("Amplitude A", 0.1, 10.0, 1.0, 0.1)
                duration = st.slider("Durée (s)", 0.1, 5.0, 1.0, 0.1)
                phase = st.slider("Phase φ (rad)", 0.0, 2*np.pi, 0.0, 0.01)
                noise = st.slider("Bruit σ", 0.0, 1.0, 0.0, 0.01)
                analyse_type = st.radio("Analyse", ["Temporel","FFT",
                    "Spectrogramme","DSP","Phase"])

            with col2:
                t, s = engine.generer(sig_type, freq, amp, duration,
                                      noise_level=noise, phase=phase)
                st.session_state['t_sig'] = t
                st.session_state['s_sig'] = s
                st.session_state['fs_sig'] = fs_global

                if analyse_type == "Temporel":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=t, y=s, mode='lines',
                        name='s(t)', line=dict(color='#00ccff', width=2)))
                    layout(fig, f"Signal temporel — {sig_type}", "t (s)", "s(t)")
                    st.plotly_chart(fig, use_container_width=True)

                elif analyse_type == "FFT":
                    xf, mag, phase_f, power_db = engine.compute_fft(s)
                    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["|X(f)|", "Phase (°)"])
                    fig.add_trace(go.Scatter(x=xf, y=mag, mode='lines',
                        line=dict(color='#00ccff', width=2), name='|FFT|',
                        fill='tozeroy', fillcolor='rgba(0,204,255,0.15)'),
                        row=1, col=1)
                    fig.add_trace(go.Scatter(x=xf, y=phase_f, mode='lines',
                        line=dict(color='#7700ff', width=1.5), name='Phase'),
                        row=2, col=1)
                    fig.update_layout(**PLOT_LAYOUT, height=500,
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                    fig.update_xaxes(**AXIS, title_text="f (Hz)")
                    fig.update_yaxes(**AXIS)
                    st.plotly_chart(fig, use_container_width=True)

                elif analyse_type == "Spectrogramme":
                    nperseg = st.slider("Fenêtre", 32, 512, 128, key="sg_win")
                    f_sg, tt_sg, Sxx = spectrogram(s, fs_global,
                        nperseg=nperseg, noverlap=int(nperseg*0.75))
                    fig = go.Figure(go.Heatmap(z=10*np.log10(Sxx+1e-12),
                        x=tt_sg, y=f_sg,
                        colorscale=[[0,'#020817'],[0.3,'#7700ff'],
                                    [0.6,'#00ccff'],[1,'#ffffff']],
                        colorbar=dict(title='dB', tickfont=dict(color='#c0d0ff'))))
                    layout(fig, "Spectrogramme", "t (s)", "f (Hz)")
                    st.plotly_chart(fig, use_container_width=True)

                elif analyse_type == "DSP":
                    f_dsp, Pxx = engine.dsp(s)
                    fig = go.Figure(go.Scatter(x=f_dsp, y=10*np.log10(Pxx+1e-12),
                        mode='lines', line=dict(color='#00ccff', width=2.5),
                        fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'))
                    layout(fig, "DSP (Welch)", "f (Hz)", "PSD (dB/Hz)")
                    st.plotly_chart(fig, use_container_width=True)

                elif analyse_type == "Phase":
                    xf, mag, phase_f, _ = engine.compute_fft(s)
                    fig = go.Figure(go.Scatter(x=xf, y=phase_f, mode='lines',
                        line=dict(color='#7700ff', width=2)))
                    layout(fig, "Spectre de phase", "f (Hz)", "Phase (°)")
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button("💾 Export CSV",
                    pd.DataFrame({"t":t,"s":s}).to_csv(index=False).encode(),
                    "signal.csv", "text/csv")

        with sub2:
            if 's_sig' not in st.session_state:
                st.info("Générez d'abord un signal.")
            else:
                s_f = st.session_state['s_sig']
                t_f = st.session_state['t_sig']
                col1, col2 = st.columns([1, 2])
                with col1:
                    ftype = st.selectbox("Filtre", ["Butterworth","Chebyshev I","Elliptique"])
                    btype = st.selectbox("Bande", ["low","high","bandpass","bandstop"])
                    order = st.slider("Ordre", 1, 10, 4)
                    nyq = fs_global/2
                    if btype in ["bandpass","bandstop"]:
                        fl = st.slider("f_basse (Hz)", 1.0, nyq*0.45, 20.0)
                        fh = st.slider("f_haute (Hz)", fl+1, nyq*0.95, 200.0)
                        cutoff = [fl, fh]
                    else:
                        cutoff = st.slider("f_coupure (Hz)", 1.0, nyq*0.95, 100.0)

                with col2:
                    s_filt, b_f, a_f = engine.filtrer(s_f, ftype, btype, cutoff, order)
                    fig_filt = make_subplots(rows=2, cols=1,
                        subplot_titles=["Signal", "Spectres"])
                    fig_filt.add_trace(go.Scatter(x=t_f, y=s_f, mode='lines',
                        name='Original',
                        line=dict(color='rgba(0,204,255,0.4)', width=1.5)),
                        row=1, col=1)
                    fig_filt.add_trace(go.Scatter(x=t_f, y=s_filt, mode='lines',
                        name='Filtré',
                        line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                    xf_o, mg_o, _, _ = engine.compute_fft(s_f)
                    xf_f2, mg_f2, _, _ = engine.compute_fft(s_filt)
                    fig_filt.add_trace(go.Scatter(x=xf_o, y=mg_o, mode='lines',
                        name='Spectre original',
                        line=dict(color='rgba(119,0,255,0.5)', width=1.5)),
                        row=2, col=1)
                    fig_filt.add_trace(go.Scatter(x=xf_f2, y=mg_f2, mode='lines',
                        name='Spectre filtré',
                        line=dict(color='#00ccff', width=2)), row=2, col=1)
                    fig_filt.update_layout(**PLOT_LAYOUT, height=520,
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                    fig_filt.update_xaxes(**AXIS)
                    fig_filt.update_yaxes(**AXIS)
                    st.plotly_chart(fig_filt, use_container_width=True)

        with sub3:
            if 's_sig' not in st.session_state:
                st.info("Générez d'abord un signal.")
            else:
                s_a = st.session_state['s_sig']
                t_a = st.session_state['t_sig']
                analyse_adv = st.radio("Analyse", [
                    "Autocorrélation","Enveloppe","Portrait de phase"],
                    horizontal=True)
                if analyse_adv == "Autocorrélation":
                    lags, acf = engine.autocorrelation(s_a)
                    fig_ac = go.Figure(go.Scatter(x=lags, y=acf, mode='lines',
                        line=dict(color='#00ccff', width=2)))
                    layout(fig_ac, "Autocorrélation", "τ (s)", "R(τ)")
                    st.plotly_chart(fig_ac, use_container_width=True)
                elif analyse_adv == "Enveloppe":
                    env, _, inst_f = engine.enveloppe(s_a)
                    fig_env = go.Figure()
                    fig_env.add_trace(go.Scatter(x=t_a, y=s_a, mode='lines',
                        line=dict(color='rgba(0,204,255,0.4)', width=1.5),
                        name='Signal'))
                    fig_env.add_trace(go.Scatter(x=t_a, y=env, mode='lines',
                        line=dict(color='#ff00cc', width=2.5), name='Enveloppe'))
                    layout(fig_env, "Enveloppe (Hilbert)", "t (s)", "Amplitude")
                    st.plotly_chart(fig_env, use_container_width=True)
                else:
                    decal = st.slider("Décalage τ", 1, 100, 10)
                    fig_pp = go.Figure(go.Scatter(x=s_a[:-decal], y=s_a[decal:],
                        mode='lines', line=dict(color='#00ccff', width=1.5)))
                    layout(fig_pp, "Portrait de phase", "s(t)", f"s(t+{decal})")
                    st.plotly_chart(fig_pp, use_container_width=True)

        with sub4:
            if 's_sig' in st.session_state:
                s_m = st.session_state['s_sig']
                met = engine.metriques(s_m)
                cols_met = st.columns(4)
                for i, (k, v) in enumerate(met.items()):
                    with cols_met[i%4]:
                        st.metric(k, f"{v:.4f}")

        with sub5:
            if 's_sig' in st.session_state:
                s_d = st.session_state['s_sig']
                fs_d = st.session_state['fs_sig']
                xf_d, mag_d, _, _ = engine.compute_fft(s_d)
                f_max = xf_d[np.argmax(mag_d)] if len(mag_d)>0 else 0
                cfl = fs_d/(2*f_max) if f_max > 0 else np.inf
                diag = [
                    {"Test":"Shannon","Valeur":f"fe={fs_d}Hz, f_max={f_max:.1f}Hz",
                     "Statut":"✅ OK" if cfl>=1 else "❌ Aliasing"},
                    {"Test":"Clipping","Valeur":f"{(np.abs(s_d)>=0.99*np.abs(s_d).max()).mean()*100:.2f}%",
                     "Statut":"✅" if (np.abs(s_d)>=0.99*np.abs(s_d).max()).mean()<0.01 else "⚠️"},
                ]
                st.dataframe(pd.DataFrame(diag), use_container_width=True)

        with sub6:
            st.markdown("### 📖 Formulaire — Signaux & Systèmes")
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_SIGNAL.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f_latex)
                col_idx += 1

    # ============================================================
    # TAB 2 — ÉNERGIE & PUISSANCE
    # ============================================================
    elif section == "⚡ Énergie & Puissance":
        st.markdown("## ⚡ Section 1 — Énergie & Puissance des Signaux")
        st.markdown("""
        - **Signal à énergie finie** : E < ∞ (signaux transitoires)
        - **Signal à puissance finie** : 0 < P < ∞ (signaux périodiques, stochastionnaires)
        """)

        sub1, sub2, sub3, sub4 = st.tabs([
            "📊 Calcul E & P",
            "🔍 Parseval & DSE",
            "📈 Catalogue signaux",
            "📖 Formules"
        ])

        # ---- SUB 1 : CALCUL E & P ----
        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                sig_ep = st.selectbox("Signal", [
                    "Sinus A·sin(2πft)",
                    "Exponentielle amortie A·e^(-αt)·cos(2πft)",
                    "Impulsion gaussienne",
                    "Bruit blanc",
                    "Signal DC",
                    "Rampe tronquée"
                ])
                A_ep = st.slider("Amplitude A", 0.1, 5.0, 1.0, 0.1)
                f_ep = st.slider("Fréquence f (Hz)", 0.5, 100.0, 10.0, 0.5)
                dur_ep = st.slider("Durée T (s)", 0.1, 10.0, 2.0, 0.1)
                alpha_ep = st.slider("α (amortissement)", 0.1, 10.0, 1.0, 0.1)

                t_ep = np.linspace(0, dur_ep, int(fs_global*dur_ep), endpoint=False)
                if sig_ep == "Sinus A·sin(2πft)":
                    s_ep = A_ep * np.sin(2*np.pi*f_ep*t_ep)
                elif sig_ep == "Exponentielle amortie A·e^(-αt)·cos(2πft)":
                    s_ep = A_ep * np.exp(-alpha_ep*t_ep) * np.cos(2*np.pi*f_ep*t_ep)
                elif sig_ep == "Impulsion gaussienne":
                    t0 = dur_ep/2
                    s_ep = A_ep * np.exp(-((t_ep-t0)/(dur_ep/8))**2)
                elif sig_ep == "Bruit blanc":
                    np.random.seed(42)
                    s_ep = A_ep * np.random.randn(len(t_ep))
                elif sig_ep == "Signal DC":
                    s_ep = A_ep * np.ones_like(t_ep)
                else:
                    s_ep = A_ep * np.where(t_ep < dur_ep/2, t_ep/(dur_ep/2), 0)

                E_val = ep_engine.energie_signal(s_ep)
                P_val = ep_engine.puissance_moyenne(s_ep)
                rms_val = ep_engine.puissance_rms(s_ep)
                cf_val = ep_engine.facteur_crete(s_ep)
                type_s = ep_engine.type_signal(s_ep)

                st.metric("Énergie E (J/Ω)", f"{E_val:.6f}")
                st.metric("Puissance P (W/Ω)", f"{P_val:.6f}")
                st.metric("RMS", f"{rms_val:.6f}")
                st.metric("Facteur de crête CF", f"{cf_val:.4f}")
                st.metric("Classification", type_s)

                # Parseval
                parseval = ep_engine.verification_parseval(s_ep)
                st.markdown("### ✅ Théorème de Parseval")
                st.metric("E temporelle", f"{parseval['E_temps']:.6f}")
                st.metric("E fréquentielle", f"{parseval['E_freq']:.6f}")
                st.metric("Erreur relative", f"{parseval['Erreur relative']:.2e}")
                st.metric("Vérifié", "✅" if parseval['Parseval vérifié'] else "❌")

            with col2:
                E_cum = ep_engine.energie_cumulative(s_ep)
                p_inst = ep_engine.puissance_instantanee(s_ep)

                fig_ep = make_subplots(rows=3, cols=1,
                    subplot_titles=["Signal s(t)",
                                    "Puissance instantanée p(t)=s²(t)",
                                    "Énergie cumulée E(t)"])
                fig_ep.add_trace(go.Scatter(x=t_ep, y=s_ep, mode='lines',
                    name='s(t)', line=dict(color='#00ccff', width=2.5)),
                    row=1, col=1)
                fig_ep.add_trace(go.Scatter(x=t_ep, y=p_inst, mode='lines',
                    name='p(t)', line=dict(color='#7700ff', width=2)),
                    row=2, col=1)
                fig_ep.add_hline(y=P_val, line_color='#ffcc00', line_dash='dash',
                                  annotation_text=f"P={P_val:.4f}", row=2, col=1)
                fig_ep.add_trace(go.Scatter(x=t_ep, y=E_cum, mode='lines',
                    name='E(t)', line=dict(color='#ff00cc', width=2)),
                    row=3, col=1)
                fig_ep.add_hline(y=E_val, line_color='#ffcc00', line_dash='dash',
                                  annotation_text=f"E={E_val:.4f}", row=3, col=1)

                fig_ep.update_layout(**PLOT_LAYOUT, height=650,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_ep.update_xaxes(**AXIS, title_text="t (s)")
                fig_ep.update_yaxes(**AXIS)
                st.plotly_chart(fig_ep, use_container_width=True)

                st.download_button("💾 Export CSV",
                    pd.DataFrame({"t":t_ep,"s":s_ep,"p_inst":p_inst,
                                  "E_cum":E_cum}).to_csv(index=False).encode(),
                    "energie_puissance.csv", "text/csv")

        # ---- SUB 2 : PARSEVAL & DSE ----
        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                f_ep2 = st.slider("f signal (Hz)", 1.0, 200.0, 20.0, 1.0,
                                   key="f_ep2")
                dur_ep2 = st.slider("Durée (s)", 0.1, 5.0, 1.0, 0.1, key="d_ep2")
                t_ep2 = np.linspace(0, dur_ep2,
                                     int(fs_global*dur_ep2), endpoint=False)
                s_ep2 = np.sin(2*np.pi*f_ep2*t_ep2) + 0.5*np.sin(2*np.pi*2*f_ep2*t_ep2)

                f_lo = st.slider("Bande f_basse (Hz)", 0.0, fs_global/4, 10.0)
                f_hi = st.slider("Bande f_haute (Hz)", f_lo+1, fs_global/2, 50.0)
                E_bande = ep_engine.energie_bande(s_ep2, f_lo, f_hi)

                freqs_dse, dse = ep_engine.densite_spectrale_energie(s_ep2)
                freqs_dsp, dsp_arr = ep_engine.densite_spectrale_puissance(s_ep2)

                st.metric("Énergie totale (J)", f"{ep_engine.energie_signal(s_ep2):.4f}")
                st.metric(f"E dans [{f_lo:.0f},{f_hi:.0f}] Hz",
                          f"{E_bande:.4f}")

            with col2:
                fig_dse = make_subplots(rows=2, cols=1,
                    subplot_titles=["DSE : Ψxx(f) = |X(f)|²",
                                    "DSP : Sxx(f) (Welch)"])
                fig_dse.add_trace(go.Scatter(x=freqs_dse, y=dse, mode='lines',
                    name='DSE', line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'),
                    row=1, col=1)

                # Surligner la bande
                mask_b = (freqs_dse >= f_lo) & (freqs_dse <= f_hi)
                fig_dse.add_trace(go.Scatter(
                    x=freqs_dse[mask_b], y=dse[mask_b], mode='lines',
                    name=f'Bande [{f_lo:.0f}-{f_hi:.0f}Hz]',
                    line=dict(color='#ffcc00', width=3),
                    fill='tozeroy', fillcolor='rgba(255,200,0,0.2)'),
                    row=1, col=1)

                fig_dse.add_trace(go.Scatter(x=freqs_dsp,
                    y=10*np.log10(dsp_arr+1e-12), mode='lines',
                    name='DSP (dB/Hz)', line=dict(color='#7700ff', width=2.5)),
                    row=2, col=1)
                fig_dse.update_layout(**PLOT_LAYOUT, height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_dse.update_xaxes(**AXIS, title_text="f (Hz)")
                fig_dse.update_yaxes(**AXIS)
                st.plotly_chart(fig_dse, use_container_width=True)

        # ---- SUB 3 : CATALOGUE ----
        with sub3:
            cat = ep_engine.catalogue_signaux_energie_puissance()
            t_cat = np.linspace(0, 2, int(2*fs_global), endpoint=False)
            n_cols = 3
            rows_cat = (len(cat) + n_cols - 1)//n_cols
            fig_cat = make_subplots(rows=rows_cat, cols=n_cols,
                subplot_titles=[f"{k} ({v})" for k,v in
                                 [(k,ep_engine.type_signal(s_c)) for k,(s_c,_) in cat.items()]])

            for idx, (nom_c, (s_c, _)) in enumerate(cat.items()):
                r, c = idx//n_cols + 1, idx%n_cols + 1
                col_c = colors[idx % len(colors)]
                fig_cat.add_trace(go.Scatter(x=t_cat[:len(s_c)], y=s_c,
                    mode='lines', name=nom_c,
                    line=dict(color=col_c, width=1.5)), row=r, col=c)

                E_c = ep_engine.energie_signal(s_c)
                P_c = ep_engine.puissance_moyenne(s_c)
                fig_cat.add_annotation(
                    text=f"E={E_c:.2f} P={P_c:.2f}",
                    x=0.5, y=0.95, xref=f"x{idx+1 if idx>0 else ''} domain",
                    yref=f"y{idx+1 if idx>0 else ''} domain",
                    showarrow=False, font=dict(color='#ffcc00', size=9))

            fig_cat.update_layout(**PLOT_LAYOUT, height=600,
                showlegend=False, title="Catalogue signaux — Énergie vs Puissance")
            fig_cat.update_xaxes(**AXIS)
            fig_cat.update_yaxes(**AXIS)
            st.plotly_chart(fig_cat, use_container_width=True)

            df_cat = pd.DataFrame([
                {"Signal": k, "Type": tp,
                 "Énergie E": round(ep_engine.energie_signal(s_c), 4),
                 "Puissance P": round(ep_engine.puissance_moyenne(s_c), 4),
                 "RMS": round(ep_engine.puissance_rms(s_c), 4)}
                for k, (s_c, tp) in cat.items()
            ])
            st.dataframe(df_cat, use_container_width=True)

        # ---- SUB 4 : FORMULES ----
        with sub4:
            st.markdown("### 📖 Formulaire — Énergie & Puissance")
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_ENERGIE_PUISSANCE.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f_latex)
                col_idx += 1
            for r in [
                "Oppenheim & Schafer — *Discrete-Time Signal Processing* (Pearson, 2009)",
                "Proakis & Manolakis — *Digital Signal Processing* (Pearson, 2006)",
            ]:
                st.markdown(f"- {r}")

    # ============================================================
    # TAB 3 — TRANSFORMÉE DE FOURIER
    # ============================================================
    elif section == "🔄 Transformée de Fourier":
        st.markdown("## 🔄 Section 2 — Transformée de Fourier")
        st.markdown("*TF continue, TFD, propriétés, fenêtrage, zero-padding*")

        sub1, sub2, sub3, sub4, sub5 = st.tabs([
            "📐 TF & Paires analytiques",
            "🔬 Propriétés TF",
            "🧮 Série de Fourier",
            "🪟 Fenêtrage",
            "🔢 TFD & Zero-padding"
        ])

        # ---- SUB 1 : TF ANALYTIQUE ----
        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 📐 Paires de Fourier")
                type_tf = st.selectbox("Signal", [
                    "Rect", "Gaussienne", "Exponentielle",
                    "Sinus", "Dirac approché"
                ])
                t_max_tf = st.slider("t_max (s)", 0.1, 5.0, 2.0, 0.1)
                params_tf = {}
                if type_tf == "Rect":
                    params_tf["tau"] = st.slider("Durée τ (s)", 0.1, 2.0, 0.5, 0.05)
                elif type_tf == "Gaussienne":
                    params_tf["sigma"] = st.slider("σ", 0.1, 2.0, 0.5, 0.05)
                elif type_tf in ["Exponentielle"]:
                    params_tf["a"] = st.slider("a (décroissance)", 0.1, 10.0, 2.0, 0.1)
                elif type_tf == "Sinus":
                    params_tf["f0"] = st.slider("f₀ (Hz)", 0.5, 20.0, 5.0, 0.5)

                st.info("""
                **Paires remarquables :**
                - rect(t/τ) ↔ τ·sinc(τf)
                - e^(-πt²/σ²) ↔ σ·e^(-πσ²f²)
                - e^(-at)u(t) ↔ 1/(a+j2πf)
                """)

            with col2:
                t_tf = np.linspace(-t_max_tf, t_max_tf, int(4000))
                x_tf = tf_engine.tf_analytique_gauche(type_tf, t_tf, **params_tf)

                # TF numérique
                dt_tf = t_tf[1]-t_tf[0]
                X_num = fft(x_tf) * dt_tf
                freqs_num = fftshift(fftfreq(len(t_tf), dt_tf))
                X_num_shift = fftshift(X_num)

                # TF analytique
                f_tf = np.linspace(-50, 50, 2000)
                X_ana = tf_engine.tf_analytique_droite(type_tf, f_tf, **params_tf)

                fig_tf = make_subplots(rows=2, cols=2,
                    subplot_titles=["x(t)", "|X(f)| analytique",
                                    "", "|X(f)| numérique (FFT)"])
                fig_tf.add_trace(go.Scatter(x=t_tf, y=x_tf, mode='lines',
                    line=dict(color='#00ccff', width=2.5), name='x(t)'),
                    row=1, col=1)
                fig_tf.add_trace(go.Scatter(x=f_tf, y=np.abs(X_ana), mode='lines',
                    line=dict(color='#7700ff', width=2.5), name='|X(f)|'),
                    row=1, col=2)
                fig_tf.add_trace(go.Scatter(x=freqs_num,
                    y=np.abs(X_num_shift), mode='lines',
                    line=dict(color='#ff00cc', width=2), name='FFT'),
                    row=2, col=2)

                fig_tf.update_layout(**PLOT_LAYOUT, height=520,
                    title=f"Paire de Fourier — {type_tf}",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_tf.update_xaxes(**AXIS)
                fig_tf.update_yaxes(**AXIS)
                fig_tf.update_xaxes(row=1, col=2, range=[-30, 30])
                fig_tf.update_xaxes(row=2, col=2, range=[-30, 30])
                st.plotly_chart(fig_tf, use_container_width=True)

        # ---- SUB 2 : PROPRIÉTÉS ----
        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔬 Vérification numérique des propriétés")
                f_prop = st.slider("f signal (Hz)", 1.0, 50.0, 10.0, 0.5)
                dur_prop = st.slider("Durée (s)", 0.5, 3.0, 1.0, 0.1)
                t_prop = np.linspace(0, dur_prop,
                                      int(fs_global*dur_prop), endpoint=False)
                s_prop = (np.sin(2*np.pi*f_prop*t_prop) +
                          0.5*np.cos(2*np.pi*2*f_prop*t_prop))

                props = tf_engine.proprietes_tf(s_prop, t_prop)
                for k, v in props.items():
                    if isinstance(v, float):
                        st.metric(k, f"{v:.4e}")
                    else:
                        st.metric(k, str(v))

                st.markdown("### 🔄 Convolution via TF")
                st.latex(r"\mathcal{F}\{x_1 * x_2\} = X_1(f)\cdot X_2(f)")
                s1_c = s_prop[:len(s_prop)//2]
                s2_c = np.exp(-3*t_prop[:len(t_prop)//4])
                conv_dir, conv_tf_arr = tf_engine.convolution_tf(s1_c, s2_c)
                err_conv = np.mean(np.abs(conv_dir-conv_tf_arr[:len(conv_dir)])**2)
                st.metric("Erreur convolution TF vs directe", f"{err_conv:.2e}")

            with col2:
                # Décalage temporel
                fig_prop = make_subplots(rows=2, cols=2,
                    subplot_titles=["Signal original",
                                    "Signal décalé (t-t0)",
                                    "|X(f)| original",
                                    "Phase — décalage temporel"])
                fig_prop.add_trace(go.Scatter(x=t_prop, y=s_prop, mode='lines',
                    line=dict(color='#00ccff', width=2)), row=1, col=1)

                t0_shift = 0.1
                s_shifted = np.interp(t_prop-t0_shift, t_prop, s_prop,
                                       left=0, right=0)
                fig_prop.add_trace(go.Scatter(x=t_prop, y=s_shifted,
                    mode='lines', line=dict(color='#ff00cc', width=2)),
                    row=1, col=2)

                freqs_p, X_p, mag_p, phase_p = tf_engine.tfd_complete(s_prop)
                _, _, mag_sh, phase_sh = tf_engine.tfd_complete(s_shifted)

                fig_prop.add_trace(go.Scatter(x=freqs_p, y=mag_p, mode='lines',
                    line=dict(color='#00ccff', width=2), name='Original'),
                    row=2, col=1)
                phase_theo = phase_p - 360*freqs_p*t0_shift
                fig_prop.add_trace(go.Scatter(x=freqs_p, y=phase_sh, mode='lines',
                    line=dict(color='#ff00cc', width=2), name='Décalé'),
                    row=2, col=2)
                fig_prop.add_trace(go.Scatter(x=freqs_p,
                    y=phase_theo % 360 - 180, mode='lines',
                    line=dict(color='#ffcc00', width=1.5, dash='dash'),
                    name='Théorique'), row=2, col=2)

                fig_prop.update_layout(**PLOT_LAYOUT, height=520,
                    title="Propriétés TF — décalage temporel",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_prop.update_xaxes(**AXIS)
                fig_prop.update_yaxes(**AXIS)
                st.plotly_chart(fig_prop, use_container_width=True)

        # ---- SUB 3 : SÉRIE DE FOURIER ----
        with sub3:
            st.markdown("### 🧮 Outil Série de Fourier et Transformées")
            if not HAS_SYMPY:
                st.warning(
                    "Sympy n'est pas installé : installez 'sympy' pour activer le calcul symbolique."
                )
                st.markdown("`pip install sympy`")
            else:
                col1, col2 = st.columns([1, 2])
                with col1:
                    fn = st.text_input(
                        "Fonction périodique x(t)",
                        value="sin(t)",
                        help="Entrez une expression en t, par ex. sin(t)+t**2",
                        key="fourier_series_function"
                    )
                    period = st.number_input(
                        "Période T", 0.1, 20.0, 2.0, 0.1,
                        help="Période du signal périodique",
                        key="fourier_series_period"
                    )
                    interval_type = st.selectbox(
                        "Intervalle de définition",
                        ["Symétrique (-T/2, T/2)", "Positif (0, T)"],
                        key="fourier_series_interval"
                    )
                    n_terms = st.slider(
                        "Nombre de termes", 3, 31, 11, 2,
                        help="Nombre de termes renvoyés dans la série approchée",
                        key="fourier_series_terms"
                    )
                    search_term = st.text_input(
                        "Recherche (ex: transformée, inverse, série)",
                        value="",
                        key="fourier_series_search"
                    )
                    auto_tf = "transform" in search_term.lower() or "transformée" in search_term.lower()
                    auto_if = "inverse" in search_term.lower()
                    show_tf = st.checkbox(
                        "Afficher la transformée de Fourier symbolique",
                        value=auto_tf,
                        key="show_fourier_transform"
                    )
                    show_if = st.checkbox(
                        "Afficher la transformée inverse de Fourier",
                        value=auto_if,
                        key="show_inverse_fourier_transform"
                    )

                with col2:
                    if fn.strip():
                        interval = "symmetric" if interval_type.startswith("Symétrique") else "positive"
                        try:
                            func, series_obj, series_expr, a_n_sym, b_n_sym, mean_sym = tf_engine.fourier_series_symbolic(
                                fn, "t", period, n_terms, interval
                            )
                            st.markdown("**Série de Fourier (approximation)**")
                            st.latex(series_expr)
                            if mean_sym is not None:
                                st.markdown("**Valeur moyenne**")
                                st.latex(mean_sym)
                            st.markdown("**Coefficient général a_n (cosinus)**")
                            if a_n_sym is not None:
                                st.latex(a_n_sym)
                            st.markdown("**Coefficient général b_n (sinus)**")
                            if b_n_sym is not None:
                                st.latex(b_n_sym)
                            # Option: afficher valeurs numériques de a_n/b_n
                            show_numeric = st.checkbox("Afficher valeurs numériques pour a_n/b_n", key="tf_show_numeric")
                            if show_numeric:
                                max_n = st.slider("Nombre de n à afficher", 1, 20, 5, key="tf_numeric_n")
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
                                func, w, X = tf_engine.fourier_transform_symbolic(fn, "t", "w")
                                st.markdown("**Transformée de Fourier**")
                                st.latex(X)
                            except Exception as err:
                                st.error(f"Impossible de calculer la transformée de Fourier : {err}")

                            if show_if:
                                try:
                                    t_var, x_rec = tf_engine.inverse_fourier_transform_symbolic(X, w, "t")
                                    st.markdown("**Transformée inverse de Fourier**")
                                    st.latex(x_rec)
                                except Exception as err:
                                    st.error(f"Impossible de calculer la TF inverse : {err}")

        # ---- SUB 4 : FENÊTRAGE ----
        with sub4:
            col1, col2 = st.columns([1, 2])
            with col1:
                f_win = st.slider("f signal (Hz)", 1.0, 100.0, 10.0, 0.5,
                                   key="f_win")
                dur_win = st.slider("Durée (s)", 0.1, 2.0, 0.5, 0.05, key="d_win")
                bruit_win = st.slider("Bruit σ", 0.0, 1.0, 0.1, 0.01)

                t_win = np.linspace(0, dur_win,
                                     int(fs_global*dur_win), endpoint=False)
                s_win_sig = (np.sin(2*np.pi*f_win*t_win) +
                              bruit_win*np.random.randn(len(t_win)))

                fenetres_sel = st.multiselect("Fenêtres à comparer",
                    ["Rectangulaire","Hanning","Hamming",
                     "Blackman","Kaiser (β=8)"],
                    default=["Rectangulaire","Hanning","Blackman"])

                st.info("""
                **Compromis fenêtrage :**
                - Rect : résolution max, fuites max
                - Hanning/Hamming : bon compromis
                - Blackman : fuites minimales, résolution réduite
                """)

            with col2:
                fig_win = make_subplots(rows=2, cols=1,
                    subplot_titles=["Fenêtres temporelles", "Spectres (dB)"])

                for i, type_fen in enumerate(fenetres_sel):
                    freqs_w, mag_w, win_w = tf_engine.effet_fenetre(
                        s_win_sig, type_fen)
                    fig_win.add_trace(go.Scatter(x=t_win, y=win_w, mode='lines',
                        name=type_fen,
                        line=dict(color=colors[i], width=2)), row=1, col=1)
                    mag_db_w = 20*np.log10(mag_w+1e-12)
                    fig_win.add_trace(go.Scatter(x=freqs_w, y=mag_db_w,
                        mode='lines', name=f'{type_fen} spectre',
                        line=dict(color=colors[i], width=2)), row=2, col=1)

                fig_win.update_layout(**PLOT_LAYOUT, height=550,
                    title="Effet du fenêtrage sur le spectre",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_win.update_xaxes(**AXIS)
                fig_win.update_yaxes(**AXIS)
                fig_win.update_xaxes(title_text="t (s)", row=1, col=1)
                fig_win.update_xaxes(title_text="f (Hz)", row=2, col=1)
                fig_win.update_yaxes(title_text="f (Hz)", row=2, col=1)
                st.plotly_chart(fig_win, use_container_width=True)

        # ---- SUB 5 : TFD & ZERO-PADDING ----
        with sub5:
            col1, col2 = st.columns([1, 2])
            with col1:
                f_tfd = st.slider("f signal (Hz)", 1.0, 100.0, 10.5, 0.5,
                                   key="f_tfd")
                N_sig = st.slider("N (points signal)", 8, 512, 64, 8)
                N_pad_val = st.slider("N_pad (zero-padding)", N_sig,
                                       N_sig*16, N_sig*4, N_sig)

                t_tfd = np.arange(N_sig)/fs_global
                s_tfd = np.sin(2*np.pi*f_tfd*t_tfd)

                freqs_tfd, X_tfd, mag_tfd, phase_tfd = tf_engine.tfd_complete(s_tfd)
                freqs_pad, mag_pad = tf_engine.zero_padding(s_tfd, N_pad_val)

                st.metric("Résolution fréq. sans ZP (Hz)",
                          f"{fs_global/N_sig:.3f}")
                st.metric("Résolution fréq. avec ZP (Hz)",
                          f"{fs_global/N_pad_val:.3f}")
                st.metric("Gain de résolution",
                          f"{N_pad_val//N_sig}×")
                st.info("""
                **Zero-padding** = interpolation du spectre,
                pas de vraie résolution supplémentaire !
                Il améliore l'affichage mais pas la résolution.
                """)

            with col2:
                fig_zp = make_subplots(rows=2, cols=1,
                    subplot_titles=[f"Signal s[n] (N={N_sig})",
                                    f"Spectre |X[k]| — ZP={N_pad_val}"])
                fig_zp.add_trace(go.Scatter(x=np.arange(N_sig), y=s_tfd,
                    mode='lines+markers', name='s[n]',
                    line=dict(color='#00ccff', width=2),
                    marker=dict(size=6)), row=1, col=1)

                # TFD sans ZP
                freqs_tfd_pos = freqs_tfd[freqs_tfd >= 0]
                mag_tfd_pos = mag_tfd[freqs_tfd >= 0]
                fig_zp.add_trace(go.Scatter(x=freqs_tfd_pos, y=mag_tfd_pos,
                    mode='markers', name=f'TFD (N={N_sig})',
                    marker=dict(color='#ff00cc', size=8, symbol='circle')),
                    row=2, col=1)

                # TFD avec ZP
                fig_zp.add_trace(go.Scatter(x=freqs_pad,
                    y=mag_pad, mode='lines', name=f'ZP (N={N_pad_val})',
                    line=dict(color='#00ccff', width=2)), row=2, col=1)
                fig_zp.add_vline(x=f_tfd, line_color='#ffcc00',
                                  line_dash='dash',
                                  annotation_text=f"f={f_tfd}Hz")

                fig_zp.update_layout(**PLOT_LAYOUT, height=520,
                    title="TFD & Zero-padding",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_zp.update_xaxes(**AXIS)
                fig_zp.update_yaxes(**AXIS)
                fig_zp.update_xaxes(title_text="Indice n", row=1, col=1)
                fig_zp.update_xaxes(title_text="f (Hz)", row=2, col=1,
                                     range=[0, fs_global/2])
                st.plotly_chart(fig_zp, use_container_width=True)

    # ============================================================
    # TAB 4 — ÉCHANTILLONNAGE
    # ============================================================
    elif section == "📡 Échantillonnage":
        st.markdown("## 📡 Section 3 — Échantillonnage")
        st.markdown("*Critère de Nyquist, aliasing, reconstruction, quantification*")

        sub1, sub2, sub3, sub4, sub5 = st.tabs([
            "📡 Principe & Shannon",
            "🌀 Aliasing",
            "🔄 Reconstruction",
            "🔢 Quantification",
            "📖 Formules"
        ])

        # ---- SUB 1 : PRINCIPE ----
        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                f_sig_e = st.slider("f signal (Hz)", 1.0, 500.0, 50.0, 1.0,
                                     key="f_sig_e")
                f_e_e = st.slider("Fréquence d'échantillonnage fe (Hz)",
                                   2.0, 2000.0, 200.0, 1.0, key="fe_e")
                A_e = st.slider("Amplitude A", 0.1, 3.0, 1.0, 0.1)
                dur_e = st.slider("Durée (s)", 0.01, 0.5, 0.1, 0.01)

                nyquist_info = ech_engine.critere_nyquist_check(f_sig_e, f_e_e)
                for k, v in nyquist_info.items():
                    if isinstance(v, (int, float)):
                        st.metric(k, f"{v:.4f}")
                    else:
                        st.metric(k, str(v))

            with col2:
                t_c, x_c, t_s, x_s = ech_engine.signal_echantillonne(
                    f_sig_e, f_e_e, A_e, dur_e)

                fig_ech = make_subplots(rows=2, cols=1,
                    subplot_titles=["Signal continu + Échantillons",
                                    "Spectre du signal échantillonné"])

                fig_ech.add_trace(go.Scatter(x=t_c, y=x_c, mode='lines',
                    name='x(t) continu',
                    line=dict(color='rgba(0,204,255,0.5)', width=2)),
                    row=1, col=1)
                fig_ech.add_trace(go.Scatter(x=t_s, y=x_s, mode='markers',
                    name='x[n] échantillons',
                    marker=dict(color='#ff00cc', size=8, symbol='circle')),
                    row=1, col=1)
                # Lignes verticales d'échantillonnage
                for tn, xn in zip(t_s[:20], x_s[:20]):
                    fig_ech.add_shape(type='line',
                        x0=tn, x1=tn, y0=0, y1=xn,
                        xref='x', yref='y',
                        line=dict(color='rgba(255,0,204,0.4)', width=1),
                        row=1, col=1)

                # Spectre
                f_spec, X_spec = ech_engine.spectre_echantillonne(
                    f_sig_e, f_e_e, n_rep=3)
                fig_ech.add_trace(go.Scatter(x=f_spec, y=X_spec, mode='lines',
                    name='Xs(f)', line=dict(color='#00ccff', width=2),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'),
                    row=2, col=1)
                fig_ech.add_vline(x=f_e_e/2, line_color='#ffcc00',
                                   line_dash='dash',
                                   annotation_text=f"fN={f_e_e/2:.1f}Hz",
                                   row=2, col=1)
                fig_ech.add_vline(x=-f_e_e/2, line_color='#ffcc00',
                                   line_dash='dash', row=2, col=1)

                fig_ech.update_layout(**PLOT_LAYOUT, height=560,
                    title=f"Échantillonnage fe={f_e_e}Hz, f={f_sig_e}Hz",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_ech.update_xaxes(**AXIS)
                fig_ech.update_yaxes(**AXIS)
                st.plotly_chart(fig_ech, use_container_width=True)

                # Tableau de Shannon
                fe_range = np.array([50, 75, 100, 150, 200, 300, 500])
                df_nyq = ech_engine.analyse_aliasing_2D(f_sig_e, fe_range)
                st.dataframe(df_nyq, use_container_width=True)

        # ---- SUB 2 : ALIASING ----
        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🌀 Phénomène d'aliasing")
                f_alias_sig = st.slider("f signal (Hz)", 10.0, 1000.0,
                                         250.0, 5.0, key="f_al")
                f_alias_e = st.slider("fe (Hz)", 10.0, 1000.0,
                                       200.0, 5.0, key="fe_al")

                aliases = ech_engine.frequences_aliasees(f_alias_sig, f_alias_e)
                df_alias = pd.DataFrame(aliases)
                st.dataframe(df_alias, use_container_width=True)

                f_N_al = f_alias_e/2
                if f_alias_sig > f_N_al:
                    f_perc = abs(f_alias_sig - f_alias_e)
                    st.error(f"❌ Aliasing! f_perçue ≈ {f_perc:.1f} Hz")
                    st.metric("f signal", f"{f_alias_sig:.1f} Hz")
                    st.metric("f perçue (alias)", f"{f_perc:.1f} Hz")
                else:
                    st.success("✅ Pas d'aliasing (fe ≥ 2f)")

            with col2:
                # Démonstration visuelle de l'aliasing
                t_demo = np.linspace(0, 0.2, 5000)
                x_vrai = np.sin(2*np.pi*f_alias_sig*t_demo)
                t_samp_a = np.arange(0, 0.2, 1/f_alias_e)
                x_samp_a = np.sin(2*np.pi*f_alias_sig*t_samp_a)
                t_rec = np.linspace(0, 0.2, 2000)
                x_rec_a = ech_engine.reconstruction_shannon(t_samp_a,
                                                             x_samp_a, t_rec)

                fig_alias = go.Figure()
                fig_alias.add_trace(go.Scatter(x=t_demo, y=x_vrai, mode='lines',
                    name=f'Signal original f={f_alias_sig}Hz',
                    line=dict(color='rgba(0,204,255,0.4)', width=1.5)))
                fig_alias.add_trace(go.Scatter(x=t_samp_a, y=x_samp_a,
                    mode='markers', name='Échantillons',
                    marker=dict(color='#ffcc00', size=8)))
                fig_alias.add_trace(go.Scatter(x=t_rec, y=x_rec_a, mode='lines',
                    name='Reconstruit (Shannon)',
                    line=dict(color='#ff00cc', width=2.5, dash='dash')))
                layout(fig_alias,
                       f"Aliasing — f={f_alias_sig}Hz, fe={f_alias_e}Hz",
                       "t (s)", "Amplitude")
                st.plotly_chart(fig_alias, use_container_width=True)

                # Comparaison fe croissant
                st.markdown("#### 📊 Effet de fe sur la reconstruction")
                fig_fe = go.Figure()
                for fe_i, col_i in [(f_alias_sig*0.8, '#ff4444'),
                                     (f_alias_sig*1.5, '#ffcc00'),
                                     (f_alias_sig*2.5, '#00ccff'),
                                     (f_alias_sig*5, '#00ff88')]:
                    t_si = np.arange(0, 0.2, 1/fe_i)
                    x_si = np.sin(2*np.pi*f_alias_sig*t_si)
                    x_ri = ech_engine.reconstruction_shannon(t_si, x_si, t_rec)
                    label = f"fe={fe_i:.0f}Hz ({'❌' if fe_i<2*f_alias_sig else '✅'})"
                    fig_fe.add_trace(go.Scatter(x=t_rec, y=x_ri, mode='lines',
                        name=label, line=dict(color=col_i, width=2)))
                fig_fe.add_trace(go.Scatter(x=t_demo, y=x_vrai, mode='lines',
                    name='Original', line=dict(color='#ffffff', width=1.5,
                    dash='dot')))
                layout(fig_fe, "Reconstruction pour différentes fe",
                       "t (s)", "Amplitude")
                st.plotly_chart(fig_fe, use_container_width=True)

        # ---- SUB 3 : RECONSTRUCTION ----
        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔄 Reconstruction de Shannon (sinc)")
                st.latex(r"x(t) = \sum_n x(nT_e)\,\text{sinc}\!\left(\frac{t-nT_e}{T_e}\right)")
                f_rec = st.slider("f signal (Hz)", 1.0, 100.0, 10.0, 1.0,
                                   key="f_rec")
                f_e_rec = st.slider("fe (Hz)", 2.0*f_rec, 20.0*f_rec,
                                     5.0*f_rec, f_rec, key="fe_rec")
                A_rec = st.slider("A", 0.1, 3.0, 1.0, 0.1, key="A_rec")

                t_c_r, x_c_r, t_s_r, x_s_r = ech_engine.signal_echantillonne(
                    f_rec, f_e_rec, A_rec, 0.5)
                t_rec_r = np.linspace(0, 0.5, 2000)
                x_rec_r = ech_engine.reconstruction_shannon(
                    t_s_r, x_s_r, t_rec_r)

                err_rec = np.sqrt(np.mean(
                    (A_rec*np.sin(2*np.pi*f_rec*t_rec_r) - x_rec_r)**2))
                st.metric("Erreur RMS reconstruction", f"{err_rec:.6f}")
                st.metric("N échantillons utilisés", len(t_s_r))
                st.metric("Critère fe/2f",
                          f"{f_e_rec/(2*f_rec):.2f} {'✅' if f_e_rec>=2*f_rec else '❌'}")

            with col2:
                fig_rec = make_subplots(rows=2, cols=1,
                    subplot_titles=["Signal & Reconstruction",
                                    "Sinc d'interpolation (quelques termes)"])

                fig_rec.add_trace(go.Scatter(x=t_c_r, y=x_c_r, mode='lines',
                    name='Original', line=dict(color='rgba(0,204,255,0.4)', width=1.5)),
                    row=1, col=1)
                fig_rec.add_trace(go.Scatter(x=t_rec_r, y=x_rec_r, mode='lines',
                    name='Reconstruit (Shannon)',
                    line=dict(color='#ff00cc', width=2.5)), row=1, col=1)
                fig_rec.add_trace(go.Scatter(x=t_s_r, y=x_s_r, mode='markers',
                    name='Échantillons',
                    marker=dict(color='#ffcc00', size=8)), row=1, col=1)

                # Sinc individuels
                T_e = 1/f_e_rec
                for n, (tn, xn) in enumerate(zip(t_s_r[:8], x_s_r[:8])):
                    sinc_n = xn * np.sinc((t_rec_r-tn)/T_e)
                    fig_rec.add_trace(go.Scatter(x=t_rec_r, y=sinc_n,
                        mode='lines', showlegend=(n==0),
                        name='sinc_n(t)' if n==0 else "",
                        line=dict(color=f'rgba(119,0,255,0.3)', width=1.5)),
                        row=2, col=1)

                fig_rec.update_layout(**PLOT_LAYOUT, height=550,
                    title=f"Reconstruction — fe={f_e_rec:.0f}Hz, f={f_rec:.0f}Hz",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_rec.update_xaxes(**AXIS, title_text="t (s)")
                fig_rec.update_yaxes(**AXIS)
                st.plotly_chart(fig_rec, use_container_width=True)

        # ---- SUB 4 : QUANTIFICATION ----
        with sub4:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔢 Quantification uniforme")
                n_bits_q = st.slider("Résolution N (bits)", 1, 16, 8)
                f_q = st.slider("f signal (Hz)", 0.5, 50.0, 5.0, 0.5, key="fq")
                A_q = st.slider("A signal", 0.1, 5.0, 1.0, 0.1, key="Aq")
                bruit_q = st.slider("Bruit additif σ", 0.0, 0.5, 0.0, 0.01)

                n_lev = 2**n_bits_q
                delta = 2*A_q / n_lev
                snr_theo = ech_engine.snr_quantification_theorique(n_bits_q)

                st.metric("Niveaux 2^N", f"{n_lev:,}")
                st.metric("Pas Δ", f"{delta:.6f}")
                st.metric("SNR théorique (dB)", f"{snr_theo:.2f}")

            with col2:
                t_q = np.linspace(0, 1, int(fs_global), endpoint=False)
                s_q_cont = A_q*(np.sin(2*np.pi*f_q*t_q) +
                                 bruit_q*np.random.randn(len(t_q)))
                s_q_quant, erreur_q, delta_q = ech_engine.quantification(
                    s_q_cont, n_bits_q, -A_q*1.05, A_q*1.05)
                snr_num = ech_engine.snr_quantification_numerique(
                    s_q_cont, s_q_quant)

                st.metric("SNR numérique (dB)", f"{snr_num:.2f}")
                st.metric("Erreur SNR (écart théo.)",
                          f"{abs(snr_num-snr_theo):.2f} dB")

                fig_q = make_subplots(rows=3, cols=1,
                    subplot_titles=[f"Signal continu vs quantifié (N={n_bits_q}bits)",
                                    "Erreur de quantification e[n]",
                                    "Histogramme de l'erreur"])
                n_show = min(500, len(t_q))
                fig_q.add_trace(go.Scatter(x=t_q[:n_show], y=s_q_cont[:n_show],
                    mode='lines', name='Continu',
                    line=dict(color='rgba(0,204,255,0.5)', width=1.5)),
                    row=1, col=1)
                fig_q.add_trace(go.Scatter(x=t_q[:n_show],
                    y=s_q_quant[:n_show], mode='lines', name='Quantifié',
                    line=dict(color='#ff00cc', width=2, shape='hv')),
                    row=1, col=1)
                fig_q.add_trace(go.Scatter(x=t_q[:n_show],
                    y=erreur_q[:n_show], mode='lines', name='Erreur e[n]',
                    line=dict(color='#ffcc00', width=1.5)), row=2, col=1)
                fig_q.add_hline(y=delta_q/2, line_color='rgba(255,100,0,0.5)',
                                 line_dash='dash', row=2, col=1)
                fig_q.add_hline(y=-delta_q/2, line_color='rgba(255,100,0,0.5)',
                                 line_dash='dash', row=2, col=1)
                fig_q.add_trace(go.Histogram(x=erreur_q, nbinsx=50,
                    marker=dict(color='rgba(119,0,255,0.6)'),
                    name='Distrib. erreur'), row=3, col=1)

                fig_q.update_layout(**PLOT_LAYOUT, height=650,
                    title=f"Quantification {n_bits_q} bits",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_q.update_xaxes(**AXIS)
                fig_q.update_yaxes(**AXIS)
                st.plotly_chart(fig_q, use_container_width=True)

                # SNR vs bits
                bits_arr = np.arange(1, 17)
                snr_arr = [ech_engine.snr_quantification_theorique(b)
                           for b in bits_arr]
                fig_snr = go.Figure()
                fig_snr.add_trace(go.Scatter(x=bits_arr, y=snr_arr,
                    mode='lines+markers', name='SNR = 6.02N+1.76',
                    line=dict(color='#00ccff', width=2.5),
                    marker=dict(size=8)))
                fig_snr.add_vline(x=n_bits_q, line_color='#ffcc00',
                                   line_dash='dash',
                                   annotation_text=f"N={n_bits_q} → {snr_theo:.1f}dB")
                layout(fig_snr, "SNR de quantification vs N bits",
                       "N (bits)", "SNR (dB)", h=300)
                st.plotly_chart(fig_snr, use_container_width=True)

                st.download_button("💾 Export CSV",
                    pd.DataFrame({"t":t_q,"s_cont":s_q_cont,
                                  "s_quant":s_q_quant,
                                  "erreur":erreur_q}).to_csv(index=False).encode(),
                    "quantification.csv", "text/csv")

        # ---- SUB 5 : FORMULES ÉCHANTILLONNAGE ----
        with sub5:
            st.markdown("### 📖 Formulaire — Échantillonnage")
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_ECHANTILLONNAGE.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f_latex)
                col_idx += 1
            st.markdown("---")
            st.markdown("### 📊 Tableau de synthèse")
            df_synth = pd.DataFrame({
                "Critère": ["fe ≥ 2f_max", "fe < 2f_max", "fe = 2f_max",
                             "SNR quantif."],
                "Résultat": ["Reconstruction parfaite", "Aliasing",
                              "Cas limite (risqué)", "6.02N+1.76 dB"],
                "Recommandation": ["Utiliser fe ≥ 5f_max",
                                    "Augmenter fe ou filtrer anti-repliement",
                                    "Éviter (dépend de la phase)",
                                    "16 bits → 98 dB"]
            })
            st.dataframe(df_synth, use_container_width=True)
            for r in ["Shannon — *Communication in the Presence of Noise* (1949)",
                      "Nyquist — *Certain Topics in Telegraph Transmission Theory* (1928)",
                      "Oppenheim — *Discrete-Time Signal Processing* (Pearson, 2009)"]:
                st.markdown(f"- {r}")
