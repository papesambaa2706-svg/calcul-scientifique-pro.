__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp, quad
from scipy.linalg import eigh, expm
from scipy.special import hermite, eval_hermite, factorial
from scipy.optimize import brentq, fsolve
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES PHYSIQUES
# ============================================================
 
# Helper: ensure a numpy scalar or 1-element array becomes a Python scalar, otherwise return array
def _ensure_scalar(val):
    arr = np.asarray(val)
    if arr.size == 1:
        return arr.item()
    return arr

CONSTANTES = {
    "ħ (J·s)":          1.0546e-34,
    "h (J·s)":          6.6261e-34,
    "m_e (kg)":         9.1094e-31,
    "e (C)":            1.6022e-19,
    "ε₀ (F/m)":         8.8542e-12,
    "a₀ (m)":           5.2918e-11,
    "E_Hartree (eV)":   27.211,
    "E_Rydberg (eV)":   13.606,
    "k_B (J/K)":        1.3806e-23,
    "c (m/s)":          2.9979e8,
}

FORMULES = {
    "Équation de Schrödinger":   r"i\hbar\frac{\partial\psi}{\partial t}=\hat{H}\psi=\left[-\frac{\hbar^2}{2m}\nabla^2+V\right]\psi",
    "Hamiltonien":               r"\hat{H}=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x)",
    "Puits infini (E_n)":        r"E_n=\frac{n^2\pi^2\hbar^2}{2mL^2},\quad n=1,2,3\ldots",
    "Puits infini (ψ_n)":        r"\psi_n(x)=\sqrt{\frac{2}{L}}\sin\!\left(\frac{n\pi x}{L}\right)",
    "Oscillateur harmonique":    r"E_n=\hbar\omega\!\left(n+\frac{1}{2}\right)",
    "Atome d'hydrogène":         r"E_n=-\frac{13.6\text{ eV}}{n^2},\quad n=1,2,\ldots",
    "Effet tunnel":              r"T\approx e^{-2\kappa L},\quad\kappa=\sqrt{\frac{2m(V_0-E)}{\hbar^2}}",
    "Règle de Born":             r"P(x)=|\psi(x)|^2,\quad\int_{-\infty}^{\infty}|\psi|^2dx=1",
    "Incertitude Heisenberg":    r"\Delta x\cdot\Delta p\geq\frac{\hbar}{2}",
    "Opérateur quantité de mvt": r"\hat{p}=-i\hbar\frac{\partial}{\partial x}",
    "Valeur moyenne":            r"\langle A\rangle=\int\psi^*\hat{A}\psi\,dx",
    "Fonction d'onde H":         r"\psi_{nlm}=R_{nl}(r)Y_l^m(\theta,\phi)",
}

# ============================================================
# NOUVELLES FORMULES LaTeX — Particules confinées
# ============================================================
FORMULES_PARTICULES_CONFINEES = {
    "Puits fini (pair)":
        r"k\tan\!\left(\frac{kL}{2}\right)=\kappa,\quad k=\sqrt{\frac{2mE}{\hbar^2}},\quad\kappa=\sqrt{\frac{2m(V_0-E)}{\hbar^2}}",
    "Puits fini (impair)":
        r"-k\cot\!\left(\frac{kL}{2}\right)=\kappa",
    "Dédoublement tunnel":
        r"\Delta E=E_+-E_-\approx\frac{\hbar^2\kappa}{mL_b}e^{-\kappa L_b}",
    "Courant de probabilité":
        r"j=\frac{\hbar}{2mi}\!\left(\psi^*\frac{\partial\psi}{\partial x}-\psi\frac{\partial\psi^*}{\partial x}\right)",
    "Équation de continuité":
        r"\frac{\partial|\psi|^2}{\partial t}+\frac{\partial j}{\partial x}=0",
    "Longueur de pénétration":
        r"l_p=\frac{1}{\kappa}=\frac{\hbar}{\sqrt{2m(V_0-E)}}",
    "Résonance de Ramsauer":
        r"T=1\text{ si }k_2 L=n\pi,\quad k_2=\sqrt{\frac{2m(E+V_0)}{\hbar^2}}",
    "Quantification WKB":
        r"\int_{x_1}^{x_2}\!\sqrt{2m(E-V(x))}\,dx=\!\left(n-\frac{1}{2}\right)\!\pi\hbar",
}

# NOUVELLES FORMULES LaTeX — Formalisme mathématique
FORMULES_FORMALISME = {
    "Produit scalaire":
        r"\langle\phi|\psi\rangle=\int_{-\infty}^{+\infty}\phi^*(x)\psi(x)\,dx",
    "Complétude":
        r"\sum_n|n\rangle\langle n|=\hat{I},\quad\langle m|n\rangle=\delta_{mn}",
    "Représentation matricielle":
        r"A_{mn}=\langle m|\hat{A}|n\rangle",
    "Commutateur fondamental":
        r"[\hat{x},\hat{p}]=i\hbar",
    "Théorème de Robertson":
        r"\Delta A\cdot\Delta B\geq\frac{1}{2}\left|\langle[\hat{A},\hat{B}]\rangle\right|",
    "Opérateurs échelle":
        r"\hat{a}=\sqrt{\frac{m\omega}{2\hbar}}\hat{x}+\frac{i\hat{p}}{\sqrt{2m\hbar\omega}},\quad\hat{a}^\dagger\hat{a}|n\rangle=n|n\rangle",
    "Opérateur densité":
        r"\hat{\rho}=|\psi\rangle\langle\psi|,\quad\mathrm{Tr}(\hat{\rho})=1",
    "Image de Heisenberg":
        r"\frac{d\hat{A}_H}{dt}=\frac{i}{\hbar}[\hat{H},\hat{A}_H]+\frac{\partial\hat{A}}{\partial t}",
    "Transformée de Fourier quantique":
        r"\tilde{\psi}(p)=\frac{1}{\sqrt{2\pi\hbar}}\int\psi(x)e^{-ipx/\hbar}\,dx",
}

# NOUVELLES FORMULES LaTeX — Postulats
FORMULES_POSTULATS = {
    "Postulat 1 — État":
        r"|\psi\rangle\in\mathcal{H},\quad\langle\psi|\psi\rangle=1",
    "Postulat 2 — Observables":
        r"\hat{A}=\hat{A}^\dagger,\quad\hat{A}|a_n\rangle=a_n|a_n\rangle",
    "Postulat 3 — Probabilité":
        r"P(a_n)=|\langle a_n|\psi\rangle|^2=|c_n|^2",
    "Postulat 4 — Réduction":
        r"|\psi\rangle\xrightarrow{\text{mesure }a_n}\frac{\hat{P}_n|\psi\rangle}{\sqrt{\langle\psi|\hat{P}_n|\psi\rangle}}",
    "Postulat 5 — Évolution":
        r"i\hbar\frac{d}{dt}|\psi(t)\rangle=\hat{H}|\psi(t)\rangle",
    "Opérateur d'évolution":
        r"|\psi(t)\rangle=e^{-i\hat{H}t/\hbar}|\psi(0)\rangle",
    "Décomposition spectrale":
        r"|\psi\rangle=\sum_n c_n|a_n\rangle,\quad c_n=\langle a_n|\psi\rangle",
    "Valeur moyenne":
        r"\langle\hat{A}\rangle=\sum_n a_n|c_n|^2",
    "Théorème d'Ehrenfest":
        r"\frac{d\langle\hat{p}\rangle}{dt}=-\!\left\langle\frac{\partial V}{\partial x}\right\rangle",
    "Déviation standard":
        r"(\Delta A)^2=\langle\hat{A}^2\rangle-\langle\hat{A}\rangle^2",
}

COULEURS_QUANTI = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88',
                   '#ffcc00', '#ff4400', '#88ccff', '#cc88ff']


def style_figure(fig, title="", xlab="", ylab="", height=420,
                 ytype: str | None = None, showlegend: bool = False):
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(5,0,20,0.8)',
        font=dict(color='#c0d0ff'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
        height=height,
        showlegend=showlegend,
    )
    fig.update_xaxes(type=ytype or 'linear', color='#c0d0ff',
                     gridcolor='rgba(100,0,255,0.2)')
    fig.update_yaxes(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)')
    return fig


def formula_section(title: str, formulas: dict[str, str], default_open: bool = False):
    with st.expander(title, expanded=default_open):
        cols = st.columns(2)
        col_idx = 0
        for nom, formule in formulas.items():
            with cols[col_idx % 2]:
                st.markdown(f"**{nom}**")
                st.latex(formule)
            col_idx += 1


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    dx = np.diff(x)
    return np.sum((y[:-1] + y[1:]) / 2 * dx)


# ============================================================
# MOTEUR MÉCANIQUE QUANTIQUE  (ORIGINAL — INCHANGÉ)
# ============================================================
class QuantumEngine:
    """Moteur de calcul en mécanique quantique."""

    def __init__(self, m: float = 9.1094e-31, hbar: float = 1.0546e-34):
        self.m = m
        self.hbar = hbar

    # --- Puits infini ---
    def puits_infini_energie(self, n: int, L: float) -> float:
        return n**2 * np.pi**2 * self.hbar**2 / (2 * self.m * L**2)

    def puits_infini_psi(self, n: int, x: np.ndarray, L: float) -> np.ndarray:
        return np.sqrt(2/L) * np.sin(n * np.pi * x / L) * (x >= 0) * (x <= L)

    def puits_infini_multi(_self, x: np.ndarray, L: float,
                            n_max: int = 5) -> tuple:
        energies = []
        psis = []
        for n in range(1, n_max+1):
            En = _self.puits_infini_energie(n, L)
            psi = _self.puits_infini_psi(n, x, L)
            energies.append(En)
            psis.append(psi)
        return np.array(energies), np.array(psis)

    # --- Oscillateur harmonique ---
    def osc_harm_energie(self, n: int, omega: float) -> float:
        return self.hbar * omega * (n + 0.5)

    def osc_harm_psi(_self, n: int, x: np.ndarray, omega: float) -> np.ndarray:
        alpha = np.sqrt(_self.m * omega / _self.hbar)
        xi = alpha * x
        Hn = eval_hermite(n, xi)
        N = (1/(np.sqrt(2**n * float(factorial(n)))) *
             (alpha/np.pi)**0.25)
        return N * Hn * np.exp(-xi**2/2)

    def osc_harm_potentiel(self, x: np.ndarray, omega: float) -> np.ndarray:
        return 0.5 * self.m * omega**2 * x**2

    # --- Barrière rectangulaire (effet tunnel) ---
    def transmittance_tunnel(_self, E_arr: np.ndarray,
                              V0: float, L: np.ndarray | float) -> np.ndarray:
        E_arr = np.asarray(E_arr, dtype=float)
        L_arr = np.asarray(L, dtype=float)

        if L_arr.shape == ():
            L_arr = np.full_like(E_arr, L_arr)
        elif L_arr.shape != E_arr.shape:
            raise ValueError("L doit être scalaire ou avoir la même forme que E_arr")

        T = np.zeros_like(E_arr, dtype=float)
        mask = E_arr >= V0

        if np.any(mask):
            E_pos = E_arr[mask]
            L_pos = L_arr[mask]
            k1 = np.sqrt(2*_self.m*E_pos) / _self.hbar
            k2 = np.sqrt(2*_self.m*(E_pos - V0)) / _self.hbar
            sin_term = np.sin(k2 * L_pos)
            safe_mask = (k1 != 0) & (k2 != 0)
            denom_pos = 1 + (k1**2 - k2**2)**2 / (4 * k1**2 * k2**2 + 1e-30) * sin_term**2
            T[mask] = np.where(
                safe_mask,
                1.0 / denom_pos,
                np.where(k1 != 0, 1.0 / (1 + (V0 / (2 * E_pos))**2 * k1**2 * L_pos**2), 0.0)
            )

        if np.any(~mask):
            E_neg = E_arr[~mask]
            L_neg = L_arr[~mask]
            k1 = np.sqrt(2*_self.m*E_neg) / _self.hbar
            kappa = np.sqrt(2*_self.m*(V0 - E_neg)) / _self.hbar
            kL = kappa * L_neg
            sinh2 = np.where(kL < 500, np.sinh(kL)**2, np.exp(2*kL) / 4)
            denom_neg = 1 + (k1**2 + kappa**2)**2 / (4 * k1**2 * kappa**2 + 1e-30) * sinh2
            T[~mask] = np.where((k1 == 0) | (kappa == 0), 0.0, 1.0 / denom_neg)

        return np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

    def paquet_onde(self, x: np.ndarray, x0: float, sigma: float,
                    k0: float, t: float, omega_k: float = None) -> np.ndarray:
        """Paquet d'onde gaussien en propagation libre."""
        hbar = self.hbar
        m = self.m
        sigma_t = np.sqrt(sigma**2 + (hbar*t/(2*m*sigma))**2) if t != 0 else sigma
        phase = k0*x - (hbar*k0**2/(2*m))*t
        envelope = np.exp(-(x-x0-hbar*k0*t/m)**2/(4*sigma_t**2))
        norm = (2*np.pi*sigma_t**2)**0.25
        psi = (1/norm) * envelope * np.exp(1j*phase)
        return psi

    # --- Atome d'hydrogène ---
    def H_energie(self, n: int) -> float:
        """Énergie en Joules."""
        E_R = 13.6 * 1.6022e-19
        return -E_R / n**2

    def H_energie_eV(self, n: int) -> float:
        return -13.6 / n**2

    def H_rayon_moyen(self, n: int, l: int) -> float:
        """Rayon moyen <r> en unités de a₀."""
        a0 = 5.2918e-11
        return a0 * (3*n**2 - l*(l+1)) / 2

    def H_psi_1s(self, r: np.ndarray) -> np.ndarray:
        """Fonction d'onde 1s normalisée."""
        a0 = 5.2918e-11
        return (1/np.sqrt(np.pi)) * (1/a0)**1.5 * np.exp(-r/a0)

    def H_densite_radiale(self, n: int, l: int,
                           r: np.ndarray) -> np.ndarray:
        """Densité de probabilité radiale P(r) = r²|R_nl|²."""
        a0 = 5.2918e-11
        rho = 2 * r / (n * a0)
        rho = np.maximum(rho, 1e-10)

        if n == 1 and l == 0:
            R = 2 * (1/a0)**1.5 * np.exp(-r/a0)
        elif n == 2 and l == 0:
            R = (1/(2*np.sqrt(2))) * (1/a0)**1.5 * (2-rho) * np.exp(-rho/2)
        elif n == 2 and l == 1:
            R = (1/(2*np.sqrt(6))) * (1/a0)**1.5 * rho * np.exp(-rho/2)
        elif n == 3 and l == 0:
            R = (2/(81*np.sqrt(3))) * (1/a0)**1.5 * (27-18*rho+2*rho**2) * np.exp(-rho/3)
        else:
            R = np.exp(-rho/n) / n

        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        return r**2 * np.abs(R)**2

    # --- Quantification numérique (FDM) ---
    def resoudre_schrodinger_1d(self, x: np.ndarray,
                                  V: np.ndarray, n_states: int = 6) -> tuple:
        """Résolution numérique par différences finies."""
        N = len(x)
        dx = x[1] - x[0]
        diag = self.hbar**2 / (self.m * dx**2) + V
        off = -self.hbar**2 / (2 * self.m * dx**2) * np.ones(N-1)
        H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        eigenvalues, eigenvectors = eigh(H, subset_by_index=[0, min(n_states-1, N-2)])
        return eigenvalues, eigenvectors.T

    # --- Evolution temporelle ---
    def evolution_temporelle(self, psi0: np.ndarray,
                              x: np.ndarray, V: np.ndarray,
                              t_arr: np.ndarray) -> np.ndarray:
        """Evolution par exponentielle matricielle."""
        N = len(x)
        dx = x[1] - x[0]
        diag = self.hbar**2 / (self.m * dx**2) + V
        off = -self.hbar**2 / (2 * self.m * dx**2) * np.ones(N-1)
        H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        psi_t = []
        for t in t_arr:
            U_t = expm(-1j * H * t / self.hbar)
            psi_t.append(U_t @ psi0)
        return np.array(psi_t)

    # --- Valeurs moyennes ---
    def valeur_moyenne_x(self, psi: np.ndarray, x: np.ndarray) -> float:
        dx = x[1] - x[0]
        # support psi either 1D (shape (N,)) or vectorized with last axis the spatial one
        integrand = np.conj(psi) * x * psi
        res = np.real(np.sum(integrand, axis=-1) * dx)
        return _ensure_scalar(res)

    def incertitude_x(self, psi: np.ndarray, x: np.ndarray) -> float:
        dx = x[1] - x[0]
        x_moy = self.valeur_moyenne_x(psi, x)
        integrand2 = np.conj(psi) * x**2 * psi
        x2_moy = np.real(np.sum(integrand2, axis=-1) * dx)
        # compute variance, support scalar or array
        var = x2_moy - np.asarray(x_moy)**2
        var = np.maximum(var, 0)
        res = np.sqrt(var)
        return _ensure_scalar(res)

    def incertitude_p(self, psi: np.ndarray, x: np.ndarray) -> float:
        dx = x[1] - x[0]
        dpsi = np.gradient(psi, dx, axis=-1)
        dpsi = np.nan_to_num(dpsi, nan=0.0, posinf=0.0, neginf=0.0)
        # handle vectorized psi along last axis
        p_moy = np.real(np.sum(np.conj(psi) * (-1j*self.hbar) * dpsi, axis=-1) * dx)
        ddpsi = np.gradient(dpsi, dx, axis=-1)
        p2_moy = np.real(np.sum(np.conj(psi) * (-self.hbar**2) * ddpsi, axis=-1) * dx)
        var = p2_moy - np.asarray(p_moy)**2
        var = np.maximum(var, 0)
        res = np.sqrt(var)
        return _ensure_scalar(res)


# ============================================================
# NOUVEAU MOTEUR — PARTICULES CONFINÉES
# ============================================================
class ParticulesConfineesEngine:
    """
    Calculs avancés pour particules confinées :
    puits fini, double puits, courant de probabilité, WKB.
    """

    def __init__(self, m: float = 9.1094e-31, hbar: float = 1.0546e-34):
        self.m    = m
        self.hbar = hbar
        self.eV   = 1.6022e-19

    def puits_fini_niveaux(self, L: float, V0: float, n_max: int = 8) -> np.ndarray:
        """
        États liés du puits fini par équations transcendantes.
        Retourne les énergies en Joules.
        """
        energies = []
        E_arr = np.linspace(1e-6 * self.eV, V0 * 0.9999, 12000)
        k     = np.sqrt(2 * self.m * E_arr) / self.hbar
        kappa = np.sqrt(2 * self.m * (V0 - E_arr)) / self.hbar

        f_pair   = k * np.tan(k * L / 2) - kappa
        f_impair = -k / np.tan(np.clip(k * L / 2, 1e-30, None)) - kappa

        for i in range(len(E_arr) - 1):
            try:
                if np.isfinite(f_pair[i]) and f_pair[i] * f_pair[i+1] < 0:
                    E_sol = brentq(
                        lambda E: (np.sqrt(2*self.m*E)/self.hbar) *
                                   np.tan((np.sqrt(2*self.m*E)/self.hbar)*L/2) -
                                   np.sqrt(2*self.m*(V0-E))/self.hbar,
                        E_arr[i], E_arr[i+1], xtol=1e-40
                    )
                    energies.append(E_sol)
                if np.isfinite(f_impair[i]) and f_impair[i] * f_impair[i+1] < 0:
                    E_sol = brentq(
                        lambda E: -(np.sqrt(2*self.m*E)/self.hbar) /
                                    np.tan(np.clip((np.sqrt(2*self.m*E)/self.hbar)*L/2, 1e-30, None)) -
                                    np.sqrt(2*self.m*(V0-E))/self.hbar,
                        E_arr[i], E_arr[i+1], xtol=1e-40
                    )
                    energies.append(E_sol)
            except Exception:
                continue

        energies = sorted(set(np.round(energies, 35)))
        return np.array(energies[:n_max])

    def puits_fini_psi(self, x: np.ndarray, E: float,
                        L: float, V0: float, parite: str = 'pair') -> np.ndarray:
        """Fonction d'onde analytique d'un état lié du puits fini."""
        k     = np.sqrt(2 * self.m * E) / self.hbar
        kappa = np.sqrt(max(2 * self.m * (V0 - E), 0)) / self.hbar
        psi   = np.zeros_like(x, dtype=float)

        interieur = (x >= -L/2) & (x <= L/2)
        gauche    = x < -L/2
        droite    = x > L/2

        if parite == 'pair':
            A = np.cos(k * L / 2)
            psi[interieur] = np.cos(k * x[interieur])
            psi[gauche]    = A * np.exp(kappa * (x[gauche] + L/2))
            psi[droite]    = A * np.exp(-kappa * (x[droite] - L/2))
        else:
            A = np.sin(k * L / 2)
            psi[interieur] = np.sin(k * x[interieur])
            psi[gauche]    = -A * np.exp(kappa * (x[gauche] + L/2))
            psi[droite]    =  A * np.exp(-kappa * (x[droite] - L/2))

        norme = trapezoid_integral(psi**2, x)
        if norme > 0:
            psi /= np.sqrt(norme)
        return psi

    def double_puits_fdm(self, L_puits: float, L_barriere: float,
                          V0: float, n_states: int = 6) -> tuple:
        """Résolution FDM du double puits. Retourne (E, ψ, x, V)."""
        L_tot = 2 * (2 * L_puits + L_barriere)
        x_arr = np.linspace(-L_tot, L_tot, 300)
        dx    = x_arr[1] - x_arr[0]

        V = np.full_like(x_arr, V0)
        p1l, p1r = -L_barriere/2 - L_puits, -L_barriere/2
        p2l, p2r =  L_barriere/2,             L_barriere/2 + L_puits
        V[(x_arr >= p1l) & (x_arr <= p1r)] = 0.0
        V[(x_arr >= p2l) & (x_arr <= p2r)] = 0.0

        diag = self.hbar**2 / (self.m * dx**2) + V
        off  = -self.hbar**2 / (2 * self.m * dx**2) * np.ones(len(x_arr) - 1)
        H    = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        evals, evecs = eigh(H, subset_by_index=[0, min(n_states-1, len(x_arr)-2)])
        return evals, evecs.T, x_arr, V

    def dedoublement_tunnel(self, L_b: float, V0: float, E: float) -> float:
        """Dédoublement ΔE ≈ (ħ²κ/mL_b) exp(-κ L_b)."""
        kappa = np.sqrt(2 * self.m * max(V0 - E, 0)) / self.hbar
        return (self.hbar**2 * kappa / (self.m * L_b)) * np.exp(-kappa * L_b)

    def courant_probabilite(self, psi: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Courant j(x) = (ħ/2mi)(ψ* ∂ψ/∂x − ψ ∂ψ*/∂x)."""
        dx   = x[1] - x[0]
        # compute gradient along spatial (last) axis to support vectorized psi
        dpsi = np.gradient(psi, dx, axis=-1)
        j    = (self.hbar / (2 * self.m * 1j)) * (np.conj(psi)*dpsi - psi*np.conj(dpsi))
        return np.real(j)

    def courant_probabilite_vecteur(self, psi: np.ndarray, x: np.ndarray) -> dict:
        """Retourne le vecteur densité de courant et les composantes physiques."""
        j = self.courant_probabilite(psi, x)
        return {
            "x": x,
            "j": j,
            # convert to Python scalar when the result is 0-d, otherwise keep array
            "j_max": _ensure_scalar(np.max(np.abs(j))),
            "j_moy": _ensure_scalar(np.mean(j)),
        }

    def densite_probabilite(self, psi: np.ndarray) -> np.ndarray:
        """Densité de probabilité |ψ(x)|²."""
        return np.abs(psi)**2

    def reflexion_transmission_barriere(self, E: float, V0: float, a: float) -> dict:
        """Compute R/T for une barrière rectangulaire de largeur a."""
        k1 = np.sqrt(2 * self.m * E) / self.hbar
        if E >= V0:
            k2 = np.sqrt(2 * self.m * (E - V0)) / self.hbar
        else:
            k2 = 1j * np.sqrt(2 * self.m * (V0 - E)) / self.hbar

        denom = (k1 + k2)**2 * np.exp(-1j * k2 * a) - (k1 - k2)**2 * np.exp(1j * k2 * a)
        t_amp = (4 * k1 * k2 * np.exp(-1j * k1 * a)) / denom
        r_amp = ((k1**2 - k2**2) * np.sin(k2 * a) * np.exp(-1j * k1 * a)) / (2j * k1 * k2 * np.cos(k2 * a) + (k1**2 + k2**2) * np.sin(k2 * a))

        R = float(np.abs(r_amp)**2)
        T = float(np.abs(t_amp)**2 * (np.real(k2) / np.real(k1)) if E >= V0 else np.abs(t_amp)**2)
        if E == 0:
            R, T = 1.0, 0.0
        return {
            "r_amp": r_amp,
            "t_amp": t_amp,
            "R": min(max(R, 0.0), 1.0),
            "T": min(max(T, 0.0), 1.0),
            "conservation": float((R + T)),
        }

    def solutions_scattering(self, E: float, V0: float, a: float, x: np.ndarray) -> dict:
        """Solutions analytiques pour une barrière potentielle rectangulaire."""
        k1 = np.sqrt(2 * self.m * E) / self.hbar
        psi = np.zeros_like(x, dtype=complex)
        if E >= V0:
            k2 = np.sqrt(2 * self.m * (E - V0)) / self.hbar
        else:
            k2 = 1j * np.sqrt(2 * self.m * (V0 - E)) / self.hbar

        # incident à gauche, barrière entre 0 et a
        xL = x[x < 0]
        xC = x[(x >= 0) & (x <= a)]
        xR = x[x > a]
        params = self.reflexion_transmission_barriere(E, V0, a)
        r = params["r_amp"]
        t = params["t_amp"]
        psi[x < 0] = np.exp(1j * k1 * xL) + r * np.exp(-1j * k1 * xL)
        psi[(x >= 0) & (x <= a)] = (np.exp(1j * k2 * xC) + np.exp(-1j * k2 * xC))
        psi[x > a] = t * np.exp(1j * k1 * xR)
        return {
            "psi": psi,
            "densite": self.densite_probabilite(psi),
            "courant": self.courant_probabilite(psi, x),
            **params,
        }

    def diagnostic_puits(self, L: float, V0: float) -> list:
        """Tableau de diagnostic physique d'un puits fini."""
        z0        = np.sqrt(2 * self.m * V0) * L / (2 * self.hbar)
        n_min     = max(1, int(np.ceil(z0 / (np.pi / 2))))
        E1_inf    = np.pi**2 * self.hbar**2 / (2 * self.m * L**2)
        kappa_mid = np.sqrt(2 * self.m * (V0 / 2)) / self.hbar
        l_pen     = 1.0 / kappa_mid if kappa_mid > 0 else float('inf')
        return [
            {"Grandeur": "Paramètre z₀",              "Valeur": f"{z0:.4f}",                    "Unité": "sans dim."},
            {"Grandeur": "Nb états liés (min)",        "Valeur": str(n_min),                      "Unité": "—"},
            {"Grandeur": "E₁ puits infini",            "Valeur": f"{E1_inf/self.eV:.5f}",         "Unité": "eV"},
            {"Grandeur": "Longueur de pénétration",    "Valeur": f"{l_pen*1e9:.4f}",              "Unité": "nm"},
            {"Grandeur": "V₀ / E₁∞",                  "Valeur": f"{V0/E1_inf:.2f}",              "Unité": "—"},
        ]


# ============================================================
# NOUVEAU MOTEUR — FORMALISME MATHÉMATIQUE
# ============================================================
class FormalismeEngine:
    """
    Formalisme de Dirac, représentation matricielle,
    commutateurs, opérateur densité, transformée de Fourier quantique.
    """

    def __init__(self, hbar: float = 1.0546e-34):
        self.hbar = hbar

    def operateur_creation(self, n_max: int) -> np.ndarray:
        """â† dans la base de Fock |0⟩…|n_max-1⟩."""
        return np.diag(np.sqrt(np.arange(1, n_max)), -1).astype(complex)

    def operateur_annihilation(self, n_max: int) -> np.ndarray:
        """â."""
        return np.diag(np.sqrt(np.arange(1, n_max)), 1).astype(complex)

    def operateur_x_matriciel(self, n_max: int, omega: float,
                               m: float = 9.1094e-31) -> np.ndarray:
        """x̂ = √(ħ/2mω)(â + â†)."""
        a    = self.operateur_annihilation(n_max)
        adag = self.operateur_creation(n_max)
        return np.sqrt(self.hbar / (2 * m * omega)) * (a + adag)

    def operateur_p_matriciel(self, n_max: int, omega: float,
                               m: float = 9.1094e-31) -> np.ndarray:
        """p̂ = i√(mħω/2)(â† − â)."""
        a    = self.operateur_annihilation(n_max)
        adag = self.operateur_creation(n_max)
        return 1j * np.sqrt(m * self.hbar * omega / 2) * (adag - a)

    def commutateur(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B - B @ A

    def anticommutateur(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B + B @ A

    def ket(self, vecteur: np.ndarray) -> np.ndarray:
        return np.array(vecteur, dtype=complex).reshape(-1, 1)

    def bra(self, vecteur: np.ndarray) -> np.ndarray:
        return self.ket(vecteur).conj().T

    def produit_scalaire(self, u: np.ndarray, v: np.ndarray) -> complex:
        scalar = self.bra(u) @ self.ket(v)
        return complex(_ensure_scalar(scalar))

    def outer_product(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.ket(u) @ self.bra(v)

    def spectral_decomposition(self, A: np.ndarray) -> tuple:
        evals, evecs = np.linalg.eig(A)
        proj = [np.outer(evecs[:, i], np.conj(evecs[:, i])) for i in range(evecs.shape[1])]
        return evals, evecs, proj

    def verifier_hermitien(self, A: np.ndarray) -> dict:
        diff    = float(np.max(np.abs(A - A.conj().T)))
        evals   = np.linalg.eigvalsh(A)
        return {
            "Hermitien":              diff < 1e-10,
            "‖A − A†‖∞":             diff,
            "Val. propre min (réel)": float(evals.min()),
            "Val. propre max (réel)": float(evals.max()),
        }

    def evolution_heisenberg(self, A: np.ndarray, H: np.ndarray, t: float) -> np.ndarray:
        """A_H(t) = U†(t) A U(t)."""
        U = expm(-1j * H * t / self.hbar)
        return U.conj().T @ A @ U

    def operateur_densite_pur(self, psi: np.ndarray) -> np.ndarray:
        psi_col = self.ket(psi)
        return psi_col @ psi_col.conj().T

    def operateur_densite_melange(self, etats: list, probabilites: list) -> np.ndarray:
        dim = len(etats[0])
        rho = np.zeros((dim, dim), dtype=complex)
        for p, psi in zip(probabilites, etats):
            v = np.array(psi).reshape(-1, 1)
            rho += p * (v @ v.conj().T)
        return rho

    def purete(self, rho: np.ndarray) -> float:
        return float(np.real(np.trace(rho @ rho)))

    def entropie_von_neumann(self, rho: np.ndarray) -> float:
        evals = np.linalg.eigvalsh(rho)
        evals = evals[evals > 1e-15]
        return float(-np.sum(evals * np.log(evals)))

    def valeur_moyenne_op(self, psi: np.ndarray, A: np.ndarray) -> complex:
        psi_col = self.ket(psi)
        val = psi_col.conj().T @ A @ psi_col
        return complex(_ensure_scalar(val))

    def incertitude_op(self, psi: np.ndarray, A: np.ndarray) -> float:
        moy    = self.valeur_moyenne_op(psi, A)
        moy_sq = self.valeur_moyenne_op(psi, A @ A)
        return float(np.sqrt(max(np.real(moy_sq - moy**2), 0)))

    def transformer_base_impulsion(self, psi_x: np.ndarray,
                                    x: np.ndarray) -> tuple:
        """Retourne (p_arr, ψ̃(p)) normalisé."""
        N  = len(x)
        dx = x[1] - x[0]
        psi_p = np.fft.fftshift(np.fft.fft(psi_x)) * dx / np.sqrt(2 * np.pi * self.hbar)
        p_arr = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi * self.hbar
        norme = trapezoid_integral(np.abs(psi_p)**2, p_arr)
        if norme > 0:
            psi_p /= np.sqrt(norme)
        return p_arr, psi_p


# ============================================================
# NOUVEAU MOTEUR — POSTULATS
# ============================================================
class PostulatsEngine:
    """
    Illustration des postulats : superposition, mesure,
    réduction, évolution, Ehrenfest.
    """

    def __init__(self, hbar: float = 1.0546e-34, m: float = 9.1094e-31):
        self.hbar = hbar
        self.m    = m
        self.eV   = 1.6022e-19

    def superposition_puits(self, coefficients: list, x: np.ndarray, L: float) -> np.ndarray:
        """État superposé normalisé dans le puits infini."""
        psi = np.zeros_like(x, dtype=complex)
        for n, c in enumerate(coefficients, start=1):
            psi += c * np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        norme = trapezoid_integral(np.abs(psi)**2, x)
        return psi / np.sqrt(max(norme, 1e-30))

    def probabilites_etats(self, coefficients: list) -> np.ndarray:
        c    = np.array(coefficients, dtype=complex)
        prob = np.abs(c)**2
        return prob / max(prob.sum(), 1e-30)

    def valeur_moyenne_energie(self, coefficients: list,
                                energies: np.ndarray) -> float:
        prob = self.probabilites_etats(coefficients)
        n    = min(len(prob), len(energies))
        return float(np.sum(prob[:n] * energies[:n]))

    def reduire_etat(self, psi: np.ndarray, x: np.ndarray,
                     x_mesure: float, sigma_mesure: float) -> np.ndarray:
        """Réduction par mesure gaussienne centrée en x_mesure."""
        fenetre    = np.exp(-(x - x_mesure)**2 / (2 * sigma_mesure**2))
        psi_r      = psi * fenetre
        norme      = trapezoid_integral(np.abs(psi_r)**2, x)
        return psi_r / np.sqrt(max(norme, 1e-30))

    def evolution_superposition(self, coefficients: list, x: np.ndarray,
                                  L: float, t_arr: np.ndarray) -> np.ndarray:
        """ψ(x,t) = Σ cₙ e^(−iEₙt/ħ) ψₙ(x). Retourne tableau (t, x)."""
        psi_t = np.zeros((len(t_arr), len(x)), dtype=complex)
        for n, c in enumerate(coefficients, start=1):
            En    = n**2 * np.pi**2 * self.hbar**2 / (2 * self.m * L**2)
            psi_n = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
            phases = np.exp(-1j * En * t_arr / self.hbar)
            psi_t += c * phases[:, np.newaxis] * psi_n[np.newaxis, :]
        norme0 = trapezoid_integral(np.abs(psi_t[0])**2, x)
        if norme0 > 0:
            psi_t /= np.sqrt(norme0)
        return psi_t

    def decomposition_spectrale(self, A: np.ndarray) -> dict:
        """Décomposition spectrale d'un opérateur Hermitien."""
        evals, evecs = np.linalg.eigh(A)
        projecteurs = [np.outer(evecs[:, i], np.conj(evecs[:, i])) for i in range(len(evals))]
        return {
            "valeurs_propres": evals,
            "vecteurs_propres": evecs,
            "projecteurs": projecteurs,
        }

    def reduction_paquet_onde(self, psi: np.ndarray, projecteur: np.ndarray) -> np.ndarray:
        """Réduction du paquet d'onde selon un projecteur de mesure."""
        psi_r = projecteur @ psi
        norme = trapezoid_integral(np.abs(psi_r)**2, np.linspace(0, 1, len(psi_r)))
        return psi_r / np.sqrt(max(norme, 1e-30))

    def evolution_temps_general(self, psi0: np.ndarray, H: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
        """Evolution temporelle générale ψ(t) = e^{-iHt/ħ} ψ(0)."""
        psi_t = np.zeros((len(t_arr), len(psi0)), dtype=complex)
        for i, t in enumerate(t_arr):
            U = expm(-1j * H * t / self.hbar)
            psi_t[i] = U @ psi0
        return psi_t

    def ehrenfest_position(self, psi_t: np.ndarray, x: np.ndarray) -> np.ndarray:
        dx = x[1] - x[0]
        return np.real(np.sum(np.abs(psi_t)**2 * x[np.newaxis, :], axis=1)) * dx

    def incertitude_position(self, psi_t: np.ndarray, x: np.ndarray) -> np.ndarray:
        dx    = x[1] - x[0]
        prob  = np.abs(psi_t)**2
        xm    = np.sum(prob * x[np.newaxis, :], axis=1) * dx
        x2m   = np.sum(prob * (x**2)[np.newaxis, :], axis=1) * dx
        return np.sqrt(np.maximum(x2m - xm**2, 0))

    def diagnostic_postulats(self, coefficients: list,
                              energies: np.ndarray) -> list:
        prob    = self.probabilites_etats(coefficients)
        n       = min(len(prob), len(energies))
        E_moy   = float(np.sum(prob[:n] * energies[:n]))
        E2_moy  = float(np.sum(prob[:n] * energies[:n]**2))
        dE      = np.sqrt(max(E2_moy - E_moy**2, 0))
        entropie = -np.sum(prob[prob > 0] * np.log(prob[prob > 0]))
        return [
            {"Grandeur": "Σ|cₙ|²",           "Valeur": f"{prob.sum():.6f}", "Statut": "✅" if abs(prob.sum()-1)<1e-5 else "⚠️"},
            {"Grandeur": "⟨E⟩ (eV)",          "Valeur": f"{E_moy/self.eV:.6f}", "Statut": "ℹ️"},
            {"Grandeur": "ΔE (eV)",            "Valeur": f"{dE/self.eV:.6f}",   "Statut": "✅"},
            {"Grandeur": "Entropie Shannon",   "Valeur": f"{entropie:.4f}",     "Statut": "ℹ️"},
        ]


# ============================================================
# PAGE PRINCIPALE (ENRICHIE)
# ============================================================
def mecanique_quantique_page():
    st.markdown("## ⚛️ Mécanique Quantique Avancée")
    st.markdown("*Puits quantiques, oscillateur, effet tunnel, atome H, evolution temporelle*")
    st.markdown("---")

    section = st.selectbox("Section", [
        "📦 Puits quantique",
        "〰️ Oscillateur harmonique",
        "🚧 Effet tunnel",
        "🔬 Atome d'hydrogène",
        "🕐 Évolution temporelle",
        "📖 Théorie",
        "🔒 Particules confinées",
        "🧮 Formalisme mathématique",
        "📐 Postulats",
    ], index=0)

    engine      = QuantumEngine()
    eng_conf    = ParticulesConfineesEngine()
    eng_form    = FormalismeEngine()
    eng_post    = PostulatsEngine()

    colors_q = ['#00ccff','#7700ff','#ff00cc','#00ff88',
                '#ffcc00','#ff4400','#88ccff','#cc88ff']

    # ============================================================
    # SECTION 1 : PUITS INFINI  (ORIGINAL — INCHANGÉ)
    # ============================================================
    if section == "📦 Puits quantique":
        st.markdown("### 📦 Puits de potentiel infini")
        col1, col2 = st.columns([1, 2])

        with col1:
            L_nm = st.slider("Largeur L (nm)", 0.1, 10.0, 1.0, 0.1)
            L = L_nm * 1e-9
            n_max = st.slider("États (n_max)", 1, 8, 4)
            show_prob = st.checkbox("Afficher |ψ|²", True)
            show_energie = st.checkbox("Niveaux d'énergie", True)

            x = np.linspace(0, L, 200)
            energies, psis = engine.puits_infini_multi(x, L, n_max)

            st.markdown("### 📐 Niveaux d'énergie")
            for n in range(1, n_max+1):
                E_J = energies[n-1]
                E_eV = E_J / 1.6022e-19
                st.metric(f"E_{n} (eV)", f"{E_eV:.4f}")

        with col2:
            fig_pw = go.Figure()

            for n in range(n_max):
                En_eV = energies[n] / 1.6022e-19
                y_plot = psis[n] if not show_prob else psis[n]**2
                scale = 0.3 * En_eV
                fig_pw.add_trace(go.Scatter(
                    x=x*1e9, y=y_plot*scale + En_eV,
                    mode='lines', name=f'n={n+1} ({En_eV:.3f}eV)',
                    line=dict(color=colors_q[n], width=2.5)
                ))
                if show_energie:
                    fig_pw.add_hline(y=En_eV,
                        line_color=f'rgba({int(colors_q[n][1:3],16)},'
                                   f'{int(colors_q[n][3:5],16)},'
                                   f'{int(colors_q[n][5:7],16)},0.3)',
                        line_dash='dot')

            fig_pw.add_vrect(x0=-0.5, x1=0, fillcolor='rgba(119,0,255,0.3)', line_width=0)
            fig_pw.add_vrect(x0=L*1e9, x1=L*1e9+0.5, fillcolor='rgba(119,0,255,0.3)', line_width=0)

            fig_pw.update_layout(
                title=f"Puits infini L={L_nm} nm — {n_max} états",
                xaxis_title="x (nm)",
                yaxis_title="|ψ|² + E (eV)" if show_prob else "ψ + E (eV)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=450,
                showlegend=False
            )
            st.plotly_chart(fig_pw, use_container_width=True, key='fig_pw')

            st.markdown("#### 🔢 Résolution numérique (FDM)")
            pot_type = st.selectbox("Potentiel", [
                "Puits infini", "Puits fini", "Double puits", "Harmonique"
            ])
            V0_num = st.slider("V₀ (eV)", 0.1, 10.0, 2.0, 0.1) * 1.6022e-19

            x_num = np.linspace(-L, 2*L, 120)
            if pot_type == "Puits infini":
                V_num = np.where((x_num >= 0) & (x_num <= L), 0, 1e10*1.6022e-19)
            elif pot_type == "Puits fini":
                V_num = np.where((x_num >= 0) & (x_num <= L), 0, V0_num)
            elif pot_type == "Double puits":
                w = L/4
                V_num = V0_num * np.ones_like(x_num)
                V_num[(x_num > 0) & (x_num < w)] = 0
                V_num[(x_num > 3*w) & (x_num < 4*w)] = 0
            else:
                V_num = 0.5 * engine.m * (1e14)**2 * x_num**2

            evals, evecs = engine.resoudre_schrodinger_1d(x_num, V_num, n_states=min(n_max,4))
            fig_fdm = go.Figure()
            V_eV = np.clip(V_num/1.6022e-19, 0, 15)
            fig_fdm.add_trace(go.Scatter(x=x_num*1e9, y=V_eV, mode='lines',
                name='V(x)', line=dict(color='rgba(255,255,255,0.5)', width=2)))
            for i, (E, psi) in enumerate(zip(evals, evecs)):
                E_eV_fdm = E/1.6022e-19
                if 0 < E_eV_fdm < 20:
                    y = np.real(psi)**2 * 2 + E_eV_fdm
                    fig_fdm.add_trace(go.Scatter(x=x_num*1e9, y=y, mode='lines',
                        name=f'E{i+1}={E_eV_fdm:.3f}eV',
                        line=dict(color=colors_q[i], width=2)))
            fig_fdm.update_layout(
                title=f"FDM — {pot_type}",
                xaxis_title="x (nm)", yaxis_title="E (eV)",
                yaxis=dict(range=[0, 12], color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
            )
            st.plotly_chart(fig_fdm, use_container_width=True, key='fig_fdm')

    # ============================================================
    # SECTION 2 : OSCILLATEUR HARMONIQUE  (ORIGINAL — INCHANGÉ)
    # ============================================================
    elif section == "〰️ Oscillateur harmonique":
        st.markdown("### 〰️ Oscillateur harmonique quantique")
        col1, col2 = st.columns([1, 2])

        with col1:
            omega_hz = st.slider("ω (×10¹³ rad/s)", 0.1, 10.0, 1.0, 0.1)
            omega = omega_hz * 1e13
            n_osc = st.slider("États n_max", 1, 7, 4)
            x_max_osc = st.slider("x_max (pm)", 10, 1000, 200)

            x_osc = np.linspace(-x_max_osc*1e-12, x_max_osc*1e-12, 300)
            V_osc = engine.osc_harm_potentiel(x_osc, omega)

            st.markdown("### 📐 Niveaux d'énergie")
            for n in range(n_osc):
                En = engine.osc_harm_energie(n, omega) / 1.6022e-19
                st.metric(f"E_{n} (eV)", f"{En:.4f}")

        with col2:
            fig_osc = go.Figure()
            V_eV_osc = V_osc / 1.6022e-19
            fig_osc.add_trace(go.Scatter(
                x=x_osc*1e12, y=np.clip(V_eV_osc, 0, 5), mode='lines',
                name='V(x)', line=dict(color='rgba(255,255,255,0.5)', width=2)
            ))
            for n in range(n_osc):
                psi_n = engine.osc_harm_psi(n, x_osc, omega)
                En_eV = engine.osc_harm_energie(n, omega) / 1.6022e-19
                y_plot = np.real(psi_n)**2 * 0.4 + En_eV
                fig_osc.add_trace(go.Scatter(
                    x=x_osc*1e12, y=y_plot, mode='lines',
                    name=f'n={n} (E={En_eV:.3f}eV)',
                    line=dict(color=colors_q[n % len(colors_q)], width=2.5)
                ))
                fig_osc.add_hline(y=En_eV,
                    line_color=f'rgba({int(colors_q[n%8][1:3],16)},'
                               f'{int(colors_q[n%8][3:5],16)},'
                               f'{int(colors_q[n%8][5:7],16)},0.3)',
                    line_dash='dot')
            fig_osc.update_layout(
                title="Oscillateur harmonique quantique",
                xaxis_title="x (pm)", yaxis_title="E (eV)",
                yaxis=dict(range=[0, 5]),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig_osc, use_container_width=True, key='fig_osc')

    # ============================================================
    # SECTION 3 : EFFET TUNNEL  (ORIGINAL — INCHANGÉ)
    # ============================================================
    elif section == "🚧 Effet tunnel":
        st.markdown("### 🚧 Effet tunnel quantique")
        col1, col2 = st.columns([1, 2])

        with col1:
            V0_eV = st.slider("Hauteur V₀ (eV)", 0.1, 20.0, 5.0, 0.1)
            L_bar = st.slider("Épaisseur L (nm)", 0.01, 5.0, 0.5, 0.01)
            V0 = V0_eV * 1.6022e-19
            L_b = L_bar * 1e-9

            E_arr = np.linspace(0.01, V0_eV * 1.5, 250) * 1.6022e-19
            T_arr = engine.transmittance_tunnel(E_arr, V0, L_b)

            st.metric("T(E=V₀/2)", f"{engine.transmittance_tunnel(np.array([V0/2]), V0, L_b)[0]:.4e}")
            st.metric("T(E=V₀)",   f"{engine.transmittance_tunnel(np.array([V0]),   V0, L_b)[0]:.4f}")

        with col2:
            fig_tun = go.Figure()
            E_eV_tun = E_arr / 1.6022e-19
            fig_tun.add_trace(go.Scatter(x=E_eV_tun, y=T_arr, mode='lines',
                name='T(E)', line=dict(color='#00ccff', width=3)))
            fig_tun.add_vline(x=V0_eV, line_color='#ffcc00', line_dash='dash',
                              annotation_text=f"V₀={V0_eV} eV")
            fig_tun.add_hline(y=1, line_color='rgba(255,255,255,0.3)', line_dash='dot')
            fig_tun.update_layout(
                title=f"Transmittance tunnel — V₀={V0_eV}eV, L={L_bar}nm",
                xaxis_title="E (eV)", yaxis_title="T(E)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)', type='log'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
            )
            st.plotly_chart(fig_tun, use_container_width=True, key='fig_tun')

            L_arr = np.linspace(0.01, 5, 150) * 1e-9
            E_fixed = V0 * 0.5
            T_L = engine.transmittance_tunnel(np.full(len(L_arr), E_fixed), V0, L_arr)
            fig_tL = go.Figure()
            fig_tL.add_trace(go.Scatter(x=L_arr*1e9, y=T_L, mode='lines',
                name='T(L) pour E=V₀/2', line=dict(color='#7700ff', width=2.5)))
            fig_tL.update_layout(
                title="Transmittance vs épaisseur (E=V₀/2)",
                xaxis_title="L (nm)", yaxis_title="T", yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=300,
            )
            st.plotly_chart(fig_tL, use_container_width=True, key='fig_tL')

    # ============================================================
    # SECTION 4 : ATOME D'HYDROGÈNE  (ORIGINAL — INCHANGÉ)
    # ============================================================
    elif section == "🔬 Atome d'hydrogène":
        st.markdown("### 🔬 Atome d'hydrogène")
        col1, col2 = st.columns([1, 2])

        with col1:
            n_H = st.slider("Nombre quantique n", 1, 5, 2)
            l_H = st.slider("Nombre quantique l", 0, n_H-1, min(1, n_H-1))

            E_eV_H = engine.H_energie_eV(n_H)
            r_moy  = engine.H_rayon_moyen(n_H, l_H)
            a0     = 5.2918e-11

            st.metric(f"E_{n_H} (eV)", f"{E_eV_H:.4f}")
            st.metric("r moyen (nm)",  f"{r_moy*1e9:.4f}")
            st.metric("r moyen (a₀)",  f"{r_moy/a0:.2f}")
            st.metric("λ_Lyman (nm)",
                      f"{1240/(13.6*(1-1/n_H**2)):.3f}" if n_H > 1 else "N/A")

            st.markdown("### 📊 Série de Rydberg")
            for n in range(1, 6):
                En = engine.H_energie_eV(n)
                delta = 13.6*(1-1/n**2) if n > 1 else 0
                st.markdown(f"- n={n}: **{En:.3f} eV** | Lyman: {1240/delta:.1f}nm" if n>1
                            else f"- n={n}: **{En:.3f} eV** (état fondamental)")

        with col2:
            r_max = 30 * a0
            r_arr = np.linspace(0.001*a0, r_max, 400)
            P_rad      = engine.H_densite_radiale(n_H, l_H, r_arr)
            P_rad_norm = P_rad / (trapezoid_integral(P_rad, r_arr) + 1e-30)

            fig_H = make_subplots(rows=2, cols=1,
                subplot_titles=[f"Densité radiale P(r) — n={n_H}, l={l_H}",
                                 "Niveaux d'énergie (eV)"])

            fig_H.add_trace(go.Scatter(
                x=r_arr/a0, y=P_rad_norm*a0, mode='lines',
                name=f'P_{n_H}{l_H}(r)',
                line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'
            ), row=1, col=1)
            fig_H.add_vline(x=r_moy/a0, line_color='#ffcc00', line_dash='dash',
                            annotation_text=f"<r>={r_moy/a0:.1f}a₀", row=1, col=1)

            for n in range(1, 6):
                En = engine.H_energie_eV(n)
                fig_H.add_trace(go.Scatter(
                    x=[0, 1], y=[En, En], mode='lines',
                    name=f'n={n} ({En:.2f}eV)',
                    line=dict(color=colors_q[n-1], width=2.5)
                ), row=2, col=1)

            fig_H.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=580,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_H.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_H.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_H.update_xaxes(title_text="r/a₀", row=1, col=1)
            fig_H.update_xaxes(title_text="",     row=2, col=1)
            fig_H.update_yaxes(title_text="P(r)·a₀", row=1, col=1)
            fig_H.update_yaxes(title_text="E (eV)",  row=2, col=1)
            st.plotly_chart(fig_H, use_container_width=True, key='fig_H')

    # ============================================================
    # SECTION 5 : ÉVOLUTION TEMPORELLE  (ORIGINAL — INCHANGÉ)
    # ============================================================
    elif section == "🕐 Évolution temporelle":
        st.markdown("### 🕐 Paquet d'onde & Évolution temporelle")
        col1, col2 = st.columns([1, 2])

        with col1:
            x0_pm    = st.slider("x₀ (pm)", -500, 0, -200)
            sigma_pm = st.slider("σ (pm)", 10, 200, 50)
            k0_inv   = st.slider("k₀ (nm⁻¹)", 1.0, 50.0, 10.0, 0.5)
            t_fs     = st.slider("Temps t (fs)", 0.0, 100.0, 0.0, 0.5)

            x0    = x0_pm * 1e-12
            sigma = sigma_pm * 1e-12
            k0    = k0_inv * 1e9
            t     = t_fs * 1e-15

            x_wp   = np.linspace(-1000e-12, 1000e-12, 600)
            psi_wp = engine.paquet_onde(x_wp, x0, sigma, k0, t)

            dx_H  = sigma
            dp_H  = engine.hbar / (2 * sigma)
            prod  = dx_H * dp_H

            st.metric("Δx (pm)",        f"{dx_H*1e12:.2f}")
            st.metric("Δp (kg·m/s)",    f"{dp_H:.3e}")
            st.metric("ΔxΔp / (ħ/2)",  f"{prod/(engine.hbar/2):.4f}")
            st.metric("E_cin (eV)",     f"{(engine.hbar*k0)**2/(2*engine.m*1.6022e-19):.4f}")

        with col2:
            prob   = np.abs(psi_wp)**2
            re_psi = np.real(psi_wp)
            im_psi = np.imag(psi_wp)

            fig_wp = make_subplots(rows=2, cols=1,
                subplot_titles=[f"Paquet d'onde ψ(x,t={t_fs}fs)",
                                 "Densité de probabilité |ψ|²"])

            fig_wp.add_trace(go.Scatter(x=x_wp*1e12, y=re_psi, mode='lines',
                name='Re(ψ)', line=dict(color='#00ccff', width=2)), row=1, col=1)
            fig_wp.add_trace(go.Scatter(x=x_wp*1e12, y=im_psi, mode='lines',
                name='Im(ψ)', line=dict(color='#7700ff', width=2, dash='dash')), row=1, col=1)
            fig_wp.add_trace(go.Scatter(x=x_wp*1e12, y=prob, mode='lines',
                name='|ψ|²', line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.15)'), row=2, col=1)

            x_centre = (x0 + engine.hbar*k0/engine.m * t)*1e12
            fig_wp.add_vline(x=x_centre, line_color='#ffcc00', line_dash='dash',
                             annotation_text=f"x_c={x_centre:.1f}pm", row=2, col=1)

            fig_wp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=520,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_wp.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff', title_text="x (pm)")
            fig_wp.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_wp, use_container_width=True, key='fig_wp')

    # ============================================================
    # SECTION 6 : THÉORIE  (ORIGINAL — INCHANGÉ)
    # ============================================================
    elif section == "📖 Théorie":
        st.markdown("### 📖 Formulaire Mécanique Quantique")
        formula_section("Formulaire quantique", FORMULES, default_open=False)

        with st.expander("🔬 Constantes quantiques", expanded=False):
            df_c = pd.DataFrame([{"Constante": k, "Valeur": v} for k, v in CONSTANTES.items()])
            st.dataframe(df_c, use_container_width=True)

        with st.expander("📚 Références", expanded=False):
            for r in ["Cohen-Tannoudji et al. — *Mécanique Quantique* (EDP Sciences, 2018)",
                      "Griffiths — *Introduction to Quantum Mechanics* (Cambridge, 2018)",
                      "Shankar — *Principles of Quantum Mechanics* (Springer, 2012)"]:
                st.markdown(f"- {r}")

    # ============================================================
    # SECTION 7 : PARTICULES CONFINÉES  (NOUVEAU)
    # ============================================================
    elif section == "🔒 Particules confinées":
        st.markdown("## 🔒 Section — Particules confinées")
        st.markdown("*Puits fini, double puits, courant de probabilité, dédoublement tunnel*")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Simulation puits fini",
            "🔗 Double puits",
            "📊 Analyse & Diagnostic",
            "📖 Théorie & Formules",
        ])

        # ---- Sous-onglet 1 : Puits fini ----
        with sub1:
            st.markdown("### Puits fini rectangulaire — États liés")
            col1, col2 = st.columns([1, 2])

            with col1:
                L_f_nm = st.slider("Largeur L (nm) [puits fini]", 0.1, 5.0, 1.0, 0.1,
                                   key="conf_L")
                V0_f_eV = st.slider("Profondeur V₀ (eV)", 0.5, 20.0, 5.0, 0.5,
                                    key="conf_V0")
                L_f  = L_f_nm * 1e-9
                V0_f = V0_f_eV * 1.6022e-19

                # Calcul des niveaux
                niveaux = eng_conf.puits_fini_niveaux(L_f, V0_f, n_max=8)
                n_lies  = len(niveaux)

                st.metric("Nb d'états liés", n_lies)
                for i, E in enumerate(niveaux):
                    E_eV_f = E / 1.6022e-19
                    parite = "pair" if i % 2 == 0 else "impair"
                    st.metric(f"E_{i+1} ({parite}) (eV)", f"{E_eV_f:.5f}")

                # Export CSV
                df_niv = pd.DataFrame({
                    "État n": range(1, n_lies+1),
                    "Énergie (J)": niveaux,
                    "Énergie (eV)": niveaux / 1.6022e-19,
                    "Parité": ["pair" if i%2==0 else "impair" for i in range(n_lies)],
                })
                st.download_button("💾 Export CSV niveaux",
                                   df_niv.to_csv(index=False).encode(),
                                   "puits_fini_niveaux.csv", "text/csv")

            with col2:
                x_f = np.linspace(-3*L_f, 3*L_f, 400)
                fig_pf = go.Figure()

                # Potentiel
                V_pf = np.where((x_f >= -L_f/2) & (x_f <= L_f/2), 0.0, V0_f_eV)
                fig_pf.add_trace(go.Scatter(
                    x=x_f*1e9, y=V_pf, mode='lines',
                    name='V(x) (eV)', line=dict(color='rgba(255,255,255,0.5)', width=2)
                ))

                # Fonctions d'onde
                for i, E in enumerate(niveaux[:6]):
                    parite = 'pair' if i % 2 == 0 else 'impair'
                    psi_f  = eng_conf.puits_fini_psi(x_f, E, L_f, V0_f, parite)
                    E_eV_f = E / 1.6022e-19
                    scale  = 0.4 * V0_f_eV / max(np.max(np.abs(psi_f)**2), 1e-10)
                    fig_pf.add_trace(go.Scatter(
                        x=x_f*1e9, y=psi_f**2 * scale + E_eV_f,
                        mode='lines', name=f'n={i+1} ({parite}) E={E_eV_f:.3f}eV',
                        line=dict(color=colors_q[i % 8], width=2.5)
                    ))
                    fig_pf.add_hline(y=E_eV_f,
                        line_color=colors_q[i%8], line_dash='dot',
                        annotation_text=f"E{i+1}", annotation_font_color=colors_q[i%8])

                fig_pf.update_layout(
                    title=f"Puits fini — L={L_f_nm}nm, V₀={V0_f_eV}eV — {n_lies} états liés",
                    xaxis_title="x (nm)", yaxis_title="E (eV) + |ψ|² (norm.)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)',
                               range=[-0.5, V0_f_eV + 1]),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
                )
                st.plotly_chart(fig_pf, use_container_width=True, key='fig_pf')

                # Courant de probabilité pour le 1er état
                if n_lies > 0:
                    psi_j   = eng_conf.puits_fini_psi(x_f, niveaux[0], L_f, V0_f, 'pair')
                    j_arr   = eng_conf.courant_probabilite(psi_j.astype(complex), x_f)
                    fig_j   = go.Figure()
                    fig_j.add_trace(go.Scatter(
                        x=x_f*1e9, y=j_arr, mode='lines',
                        name='j(x)', line=dict(color='#ff00cc', width=2)
                    ))
                    fig_j.update_layout(
                        title="Courant de probabilité j(x) — état fondamental",
                        xaxis_title="x (nm)", yaxis_title="j(x) (m⁻¹s⁻¹)",
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=280,
                    )
                    st.plotly_chart(fig_j, use_container_width=True, key='fig_j')

                # ---- Outil de diffusion et coefficients R/T ----
                with st.expander("🔍 Outil de diffusion barrière / coefficients R/T", expanded=False):
                    E_sc_eV = st.slider("Énergie incidente E (eV)", 0.1, 20.0, 5.0, 0.1, key="sc_E")
                    V0_sc_eV = st.slider("Hauteur de barrière V₀ (eV)", 0.1, 20.0, 8.0, 0.1, key="sc_V0")
                    a_sc_nm = st.slider("Largeur de barrière a (nm)", 0.1, 5.0, 1.0, 0.1, key="sc_a")
                    x_sc = np.linspace(-2*a_sc_nm*1e-9, 4*a_sc_nm*1e-9, 300)
                    result_sc = eng_conf.solutions_scattering(
                        E_sc_eV * 1.6022e-19,
                        V0_sc_eV * 1.6022e-19,
                        a_sc_nm * 1e-9,
                        x_sc
                    )

                    st.metric("R (coefficient de réflexion)", f"{result_sc['R']:.6f}")
                    st.metric("T (coefficient de transmission)", f"{result_sc['T']:.6f}")
                    st.metric("R+T", f"{result_sc['conservation']:.6f}")
                    st.markdown(f"- Amplitude de réflexion : `{result_sc['r_amp']:.3e}`")
                    st.markdown(f"- Amplitude de transmission : `{result_sc['t_amp']:.3e}`")

                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(
                        x=x_sc*1e9, y=result_sc['densite'], mode='lines',
                        name='|ψ|²', line=dict(color='#00ccff', width=2)
                    ))
                    fig_sc.add_trace(go.Scatter(
                        x=x_sc*1e9, y=result_sc['courant'], mode='lines',
                        name='j(x)', line=dict(color='#ff00cc', width=2)
                    ))
                    fig_sc.update_layout(
                        title=f"Diffusion sur barrière — E={E_sc_eV:.2f}eV, V₀={V0_sc_eV:.2f}eV, a={a_sc_nm:.2f}nm",
                        xaxis_title="x (nm)", yaxis_title="Amplitude / courant",
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                        font=dict(color='#c0d0ff'), height=380,
                        legend=dict(bgcolor='rgba(0,0,0,0.5)')
                    )
                    st.plotly_chart(fig_sc, use_container_width=True, key='fig_sc')

        # ---- Sous-onglet 2 : Double puits ----
        with sub2:
            st.markdown("### Double puits — dédoublement par effet tunnel")
            col1, col2 = st.columns([1, 2])

            with col1:
                L_dp_nm  = st.slider("Largeur de chaque puits (nm)", 0.2, 3.0, 0.8, 0.1, key="dp_L")
                Lb_dp_nm = st.slider("Épaisseur barrière (nm)", 0.1, 3.0, 0.5, 0.05, key="dp_Lb")
                V0_dp_eV = st.slider("Hauteur barrière V₀ (eV)", 0.5, 15.0, 5.0, 0.5, key="dp_V0")
                L_dp     = L_dp_nm * 1e-9
                Lb_dp    = Lb_dp_nm * 1e-9
                V0_dp    = V0_dp_eV * 1.6022e-19

                evals_dp, evecs_dp, x_dp, V_dp = eng_conf.double_puits_fdm(
                    L_dp, Lb_dp, V0_dp, n_states=6
                )
                n_valides = sum(1 for E in evals_dp if 0 < E < V0_dp)

                if len(evals_dp) >= 2:
                    dE_calc  = (evals_dp[1] - evals_dp[0]) / 1.6022e-19
                    E1_eV_dp = evals_dp[0] / 1.6022e-19
                    dE_approx = eng_conf.dedoublement_tunnel(Lb_dp, V0_dp, evals_dp[0]) / 1.6022e-19
                    st.metric("E₁ (eV)",           f"{E1_eV_dp:.5f}")
                    st.metric("ΔE = E₂−E₁ (eV)",   f"{dE_calc:.6f}")
                    st.metric("ΔE approx. tunnel",  f"{dE_approx:.6f}")
                    st.metric("États sous V₀",      n_valides)

                # Export
                df_dp = pd.DataFrame({
                    "État": range(1, len(evals_dp)+1),
                    "E (eV)": evals_dp / 1.6022e-19,
                })
                st.download_button("💾 Export CSV double puits",
                                   df_dp.to_csv(index=False).encode(),
                                   "double_puits.csv", "text/csv")

            with col2:
                fig_dp = go.Figure()
                V_dp_eV = np.clip(V_dp / 1.6022e-19, 0, V0_dp_eV + 1)
                fig_dp.add_trace(go.Scatter(
                    x=x_dp*1e9, y=V_dp_eV, mode='lines',
                    name='V(x) (eV)', line=dict(color='rgba(255,255,255,0.5)', width=2)
                ))

                for i, (E, psi) in enumerate(zip(evals_dp, evecs_dp)):
                    E_eV_dp = E / 1.6022e-19
                    if 0 < E_eV_dp < V0_dp_eV + 1:
                        prob_dp = np.real(psi)**2
                        scale   = 0.5 * V0_dp_eV / max(prob_dp.max(), 1e-10)
                        fig_dp.add_trace(go.Scatter(
                            x=x_dp*1e9, y=prob_dp*scale + E_eV_dp,
                            mode='lines', name=f'n={i+1} E={E_eV_dp:.4f}eV',
                            line=dict(color=colors_q[i % 8], width=2.5)
                        ))
                        fig_dp.add_hline(y=E_eV_dp, line_color=colors_q[i%8],
                                         line_dash='dot')

                fig_dp.update_layout(
                    title=f"Double puits — L={L_dp_nm}nm, L_b={Lb_dp_nm}nm, V₀={V0_dp_eV}eV",
                    xaxis_title="x (nm)", yaxis_title="E (eV)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
                )
                st.plotly_chart(fig_dp, use_container_width=True, key='fig_dp')

                # Dédoublement vs épaisseur barrière
                Lb_arr   = np.linspace(0.1, 3.0, 100) * 1e-9
                dE_arr   = np.array([eng_conf.dedoublement_tunnel(lb, V0_dp, V0_dp*0.3)
                                     for lb in Lb_arr]) / 1.6022e-19
                fig_dE   = go.Figure()
                fig_dE.add_trace(go.Scatter(
                    x=Lb_arr*1e9, y=np.abs(dE_arr), mode='lines',
                    name='ΔE(L_b)', line=dict(color='#00ff88', width=2.5)
                ))
                fig_dE.update_layout(
                    title="Dédoublement tunnel ΔE vs épaisseur barrière",
                    xaxis_title="L_b (nm)", yaxis_title="ΔE (eV)",
                    yaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300,
                )
                st.plotly_chart(fig_dE, use_container_width=True, key='fig_dE')

        # ---- Sous-onglet 3 : Diagnostic ----
        with sub3:
            st.markdown("### 📊 Diagnostic automatique du puits fini")
            L_diag_nm  = st.slider("L (nm)", 0.1, 5.0, 1.0, 0.1, key="diag_L")
            V0_diag_eV = st.slider("V₀ (eV)", 0.5, 20.0, 5.0, 0.5, key="diag_V0")
            L_diag     = L_diag_nm * 1e-9
            V0_diag    = V0_diag_eV * 1.6022e-19

            diag = eng_conf.diagnostic_puits(L_diag, V0_diag)
            st.dataframe(pd.DataFrame(diag), use_container_width=True)

            # Cartographie : nb d'états liés en fonction de L et V0
            st.markdown("#### Cartographie : nombre d'états liés n(L, V₀)")
            L_scan  = np.linspace(0.2, 4.0, 20) * 1e-9
            V0_scan = np.linspace(1.0, 15.0, 20) * 1.6022e-19
            Z_n     = np.zeros((len(V0_scan), len(L_scan)))
            for iv, v0 in enumerate(V0_scan):
                for il, ll in enumerate(L_scan):
                    try:
                        n_e = len(eng_conf.puits_fini_niveaux(ll, v0, n_max=10))
                    except Exception:
                        n_e = 0
                    Z_n[iv, il] = n_e

            fig_carte = go.Figure(go.Heatmap(
                x=L_scan*1e9, y=V0_scan/1.6022e-19, z=Z_n,
                colorscale=[[0,'#000014'],[0.25,'#00ccff'],
                            [0.5,'#7700ff'],[0.75,'#ff00cc'],[1,'#ffcc00']],
                colorbar=dict(title="n états", tickfont=dict(color='#c0d0ff')),
            ))
            fig_carte.update_layout(
                title="Nombre d'états liés n(L, V₀)",
                xaxis_title="L (nm)", yaxis_title="V₀ (eV)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'), yaxis=dict(color='#c0d0ff'),
                height=420,
            )
            st.plotly_chart(fig_carte, use_container_width=True, key='fig_carte')

        # ---- Sous-onglet 4 : Théorie ----
        with sub4:
            st.markdown("### 📖 Théorie — Particules confinées")
            st.markdown("""
**Puits de potentiel fini** : une particule de masse $m$ confinée dans un puits de profondeur $V_0$ 
et de largeur $L$ admet un nombre fini d'états liés. À la différence du puits infini, 
la fonction d'onde pénètre exponentiellement dans les régions classiquement interdites.
            """)
            formula_section("Formules du puits fini", FORMULES_PARTICULES_CONFINEES, default_open=False)

            with st.expander("📐 Tableau des grandeurs", expanded=False):
                df_gd = pd.DataFrame([
                    {"Grandeur": "k",     "Définition": "√(2mE)/ħ",          "Unité": "m⁻¹"},
                    {"Grandeur": "κ",     "Définition": "√(2m(V₀-E))/ħ",     "Unité": "m⁻¹"},
                    {"Grandeur": "l_p",   "Définition": "1/κ",                "Unité": "m"},
                    {"Grandeur": "j(x)",  "Définition": "courant probabilité", "Unité": "m⁻¹s⁻¹"},
                    {"Grandeur": "ΔE",    "Définition": "dédoublement tunnel", "Unité": "J"},
                ])
                st.dataframe(df_gd, use_container_width=True)

            with st.expander("📚 Références", expanded=False):
                st.markdown("**Références :** Cohen-Tannoudji (chap. I–II) · Griffiths (chap. 2)")

    # ============================================================
    # SECTION 8 : FORMALISME MATHÉMATIQUE  (NOUVEAU)
    # ============================================================
    elif section == "🧮 Formalisme mathématique":
        st.markdown("## 🧮 Section — Formalisme mathématique de la MQ")
        st.markdown("*Espace de Hilbert, notation de Dirac, opérateurs, commutateurs, densité*")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Opérateurs matriciels",
            "📊 Opérateur densité",
            "🔬 Représentation impulsion",
            "📖 Théorie & Formules",
        ])

        # ---- Sous-onglet 1 : Opérateurs ----
        with sub1:
            st.markdown("### Opérateurs x̂ et p̂ dans la base de Fock")
            col1, col2 = st.columns([1, 2])

            with col1:
                n_fock    = st.slider("Dimension de la base |n⟩", 3, 12, 6, key="fock_n")
                omega_fock_hz = st.slider("ω (×10¹³ rad/s)", 0.1, 5.0, 1.0, 0.1, key="fock_w")
                omega_fock = omega_fock_hz * 1e13

                X_mat = eng_form.operateur_x_matriciel(n_fock, omega_fock)
                P_mat = eng_form.operateur_p_matriciel(n_fock, omega_fock)
                comm  = eng_form.commutateur(X_mat, P_mat)

                # Vérification [x,p] = iħ
                comm_norm = np.max(np.abs(comm - 1j * eng_form.hbar * np.eye(n_fock)))

                diag_x = eng_form.verifier_hermitien(X_mat)
                diag_p = eng_form.verifier_hermitien(P_mat)

                st.metric("‖[x̂,p̂] − iħ‖∞", f"{comm_norm:.3e}", help="Doit être ≈ 0")
                st.metric("x̂ hermitien", "✅" if diag_x["Hermitien"] else "❌")
                st.metric("p̂ hermitien", "✅" if diag_p["Hermitien"] else "❌")

                # Export matrice x
                df_X = pd.DataFrame(np.real(X_mat) * 1e10,
                                    columns=[f"|{i}⟩" for i in range(n_fock)],
                                    index=[f"⟨{i}|" for i in range(n_fock)])
                st.download_button("💾 Export x̂ (×10¹⁰ m)",
                                   df_X.to_csv().encode(), "matrice_x.csv", "text/csv")

            with col2:
                # Heatmap matrice x̂
                fig_mat = make_subplots(rows=1, cols=2,
                    subplot_titles=["Re(x̂) (×10¹⁰ m)", "Re(p̂) (×10²⁴ kg·m/s)"])

                fig_mat.add_trace(go.Heatmap(
                    z=np.real(X_mat)*1e10,
                    colorscale=[[0,'#000014'],[0.5,'#7700ff'],[1,'#00ccff']],
                    showscale=True,
                    colorbar=dict(x=0.45, tickfont=dict(color='#c0d0ff')),
                ), row=1, col=1)
                fig_mat.add_trace(go.Heatmap(
                    z=np.real(P_mat)*1e24,
                    colorscale=[[0,'#000014'],[0.5,'#ff00cc'],[1,'#ffcc00']],
                    showscale=True,
                    colorbar=dict(x=1.0, tickfont=dict(color='#c0d0ff')),
                ), row=1, col=2)

                fig_mat.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=420,
                )
                st.plotly_chart(fig_mat, use_container_width=True, key='fig_mat')

                # Valeurs propres de x̂ vs n_fock
                dims   = range(3, 14)
                vp_max = []
                for d in dims:
                    Xd   = eng_form.operateur_x_matriciel(d, omega_fock)
                    evls = np.linalg.eigvalsh(np.real(Xd))
                    vp_max.append(float(evls.max()) * 1e12)

                fig_vp = go.Figure()
                fig_vp.add_trace(go.Scatter(
                    x=list(dims), y=vp_max, mode='lines+markers',
                    name='VP max x̂ (pm)', line=dict(color='#00ccff', width=2.5),
                    marker=dict(color='#00ccff', size=7)
                ))
                fig_vp.update_layout(
                    title="Valeur propre max de x̂ vs dimension de la base",
                    xaxis_title="Dimension n_max", yaxis_title="VP max (pm)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300,
                )
                st.plotly_chart(fig_vp, use_container_width=True, key='fig_vp')

        # ---- Sous-onglet 2 : Opérateur densité ----
        with sub2:
            st.markdown("### Opérateur densité ρ — état pur vs mélange")
            col1, col2 = st.columns([1, 2])

            with col1:
                n_dim_rho = st.slider("Dimension de l'espace", 2, 8, 4, key="rho_dim")
                type_etat = st.radio("Type d'état", ["État pur |n=0⟩", "Superposition équiprobable",
                                                      "Mélange statistique"], key="rho_type")

                base = np.eye(n_dim_rho, dtype=complex)
                if type_etat == "État pur |n=0⟩":
                    psi_rho = base[:, 0]
                    rho     = eng_form.operateur_densite_pur(psi_rho)
                elif type_etat == "Superposition équiprobable":
                    psi_rho = base.sum(axis=1) / np.sqrt(n_dim_rho)
                    rho     = eng_form.operateur_densite_pur(psi_rho)
                else:
                    rho = eng_form.operateur_densite_melange(
                        [base[:, i] for i in range(n_dim_rho)],
                        [1/n_dim_rho] * n_dim_rho
                    )

                purete_rho = eng_form.purete(rho)
                entropie   = eng_form.entropie_von_neumann(rho)
                trace_rho  = float(np.real(np.trace(rho)))

                st.metric("Tr(ρ)",      f"{trace_rho:.6f}")
                st.metric("Pureté γ",   f"{purete_rho:.6f}")
                st.metric("Entropie S", f"{entropie:.4f} nats")
                st.metric("État pur ?", "✅" if purete_rho > 0.999 else "❌ Mélange")

                df_rho = pd.DataFrame(np.real(rho).round(4))
                st.download_button("💾 Export matrice ρ",
                                   df_rho.to_csv().encode(), "rho.csv", "text/csv")

            with col2:
                fig_rho = make_subplots(rows=1, cols=2,
                    subplot_titles=["Re(ρ)", "Im(ρ)"])
                kw = dict(colorscale=[[0,'#000014'],[0.5,'#7700ff'],[1,'#00ccff']],
                          showscale=False)
                fig_rho.add_trace(go.Heatmap(z=np.real(rho), **kw), row=1, col=1)
                fig_rho.add_trace(go.Heatmap(z=np.imag(rho), **kw), row=1, col=2)
                fig_rho.update_layout(
                    title=f"Matrice densité — {type_etat}",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=400,
                )
                st.plotly_chart(fig_rho, use_container_width=True, key='fig_rho')

                # Pureté vs dimension pour mélange
                dims_p   = range(2, 12)
                puretes  = [1/d for d in dims_p]
                fig_pur  = go.Figure()
                fig_pur.add_trace(go.Scatter(
                    x=list(dims_p), y=puretes, mode='lines+markers',
                    name='γ = 1/d (mélange équiprobable)',
                    line=dict(color='#ff00cc', width=2.5),
                    marker=dict(color='#ff00cc', size=7)
                ))
                fig_pur.add_hline(y=1.0, line_color='#00ff88', line_dash='dash',
                                  annotation_text="État pur γ=1")
                fig_pur.update_layout(
                    title="Pureté γ vs dimension (mélange équiprobable)",
                    xaxis_title="Dimension d", yaxis_title="γ = Tr(ρ²)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300,
                )
                st.plotly_chart(fig_pur, use_container_width=True, key='fig_pur')

        # ---- Sous-onglet 3 : Représentation impulsion ----
        with sub3:
            st.markdown("### Transformée de Fourier quantique — représentation impulsion")
            col1, col2 = st.columns([1, 2])

            with col1:
                n_fp    = st.slider("État n du puits infini", 1, 6, 1, key="fp_n")
                L_fp_nm = st.slider("Largeur L (nm)", 0.5, 5.0, 1.0, 0.1, key="fp_L")
                L_fp    = L_fp_nm * 1e-9
                x_fp    = np.linspace(0, L_fp, 512)
                psi_fp  = np.sqrt(2/L_fp) * np.sin(n_fp * np.pi * x_fp / L_fp)

                p_arr, psi_p_arr = eng_form.transformer_base_impulsion(psi_fp, x_fp)

                # Incertitudes
                dx_fp = x_fp[1] - x_fp[0]
                x_moy = float(np.sum(psi_fp**2 * x_fp) * dx_fp)
                x2m   = float(np.sum(psi_fp**2 * x_fp**2) * dx_fp)
                dx_fp_val = np.sqrt(max(x2m - x_moy**2, 0))

                dp_fp_arr = p_arr[1] - p_arr[0]
                p_moy = float(np.real(np.sum(np.abs(psi_p_arr)**2 * p_arr)) * dp_fp_arr)
                p2m   = float(np.real(np.sum(np.abs(psi_p_arr)**2 * p_arr**2)) * dp_fp_arr)
                dp_fp_val = np.sqrt(max(p2m - p_moy**2, 0))

                prod_fp = dx_fp_val * dp_fp_val
                hbar_fp = 1.0546e-34

                st.metric("Δx (nm)",        f"{dx_fp_val*1e9:.4f}")
                st.metric("Δp (kg·m/s)",    f"{dp_fp_val:.3e}")
                st.metric("ΔxΔp / (ħ/2)",  f"{prod_fp/(hbar_fp/2):.4f}")

            with col2:
                fig_fp = make_subplots(rows=2, cols=1,
                    subplot_titles=[f"ψ(x) — état n={n_fp} dans le puits infini",
                                     "ψ̃(p) — représentation impulsion"])

                fig_fp.add_trace(go.Scatter(
                    x=x_fp*1e9, y=psi_fp, mode='lines',
                    name='ψ(x)', line=dict(color='#00ccff', width=2.5)
                ), row=1, col=1)

                mask_p = (p_arr > -5e-24) & (p_arr < 5e-24)
                fig_fp.add_trace(go.Scatter(
                    x=p_arr[mask_p]*1e24, y=np.abs(psi_p_arr[mask_p])**2,
                    mode='lines', name='|ψ̃(p)|²',
                    line=dict(color='#7700ff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(119,0,255,0.15)'
                ), row=2, col=1)

                fig_fp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_fp.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_fp.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_fp.update_xaxes(title_text="x (nm)",          row=1, col=1)
                fig_fp.update_xaxes(title_text="p (×10⁻²⁴ kg·m/s)", row=2, col=1)
                fig_fp.update_yaxes(title_text="ψ(x)",            row=1, col=1)
                fig_fp.update_yaxes(title_text="|ψ̃(p)|²",         row=2, col=1)
                st.plotly_chart(fig_fp, use_container_width=True, key='fig_fp')

        # ---- Sous-onglet 4 : Théorie ----
        with sub4:
            st.markdown("### 📖 Théorie — Formalisme mathématique")
            st.markdown(r"""
Le formalisme de **Dirac** unifie les représentations position et impulsion dans un 
espace de Hilbert abstrait $\mathcal{H}$. Tout état est un vecteur $|\psi\rangle$, 
tout observable une application linéaire hermitienne $\hat{A}$.
            """)
            formula_section("Formules du formalisme", FORMULES_FORMALISME, default_open=False)

            with st.expander("� Outil Dirac & valeurs propres", expanded=True):
                vecteurs = {
                    "|0⟩ = [1,0]": np.array([1, 0], dtype=complex),
                    "|1⟩ = [0,1]": np.array([0, 1], dtype=complex),
                    "|+⟩ = [1,1]": np.array([1, 1], dtype=complex),
                    "|-⟩ = [1,-1]": np.array([1, -1], dtype=complex),
                }
                ket1_key = st.selectbox("Sélectionner |ψ⟩", list(vecteurs.keys()), index=2)
                ket2_key = st.selectbox("Sélectionner |φ⟩", list(vecteurs.keys()), index=3)
                ket1 = vecteurs[ket1_key]
                ket2 = vecteurs[ket2_key]

                st.markdown(f"- **|ψ⟩** = {ket1.tolist()}  ")
                st.markdown(f"- **⟨ψ|** = {np.array2string(eng_form.bra(ket1), precision=3)}  ")
                st.markdown(f"- **|φ⟩** = {ket2.tolist()}  ")
                st.markdown(f"- **⟨φ|** = {np.array2string(eng_form.bra(ket2), precision=3)}  ")
                st.markdown(f"- **⟨ψ|φ⟩** = `{eng_form.produit_scalaire(ket1, ket2):.6f}`")
                st.markdown(f"- **|ψ⟩⟨φ|** = `{np.array2string(eng_form.outer_product(ket1, ket2), precision=3)}`")

                st.markdown("#### Outil opérateur personnalisé")
                op_dim = st.number_input("Dimension de l'opérateur", min_value=2, max_value=6, value=2, step=1, key="form_custom_dim")
                colsA = st.columns(op_dim)
                A_entries = []
                for i in range(op_dim):
                    with colsA[i]:
                        A_entries.append([
                            st.number_input(f"Re(A[{i+1},{j+1}])", value=1.0 if i == j else 0.0, key=f"form_A_{i}_{j}_re") +
                            1j * st.number_input(f"Im(A[{i+1},{j+1}])", value=0.0, key=f"form_A_{i}_{j}_im")
                            for j in range(op_dim)
                        ])
                A_custom = np.array(A_entries, dtype=complex)

                colsB = st.columns(op_dim)
                B_entries = []
                for i in range(op_dim):
                    with colsB[i]:
                        B_entries.append([
                            st.number_input(f"Re(B[{i+1},{j+1}])", value=0.0, key=f"form_B_{i}_{j}_re") +
                            1j * st.number_input(f"Im(B[{i+1},{j+1}])", value=0.0, key=f"form_B_{i}_{j}_im")
                            for j in range(op_dim)
                        ])
                B_custom = np.array(B_entries, dtype=complex)

                st.markdown("**Opérateur A personnalisé**")
                st.write(np.round(A_custom, 4))
                st.markdown("**Opérateur B personnalisé**")
                st.write(np.round(B_custom, 4))

                evs, evecs, projecteurs = eng_form.spectral_decomposition(A_custom)
                comm_AB = eng_form.commutateur(A_custom, B_custom)
                st.markdown(f"**Valeurs propres de A personnalisé :** {np.round(evs, 4).tolist()}")
                st.markdown("**Vecteurs propres (colonnes)** :")
                st.dataframe(np.round(np.real(evecs), 4), use_container_width=True)
                st.markdown("**Projecteurs de A** :")
                for i, P in enumerate(projecteurs):
                    st.markdown(f"- Projecteur {i+1}")
                    st.write(np.round(P, 4))
                st.markdown("**Commutateur [A,B]** :")
                st.write(np.round(comm_AB, 4))

                opt_op = st.selectbox("Opérateur pour décomposition", ["x̂", "p̂", "σ_x", "σ_z"], key="form_op")
                if opt_op in ["x̂", "p̂"]:
                    n_fock_disp = st.slider("Dimension pour décomposition", 3, 10, 5, key="form_dim")
                    omega_disp   = st.slider("ω (×10¹³ rad/s)", 0.1, 5.0, 1.0, 0.1, key="form_w") * 1e13
                    A_disp = eng_form.operateur_x_matriciel(n_fock_disp, omega_disp) if opt_op == "x̂" else eng_form.operateur_p_matriciel(n_fock_disp, omega_disp)
                else:
                    if opt_op == "σ_x":
                        A_disp = np.array([[0, 1], [1, 0]], dtype=complex)
                    else:
                        A_disp = np.array([[1, 0], [0, -1]], dtype=complex)

                evs, evecs, projecteurs = eng_form.spectral_decomposition(A_disp)
                st.markdown(f"**Valeurs propres de {opt_op} :** {np.round(evs, 4).tolist()}")
                st.markdown("**Vecteurs propres (colonnes)** :")
                st.dataframe(np.round(np.real(evecs), 4), use_container_width=True)
                st.markdown("**Projecteurs** :")
                for i, P in enumerate(projecteurs):
                    st.markdown(f"- Projecteur {i+1}")
                    st.write(np.round(P, 4))

            with st.expander("�📐 Opérateurs et représentations", expanded=False):
                df_ops = pd.DataFrame([
                    {"Opérateur": "x̂",    "Repr. position": "x",           "Repr. Fock": "√(ħ/2mω)(â+â†)"},
                    {"Opérateur": "p̂",    "Repr. position": "−iħ ∂/∂x",   "Repr. Fock": "i√(mħω/2)(â†−â)"},
                    {"Opérateur": "Ĥ",    "Repr. position": "−ħ²/2m ∂²+V", "Repr. Fock": "ħω(â†â + ½)"},
                    {"Opérateur": "â",    "Repr. Fock": "√n |n-1⟩",        "Repr. position": "√(mω/2ħ)x+ip/√(2mħω)"},
                    {"Opérateur": "â†",   "Repr. Fock": "√(n+1)|n+1⟩",    "Repr. position": "hermitien conjugué de â"},
                ])
                st.dataframe(df_ops, use_container_width=True)

            with st.expander("📚 Références", expanded=False):
                st.markdown("**Références :** Shankar (chap. 1) · Cohen-Tannoudji (chap. II)")

    # ============================================================
    # SECTION 9 : POSTULATS  (NOUVEAU)
    # ============================================================
    elif section == "📐 Postulats":
        st.markdown("## 📐 Section — Postulats de la mécanique quantique")
        st.markdown("*Superposition, mesure, réduction, évolution temporelle, Ehrenfest*")

        sub1, sub2, sub3, sub4 = st.tabs([
            "⚙️ Superposition & mesure",
            "📊 Évolution & Ehrenfest",
            "🔬 Réduction du paquet",
            "📖 Théorie & Formules",
        ])

        # ---- Sous-onglet 1 : Superposition ----
        with sub1:
            st.markdown("### Postulat 3 — Probabilités de mesure")
            col1, col2 = st.columns([1, 2])

            with col1:
                L_post_nm = st.slider("Largeur L (nm)", 0.5, 3.0, 1.0, 0.1, key="post_L")
                L_post    = L_post_nm * 1e-9
                st.markdown("**Coefficients cₙ de la superposition** (parties réelles)")

                n_super = st.slider("Nombre d'états", 2, 6, 3, key="post_n")
                coeffs  = []
                for n in range(1, n_super + 1):
                    c_def = 1.0 if n == 1 else (0.5 if n == 2 else 0.0)
                    c = st.slider(f"c_{n}", -2.0, 2.0, c_def, 0.1, key=f"post_c{n}")
                    coeffs.append(complex(c))

                proba    = eng_post.probabilites_etats(coeffs)
                hbar_p   = 1.0546e-34
                m_p      = 9.1094e-31
                eV_p     = 1.6022e-19
                energies_post = np.array([
                    n**2 * np.pi**2 * hbar_p**2 / (2 * m_p * L_post**2)
                    for n in range(1, n_super + 1)
                ])
                E_moy_post = eng_post.valeur_moyenne_energie(coeffs, energies_post)

                for n in range(n_super):
                    st.metric(f"|c_{n+1}|² = P(E_{n+1})", f"{proba[n]:.4f}")
                st.metric("⟨E⟩ (eV)", f"{E_moy_post/eV_p:.5f}")

                with st.expander("🔧 Outil de mesure personnalisée", expanded=False):
                    st.markdown("### Observable personnalisée et statistiques de mesure")
                    dim_state = st.number_input("Dimension de l'état quantique", min_value=2, max_value=6, value=2, step=1, key="post_state_dim")
                    st.markdown("**État quantique |ψ⟩ (base dimensionnelle choisie)**")
                    psi_entries = []
                    for i in range(dim_state):
                        psi_re = st.number_input(f"Re(ψ_{i+1})", value=1.0 if i == 0 else 0.0, step=0.1, key=f"post_psi_{i}_re")
                        psi_im = st.number_input(f"Im(ψ_{i+1})", value=0.0, step=0.1, key=f"post_psi_{i}_im")
                        psi_entries.append(psi_re + 1j * psi_im)
                    psi_state = np.array(psi_entries, dtype=complex)
                    norm_psi = np.linalg.norm(psi_state)
                    psi_state = psi_state / max(norm_psi, 1e-12)
                    st.markdown(f"- Normalisé |ψ⟩ = {np.array2string(np.round(psi_state, 4))}")

                    st.markdown("**Observable Hermitienne A personnalisée**")
                    A_obs_entries = []
                    for i in range(dim_state):
                        row = []
                        for j in range(dim_state):
                            if j < i:
                                # symétrie hermitienne partagée
                                row.append(None)
                            else:
                                row.append(
                                    st.number_input(f"Re(A[{i+1},{j+1}])", value=1.0 if i == j else 0.0, step=0.1, key=f"post_A_{i}_{j}_re") +
                                    1j * st.number_input(f"Im(A[{i+1},{j+1}])", value=0.0, step=0.1, key=f"post_A_{i}_{j}_im")
                                )
                        A_obs_entries.append(row)
                    A_obs = np.zeros((dim_state, dim_state), dtype=complex)
                    for i in range(dim_state):
                        for j in range(dim_state):
                            if j < i:
                                A_obs[i, j] = np.conj(A_obs[j, i])
                            else:
                                A_obs[i, j] = A_obs_entries[i][j]

                    evs_obs, evecs_obs, proj_obs = eng_form.spectral_decomposition(A_obs)
                    proba_obs = np.array([abs(np.vdot(evecs_obs[:, i], psi_state))**2 for i in range(evecs_obs.shape[1])])
                    proba_obs /= max(proba_obs.sum(), 1e-12)
                    expA = eng_form.valeur_moyenne_op(psi_state, A_obs)
                    sigmaA = eng_form.incertitude_op(psi_state, A_obs)

                    st.markdown(f"- **⟨A⟩** = `{expA:.6f}`")
                    st.markdown(f"- **ΔA** = `{sigmaA:.6f}`")
                    st.markdown("- **Probabilités de mesure P(aₙ)** :")
                    st.write(pd.DataFrame({
                        "Valeur propre": np.round(np.real(evs_obs), 4),
                        "P(aₙ)": np.round(proba_obs, 4)
                    }))
                    st.markdown("- **Projecteurs** :")
                    for i, P in enumerate(proj_obs):
                        st.markdown(f"  - P_{i+1} =")
                        st.write(np.round(P, 4))

            with col2:
                x_post = np.linspace(0, L_post, 250)
                psi_s  = eng_post.superposition_puits(coeffs, x_post, L_post)

                fig_super = make_subplots(rows=2, cols=1,
                    subplot_titles=["ψ(x) — état superposé",
                                     "Probabilités |cₙ|² par état"])

                fig_super.add_trace(go.Scatter(
                    x=x_post*1e9, y=np.real(psi_s), mode='lines',
                    name='Re(ψ)', line=dict(color='#00ccff', width=2.5)
                ), row=1, col=1)
                fig_super.add_trace(go.Scatter(
                    x=x_post*1e9, y=np.abs(psi_s)**2, mode='lines',
                    name='|ψ|²', line=dict(color='#ff00cc', width=2),
                    fill='tozeroy', fillcolor='rgba(255,0,204,0.12)'
                ), row=1, col=1)

                fig_super.add_trace(go.Bar(
                    x=[f'n={n+1}' for n in range(n_super)],
                    y=proba,
                    marker_color=colors_q[:n_super],
                    name='P(aₙ)',
                ), row=2, col=1)

                fig_super.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=540,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_super.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_super.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_super.update_xaxes(title_text="x (nm)", row=1, col=1)
                fig_super.update_xaxes(title_text="État", row=2, col=1)
                fig_super.update_yaxes(title_text="ψ / |ψ|²", row=1, col=1)
                fig_super.update_yaxes(title_text="|cₙ|²",    row=2, col=1)
                st.plotly_chart(fig_super, use_container_width=True, key='fig_super')

                # Diagnostic
                st.markdown("#### Diagnostic automatique (Postulat 3)")
                diag_p = eng_post.diagnostic_postulats(coeffs, energies_post)
                st.dataframe(pd.DataFrame(diag_p), use_container_width=True)

                # Export
                df_super = pd.DataFrame({
                    "x (nm)": x_post*1e9,
                    "Re(ψ)":  np.real(psi_s),
                    "Im(ψ)":  np.imag(psi_s),
                    "|ψ|²":   np.abs(psi_s)**2,
                })
                st.download_button("💾 Export CSV superposition",
                                   df_super.to_csv(index=False).encode(),
                                   "superposition.csv", "text/csv")

        # ---- Sous-onglet 2 : Évolution / Ehrenfest ----
        with sub2:
            st.markdown("### Postulat 5 — Évolution temporelle & théorème d'Ehrenfest")
            col1, col2 = st.columns([1, 2])

            with col1:
                L_ehr_nm  = st.slider("L (nm)", 0.5, 3.0, 1.0, 0.1, key="ehr_L")
                L_ehr     = L_ehr_nm * 1e-9
                t_max_ehr = st.slider("t_max (fs)", 1.0, 200.0, 50.0, 1.0, key="ehr_t")
                n_t_ehr   = st.slider("Nb points temporels", 30, 120, 60, key="ehr_nt")

                st.markdown("**Coefficients (état initial)**")
                c1_e = st.slider("c₁", 0.0, 2.0, 1.0, 0.1, key="ehr_c1")
                c2_e = st.slider("c₂", 0.0, 2.0, 1.0, 0.1, key="ehr_c2")
                c3_e = st.slider("c₃", 0.0, 2.0, 0.0, 0.1, key="ehr_c3")
                coeffs_ehr = [complex(c1_e), complex(c2_e), complex(c3_e)]

                hbar_e = 1.0546e-34
                m_e2   = 9.1094e-31
                eV_e   = 1.6022e-19

                t_arr_ehr = np.linspace(0, t_max_ehr * 1e-15, n_t_ehr)
                x_ehr     = np.linspace(0, L_ehr, 120)

                energies_ehr = np.array([
                    n**2 * np.pi**2 * hbar_e**2 / (2 * m_e2 * L_ehr**2)
                    for n in range(1, 4)
                ])
                E_moy_ehr = eng_post.valeur_moyenne_energie(coeffs_ehr, energies_ehr)
                st.metric("⟨E⟩ (eV)", f"{E_moy_ehr/eV_e:.5f}")

            with col2:
                psi_t_ehr = eng_post.evolution_superposition(
                    coeffs_ehr, x_ehr, L_ehr, t_arr_ehr
                )
                x_moy_t = eng_post.ehrenfest_position(psi_t_ehr, x_ehr)
                dx_t    = eng_post.incertitude_position(psi_t_ehr, x_ehr)

                fig_ehr = make_subplots(rows=2, cols=1,
                    subplot_titles=["⟨x⟩(t) — théorème d'Ehrenfest",
                                     "Δx(t) — incertitude de position"])

                fig_ehr.add_trace(go.Scatter(
                    x=t_arr_ehr*1e15, y=x_moy_t*1e9, mode='lines',
                    name='⟨x⟩(t) (nm)', line=dict(color='#00ccff', width=2.5)
                ), row=1, col=1)
                fig_ehr.add_hline(y=L_ehr*1e9/2, line_color='#ffcc00', line_dash='dash',
                                  annotation_text="L/2", row=1, col=1)

                fig_ehr.add_trace(go.Scatter(
                    x=t_arr_ehr*1e15, y=dx_t*1e9, mode='lines',
                    name='Δx(t) (nm)', line=dict(color='#ff00cc', width=2.5)
                ), row=2, col=1)

                fig_ehr.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_ehr.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_ehr.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_ehr.update_xaxes(title_text="t (fs)", row=1, col=1)
                fig_ehr.update_xaxes(title_text="t (fs)", row=2, col=1)
                fig_ehr.update_yaxes(title_text="⟨x⟩ (nm)", row=1, col=1)
                fig_ehr.update_yaxes(title_text="Δx (nm)",  row=2, col=1)
                st.plotly_chart(fig_ehr, use_container_width=True, key='fig_ehr')

                # Densité de probabilité animée (instantanée)
                t_show_idx = st.slider("Instant affiché (fs)", 0, n_t_ehr-1, 0, key="ehr_show")
                prob_show  = np.abs(psi_t_ehr[t_show_idx])**2
                t_val      = t_arr_ehr[t_show_idx] * 1e15

                fig_snap = go.Figure()
                fig_snap.add_trace(go.Scatter(
                    x=x_ehr*1e9, y=prob_show, mode='lines',
                    name=f'|ψ(x,t={t_val:.1f}fs)|²',
                    line=dict(color='#00ff88', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,255,136,0.12)'
                ))
                fig_snap.add_vline(x=x_moy_t[t_show_idx]*1e9, line_color='#ffcc00',
                                   line_dash='dash',
                                   annotation_text=f"⟨x⟩={x_moy_t[t_show_idx]*1e9:.3f}nm")
                fig_snap.update_layout(
                    title=f"|ψ(x,t)|² à t={t_val:.1f} fs",
                    xaxis_title="x (nm)", yaxis_title="|ψ|²",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320,
                )
                st.plotly_chart(fig_snap, use_container_width=True, key='fig_snap')

                # Export
                df_ehr = pd.DataFrame({
                    "t (fs)":   t_arr_ehr*1e15,
                    "⟨x⟩ (nm)": x_moy_t*1e9,
                    "Δx (nm)":  dx_t*1e9,
                })
                st.download_button("💾 Export CSV Ehrenfest",
                                   df_ehr.to_csv(index=False).encode(),
                                   "ehrenfest.csv", "text/csv")

        # ---- Sous-onglet 3 : Réduction ----
        with sub3:
            st.markdown("### Postulat 4 — Réduction du paquet d'onde")
            col1, col2 = st.columns([1, 2])

            with col1:
                L_red_nm  = st.slider("L (nm)", 0.5, 3.0, 1.0, 0.1, key="red_L")
                L_red     = L_red_nm * 1e-9
                x_mes_nm  = st.slider("Position mesurée x_m (nm)", 0.05, L_red_nm-0.05,
                                      L_red_nm/2, 0.01, key="red_xm")
                sig_mes   = st.slider("Résolution σ (nm)", 0.01, 0.5, 0.1, 0.01, key="red_sig")

                x_red  = np.linspace(0, L_red, 250)
                psi_av = (np.sqrt(2/L_red) * np.sin(np.pi * x_red / L_red) +
                          np.sqrt(2/L_red) * np.sin(2 * np.pi * x_red / L_red)) / np.sqrt(2)
                norme_av = trapezoid_integral(np.abs(psi_av)**2, x_red)
                psi_av   = psi_av / np.sqrt(max(norme_av, 1e-30))

                psi_ap   = eng_post.reduire_etat(psi_av.astype(complex), x_red,
                                                  x_mes_nm*1e-9, sig_mes*1e-9)

                prob_av  = np.abs(psi_av)**2
                prob_ap  = np.abs(psi_ap)**2

                dx_red   = x_red[1] - x_red[0]
                x_moy_av = float(np.sum(prob_av * x_red) * dx_red) * 1e9
                x_moy_ap = float(np.sum(prob_ap * x_red) * dx_red) * 1e9
                dx_av    = np.sqrt(max(float(np.sum(prob_av * x_red**2)*dx_red) - (x_moy_av*1e-9)**2, 0)) * 1e9
                dx_ap    = np.sqrt(max(float(np.sum(prob_ap * x_red**2)*dx_red) - (x_moy_ap*1e-9)**2, 0)) * 1e9

                st.metric("⟨x⟩ avant mesure (nm)", f"{x_moy_av:.4f}")
                st.metric("⟨x⟩ après mesure (nm)", f"{x_moy_ap:.4f}")
                st.metric("Δx avant (nm)",          f"{dx_av:.4f}")
                st.metric("Δx après (nm)",          f"{dx_ap:.4f}")
                st.metric("Réduction Δx",           f"÷{dx_av/max(dx_ap,1e-10):.1f}")

            with col2:
                fig_red = go.Figure()
                fig_red.add_trace(go.Scatter(
                    x=x_red*1e9, y=prob_av, mode='lines',
                    name='|ψ|² avant mesure',
                    line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.10)'
                ))
                fig_red.add_trace(go.Scatter(
                    x=x_red*1e9, y=prob_ap, mode='lines',
                    name='|ψ|² après mesure',
                    line=dict(color='#ff00cc', width=2.5),
                    fill='tozeroy', fillcolor='rgba(255,0,204,0.12)'
                ))
                fig_red.add_vline(x=x_mes_nm, line_color='#ffcc00', line_dash='dash',
                                  annotation_text=f"x_m={x_mes_nm:.2f}nm")
                fig_red.update_layout(
                    title="Réduction du paquet d'onde par mesure de position",
                    xaxis_title="x (nm)", yaxis_title="|ψ(x)|²",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
                )
                st.plotly_chart(fig_red, use_container_width=True, key='fig_red_postulats')

                # Δx après vs σ mesure
                sigs  = np.linspace(0.01, 0.5, 40) * 1e-9
                dx_ap_arr = []
                for s in sigs:
                    pa = eng_post.reduire_etat(psi_av.astype(complex), x_red,
                                               x_mes_nm*1e-9, s)
                    proba = np.abs(pa)**2
                    xm    = float(np.sum(proba * x_red) * dx_red)
                    x2m   = float(np.sum(proba * x_red**2) * dx_red)
                    dx_ap_arr.append(np.sqrt(max(x2m - xm**2, 0)) * 1e9)

                fig_dx = go.Figure()
                fig_dx.add_trace(go.Scatter(
                    x=sigs*1e9, y=dx_ap_arr, mode='lines',
                    name='Δx après (nm)', line=dict(color='#00ff88', width=2.5)
                ))
                fig_dx.update_layout(
                    title="Incertitude résiduelle Δx vs résolution σ de mesure",
                    xaxis_title="σ mesure (nm)", yaxis_title="Δx après (nm)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300,
                )
                st.plotly_chart(fig_dx, use_container_width=True, key='fig_dx_postulats')

        # ---- Sous-onglet 4 : Théorie ----
        with sub4:
            st.markdown("### 📖 Théorie — Les 5 postulats de la MQ")
            st.markdown("""
La mécanique quantique repose sur **5 postulats fondamentaux** qui définissent
l'espace des états, les observables, les résultats de mesure, la réduction
du paquet d'onde et la loi d'évolution temporelle.
            """)
            formula_section("Formules des postulats", FORMULES_POSTULATS, default_open=False)

            with st.expander("📘 Résumé des postulats", expanded=False):
                df_post_tab = pd.DataFrame([
                    {"Postulat": "1", "Énoncé": "L'état d'un système est décrit par |ψ⟩∈ℋ",
                     "Conséquence": "Superposition, interférence"},
                    {"Postulat": "2", "Énoncé": "Toute observable ↔ opérateur hermitien Â",
                     "Conséquence": "Valeurs propres réelles"},
                    {"Postulat": "3", "Énoncé": "P(aₙ) = |⟨aₙ|ψ⟩|²",
                     "Conséquence": "Caractère probabiliste"},
                    {"Postulat": "4", "Énoncé": "Après mesure aₙ : |ψ⟩ → |aₙ⟩",
                     "Conséquence": "Réduction, non-réversibilité"},
                    {"Postulat": "5", "Énoncé": "iħ d|ψ⟩/dt = Ĥ|ψ⟩",
                     "Conséquence": "Évolution unitaire, conservation énergie"},
                ])
                st.dataframe(df_post_tab, use_container_width=True)

                with st.expander("🔧 Outil spectral & évolution temporelle", expanded=True):
                    H_choice = st.selectbox("Choisir un Hamiltonien", ["σ_x", "σ_z", "Hamiltonien 2×2 personnalisé"], key="post_h")
                    if H_choice == "Hamiltonien 2×2 personnalisé":
                        h00 = st.number_input("H₀₀", value=1.0, step=0.1, key="h00")
                        h01 = st.number_input("H₀₁", value=0.5, step=0.1, key="h01")
                        h10 = st.number_input("H₁₀", value=0.5, step=0.1, key="h10")
                        h11 = st.number_input("H₁₁", value=-1.0, step=0.1, key="h11")
                        H_mat = np.array([[h00, h01], [h10, h11]], dtype=complex)
                    elif H_choice == "σ_x":
                        H_mat = np.array([[0, 1], [1, 0]], dtype=complex)
                    else:
                        H_mat = np.array([[1, 0], [0, -1]], dtype=complex)

                    spectral = eng_post.decomposition_spectrale(H_mat)
                    st.markdown(f"**Valeurs propres :** {np.round(spectral['valeurs_propres'], 4).tolist()}")
                    st.markdown("**Vecteurs propres** :")
                    st.dataframe(np.round(np.real(spectral['vecteurs_propres']), 4), use_container_width=True)

                    psi0_choice = st.selectbox("État initial |ψ₀⟩", ["[1,0]", "[0,1]", "[1,1]"], key="psi0_choice")
                    psi0 = np.array([1, 0], dtype=complex) if psi0_choice == "[1,0]" else np.array([0, 1], dtype=complex)
                    if psi0_choice == "[1,1]":
                        psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)

                    t_final = st.slider("Temps final t (arb.)", 0.1, 10.0, 3.0, 0.1, key="t_final")
                    t_points = np.linspace(0, t_final, 60)
                    psi_t = eng_post.evolution_temps_general(psi0, H_mat, t_points)
                    probs = np.abs(psi_t)**2
                    
                    # Vérifier que les données sont valides avant le tracé
                    if probs.shape[1] >= 2 and np.all(np.isfinite(probs)):
                        fig_t = go.Figure()
                        fig_t.add_trace(go.Scatter(
                            x=t_points, y=np.clip(probs[:, 0], 0, 1), mode='lines',
                            name='P(|0⟩)', line=dict(color='#00ccff', width=2.5)
                        ))
                        fig_t.add_trace(go.Scatter(
                            x=t_points, y=np.clip(probs[:, 1], 0, 1), mode='lines',
                            name='P(|1⟩)', line=dict(color='#ff00cc', width=2.5)
                        ))
                        fig_t.update_layout(
                            title="Évolution temporelle d'un état quantique",
                            xaxis_title='t (arb.)', yaxis_title='Probabilité',
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                            font=dict(color='#c0d0ff'), legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=360
                        )
                        st.plotly_chart(fig_t, use_container_width=True, key='fig_evolution_postulats')
                    else:
                        st.warning("Les données de probabilités ne sont pas valides.")

                    with st.expander("Mesure et réduction du paquet d'onde", expanded=False):
                        obs_choice = st.selectbox("Observable mesurée", ["σ_z", "σ_x"], key="obs_choice")
                        if obs_choice == "σ_z":
                            A_obs = np.array([[1, 0], [0, -1]], dtype=complex)
                        else:
                            A_obs = np.array([[0, 1], [1, 0]], dtype=complex)
                        eig = np.linalg.eigh(A_obs)
                        projecteur = np.outer(eig[1][:, 0], np.conj(eig[1][:, 0]))
                        psi_mes = eng_post.reduction_paquet_onde(psi0, projecteur)
                        prob_meas = np.abs(psi_mes)**2
                        st.markdown(f"**État après mesure** : {np.round(psi_mes, 4).tolist()}")
                        st.markdown(f"**Probabilités de résultat** : {np.round(prob_meas, 4).tolist()}")

            with st.expander("📚 Références", expanded=False):
                st.markdown("**Références :** Cohen-Tannoudji (Postulats, chap. III) · Shankar (chap. 4) · Griffiths (chap. 3)")
