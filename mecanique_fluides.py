__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp, quad
from scipy.optimize import fsolve, brentq
from scipy import signal, stats
from scipy.fft import fft, rfft, rfftfreq
from navier_stokes import navier_stokes_page
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE (ORIGINAL)
# ============================================================
CONSTANTES = {
    "ρ_eau (kg/m³)":      1000.0,
    "ρ_air (kg/m³)":      1.225,
    "μ_eau (Pa·s)":       1.002e-3,
    "μ_air (Pa·s)":       1.81e-5,
    "ν_eau (m²/s)":       1.004e-6,
    "ν_air (m²/s)":       1.48e-5,
    "g (m/s²)":           9.81,
    "P_atm (Pa)":         101325.0,
    "γ_eau (N/m)":        0.0728,
    "κ_air (J/(kg·K))":   1.4,
    "R_air (J/(kg·K))":   287.0,
}

FORMULES = {
    "Navier-Stokes":         r"\rho\!\left(\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}\right)=-\nabla p+\mu\nabla^2\mathbf{u}+\rho\mathbf{g}",
    "Continuité":            r"\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0",
    "Reynolds":              r"Re=\frac{\rho U L}{\mu}=\frac{UL}{\nu}",
    "Bernoulli":             r"p+\frac{1}{2}\rho u^2+\rho g z=\text{const}",
    "Darcy-Weisbach":        r"\Delta p = f\frac{L}{D}\frac{\rho u^2}{2}",
    "Moody (turbulent)":     r"\frac{1}{\sqrt{f}}=-2\log\!\left(\frac{\varepsilon/D}{3.7}+\frac{2.51}{Re\sqrt{f}}\right)",
    "Couche limite Blasius":  r"\delta(x)=5\sqrt{\frac{\nu x}{U_\infty}}=\frac{5x}{\sqrt{Re_x}}",
    "Traînée":               r"F_D=\frac{1}{2}C_D\rho U^2 A",
    "Portance":              r"F_L=\frac{1}{2}C_L\rho U^2 A",
    "Strouhal":              r"St=\frac{fD}{U}",
    "Weber":                 r"We=\frac{\rho U^2 L}{\gamma}",
    "Froude":                r"Fr=\frac{U}{\sqrt{gL}}",
    "Mach":                  r"Ma=\frac{U}{a}=\frac{U}{\sqrt{\gamma RT}}",
    "Poiseuille (débit)":    r"Q=\frac{\pi R^4}{8\mu}\left(-\frac{dp}{dx}\right)=\frac{\pi R^4\Delta p}{8\mu L}",
    "Torricelli":            r"U_{sortie}=\sqrt{2gh}",
    "Coefficient de perte":  r"\Delta p=K\frac{\rho U^2}{2}",
}

FLUIDES_PREDEF = {
    "Eau (20°C)":      {"rho": 998.2,  "mu": 1.002e-3, "nu": 1.004e-6, "gamma": 0.0728},
    "Air (20°C)":      {"rho": 1.204,  "mu": 1.81e-5,  "nu": 1.506e-5, "gamma": None},
    "Huile moteur":    {"rho": 870.0,  "mu": 0.1,      "nu": 1.15e-4,  "gamma": 0.030},
    "Glycérol":        {"rho": 1261.0, "mu": 1.412,    "nu": 1.12e-3,  "gamma": 0.064},
    "Mercure":         {"rho": 13534.0,"mu": 1.526e-3, "nu": 1.13e-7,  "gamma": 0.485},
    "Personnalisé":    {"rho": 1000.0, "mu": 1e-3,     "nu": 1e-6,     "gamma": 0.07},
}


# ============================================================
# MOTEUR MÉCANIQUE DES FLUIDES (ORIGINAL)
# ============================================================
class FluidEngine:
    """Moteur scientifique complet en mécanique des fluides."""

    def __init__(self, rho: float, mu: float):
        self.rho = rho
        self.mu  = mu
        self.nu  = mu / rho

    def reynolds(self, U: float, L: float) -> float:
        return U * L / self.nu

    def froude(self, U: float, L: float) -> float:
        return U / np.sqrt(9.81 * L)

    def mach(self, U: float, T: float = 293.15,
             gamma: float = 1.4, R: float = 287.0) -> float:
        a = np.sqrt(gamma * R * T)
        return U / a

    def weber(self, U: float, L: float, gamma: float) -> float:
        return self.rho * U**2 * L / gamma

    def strouhal(self, f: float, D: float, U: float) -> float:
        return f * D / U

    @st.cache_data
    def profil_poiseuille(_self, R: float, n: int = 200) -> tuple:
        r     = np.linspace(-R, R, n)
        u_max = 1.0
        u     = u_max * (1 - (r/R)**2)
        return r, u

    def debit_poiseuille(self, R: float, dpdx: float) -> float:
        return np.pi * R**4 * abs(dpdx) / (8 * self.mu)

    def vitesse_moyenne(self, R: float, dpdx: float) -> float:
        return self.debit_poiseuille(R, dpdx) / (np.pi * R**2)

    def pertes_charge_darcy(self, U: float, L: float, D: float,
                             eps: float = 0.0) -> dict:
        Re = self.reynolds(U, D)
        if Re < 2300:
            f      = 64 / Re
            regime = "Laminaire"
        elif Re < 4000:
            f      = 64/Re*0.5 + 0.316*Re**(-0.25)*0.5
            regime = "Transitoire"
        else:
            def colebrook(f_val):
                if f_val <= 0:
                    return 1e10
                return (1/np.sqrt(f_val) +
                        2*np.log10(eps/(3.7*D) + 2.51/(Re*np.sqrt(f_val))))
            try:
                f = brentq(lambda fv: colebrook(fv), 1e-6, 1.0)
            except:
                f = 0.316 * Re**(-0.25)
            regime = "Turbulent"
        dP = f * L/D * 0.5 * self.rho * U**2
        return {"f": f, "Re": Re, "dP": dP, "regime": regime, "dP_par_m": dP/L}

    def couche_limite_blasius(_self, x_arr: np.ndarray, U_inf: float) -> dict:
        x_arr = np.asarray(x_arr, dtype=float)
        Re_x      = U_inf * x_arr / _self.nu
        delta     = 5     * x_arr / np.sqrt(np.maximum(Re_x, 1e-10))
        delta_star= 1.7208* x_arr / np.sqrt(np.maximum(Re_x, 1e-10))
        theta     = 0.664 * x_arr / np.sqrt(np.maximum(Re_x, 1e-10))
        Cf        = 0.664 / np.sqrt(np.maximum(Re_x, 1e-10))
        return {"delta": delta, "delta_star": delta_star,
                "theta": theta, "Cf": Cf, "Re_x": Re_x}

    @st.cache_data
    def profil_blasius(_self, eta_max: float = 8.0, n: int = 200) -> tuple:
        def blasius_ode(y, eta):
            f, fp, fpp = y
            return [fp, fpp, -0.5*f*fpp]
        eta = np.linspace(0, eta_max, n)
        y0  = [0, 0, 0.332]
        sol = odeint(blasius_ode, y0, eta)
        return eta, sol[:, 1]

    def force_trainee(self, CD: float, U: float, A: float) -> float:
        return 0.5 * CD * self.rho * U**2 * A

    def force_portance(self, CL: float, U: float, A: float) -> float:
        return 0.5 * CL * self.rho * U**2 * A

    def finesse_aerodyn(self, CL: float, CD: float) -> float:
        return CL / CD if CD > 0 else np.inf

    def vitesse_torricelli(self, h: float, Cd: float = 0.61) -> float:
        return Cd * np.sqrt(2 * 9.81 * h)

    @st.cache_data
    def vidange_reservoir(_self, A_res: float, A_or: float,
                           h0: float, Cd: float = 0.61) -> tuple:
        def dhdt(h, t):
            if h[0] <= 0:
                return [0]
            v = Cd * np.sqrt(2 * 9.81 * h[0])
            return [-(A_or/A_res) * v]
        t_end = 2 * A_res * np.sqrt(2*h0/9.81) / (Cd * A_or * 1.5)
        t   = np.linspace(0, t_end, 500)
        sol = odeint(dhdt, [h0], t)
        h   = np.maximum(sol[:, 0], 0)
        return t, h

    def coup_de_belier(self, U: float, L: float, D: float,
                        e: float, E: float) -> dict:
        c       = 1 / np.sqrt(self.rho * (1/(E) + D/(e*2e11)))
        delta_p = self.rho * c * U
        t_ret   = 2 * L / c
        return {"c_onde": c, "delta_p": delta_p,
                "t_retour": t_ret, "surpression_%": delta_p/101325*100}

    def similitude(self, echelle_L: float, U_maquette: float) -> dict:
        Re_mod  = U_maquette * echelle_L / self.nu
        U_proto = U_maquette / echelle_L
        Re_proto= U_proto / self.nu
        return {"Re_modele": Re_mod, "U_prototype": U_proto,
                "Re_prototype": Re_proto, "Rapport_forces": echelle_L**3}

    def diagnostiquer(self, U: float, L: float, D: float = None) -> list:
        Re = self.reynolds(U, L)
        D  = D or L
        return [
            {"Test": "Régime (Re)", "Valeur": f"{Re:.1f}",
             "Statut": "🟢 Laminaire" if Re<2300 else "🔴 Turbulent" if Re>4000 else "🟡 Transitoire",
             "Note": f"Re={'<2300' if Re<2300 else '>4000' if Re>4000 else '2300-4000'}"},
            {"Test": "CFL (dt=0.01)", "Valeur": f"{U*0.01/D:.4f}",
             "Statut": "✅ OK" if U*0.01/D < 1 else "⚠️ Instable",
             "Note": "Condition CFL = U·Δt/Δx < 1"},
            {"Test": "Froude", "Valeur": f"{self.froude(U, L):.3f}",
             "Statut": "Fluvial" if self.froude(U,L)<1 else "Torrentiel",
             "Note": "Fr < 1 : fluvial"},
        ]


# ============================================================
# ██  NOUVEAUX MOTEURS — CHAPITRES ENRICHIS                 ██
# ============================================================

# --- Formulaires LaTeX nouveaux chapitres ---
FORMULES_STATIQUE = {
    "Pression hydrostatique":       r"p(h)=p_0+\rho g h",
    "Équation fondamentale":        r"\frac{dp}{dz}=-\rho g",
    "Poussée d'Archimède":          r"F_A=\rho_f\,g\,V_{\text{immergé}}",
    "Force surface plane":          r"F=\rho g\,h_G\,A",
    "Centre de poussée":            r"h_p=h_G+\frac{I_G}{h_G\,A},\quad I_G=\frac{bh^3}{12}",
    "Hauteur métacentrique":        r"GM=\frac{I_{WP}}{V_{\text{imm}}}-\overline{GB}",
    "Capillarité":                  r"h=\frac{2\gamma\cos\theta}{\rho g r}",
    "Atmosphère standard ISA":      r"p(z)=p_0\!\left(1-\frac{Lz}{T_0}\right)^{\!g/(RL)}",
    "Pression manométrique":        r"p_m=p-p_{\text{atm}}=\rho g h",
    "Manomètre en U":               r"p_A-p_B=(\rho_2-\rho_1)g\,\Delta h",
}

FORMULES_CINEMATIQUE = {
    "Dérivée particulaire":         r"\frac{D\mathbf{u}}{Dt}=\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}",
    "Continuité incompressible":    r"\nabla\cdot\mathbf{u}=0",
    "Fonction de courant 2D":       r"u=\frac{\partial\psi}{\partial y},\quad v=-\frac{\partial\psi}{\partial x}",
    "Potentiel de vitesse":         r"\mathbf{u}=\nabla\phi,\quad\nabla^2\phi=0",
    "Vorticité":                    r"\omega_z=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y}",
    "Source 2D":                    r"u_r=\frac{Q}{2\pi r},\quad\psi=\frac{Q\,\theta}{2\pi}",
    "Tourbillon libre":             r"u_\theta=\frac{\Gamma}{2\pi r},\quad\psi=-\frac{\Gamma}{2\pi}\ln r",
    "Circulation":                  r"\Gamma=\oint_C\mathbf{u}\cdot d\mathbf{l}",
    "Ligne de courant":             r"\frac{dx}{u}=\frac{dy}{v}\quad(\psi=\mathrm{const})",
    "Déformation cisaillement":     r"\dot{\gamma}=\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}",
}

FORMULES_DYNAMIQUE = {
    "Euler (fluide parfait)":       r"\rho\frac{D\mathbf{u}}{Dt}=-\nabla p+\rho\mathbf{g}",
    "Bernoulli généralisé":         r"p+\tfrac{1}{2}\rho u^2+\rho gz+\Delta p_f=\mathrm{const}",
    "Coefficient de pression":      r"C_p=\frac{p-p_\infty}{\frac{1}{2}\rho U_\infty^2}=1-\!\left(\frac{u}{U_\infty}\right)^{\!2}",
    "Bilan quantité de mouvement":  r"\sum\mathbf{F}=\frac{d}{dt}\!\int_V\!\rho\mathbf{u}\,dV+\oint_S\rho\mathbf{u}(\mathbf{u}\cdot\hat{n})\,dS",
    "Venturi":                      r"Q=C_d\frac{A_1 A_2}{\sqrt{A_1^2-A_2^2}}\sqrt{2g\Delta h}",
    "Joukowski":                    r"\Delta p=\rho\,c\,\Delta U,\quad c=\sqrt{\frac{1}{\rho\!\left(\frac{1}{K}+\frac{D}{eE}\right)}}",
    "Loi log turbulente":           r"u^+=\frac{1}{\kappa}\ln y^++B,\quad\kappa=0.41,\;B=5.2",
    "Strouhal / Kármán":            r"St=\frac{f\,D}{U}\approx0.198\!\left(1-\frac{19.7}{Re}\right)",
    "Pitot":                        r"p_0=p+\tfrac{1}{2}\rho u^2",
    "Jet impactant":                r"F=\rho Q U=\rho A U^2",
}


# ============================================================
# MOTEUR — STATIQUE DES FLUIDES
# ============================================================
class FluidStaticEngine:
    """Moteur scientifique complet — Statique des fluides."""

    def __init__(self, rho: float = 1000.0, g: float = 9.81):
        self.rho = rho
        self.g   = g

    def pression_hydrostatique(self, h: float, p0: float = 101325.0) -> float:
        """Pression absolue à la profondeur h (m)."""
        return p0 + self.rho * self.g * h

    def hauteur_equivalente(self, delta_p: float) -> float:
        return delta_p / (self.rho * self.g)

    def pression_altitude(self, z: np.ndarray, p0: float = 101325.0,
                          T0: float = 288.15, L: float = 0.0065) -> np.ndarray:
        """Atmosphère standard ISA."""
        R_air, g = 287.0, 9.81
        return p0 * (1 - L * z / T0) ** (g / (R_air * L))

    def poussee_archimede(self, V_imm: float) -> float:
        return self.rho * self.g * V_imm

    def flottabilite_sphere(self, R: float, rho_obj: float) -> dict:
        V  = (4/3) * np.pi * R**3
        P  = rho_obj * self.g * V
        FA = self.poussee_archimede(V)
        return {"Volume (m³)": V, "Poids (N)": P,
                "Poussée FA (N)": FA, "Force nette (N)": FA - P,
                "Flotte": FA >= P}

    def centre_poussee_rectangle(self, h_G: float, b: float, h: float) -> dict:
        A   = b * h
        I_G = b * h**3 / 12
        h_p = h_G + I_G / (max(h_G, 1e-10) * A)
        F   = self.rho * self.g * h_G * A
        return {"F (N)": F, "h_G (m)": h_G,
                "h_p (m)": h_p, "Excentricité (m)": h_p - h_G}

    def metacentre(self, I_WP: float, V_imm: float, GB: float) -> dict:
        BM = I_WP / max(V_imm, 1e-10)
        GM = BM - GB
        return {"BM (m)": BM, "GM (m)": GM,
                "Stable": GM > 0,
                "Statut": "✅ Stable" if GM > 0 else "❌ Instable"}

    def hauteur_capillaire(self, gamma: float, theta_deg: float, r: float) -> float:
        theta = np.radians(theta_deg)
        return 2 * gamma * np.cos(theta) / (self.rho * self.g * max(r, 1e-12))

    def pression_cylindre_Cp(self, theta_arr: np.ndarray) -> np.ndarray:
        """Coefficient de pression théorique autour d'un cylindre (potentiel)."""
        return 1 - 4 * np.sin(theta_arr)**2

    def diagnostiquer_statique(self, h: float, b: float, hh: float,
                                R_sph: float, rho_obj: float) -> list:
        p   = self.pression_hydrostatique(h)
        res = self.flottabilite_sphere(R_sph, rho_obj)
        sp  = self.centre_poussee_rectangle(h, b, hh)
        # Ensure boolean display is robust if arrays or numpy scalars are returned
        flotte_flag = bool(np.asarray(res.get("Flotte", False)).all())
        return [
            {"Test": "Pression à h (Pa)", "Valeur": f"{p:.1f}",
             "Statut": "ℹ️", "Note": "p = p_atm + ρgh"},
            {"Test": "Pression (bar)", "Valeur": f"{p/1e5:.4f}",
             "Statut": "ℹ️", "Note": "1 bar = 1e5 Pa"},
            {"Test": "Archimède sphère", "Valeur": f"FA={res['Poussée FA (N)']:.2f} N",
             "Statut": "✅ Flotte" if flotte_flag else "🔴 Coule",
             "Note": f"ρ_obj={rho_obj:.0f} vs ρ_fluide={self.rho:.0f} kg/m³"},
            {"Test": "Centre de poussée", "Valeur": f"h_p={sp['h_p (m)']:.3f} m",
             "Statut": "✅", "Note": f"Excentricité={sp['Excentricité (m)']:.3f} m"},
        ]


# ============================================================
# MOTEUR — CINÉMATIQUE DES FLUIDES
# ============================================================
class FluidKinematicsEngine:
    """Moteur scientifique complet — Cinématique des fluides."""

    def __init__(self, rho: float = 1000.0):
        self.rho = rho

    def source_2d(self, Q: float, x: np.ndarray, y: np.ndarray,
                  x0: float = 0.0, y0: float = 0.0) -> tuple:
        """Champ de vitesse d'une source 2D."""
        dx = x - x0;  dy = y - y0
        r2 = dx**2 + dy**2 + 1e-10
        return Q/(2*np.pi)*dx/r2, Q/(2*np.pi)*dy/r2

    def tourbillon_2d(self, Gamma: float, x: np.ndarray, y: np.ndarray,
                      x0: float = 0.0, y0: float = 0.0) -> tuple:
        """Champ de vitesse d'un tourbillon libre."""
        dx = x - x0;  dy = y - y0
        r2 = dx**2 + dy**2 + 1e-10
        return -Gamma/(2*np.pi)*dy/r2, Gamma/(2*np.pi)*dx/r2

    def ecoulement_uniforme(self, U: float, alpha_deg: float,
                             shape: tuple) -> tuple:
        alpha = np.radians(alpha_deg)
        ux = U * np.cos(alpha) * np.ones(shape)
        uy = U * np.sin(alpha) * np.ones(shape)
        return ux, uy

    def cylindre_potentiel(self, R: float, U_inf: float, Gamma: float,
                            x: np.ndarray, y: np.ndarray) -> tuple:
        """Écoulement potentiel autour d'un cylindre avec circulation."""
        r     = np.sqrt(x**2 + y**2) + 1e-12
        theta = np.arctan2(y, x)
        outside = r > R
        ur = np.where(outside, U_inf*(1 - R**2/r**2)*np.cos(theta), 0.0)
        ut = np.where(outside, -U_inf*(1 + R**2/r**2)*np.sin(theta)
                                - Gamma/(2*np.pi*r), 0.0)
        ux = ur*np.cos(theta) - ut*np.sin(theta)
        uy = ur*np.sin(theta) + ut*np.cos(theta)
        return ux, uy

    def portance_kutta_joukowski(self, rho: float, U_inf: float,
                                  Gamma: float) -> float:
        """Portance par unité d'envergure (Kutta-Joukowski)."""
        return rho * U_inf * Gamma

    def vorticite_numerique(self, ux: np.ndarray, uy: np.ndarray,
                             dx: float, dy: float) -> np.ndarray:
        return np.gradient(uy, dx, axis=1) - np.gradient(ux, dy, axis=0)

    def taux_deformation(self, dudx: float, dvdy: float,
                          dudy: float, dvdx: float) -> dict:
        return {
            "Divergence ∇·u": dudx + dvdy,
            "Rotation ω_z":   0.5*(dvdx - dudy),
            "Cisaillement":   0.5*(dudy + dvdx),
        }

    def diagnostiquer_cinematique(self, U: float, Q: float, Gamma: float) -> list:
        return [
            {"Test": "Continuité (∇·u=0)", "Valeur": "0 (exact)",
             "Statut": "✅ Vérifié", "Note": "Écoulement incompressible"},
            {"Test": "Irrotationnel", "Valeur": f"Γ={Gamma:.3f} m²/s",
             "Statut": "✅ Oui" if Gamma == 0 else "ℹ️ Tourbillon",
             "Note": "ω=0 hors singularités"},
            {"Test": "Débit source (m²/s)", "Valeur": f"Q={Q:.4f}",
             "Statut": "✅ Source" if Q > 0 else "🔵 Puits",
             "Note": "Par unité de profondeur"},
            {"Test": "Portance K-J (N/m)", "Valeur": f"L'={self.rho*U*Gamma:.2f}",
             "Statut": "ℹ️", "Note": "L' = ρ·U∞·Γ"},
        ]


# ============================================================
# MOTEUR — DYNAMIQUE DES FLUIDES
# ============================================================
class FluidDynamicsAdvEngine:
    """Moteur scientifique complet — Dynamique des fluides avancée."""

    def __init__(self, rho: float = 1000.0, mu: float = 1e-3):
        self.rho = rho
        self.mu  = mu
        self.nu  = mu / rho

    def bernoulli_check(self, p1: float, u1: float, z1: float,
                         p2: float, u2: float, z2: float,
                         pertes: float = 0.0) -> dict:
        H1 = p1 + 0.5*self.rho*u1**2 + self.rho*9.81*z1
        H2 = p2 + 0.5*self.rho*u2**2 + self.rho*9.81*z2
        return {"H1 (Pa)": H1, "H2 (Pa)": H2,
                "ΔH (Pa)": H1-H2, "Pertes (Pa)": pertes,
                "Résidu (Pa)": abs(H1-H2-pertes),
                "Valide": abs(H1-H2-pertes) < 1.0}

    def venturi(self, D1: float, D2: float, delta_h: float,
                Cd: float = 0.98) -> dict:
        A1 = np.pi*D1**2/4;  A2 = np.pi*D2**2/4
        Q_th  = A1*A2*np.sqrt(2*9.81*delta_h / max(A1**2-A2**2, 1e-12))
        Q_rel = Cd * Q_th
        u1 = Q_rel/A1;  u2 = Q_rel/A2
        dp = 0.5*self.rho*(u2**2 - u1**2)
        return {"Q_th (m³/s)": Q_th, "Q_réel (m³/s)": Q_rel,
                "u1 (m/s)": u1, "u2 (m/s)": u2,
                "Δp (Pa)": dp, "Δh_éq (m)": dp/(self.rho*9.81)}

    def jet_impact(self, U_jet: float, D_jet: float) -> dict:
        A = np.pi*D_jet**2/4
        Q = U_jet * A
        F = self.rho * Q * U_jet
        return {"Q (m³/s)": Q, "F_impact (N)": F,
                "Pression_dyn (Pa)": 0.5*self.rho*U_jet**2}

    def profil_turbulent_log(self, y_arr: np.ndarray,
                              u_tau: float) -> np.ndarray:
        kappa, B = 0.41, 5.2
        y_plus = u_tau * y_arr / self.nu
        u_plus = np.where(y_plus > 11.63,
                          (1/kappa)*np.log(np.maximum(y_plus, 1e-10)) + B,
                          y_plus)
        return u_plus * u_tau

    def frequence_karman(self, U: float, D: float, Re: float) -> dict:
        St  = 0.198*(1 - 19.7/max(Re, 1)) if Re > 300 else 0.12
        f   = St * U / max(D, 1e-10)
        return {"St": St, "f_Kármán (Hz)": f,
                "T (ms)": 1000/max(f, 1e-10)}

    def onde_choc_belier(self, U0: float, L: float, D: float,
                          e: float, E: float = 2.1e11) -> dict:
        K_eau    = 2.2e9
        c_rigide = np.sqrt(K_eau / self.rho)
        c_tuyau  = np.sqrt(1/(self.rho*(1/K_eau + D/(e*E))))
        dP       = self.rho * c_tuyau * U0
        return {"c_rigide (m/s)": c_rigide,
                "c_tuyau (m/s)": c_tuyau,
                "ΔP Joukowski (Pa)": dP,
                "ΔP (bar)": dP/1e5,
                "t_retour (s)": 2*L/c_tuyau}

    def Cp_profil_NACA(self, alpha_deg: float,
                        x_arr: np.ndarray) -> tuple:
        """Cp approximatif extrados/intrados NACA mince (théorie linéarisée)."""
        alpha  = np.radians(alpha_deg)
        Cp_ext = -2*alpha / np.sqrt(np.maximum(1 - x_arr**2, 1e-8))
        Cp_int =  2*alpha / np.sqrt(np.maximum(1 - x_arr**2, 1e-8))
        return Cp_ext, Cp_int

    def analyse_pi(self, rho: float, U: float, L: float,
                    mu: float, g: float, gamma: float) -> dict:
        return {
            "Re": rho*U*L/mu,
            "Fr": U/np.sqrt(g*L),
            "We": rho*U**2*L/gamma,
            "Ca": mu*U/gamma,
        }

    def diagnostiquer_dynamique(self, U: float, L: float) -> list:
        Re = U*L/self.nu
        Fr = U/np.sqrt(9.81*L)
        return [
            {"Test": "Reynolds", "Valeur": f"{Re:.2e}",
             "Statut": "🟢 Laminaire" if Re<2300 else "🔴 Turbulent",
             "Note": "Re = UL/ν"},
            {"Test": "Froude", "Valeur": f"{Fr:.4f}",
             "Statut": "🌊 Fluvial" if Fr<1 else "⚡ Torrentiel",
             "Note": "Fr = U/√(gL)"},
            {"Test": "Énergie cin. (J/m³)", "Valeur": f"{0.5*self.rho*U**2:.2f}",
             "Statut": "ℹ️", "Note": "½ρU²"},
            {"Test": "Pression dyn. (Pa)", "Valeur": f"{0.5*self.rho*U**2:.2f}",
             "Statut": "ℹ️", "Note": "q = ½ρU²"},
        ]


# ============================================================
# PAGE PRINCIPALE (ORIGINALE — INCHANGÉE)
# ============================================================
def mecanique_fluides_page():
    st.markdown("## 💧 Mécanique des Fluides Avancée")
    st.markdown("*Écoulements internes/externes, couche limite, pertes de charge, aérodynamique*")
    st.markdown("---")

    section = st.selectbox(
        "Section",
        [
            "⚙️ Fluide",
            "🔢 Nombres adimensionnels",
            "🚿 Écoulements internes",
            "🌬️ Couche limite",
            "✈️ Aérodynamique",
            "🌊 Navier-Stokes",
            "🌊 Hydraulique",
            "📖 Théorie",
            "📚 Chapitres enrichis",
        ],
        key="section_mecanique_fluides"
    )

    # Valeurs par défaut utilisées si aucune configuration fluide n'a encore été choisie
    rho, mu = 1000.0, 1e-3
    # Récupérer la dernière sélection de fluide (évite UnboundLocalError si on visite
    # directement une autre section sans passer par "⚙️ Fluide").
    fluide_sel = st.session_state.get('fluide_sel', list(FLUIDES_PREDEF.keys())[0])

    if section == "⚙️ Fluide":
        st.markdown("### ⚙️ Configuration du Fluide")
        st.markdown("*Sélectionnez le fluide ou définissez des propriétés personnalisées.*")
        fluide_sel = st.selectbox("Fluide", list(FLUIDES_PREDEF.keys()))
        # Sauvegarder la sélection dans la session pour la réutiliser ailleurs
        st.session_state['fluide_sel'] = fluide_sel
        if fluide_sel == "Personnalisé":
            st.markdown("**Propriétés personnalisées**")
            col1, col2 = st.columns(2)
            with col1:
                rho = st.slider("ρ (kg/m³)", 1.0, 15000.0, rho)
            with col2:
                mu  = st.slider("μ (Pa·s)", 1e-6, 10.0, mu, format="%.2e")
        else:
            fp  = FLUIDES_PREDEF[fluide_sel]
            rho, mu = fp["rho"], fp["mu"]
            # conserver la sélection dans la session
            st.session_state['fluide_sel'] = fluide_sel
            st.success(f"**Fluide sélectionné : {fluide_sel}**")
            st.info(f"ρ = {rho} kg/m³ | μ = {mu:.2e} Pa·s | ν = {fp['nu']:.2e} m²/s")

        st.markdown("### 🔬 Propriétés du fluide")
        df_props = pd.DataFrame([
            {"Propriété": "Masse volumique ρ", "Valeur": f"{rho} kg/m³", "Unité": "kg/m³"},
            {"Propriété": "Viscosité dynamique μ", "Valeur": f"{mu:.2e}", "Unité": "Pa·s"},
            {"Propriété": "Viscosité cinématique ν", "Valeur": f"{mu/rho:.2e}", "Unité": "m²/s"},
        ])
        st.table(df_props)

    engine = FluidEngine(rho, mu)

    if section == "🔢 Nombres adimensionnels":
        st.markdown("### 🔢 Nombres adimensionnels")
        col1, col2 = st.columns([1, 2])
        with col1:
            U            = st.slider("Vitesse U (m/s)", 0.01, 100.0, 1.0, 0.01)
            L            = st.slider("Longueur L (m)", 0.001, 10.0, 0.1, 0.001)
            T            = st.slider("Température T (K)", 200.0, 400.0, 293.0, 1.0)
            gamma_surf   = st.slider("Tension superficielle γ (N/m)", 0.001, 0.5, 0.072, 0.001)
            Re = engine.reynolds(U, L);  Fr = engine.froude(U, L)
            Ma = engine.mach(U, T);      We = engine.weber(U, L, gamma_surf)
            nombres = {"Reynolds Re": Re, "Froude Fr": Fr, "Mach Ma": Ma,
                       "Weber We": We, "Strouhal St (f=1Hz)": engine.strouhal(1.0, L, U),
                       "Euler Eu": 1.0/(0.5*rho*U**2) if U>0 else 0}
            for k, v in nombres.items():
                st.metric(k, f"{v:.4f}")
        with col2:
            U_arr  = np.logspace(-2, 3, 200)
            Re_arr = engine.reynolds(U_arr, L)
            fig_re = go.Figure()
            fig_re.add_trace(go.Scatter(x=U_arr, y=Re_arr, mode='lines',
                name='Re(U)', line=dict(color='#00ccff', width=3)))
            fig_re.add_hline(y=2300, line_color='#00ff88', line_dash='dash',
                             annotation_text="Laminaire→Transitoire (Re=2300)")
            fig_re.add_hline(y=4000, line_color='#ff4444', line_dash='dash',
                             annotation_text="Transitoire→Turbulent (Re=4000)")
            fig_re.add_vline(x=U, line_color='#ffcc00', line_dash='dot',
                             annotation_text=f"U={U} m/s")
            fig_re.update_layout(
                title=f"Reynolds vs Vitesse — {fluide_sel}",
                xaxis_title="U (m/s)", yaxis_title="Re",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400)
            st.plotly_chart(fig_re, use_container_width=True)

            st.markdown("#### 📊 Diagramme de Moody")
            Re_moody    = np.logspace(3, 8, 200)
            eps_D_vals  = [0, 1e-4, 1e-3, 5e-3, 1e-2]
            colors_m    = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00']
            fig_moody   = go.Figure()
            for i, eps_D in enumerate(eps_D_vals):
                f_vals = []
                for Re_i in Re_moody:
                    if Re_i < 2300:
                        f_vals.append(64/Re_i)
                    else:
                        try:
                            def col(fv):
                                return 1/np.sqrt(fv)+2*np.log10(eps_D/3.7+2.51/(Re_i*np.sqrt(fv)))
                            fv = brentq(col, 1e-6, 1.0)
                            f_vals.append(fv)
                        except:
                            f_vals.append(0.316*Re_i**(-0.25))
                label = f"ε/D={eps_D}" if eps_D > 0 else "Lisse"
                fig_moody.add_trace(go.Scatter(x=Re_moody, y=f_vals, mode='lines',
                    name=label, line=dict(color=colors_m[i], width=2)))
            fig_moody.update_layout(
                title="Diagramme de Moody",
                xaxis_title="Re", yaxis_title="f (Darcy)",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=350, showlegend=False)
            st.plotly_chart(fig_moody, use_container_width=True)

    elif section == "🚿 Écoulements internes":
        st.markdown("### 🚿 Écoulements en conduite")
        col1, col2 = st.columns([1, 2])
        with col1:
            R_pipe = st.slider("Rayon R (m)", 0.001, 0.5, 0.05, 0.001)
            L_pipe = st.slider("Longueur L (m)", 0.1, 100.0, 10.0, 0.1)
            dpdx   = st.slider("-dp/dx (Pa/m)", 0.01, 1000.0, 10.0, 0.1)
            eps    = st.slider("Rugosité ε (m)", 0.0, 0.01, 0.0001, 0.0001)
            U_moy  = engine.vitesse_moyenne(R_pipe, dpdx)
            Q_vol  = engine.debit_poiseuille(R_pipe, dpdx)
            pertes = engine.pertes_charge_darcy(U_moy, L_pipe, 2*R_pipe, eps)
            st.metric("U_moy (m/s)", f"{U_moy:.4f}")
            st.metric("Q (m³/s)", f"{Q_vol:.4e}")
            st.metric("Re", f"{pertes['Re']:.1f}")
            st.metric("Régime", pertes["regime"])
            st.metric("f (Darcy)", f"{pertes['f']:.5f}")
            st.metric("ΔP total (Pa)", f"{pertes['dP']:.2f}")
            st.metric("ΔP/m (Pa/m)", f"{pertes['dP_par_m']:.4f}")
        with col2:
            r, u     = engine.profil_poiseuille(R_pipe)
            u_scale  = u * U_moy / u.max() if u.max() > 0 else u
            fig_p    = make_subplots(rows=1, cols=2,
                subplot_titles=["Profil u(r)", "Champ de vitesse 2D"])
            fig_p.add_trace(go.Scatter(x=u_scale, y=r, mode='lines', name='u(r)',
                line=dict(color='#00ccff', width=3),
                fill='tozerox', fillcolor='rgba(0,204,255,0.15)'), row=1, col=1)
            fig_p.add_hline(y=0, line_color='rgba(255,255,255,0.3)', row=1, col=1)
            theta  = np.linspace(0, 2*np.pi, 40)
            r_2d   = np.linspace(0, R_pipe, 25)
            R_g, Th_g = np.meshgrid(r_2d, theta)
            U_2d   = U_moy * 2 * (1 - (R_g/R_pipe)**2)
            X_2d   = R_g * np.cos(Th_g) * 1000
            Y_2d   = R_g * np.sin(Th_g) * 1000
            fig_p.add_trace(go.Scatter(
                x=X_2d.flatten(), y=Y_2d.flatten(), mode='markers',
                marker=dict(color=U_2d.flatten(), size=4,
                            colorscale=[[0,'#020817'],[0.5,'#7700ff'],[1,'#00ccff']],
                            showscale=True), name='u(x,y)'), row=1, col=2)
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)', font=dict(color='#c0d0ff'), height=420,
                legend=dict(bgcolor='rgba(0,0,0,0.5)'))
            fig_p.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_p.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_p, use_container_width=True)
            L_arr  = np.linspace(0.1, L_pipe, 200)
            dP_arr = pertes['dP_par_m'] * L_arr
            fig_dP = go.Figure()
            fig_dP.add_trace(go.Scatter(x=L_arr, y=dP_arr, mode='lines',
                name='ΔP(L)', line=dict(color='#ff00cc', width=2.5),
                fill='tozeroy', fillcolor='rgba(255,0,204,0.1)'))
            fig_dP.update_layout(title="Pertes de charge vs longueur",
                xaxis_title="L (m)", yaxis_title="ΔP (Pa)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=300)
            st.plotly_chart(fig_dP, use_container_width=True)

    elif section == "🌬️ Couche limite":
        st.markdown("### 🌬️ Couche limite laminaire de Blasius")
        col1, col2 = st.columns([1, 2])
        with col1:
            U_inf = st.slider("U∞ (m/s)", 0.01, 50.0, 5.0, 0.1)
            x_max = st.slider("x_max (m)", 0.01, 5.0, 1.0, 0.01)
            x_arr = np.linspace(0.001, x_max, 300)
            bl    = engine.couche_limite_blasius(x_arr, U_inf)
            Re_L  = engine.reynolds(U_inf, x_max)
            st.metric("Re_L", f"{Re_L:.2e}")
            st.metric("δ à x_max (mm)", f"{bl['delta'][-1]*1000:.3f}")
            st.metric("δ* à x_max (mm)", f"{bl['delta_star'][-1]*1000:.3f}")
            st.metric("θ à x_max (mm)", f"{bl['theta'][-1]*1000:.3f}")
            st.metric("Cf à x_max", f"{bl['Cf'][-1]:.4f}")
            st.metric("Cf moyen", f"{2*bl['Cf'][-1]:.4f}")
        with col2:
            fig_bl = make_subplots(rows=2, cols=1,
                subplot_titles=["Épaisseurs de couche limite", "Profil de Blasius f'(η)"])
            fig_bl.add_trace(go.Scatter(x=x_arr, y=bl['delta']*1000, mode='lines',
                name='δ (mm)', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
            fig_bl.add_trace(go.Scatter(x=x_arr, y=bl['delta_star']*1000, mode='lines',
                name='δ* (mm)', line=dict(color='#7700ff', width=2, dash='dash')), row=1, col=1)
            fig_bl.add_trace(go.Scatter(x=x_arr, y=bl['theta']*1000, mode='lines',
                name='θ (mm)', line=dict(color='#ff00cc', width=2, dash='dot')), row=1, col=1)
            eta, fp_blasius = engine.profil_blasius()
            fig_bl.add_trace(go.Scatter(x=fp_blasius, y=eta, mode='lines',
                name="u/U∞", line=dict(color='#00ccff', width=3),
                fill='tozerox', fillcolor='rgba(0,204,255,0.1)'), row=2, col=1)
            fig_bl.add_vline(x=0.99, line_color='#ffcc00', line_dash='dash',
                             annotation_text="u/U∞=0.99", row=2, col=1)
            fig_bl.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)', font=dict(color='#c0d0ff'),
                height=550, legend=dict(bgcolor='rgba(0,0,0,0.5)'))
            fig_bl.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_bl.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_bl, use_container_width=True)

    elif section == "✈️ Aérodynamique":
        st.markdown("### ✈️ Aérodynamique & Forces")
        col1, col2 = st.columns([1, 2])
        with col1:
            U_aero = st.slider("Vitesse U (m/s)", 1.0, 300.0, 50.0, 1.0)
            S_aero = st.slider("Surface S (m²)", 0.01, 200.0, 20.0, 0.1)
            CD     = st.slider("Cx (traînée)", 0.001, 2.0, 0.3, 0.001)
            CL     = st.slider("Cz (portance)", 0.0, 3.0, 0.5, 0.01)
            masse  = st.slider("Masse (kg)", 1.0, 500000.0, 5000.0, 100.0)
            FD     = engine.force_trainee(CD, U_aero, S_aero)
            FL     = engine.force_portance(CL, U_aero, S_aero)
            finesse= engine.finesse_aerodyn(CL, CD)
            poids  = masse * 9.81
            st.metric("Traînée FD (N)", f"{FD:.2f}")
            st.metric("Portance FL (N)", f"{FL:.2f}")
            st.metric("Finesse CL/CD", f"{finesse:.2f}")
            st.metric("Charge utile (FL/FD)", f"{FL/FD:.2f}" if FD>0 else "∞")
            st.metric("FL / Poids", f"{FL/poids:.4f}")
            st.metric("Vitesse décrochage (m/s)",
                      f"{np.sqrt(2*masse*9.81/(rho*S_aero*(CL+0.01))):.2f}")
        with col2:
            U_sweep  = np.linspace(1, 300, 300)
            FD_sweep = engine.force_trainee(CD, U_sweep, S_aero)
            FL_sweep = engine.force_portance(CL, U_sweep, S_aero)
            fig_aero = go.Figure()
            fig_aero.add_trace(go.Scatter(x=U_sweep, y=FD_sweep, mode='lines',
                name='Traînée FD (N)', line=dict(color='#ff4444', width=2.5)))
            fig_aero.add_trace(go.Scatter(x=U_sweep, y=FL_sweep, mode='lines',
                name='Portance FL (N)', line=dict(color='#00ccff', width=2.5)))
            fig_aero.add_hline(y=poids, line_color='#ffcc00', line_dash='dash',
                               annotation_text=f"Poids={poids:.0f}N")
            fig_aero.add_vline(x=U_aero, line_color='rgba(255,255,255,0.4)',
                               line_dash='dot', annotation_text=f"U={U_aero}m/s")
            fig_aero.update_layout(title="Forces aérodynamiques vs vitesse",
                xaxis_title="U (m/s)", yaxis_title="Force (N)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400)
            st.plotly_chart(fig_aero, use_container_width=True)
            alpha_arr = np.linspace(-5, 20, 100)
            CL_arr    = 0.1 * alpha_arr + CL
            CD_arr    = CD + 0.01 * alpha_arr**2 / 100
            fig_pol   = go.Figure()
            fig_pol.add_trace(go.Scatter(x=CD_arr, y=CL_arr, mode='lines',
                line=dict(color='#00ccff', width=2.5), name='Polaire'))
            fig_pol.update_layout(title="Polaire de portance CL=f(CD)",
                xaxis_title="CD (traînée)", yaxis_title="CL (portance)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=320)
            st.plotly_chart(fig_pol, use_container_width=True)

    elif section == "🌊 Navier-Stokes":
        st.markdown("### 🌊 Navier-Stokes — Simulation CFD")
        st.info("Navier-Stokes a été importé et embarqué dans Mécanique des Fluides.")
        navier_stokes_page()

    elif section == "🌊 Hydraulique":
        st.markdown("### 🌊 Hydraulique & Torricelli")
        col1, col2 = st.columns([1, 2])
        with col1:
            h0    = st.slider("Hauteur initiale h₀ (m)", 0.1, 20.0, 5.0, 0.1)
            A_res = st.slider("Section réservoir A_res (m²)", 0.01, 100.0, 1.0, 0.01)
            A_or  = st.slider("Section orifice A_or (m²)", 0.0001, 0.1, 0.005, 0.0001)
            Cd    = st.slider("Coefficient de décharge Cd", 0.4, 1.0, 0.61, 0.01)
            v_tor = engine.vitesse_torricelli(h0, Cd)
            Q_or  = Cd * A_or * np.sqrt(2 * 9.81 * h0)
            st.metric("Vitesse Torricelli (m/s)", f"{v_tor:.3f}")
            st.metric("Débit (m³/s)", f"{Q_or:.4f}")
            st.metric("Débit (L/s)", f"{Q_or*1000:.2f}")
        with col2:
            t_vid, h_vid = engine.vidange_reservoir(A_res, A_or, h0, Cd)
            Q_vid = Cd * A_or * np.sqrt(2 * 9.81 * np.maximum(h_vid, 0))
            fig_vid = make_subplots(rows=2, cols=1,
                subplot_titles=["Hauteur h(t)", "Débit Q(t)"])
            fig_vid.add_trace(go.Scatter(x=t_vid, y=h_vid, mode='lines',
                name='h(t)', line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'), row=1, col=1)
            fig_vid.add_trace(go.Scatter(x=t_vid, y=Q_vid, mode='lines',
                name='Q(t)', line=dict(color='#7700ff', width=2.5)), row=2, col=1)
            idx_vide = np.where(h_vid < 0.001)[0]
            if len(idx_vide) > 0:
                t_vide = t_vid[idx_vide[0]]
                fig_vid.add_vline(x=t_vide, line_color='#ffcc00', line_dash='dash',
                                  annotation_text=f"t_vidange={t_vide:.1f}s", row=1, col=1)
            fig_vid.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)', font=dict(color='#c0d0ff'),
                height=500, legend=dict(bgcolor='rgba(0,0,0,0.5)'))
            fig_vid.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                  title_text="Temps (s)")
            fig_vid.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_vid, use_container_width=True)

    elif section == "📖 Théorie":
        st.markdown("### 📖 Formulaire mécanique des fluides")
        for nom, formule in FORMULES.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)
        st.markdown("---")
        st.markdown("### 🔬 Propriétés des fluides")
        df_f = pd.DataFrame([{"Fluide": k, **{kk: vv for kk, vv in v.items() if vv is not None}}
                              for k, v in FLUIDES_PREDEF.items() if k != "Personnalisé"])
        st.dataframe(df_f, use_container_width=True)
        st.markdown("---")
        for r in ["Batchelor — *An Introduction to Fluid Dynamics* (Cambridge, 2000)",
                  "White — *Fluid Mechanics* (McGraw-Hill, 2015)",
                  "Schlichting — *Boundary Layer Theory* (Springer, 2017)"]:
            st.markdown(f"- {r}")

    elif section == "📚 Chapitres enrichis":
        mecanique_fluides_enrichie(rho, mu)


# ============================================================
# ██  NOUVEAUX ONGLETS — CHAPITRES ENRICHIS                 ██
# ============================================================
def mecanique_fluides_enrichie(rho: float, mu: float):
    """
    Nouveaux chapitres : Statique | Cinématique | Dynamique avancée.
    """
    COLORS = ['#00ccff','#7700ff','#ff00cc','#00ff88',
              '#ffcc00','#ff4400','#88ccff','#cc88ff']
    STYLE  = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                  font=dict(color='#c0d0ff'), legend=dict(bgcolor='rgba(0,0,0,0.5)'))
    AX     = dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')

    st.markdown("---")
    st.markdown("## 🔬 Options approfondies — Mécanique des Fluides")

    chapitre = st.radio("Option",
        ["📐 Statique des fluides",
         "🌀 Cinématique des fluides",
         "⚡ Dynamique des fluides"],
        key="chap_fl_enr")

    # ══════════════════════════════════════════════════════════
    # CHAPITRE A — STATIQUE DES FLUIDES
    # ══════════════════════════════════════════════════════════
    if chapitre == "📐 Statique des fluides":
        st.markdown("## 📐 Section A — Statique des Fluides")
        st.markdown("Étude des fluides au repos : distribution de pression, "
                    "poussée d'Archimède, stabilité des corps flottants et capillarité.")

        subA1, subA2, subA3 = st.tabs([
            "⚙️ Simulation", "📊 Analyse pression",
            "🔬 Corps flottants & Capillarité"])

        with st.sidebar.expander("⚙️ Paramètres statique", expanded=True):
            rho_s = st.slider("ρ fluide (kg/m³)", 500.0, 15000.0, rho, 10.0, key="rho_st")
            g_s   = st.slider("g (m/s²)", 1.0, 25.0, 9.81, 0.01, key="g_st")

        eng_s = FluidStaticEngine(rho_s, g_s)

        # --- subA1 : Simulation ---
        with subA1:
            st.markdown("### ⚙️ Pression hydrostatique & Archimède")
            col1, col2 = st.columns([1, 2])
            with col1:
                h_max  = st.slider("Profondeur max (m)", 1.0, 1000.0, 100.0, 1.0, key="hmax_st")
                p_atm  = st.slider("p_atm (Pa)", 50000.0, 110000.0, 101325.0, 100.0, key="patm_st")
                h_pt   = st.slider("Point d'étude h (m)", 0.0, h_max, h_max/2, 0.1, key="hpt_st")
                p_pt   = eng_s.pression_hydrostatique(h_pt, p_atm)
                st.metric("p(h) (Pa)", f"{p_pt:.1f}")
                st.metric("p(h) (bar)", f"{p_pt/1e5:.4f}")
                st.metric("Pression manométrique (Pa)", f"{p_pt - p_atm:.1f}")
                st.metric("Hauteur éq. eau (m)", f"{eng_s.hauteur_equivalente(p_pt - p_atm):.3f}")
                st.markdown("---")
                st.markdown("#### 🚢 Archimède — Sphère")
                R_sph   = st.slider("Rayon sphère (m)", 0.01, 5.0, 0.5, 0.01, key="Rsph_st")
                rho_obj = st.slider("ρ objet (kg/m³)", 10.0, 15000.0, 800.0, 10.0, key="robj_st")
                res_a   = eng_s.flottabilite_sphere(R_sph, rho_obj)
                for k, v in res_a.items():
                    if isinstance(v, (bool, np.bool_)):
                        st.metric(k, "✅ OUI" if bool(v) else "❌ NON")
                    else:
                        st.metric(k, f"{v:.4g}")
            with col2:
                h_arr  = np.linspace(0, h_max, 300)
                p_arr  = eng_s.pression_hydrostatique(h_arr, p_atm)
                fig_ph = go.Figure()
                fig_ph.add_trace(go.Scatter(
                    x=p_arr/1e5, y=-h_arr, mode='lines',
                    name='p(h) (bar)', line=dict(color='#00ccff', width=3),
                    fill='tozerox', fillcolor='rgba(0,204,255,0.1)'))
                fig_ph.add_hline(y=-h_pt, line_color='#ffcc00', line_dash='dash',
                                 annotation_text=f"h={h_pt:.1f}m → {p_pt/1e5:.3f} bar")
                fig_ph.update_layout(title="Profil de pression hydrostatique",
                    xaxis_title="Pression (bar)", yaxis_title="Profondeur (m)",
                    **STYLE, height=400, xaxis=dict(**AX), yaxis=dict(**AX))
                st.plotly_chart(fig_ph, use_container_width=True)

                st.markdown("#### 🟦 Force sur surface plane immergée")
                c1, c2 = st.columns(2)
                with c1:
                    b_s  = st.slider("Largeur b (m)", 0.1, 10.0, 2.0, 0.1, key="b_surf_st")
                    h_s  = st.slider("Hauteur h (m)", 0.1, 10.0, 3.0, 0.1, key="h_surf_st")
                with c2:
                    hG_s = st.slider("h_G (m)", 0.1, 50.0, 5.0, 0.1, key="hG_surf_st")
                res_sp = eng_s.centre_poussee_rectangle(hG_s, b_s, h_s)
                for k, v in res_sp.items():
                    st.metric(k, f"{v:.4f}")

        # --- subA2 : Analyse pression ---
        with subA2:
            st.markdown("### 📊 Analyse des champs de pression")
            col1, col2 = st.columns([1, 2])
            with col1:
                z_max_km = st.slider("Altitude max (km)", 1.0, 30.0, 11.0, 0.5, key="zmax_atm_st")
                T0_atm   = st.slider("T₀ au sol (K)", 250.0, 310.0, 288.15, 0.1, key="T0_atm_st")
                z_arr    = np.linspace(0, z_max_km*1e3, 300)
                p_z      = eng_s.pression_altitude(z_arr, T0=T0_atm)
                st.metric("p au sol (Pa)", f"{p_z[0]:.1f}")
                st.metric(f"p à {z_max_km:.0f} km (Pa)", f"{p_z[-1]:.1f}")
                st.metric(f"p à {z_max_km:.0f} km (hPa)", f"{p_z[-1]/100:.2f}")
                st.metric("Rapport p/p₀", f"{p_z[-1]/p_z[0]:.5f}")
            with col2:
                fig_atm = make_subplots(rows=1, cols=2,
                    subplot_titles=["p vs Altitude", "Cp cylindre (théorie potentielle)"])
                fig_atm.add_trace(go.Scatter(x=p_z/100, y=z_arr/1e3, mode='lines',
                    name='p(z)', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                theta_c = np.linspace(0, 2*np.pi, 300)
                Cp_c    = eng_s.pression_cylindre_Cp(theta_c)
                fig_atm.add_trace(go.Scatter(
                    x=np.degrees(theta_c), y=Cp_c, mode='lines',
                    name='Cp cylindre', line=dict(color='#ff00cc', width=2.5),
                    fill='tozeroy', fillcolor='rgba(255,0,204,0.1)'), row=1, col=2)
                fig_atm.add_hline(y=-3, line_color='#ffcc00', line_dash='dash',
                                  annotation_text="Cp_min=-3", row=1, col=2)
                fig_atm.add_hline(y=1, line_color='#00ff88', line_dash='dash',
                                  annotation_text="Cp=1 (arrêt)", row=1, col=2)
                fig_atm.update_layout(**STYLE, height=400)
                fig_atm.update_xaxes(**AX)
                fig_atm.update_yaxes(**AX)
                st.plotly_chart(fig_atm, use_container_width=True)

            st.markdown("### 🔍 Diagnostic hydrostatique")
            diag_s = eng_s.diagnostiquer_statique(h_pt, b_s, h_s, R_sph, rho_obj)
            st.dataframe(pd.DataFrame(diag_s), use_container_width=True)
            df_exp = pd.DataFrame({"Profondeur (m)": h_arr, "p (Pa)": p_arr, "p (bar)": p_arr/1e5})
            st.download_button("📥 Exporter profil pression (CSV)", df_exp.to_csv(index=False),
                               "profil_pression.csv", "text/csv", key="dl_pres_st")

        # --- subA3 : Corps flottants & Capillarité ---
        with subA3:
            st.markdown("### 🔬 Stabilité des corps flottants & Capillarité")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### ⚓ Métacentre — Corps rectangulaire")
                B_fl  = st.slider("Base b (m)", 0.5, 20.0, 4.0, 0.1, key="B_fl_st")
                H_fl  = st.slider("Hauteur totale H (m)", 0.5, 20.0, 6.0, 0.1, key="H_fl_st")
                T_fl  = st.slider("Tirant d'eau T (m)", 0.1, H_fl*0.95, H_fl*0.5, 0.05, key="T_fl_st")
                I_WP  = B_fl**3 / 12
                V_imm = B_fl * T_fl
                GB    = H_fl/2 - T_fl/2
                res_m = eng_s.metacentre(I_WP, V_imm, GB)
                st.metric("BM (m)", f"{res_m['BM (m)']:.4f}")
                st.metric("GM (m)", f"{res_m['GM (m)']:.4f}")
                st.info(res_m["Statut"])

                st.markdown("#### 💧 Capillarité")
                gamma_c  = st.slider("γ (mN/m)", 1.0, 100.0, 72.8, 0.5, key="gamma_cap_st") * 1e-3
                theta_c2 = st.slider("Angle θ (°)", 0.0, 180.0, 0.0, 1.0, key="theta_cap_st")
                r_cap    = st.slider("Rayon tube (μm)", 1.0, 5000.0, 100.0, 1.0, key="rcap_st") * 1e-6
                h_cap    = eng_s.hauteur_capillaire(gamma_c, theta_c2, r_cap)
                st.metric("Hauteur capillaire (mm)", f"{h_cap*1000:.3f}")
                st.metric("Hauteur capillaire (cm)", f"{h_cap*100:.4f}")

            with col2:
                T_sw  = np.linspace(0.1, H_fl*0.95, 200)
                GM_sw = []
                for Ts in T_sw:
                    r_ = eng_s.metacentre(I_WP, B_fl*Ts, H_fl/2 - Ts/2)
                    GM_sw.append(r_["GM (m)"])
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(x=T_sw, y=GM_sw, mode='lines',
                    name='GM(T)', line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.08)'))
                fig_mc.add_hline(y=0, line_color='#ff4400', line_dash='dash',
                                 annotation_text="GM=0 — limite stabilité")
                fig_mc.add_vline(x=T_fl, line_color='#ffcc00', line_dash='dot',
                                 annotation_text=f"T={T_fl:.1f}m")
                fig_mc.update_layout(title="Hauteur métacentrique GM vs Tirant d'eau",
                    xaxis_title="Tirant d'eau T (m)", yaxis_title="GM (m)",
                    **STYLE, height=340, xaxis=dict(**AX), yaxis=dict(**AX))
                st.plotly_chart(fig_mc, use_container_width=True)

                r_arr_cap = np.logspace(-6, -2, 200)
                h_arr_cap = eng_s.hauteur_capillaire(gamma_c, theta_c2, r_arr_cap)
                fig_cap   = go.Figure()
                fig_cap.add_trace(go.Scatter(x=r_arr_cap*1e6, y=h_arr_cap*100, mode='lines',
                    name='h(r)', line=dict(color='#ff00cc', width=2.5),
                    fill='tozeroy', fillcolor='rgba(255,0,204,0.08)'))
                fig_cap.add_vline(x=r_cap*1e6, line_color='#ffcc00', line_dash='dot',
                                  annotation_text=f"r={r_cap*1e6:.0f}μm")
                fig_cap.update_layout(title="Hauteur capillaire vs rayon du tube",
                    xaxis_title="Rayon (μm)", yaxis_title="h capillaire (cm)",
                    xaxis_type='log',
                    **STYLE, height=300, xaxis=dict(**AX, type='log'), yaxis=dict(**AX))
                st.plotly_chart(fig_cap, use_container_width=True)

                df_cap_exp = pd.DataFrame({"r_tube (μm)": r_arr_cap*1e6, "h_cap (mm)": h_arr_cap*1000})
                st.download_button("📥 Exporter capillarité (CSV)", df_cap_exp.to_csv(index=False),
                                   "capillarite.csv", "text/csv", key="dl_cap_st")

        # L'onglet Théorie a été supprimé pour éviter les erreurs de rendu.

    # ══════════════════════════════════════════════════════════
    # CHAPITRE B — CINÉMATIQUE DES FLUIDES
    # ══════════════════════════════════════════════════════════
    elif chapitre == "🌀 Cinématique des fluides":
        st.markdown("## 🌀 Section B — Cinématique des Fluides")
        st.markdown("Description du mouvement des fluides : champs de vitesse, "
                    "lignes de courant, vorticité et écoulements potentiels.")

        subB1, subB2, subB3 = st.tabs([
            "⚙️ Simulation", "📊 Analyse vorticité",
            "🔬 Écoulement potentiel cylindre"])

        with st.sidebar.expander("⚙️ Paramètres cinématique", expanded=True):
            rho_k = st.slider("ρ (kg/m³)", 100.0, 13500.0, rho, 10.0, key="rho_kin")

        eng_k = FluidKinematicsEngine(rho_k)

        # --- subB1 : Simulation ---
        with subB1:
            st.markdown("### ⚙️ Superposition d'écoulements élémentaires")
            col1, col2 = st.columns([1, 2])
            with col1:
                type_ecoul = st.selectbox("Type d'écoulement",
                    ["Source 2D", "Tourbillon libre", "Source + Tourbillon",
                     "Écoulement uniforme", "Doublet"], key="type_ecoul_kin")
                U_unif  = st.slider("U∞ (m/s)", 0.1, 20.0, 5.0, 0.1, key="Uunif_kin")
                Q_src   = st.slider("Débit source Q (m²/s)", 0.0, 10.0, 2.0, 0.1, key="Q_kin")
                Gamma_v = st.slider("Circulation Γ (m²/s)", 0.0, 10.0, 0.0, 0.1, key="Gamma_kin")

                Re_kin = rho_k * U_unif * 1.0 / 1e-3
                St_kin = 0.198*(1 - 19.7/max(Re_kin, 1)) if Re_kin > 300 else 0.12
                st.metric("Re (L=1m)", f"{Re_kin:.2e}")
                st.metric("Strouhal", f"{St_kin:.4f}")
                st.metric("Portance K-J (N/m)",
                          f"{eng_k.portance_kutta_joukowski(rho_k, U_unif, Gamma_v):.3f}")

                diag_k = eng_k.diagnostiquer_cinematique(U_unif, Q_src, Gamma_v)
                st.dataframe(pd.DataFrame(diag_k), use_container_width=True)

            with col2:
                Nx, Ny = 40, 40
                x_lin  = np.linspace(-3, 3, Nx)
                y_lin  = np.linspace(-3, 3, Ny)
                X, Y   = np.meshgrid(x_lin, y_lin)

                if type_ecoul == "Source 2D":
                    UX, UY = eng_k.source_2d(Q_src, X, Y)
                elif type_ecoul == "Tourbillon libre":
                    UX, UY = eng_k.tourbillon_2d(Gamma_v, X, Y)
                elif type_ecoul == "Source + Tourbillon":
                    ux1, uy1 = eng_k.source_2d(Q_src, X, Y)
                    ux2, uy2 = eng_k.tourbillon_2d(Gamma_v, X, Y)
                    UX, UY   = ux1+ux2, uy1+uy2
                elif type_ecoul == "Écoulement uniforme":
                    UX, UY   = eng_k.ecoulement_uniforme(U_unif, 0, X.shape)
                else:  # Doublet
                    ux1, uy1 = eng_k.source_2d( Q_src, X, Y, -0.5, 0)
                    ux2, uy2 = eng_k.source_2d(-Q_src, X, Y,  0.5, 0)
                    UX, UY   = ux1+ux2, uy1+uy2

                vitesse = np.sqrt(UX**2 + UY**2)
                vitesse = np.clip(vitesse, 0, np.percentile(vitesse, 95))

                fig_kin = go.Figure()
                fig_kin.add_trace(go.Heatmap(
                    x=x_lin, y=y_lin, z=vitesse,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.7,'#00ccff'],[1,'#ffffff']],
                    showscale=True, name='|u| (m/s)',
                    colorbar=dict(title='|u| m/s', tickfont=dict(color='#c0d0ff'))))
                # Lignes de courant (streamlines approchées par quiver)
                scale = 0.3 / (np.max(vitesse) + 1e-10)
                fig_kin.add_trace(go.Scatter(
                    x=X.flatten(), y=Y.flatten(),
                    mode='markers',
                    marker=dict(size=1, color='rgba(200,200,255,0.1)'),
                    showlegend=False))
                fig_kin.update_layout(
                    title=f"Champ de vitesse — {type_ecoul}",
                    xaxis_title="x (m)", yaxis_title="y (m)",
                    **STYLE, height=480,
                    xaxis=dict(**AX), yaxis=dict(**AX))
                st.plotly_chart(fig_kin, use_container_width=True)

        # --- subB2 : Analyse vorticité ---
        with subB2:
            st.markdown("### 📊 Analyse vorticité & Déformation")
            col1, col2 = st.columns([1, 2])
            with col1:
                dudx_v = st.slider("∂u/∂x (s⁻¹)", -5.0, 5.0, 1.0, 0.1, key="dudx_kin")
                dvdy_v = st.slider("∂v/∂y (s⁻¹)", -5.0, 5.0,-1.0, 0.1, key="dvdy_kin")
                dudy_v = st.slider("∂u/∂y (s⁻¹)", -5.0, 5.0, 2.0, 0.1, key="dudy_kin")
                dvdx_v = st.slider("∂v/∂x (s⁻¹)", -5.0, 5.0,-2.0, 0.1, key="dvdx_kin")
                res_df = eng_k.taux_deformation(dudx_v, dvdy_v, dudy_v, dvdx_v)
                for k, v in res_df.items():
                    st.metric(k, f"{v:.4f} s⁻¹")
                st.markdown("---")
                st.markdown("**Hypothèse d'incompressibilité :**")
                incomp = abs(dudx_v + dvdy_v) < 0.01
                st.metric("∇·u (s⁻¹)", f"{dudx_v+dvdy_v:.4f}",
                          delta="≈ 0 ✅" if incomp else "≠ 0 ⚠️")
                ux_sample = np.array([[1.0, 2.0, 3.0],
                                      [1.5, 2.5, 3.5],
                                      [2.0, 3.0, 4.0]])
                uy_sample = np.array([[0.0, 0.5, 1.0],
                                      [0.0, 0.5, 1.0],
                                      [0.0, 0.5, 1.0]])
                vort = eng_k.vorticite_numerique(ux_sample, uy_sample, 0.1, 0.1)
                st.markdown("**Test vorticité numérique**")
                st.write(vort)
            with col2:
                # Évolution temporelle d'une particule fluide (déformation)
                t_arr  = np.linspace(0, 2, 200)
                x_part = np.exp(dudx_v * t_arr)
                y_part = np.exp(dvdy_v * t_arr)
                fig_def = make_subplots(rows=1, cols=2,
                    subplot_titles=["Trajectoire particule (x,y)", "Évolution x(t) et y(t)"])
                fig_def.add_trace(go.Scatter(x=x_part, y=y_part, mode='lines',
                    name='Trajectoire', line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.08)'), row=1, col=1)
                fig_def.add_trace(go.Scatter(x=t_arr, y=x_part, mode='lines',
                    name='x(t)', line=dict(color='#00ccff', width=2)), row=1, col=2)
                fig_def.add_trace(go.Scatter(x=t_arr, y=y_part, mode='lines',
                    name='y(t)', line=dict(color='#ff00cc', width=2, dash='dash')), row=1, col=2)
                fig_def.update_layout(**STYLE, height=380)
                fig_def.update_xaxes(**AX)
                fig_def.update_yaxes(**AX)
                st.plotly_chart(fig_def, use_container_width=True)

                # Export
                df_def = pd.DataFrame({"t (s)": t_arr, "x (m)": x_part, "y (m)": y_part})
                st.download_button("📥 Exporter trajectoire (CSV)", df_def.to_csv(index=False),
                                   "trajectoire_particule.csv", "text/csv", key="dl_traj_kin")

        # --- subB3 : Écoulement potentiel cylindre ---
        with subB3:
            st.markdown("### 🔬 Écoulement potentiel autour d'un cylindre")
            col1, col2 = st.columns([1, 2])
            with col1:
                R_cyl   = st.slider("Rayon cylindre R (m)", 0.1, 3.0, 1.0, 0.05, key="Rcyl_kin")
                U_cyl   = st.slider("U∞ (m/s)", 0.1, 20.0, 5.0, 0.1, key="Ucyl_kin")
                Gam_cyl = st.slider("Circulation Γ (m²/s)", 0.0, 50.0, 0.0, 0.5, key="Gam_cyl_kin")
                portance_kj = eng_k.portance_kutta_joukowski(rho_k, U_cyl, Gam_cyl)
                st.metric("Portance L' (N/m)", f"{portance_kj:.4f}")
                st.metric("Cp_min théorique", "-3.00")
                st.metric("Cp point arrêt", "1.00")
                # Angle des points de stagnation (robust to numpy arrays)
                g_arr = np.asarray(Gam_cyl)
                try:
                    if g_arr.size == 1:
                        g_val = float(g_arr.item())
                        cond = (g_val > 0) and (g_val < 4*np.pi*R_cyl*U_cyl)
                    else:
                        cond = np.all((g_arr > 0) & (g_arr < 4*np.pi*R_cyl*U_cyl))
                except Exception:
                    cond = False

                if cond:
                    sin_th = -Gam_cyl / (4*np.pi*R_cyl*U_cyl)
                    th_st  = np.degrees(np.arcsin(np.clip(sin_th, -1, 1)))
                    st.metric("θ stagnation (°)", f"{float(np.asarray(th_st).item()):.2f}")
                else:
                    st.metric("θ stagnation (°)", "0 / 180")

            with col2:
                Nx2, Ny2 = 50, 50
                x2 = np.linspace(-3*R_cyl, 3*R_cyl, Nx2)
                y2 = np.linspace(-3*R_cyl, 3*R_cyl, Ny2)
                X2, Y2   = np.meshgrid(x2, y2)
                UX2, UY2 = eng_k.cylindre_potentiel(R_cyl, U_cyl, Gam_cyl, X2, Y2)
                V2 = np.sqrt(UX2**2 + UY2**2)
                V2 = np.clip(V2, 0, 3*U_cyl)

                # Cp sur la surface
                theta_s = np.linspace(0, 2*np.pi, 300)
                Cp_surf = 1 - 4*np.sin(theta_s)**2

                fig_cyl = make_subplots(rows=1, cols=2,
                    subplot_titles=["Champ de vitesse |u|", "Cp sur le cylindre"])
                fig_cyl.add_trace(go.Heatmap(
                    x=x2, y=y2, z=V2,
                    colorscale=[[0,'#020817'],[0.4,'#7700ff'],[0.8,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                    colorbar=dict(title='|u| m/s', tickfont=dict(color='#c0d0ff'))),
                    row=1, col=1)
                # Contour du cylindre
                theta_circ = np.linspace(0, 2*np.pi, 100)
                fig_cyl.add_trace(go.Scatter(
                    x=R_cyl*np.cos(theta_circ), y=R_cyl*np.sin(theta_circ),
                    mode='lines', line=dict(color='white', width=2),
                    name='Cylindre'), row=1, col=1)
                fig_cyl.add_trace(go.Scatter(
                    x=np.degrees(theta_s), y=Cp_surf, mode='lines',
                    name='Cp(θ)', line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'), row=1, col=2)
                fig_cyl.add_hline(y=1, line_color='#00ff88', line_dash='dash',
                                  annotation_text="Cp=1", row=1, col=2)
                fig_cyl.add_hline(y=-3, line_color='#ff4400', line_dash='dash',
                                  annotation_text="Cp=-3", row=1, col=2)
                fig_cyl.update_layout(**STYLE, height=420)
                fig_cyl.update_xaxes(**AX)
                fig_cyl.update_yaxes(**AX)
                st.plotly_chart(fig_cyl, use_container_width=True)

                df_cp = pd.DataFrame({"theta_deg": np.degrees(theta_s), "Cp": Cp_surf})
                st.download_button("📥 Exporter Cp cylindre (CSV)", df_cp.to_csv(index=False),
                                   "Cp_cylindre.csv", "text/csv", key="dl_cp_cyl_kin")

        # L'onglet Théorie a été supprimé pour éviter les erreurs de rendu.

    # ══════════════════════════════════════════════════════════
    # CHAPITRE C — DYNAMIQUE DES FLUIDES
    # ══════════════════════════════════════════════════════════
    else:
        st.markdown("## ⚡ Section C — Dynamique des Fluides")
        st.markdown("Applications de Bernoulli, bilans de quantité de mouvement, "
                    "venturi, coup de bélier, profils turbulents et allée de Kármán.")

        subC1, subC2, subC3 = st.tabs([
            "⚙️ Simulation Bernoulli & Venturi", "📊 Sillage & Kármán",
            "🔬 Choc hydraulique & Jet"])

        with st.sidebar.expander("⚙️ Paramètres dynamique", expanded=True):
            rho_d = st.slider("ρ (kg/m³)", 100.0, 13500.0, rho, 10.0, key="rho_dyn")
            mu_d  = st.slider("μ (Pa·s)", 1e-5, 5.0, mu, format="%.2e", key="mu_dyn")

        eng_d = FluidDynamicsAdvEngine(rho_d, mu_d)

        # --- subC1 : Bernoulli & Venturi ---
        with subC1:
            st.markdown("### ⚙️ Bernoulli & Tube Venturi")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### 🌀 Venturi")
                D1_v = st.slider("D₁ amont (mm)", 10.0, 500.0, 100.0, 1.0, key="D1_ven") * 1e-3
                D2_v = st.slider("D₂ col (mm)", 5.0, D1_v*1000*0.9, 50.0, 1.0, key="D2_ven") * 1e-3
                Dh_v = st.slider("Δh manomètre (mm)", 1.0, 500.0, 50.0, 1.0, key="Dh_ven") * 1e-3
                Cd_v = st.slider("Cd venturi", 0.9, 1.0, 0.98, 0.001, key="Cd_ven")
                res_v = eng_d.venturi(D1_v, D2_v, Dh_v, Cd_v)
                for k, vv in res_v.items():
                    st.metric(k, f"{vv:.5g}")

                st.markdown("---")
                st.markdown("#### ⚡ Vérification Bernoulli")
                p1_b = st.slider("p₁ (Pa)", 1e4, 2e5, 1.1e5, 100.0, key="p1_bern")
                u1_b = st.slider("u₁ (m/s)", 0.1, 30.0, 2.0, 0.1, key="u1_bern")
                z1_b = st.slider("z₁ (m)", 0.0, 20.0, 0.0, 0.1, key="z1_bern")
                p2_b = st.slider("p₂ (Pa)", 1e4, 2e5, 1.0e5, 100.0, key="p2_bern")
                u2_b = res_v["u2 (m/s)"]
                z2_b = st.slider("z₂ (m)", 0.0, 20.0, 0.0, 0.1, key="z2_bern")
                res_b = eng_d.bernoulli_check(p1_b, u1_b, z1_b, p2_b, u2_b, z2_b)
                for k, vv in res_b.items():
                    if isinstance(vv, (bool, np.bool_)):
                        st.metric(k, "✅ OUI" if bool(vv) else "❌ NON")
                    else:
                        st.metric(k, f"{vv:.4g}")

            with col2:
                D1_arr  = np.linspace(D2_v*1.1, 0.5, 200)
                res_arr = [eng_d.venturi(D1_, D2_v, Dh_v, Cd_v) for D1_ in D1_arr]
                Q_arr   = [r["Q_réel (m³/s)"] for r in res_arr]
                u2_arr  = [r["u2 (m/s)"] for r in res_arr]

                fig_ven = make_subplots(rows=2, cols=1,
                    subplot_titles=["Q réel vs D₁", "u₂ au col vs D₁"])
                fig_ven.add_trace(go.Scatter(x=D1_arr*1000, y=Q_arr, mode='lines',
                    name='Q (m³/s)', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                fig_ven.add_vline(x=D1_v*1000, line_color='#ffcc00', line_dash='dot',
                                  annotation_text=f"D₁={D1_v*1000:.0f}mm", row=1, col=1)
                fig_ven.add_trace(go.Scatter(x=D1_arr*1000, y=u2_arr, mode='lines',
                    name='u₂ (m/s)', line=dict(color='#ff00cc', width=2.5)), row=2, col=1)
                fig_ven.update_layout(**STYLE, height=430)
                fig_ven.update_xaxes(title_text="D₁ (mm)", **AX)
                fig_ven.update_yaxes(**AX)
                st.plotly_chart(fig_ven, use_container_width=True)

                df_ven = pd.DataFrame({"D1 (mm)": D1_arr*1000, "Q (m³/s)": Q_arr, "u2 (m/s)": u2_arr})
                st.download_button("📥 Exporter Venturi (CSV)", df_ven.to_csv(index=False),
                                   "venturi.csv", "text/csv", key="dl_ven_dyn")

        # --- subC2 : Sillage & Kármán ---
        with subC2:
            st.markdown("### 📊 Sillage & Allée de Kármán")
            col1, col2 = st.columns([1, 2])
            with col1:
                U_kar  = st.slider("U∞ (m/s)", 0.01, 50.0, 5.0, 0.1, key="U_kar_dyn")
                D_kar  = st.slider("Diamètre cylindre D (m)", 0.001, 1.0, 0.05, 0.001, key="D_kar_dyn")
                Re_kar = rho_d * U_kar * D_kar / mu_d
                res_kar = eng_d.frequence_karman(U_kar, D_kar, Re_kar)
                st.metric("Reynolds", f"{Re_kar:.2e}")
                st.metric("Strouhal", f"{res_kar['St']:.4f}")
                st.metric("f Kármán (Hz)", f"{res_kar['f_Kármán (Hz)']:.3f}")
                st.metric("T période (ms)", f"{res_kar['T (ms)']:.3f}")

                st.markdown("---")
                st.markdown("#### 🌊 Profil turbulent (loi log)")
                u_tau_v = st.slider("u_τ (m/s)", 0.01, 2.0, 0.2, 0.01, key="utau_dyn")
                y_max_v = st.slider("y_max (mm)", 0.1, 100.0, 10.0, 0.1, key="ymax_dyn") * 1e-3
                y_log   = np.logspace(np.log10(mu_d/rho_d/u_tau_v*5), np.log10(y_max_v), 200)
                u_log   = eng_d.profil_turbulent_log(y_log, u_tau_v)
                u_tau_disp = u_tau_v
                st.metric("u_τ (m/s)", f"{u_tau_disp:.3f}")
                st.metric("U_max estimé (m/s)", f"{u_log[-1]:.3f}")

            with col2:
                # Sillage simulé
                t_kar   = np.linspace(0, 10/max(res_kar['f_Kármán (Hz)'], 0.01), 500)
                f_kar   = res_kar['f_Kármán (Hz)']
                amp_kar = np.exp(-t_kar * 0.1)
                y_kar   = amp_kar * np.sin(2*np.pi*f_kar*t_kar)
                y_kar2  = amp_kar * np.sin(2*np.pi*f_kar*t_kar + np.pi)

                fig_kar = make_subplots(rows=2, cols=1,
                    subplot_titles=[f"Allée de Kármán — f={f_kar:.2f} Hz",
                                    "Profil turbulent loi log"])
                fig_kar.add_trace(go.Scatter(x=t_kar, y=y_kar, mode='lines',
                    name='Tourbillon +', line=dict(color='#00ccff', width=2)), row=1, col=1)
                fig_kar.add_trace(go.Scatter(x=t_kar, y=y_kar2, mode='lines',
                    name='Tourbillon −', line=dict(color='#ff00cc', width=2,
                    dash='dash')), row=1, col=1)
                fig_kar.add_trace(go.Scatter(x=u_log, y=y_log*1000, mode='lines',
                    name='u(y) loi log', line=dict(color='#00ff88', width=2.5),
                    fill='tozerox', fillcolor='rgba(0,255,136,0.08)'), row=2, col=1)
                fig_kar.update_layout(**STYLE, height=480)
                fig_kar.update_xaxes(**AX)
                fig_kar.update_yaxes(**AX)
                fig_kar.update_yaxes(title_text="y (mm)", row=2, col=1)
                st.plotly_chart(fig_kar, use_container_width=True)

                # Diagramme Re vs St
                Re_sw  = np.logspace(2, 6, 300)
                St_sw  = np.where(Re_sw > 300, 0.198*(1-19.7/Re_sw), 0.12)
                fig_st = go.Figure()
                fig_st.add_trace(go.Scatter(x=Re_sw, y=St_sw, mode='lines',
                    name='St(Re)', line=dict(color='#7700ff', width=2.5)))
                fig_st.add_vline(x=Re_kar, line_color='#ffcc00', line_dash='dot',
                                 annotation_text=f"Re={Re_kar:.2e}")
                fig_st.update_layout(title="Strouhal vs Reynolds — Cylindre",
                    xaxis_title="Re", yaxis_title="St",
                    xaxis_type='log', **STYLE, height=300,
                    xaxis=dict(**AX, type='log'), yaxis=dict(**AX))
                st.plotly_chart(fig_st, use_container_width=True)

                df_kar = pd.DataFrame({"t (s)": t_kar, "y+ (m)": y_kar, "y- (m)": y_kar2})
                st.download_button("📥 Exporter sillage (CSV)", df_kar.to_csv(index=False),
                                   "sillage_karman.csv", "text/csv", key="dl_kar_dyn")

        # --- subC3 : Coup de bélier & Jet ---
        with subC3:
            st.markdown("### 🔬 Choc hydraulique & Impact de jet")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### 💥 Coup de bélier (Joukowski)")
                U0_bel = st.slider("Vitesse initiale U₀ (m/s)", 0.1, 20.0, 2.0, 0.1, key="U0_bel_dyn")
                L_bel  = st.slider("Longueur tuyau L (m)", 1.0, 1000.0, 100.0, 1.0, key="L_bel_dyn")
                D_bel  = st.slider("Diamètre D (m)", 0.01, 1.0, 0.1, 0.01, key="D_bel_dyn")
                e_bel  = st.slider("Épaisseur paroi e (mm)", 1.0, 50.0, 5.0, 0.5, key="e_bel_dyn") * 1e-3
                res_bel = eng_d.onde_choc_belier(U0_bel, L_bel, D_bel, e_bel)
                for k, vv in res_bel.items():
                    st.metric(k, f"{vv:.5g}")

                st.markdown("---")
                st.markdown("#### 💦 Jet impactant plaque plane")
                U_jet  = st.slider("U_jet (m/s)", 0.5, 50.0, 10.0, 0.5, key="Ujet_dyn")
                D_jet  = st.slider("D_jet (mm)", 1.0, 200.0, 20.0, 1.0, key="Djet_dyn") * 1e-3
                res_jet = eng_d.jet_impact(U_jet, D_jet)
                for k, vv in res_jet.items():
                    st.metric(k, f"{vv:.5g}")

            with col2:
                # Propagation onde de choc
                t_bel = np.linspace(0, 4*res_bel['t_retour (s)'], 500)
                p_bel = np.zeros_like(t_bel)
                T_ret = res_bel['t_retour (s)']
                dP    = res_bel['ΔP Joukowski (Pa)']
                for i, t in enumerate(t_bel):
                    cycle = t % (2*T_ret)
                    if cycle < T_ret:
                        p_bel[i] = 101325 + dP * np.exp(-cycle/T_ret*3)
                    else:
                        p_bel[i] = 101325 - dP * 0.3 * np.exp(-(cycle-T_ret)/T_ret*3)

                fig_bel = go.Figure()
                fig_bel.add_trace(go.Scatter(x=t_bel*1000, y=p_bel/1e5, mode='lines',
                    name='p(t) (bar)', line=dict(color='#ff4400', width=2.5)))
                fig_bel.add_hline(y=101325/1e5, line_color='rgba(255,255,255,0.3)',
                                  line_dash='dot', annotation_text="p_atm")
                fig_bel.add_hline(y=(101325+dP)/1e5, line_color='#ffcc00', line_dash='dash',
                                  annotation_text=f"+ΔP={dP/1e5:.2f}bar")
                fig_bel.update_layout(title="Coup de bélier — Surpression en fonction du temps",
                    xaxis_title="t (ms)", yaxis_title="p (bar)",
                    **STYLE, height=320, xaxis=dict(**AX), yaxis=dict(**AX))
                st.plotly_chart(fig_bel, use_container_width=True)

                # Force jet vs vitesse
                U_jet_sw = np.linspace(0.1, 50, 200)
                F_jet_sw = rho_d * np.pi*D_jet**2/4 * U_jet_sw**2
                fig_jet  = go.Figure()
                fig_jet.add_trace(go.Scatter(x=U_jet_sw, y=F_jet_sw, mode='lines',
                    name='F_impact (N)', line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.08)'))
                fig_jet.add_vline(x=U_jet, line_color='#ffcc00', line_dash='dot',
                                  annotation_text=f"U_jet={U_jet}m/s")
                fig_jet.update_layout(title="Force d'impact vs Vitesse du jet",
                    xaxis_title="U_jet (m/s)", yaxis_title="F (N)",
                    **STYLE, height=290, xaxis=dict(**AX), yaxis=dict(**AX))
                st.plotly_chart(fig_jet, use_container_width=True)

                df_bel = pd.DataFrame({"t (ms)": t_bel*1000, "p (bar)": p_bel/1e5})
                st.download_button("📥 Exporter choc hydraulique (CSV)", df_bel.to_csv(index=False),
                                   "coup_de_belier.csv", "text/csv", key="dl_bel_dyn")

                x_naca = np.linspace(-0.99, 0.99, 100)
                Cp_ext, Cp_int = eng_d.Cp_profil_NACA(5.0, x_naca)
                st.markdown("#### Cp approximatif NACA")
                st.metric("Cp ext max", f"{float(np.max(Cp_ext)):.3f}")
                st.metric("Cp int min", f"{float(np.min(Cp_int)):.3f}")
                st.markdown("#### Analyse d'adimensionnels (Pi)")
                st.json(eng_d.analyse_pi(rho_d, U_jet, D_jet, mu_d, 9.81, 0.072))

        # L'onglet Théorie a été supprimé pour éviter les erreurs de rendu.

