__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import jv
from scipy.integrate import quad
from scipy.fft import fft2, fftshift, ifft2
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
CONSTANTES_OPT = {
    "c (m/s)":          2.998e8,
    "λ_rouge (nm)":     650.0,
    "λ_vert (nm)":      550.0,
    "λ_bleu (nm)":      450.0,
    "n_verre":          1.500,
    "n_eau":            1.333,
    "n_air":            1.000,
    "I_0 (W/m²)":       1.0,
}

FORMULES_OPTIQUE = {
    # Interférences
    "Différence de marche 2 fentes":   r"\delta = \frac{d \cdot y}{D}",
    "Interfranges":                    r"i = \frac{\lambda D}{d}",
    "Condition frange brillante":      r"\delta = m\lambda,\quad m\in\mathbb{Z}",
    "Condition frange sombre":         r"\delta = (2m+1)\frac{\lambda}{2}",
    "Intensité 2 ondes cohérentes":    r"I = I_1+I_2+2\sqrt{I_1 I_2}\cos(\Delta\phi)",
    "Contraste (visibilité)":          r"\mathcal{V} = \frac{I_{max}-I_{min}}{I_{max}+I_{min}}",
    "Différence de phase":             r"\Delta\phi = \frac{2\pi}{\lambda}\delta = \frac{2\pi}{\lambda}(n_2 L_2 - n_1 L_1)",
    "Interféromètre Fabry-Pérot":      r"I_T = \frac{I_0}{1 + F\sin^2(\delta/2)},\quad F = \frac{4R}{(1-R)^2}",
    # Diffraction
    "Diffraction simple fente (FF)":   r"I(\theta) = I_0\left(\frac{\sin\alpha}{\alpha}\right)^2,\quad\alpha=\frac{\pi a\sin\theta}{\lambda}",
    "Réseau de diffraction":           r"d\sin\theta_m = m\lambda,\quad m=0,\pm1,\pm2\ldots",
    "Pouvoir de résolution réseau":    r"R = \frac{\lambda}{\Delta\lambda} = mN",
    "Critère de Rayleigh":             r"\theta_{min} = 1.22\frac{\lambda}{D}",
    "Ouverture numérique":             r"ON = n\sin\theta_{max}",
    "Diffraction de Fresnel":          r"U(P) = \frac{A}{i\lambda}\iint_{\Sigma}\frac{e^{ikr}}{r}K(\chi)\,d\Sigma",
    "Zone de Fresnel":                 r"r_m = \sqrt{m\lambda\frac{b(a+b)}{a}},\quad m=1,2,3\ldots",
    "Transformée de Fourier optique":  r"U(f_x,f_y) = \mathcal{F}\{t(x,y)\}_{f_x=x/\lambda F}",
    # Polarisation
    "Loi de Malus":                    r"I = I_0\cos^2\theta",
    "Onde polarisée rectiligne":       r"\mathbf{E} = E_0\cos(\omega t - kz)\,\hat{x}",
    "Onde elliptique":                 r"E_x = A\cos(\omega t),\quad E_y = B\cos(\omega t+\delta)",
    "Onde circulaire":                 r"E_x = A\cos(\omega t),\quad E_y = \pm A\sin(\omega t)",
    "Lame quart d'onde":               r"\delta = \frac{\pi}{2},\quad e = \frac{\lambda}{4(n_e-n_o)}",
    "Lame demi-onde":                  r"\delta = \pi,\quad e = \frac{\lambda}{2(n_e-n_o)}",
    "Degré de polarisation":           r"P = \frac{I_{max}-I_{min}}{I_{max}+I_{min}}",
}

CHAPITRES_OPTIQUE = {
    "1 — Interférences à 2 ondes":              "interferences",
    "2 — Diffraction (champs proche & lointain)":"diffraction",
    "3 — Polarisation & Théorème de Malus":     "polarisation",
}


def safe_latex(f_latex: str):
    """Affiche une formule LaTeX en essayant un rendu Markdown, puis `st.latex` en fallback.

    Certains environnements clients peuvent lever des erreurs DOM (NotFoundError)
    lors du rendu MathJax ; on essaie un rendu plus robuste pour éviter de casser
    les pages de formules.
    """
    # Render via Markdown first (more robust across clients), then fall
    # back to `st.latex` if Markdown fails. Finally fall back to plain text.
    try:
        latex_block = "$$" + f_latex + "$$"
        st.markdown(latex_block)
    except Exception:
        try:
            st.latex(f_latex)
        except Exception:
            st.write(f_latex)


# ============================================================
# MOTEUR OPTIQUE ONDULATOIRE
# ============================================================
class OptiqueEngine:
    """Moteur scientifique en optique ondulatoire."""

    def __init__(self, lam: float = 550e-9):
        self.lam = lam     # longueur d'onde (m)
        self.k = 2*np.pi/lam

    # -------------------------------------------------------
    # CHAPITRE 1 — INTERFÉRENCES À 2 ONDES
    # -------------------------------------------------------
    def interfranges(self, d: float, D: float) -> float:
        """i = λD/d"""
        return self.lam * D / d

    def intensite_2ondes(self, I1: float, I2: float,
                          delta_phi: np.ndarray) -> np.ndarray:
        """I = I₁ + I₂ + 2√(I₁I₂)·cos(Δφ)"""
        return I1 + I2 + 2*np.sqrt(I1*I2)*np.cos(delta_phi)

    def figure_interference_young(self, d: float, D: float,
                                   y: np.ndarray,
                                   I1: float = 1.0,
                                   I2: float = 1.0) -> np.ndarray:
        """Figure d'interférence de Young (2 fentes)."""
        delta = d * y / D
        delta_phi = 2*np.pi*delta / self.lam
        return self.intensite_2ondes(I1, I2, delta_phi)

    def visibilite(self, I1: float, I2: float) -> float:
        """V = (I_max - I_min)/(I_max + I_min)"""
        I_max = I1 + I2 + 2*np.sqrt(I1*I2)
        I_min = I1 + I2 - 2*np.sqrt(I1*I2)
        return (I_max - I_min)/(I_max + I_min) if (I_max+I_min) > 0 else 0

    def position_franges(self, d: float, D: float,
                          m_max: int = 10) -> tuple:
        """Positions des franges brillantes et sombres."""
        m = np.arange(-m_max, m_max+1)
        y_brillantes = m * self.lam * D / d
        y_sombres = (m + 0.5) * self.lam * D / d
        return y_brillantes, y_sombres

    def interference_lame_air(self, e: np.ndarray,
                               n: float = 1.0) -> np.ndarray:
        """Interférence en lame d'air (miroir de Lloyd)."""
        delta_phi = 4*np.pi*n*e/self.lam + np.pi  # +π pour réflexion
        return np.cos(delta_phi/2)**2

    def fabry_perot(self, delta: np.ndarray,
                    R: float) -> np.ndarray:
        """Transmittance du Fabry-Pérot."""
        F = 4*R/(1-R)**2
        return 1/(1 + F*np.sin(delta/2)**2)

    def coherence_temporelle(self, delta_lam: float) -> float:
        """Longueur de cohérence temporelle."""
        return self.lam**2 / delta_lam if delta_lam > 0 else np.inf

    def difference_marche_2fentes(self, d: float, D: float,
                                   y: np.ndarray) -> np.ndarray:
        """δ = d·y/D (approximation paraxiale)"""
        return d * y / D

    def difference_marche_exacte(self, d: float, D: float,
                                  y: np.ndarray) -> np.ndarray:
        """δ exact = r₂ - r₁ (sans approximation)"""
        r1 = np.sqrt(D**2 + (y - d/2)**2)
        r2 = np.sqrt(D**2 + (y + d/2)**2)
        return np.abs(r2 - r1)

    def N_fentes(self, d: float, D: float, y: np.ndarray,
                  N: int, a: float = None) -> np.ndarray:
        """Interférence à N fentes (réseau)."""
        delta_phi = 2*np.pi*d*y/(D*self.lam)
        # Terme de réseau
        denom_r = np.sinc(delta_phi/(2*np.pi) + 1e-15)
        numer_r = np.sin(N*delta_phi/2)
        denom_r2 = np.sin(delta_phi/2 + 1e-15)
        I_reseau = (numer_r/denom_r2)**2 / N**2
        if a is not None:
            # Enveloppe de diffraction
            alpha = np.pi*a*y/(D*self.lam)
            env = (np.sinc(alpha/np.pi + 1e-15))**2
            return I_reseau * env
        return I_reseau

    # -------------------------------------------------------
    # CHAPITRE 2 — DIFFRACTION
    # -------------------------------------------------------

    # --- Champ lointain (Fraunhofer) ---
    def diffraction_fente_simple(self, a: float,
                                  theta: np.ndarray) -> np.ndarray:
        """I(θ) = I₀·sinc²(α) — fente simple."""
        alpha = np.pi*a*np.sin(theta)/self.lam
        return (np.sinc(alpha/np.pi + 1e-15))**2

    def diffraction_fente_double(self, a: float, d: float,
                                  theta: np.ndarray) -> np.ndarray:
        """I(θ) — double fente avec diffraction."""
        alpha = np.pi*a*np.sin(theta)/self.lam
        beta  = np.pi*d*np.sin(theta)/self.lam
        env   = (np.sinc(alpha/np.pi + 1e-15))**2
        inter = np.cos(beta)**2
        return env * inter

    def diffraction_reseau(self, d_res: float, N_r: int,
                            theta: np.ndarray) -> np.ndarray:
        """Réseau de diffraction N fentes."""
        beta = np.pi*d_res*np.sin(theta)/self.lam
        numer = np.sin(N_r*beta + 1e-15)
        denom = np.sin(beta + 1e-15)
        return (numer/denom)**2 / N_r**2

    def ordres_reseau(self, d_res: float,
                       m_max: int = 5) -> pd.DataFrame:
        """Angles de diffraction des ordres du réseau."""
        rows = []
        for m in range(-m_max, m_max+1):
            sin_theta = m*self.lam/d_res
            if -1 <= sin_theta <= 1:
                theta = np.degrees(np.arcsin(sin_theta))
                rows.append({"m": m, "θ (°)": round(theta, 4),
                             "sin θ": round(sin_theta, 6)})
        return pd.DataFrame(rows)

    def resolution_rayleigh(self, D_ouv: float) -> float:
        """θ_min = 1.22·λ/D"""
        return 1.22 * self.lam / D_ouv

    def diffraction_circulaire_airy(self, D_ouv: float,
                                    theta: np.ndarray) -> np.ndarray:
        """Tache d'Airy — ouverture circulaire."""
        x = np.pi * D_ouv * np.sin(np.maximum(theta, 1e-10)) / self.lam
        I = (2 * jv(1, x + 1e-15) / (x + 1e-15))**2
        return I

    def pouvoir_resolution_reseau(self, m: int, N_r: int) -> float:
        """R = mN"""
        return m * N_r

    def TF_ouverture_2D(self, ouverture: np.ndarray,
                         dx: float, f_lentille: float) -> np.ndarray:
        """Diffraction Fraunhofer par TF 2D."""
        F = fft2(ouverture)
        F_shifted = fftshift(F)
        I = np.abs(F_shifted)**2
        return I / (I.max() + 1e-30)

    def ouverture_rectangulaire(self, N: int, a_px: int,
                                 b_px: int) -> np.ndarray:
        """Ouverture rectangulaire a×b."""
        t = np.zeros((N, N))
        cx, cy = N//2, N//2
        t[cy-b_px//2:cy+b_px//2, cx-a_px//2:cx+a_px//2] = 1
        return t

    def ouverture_circulaire(self, N: int, r_px: int) -> np.ndarray:
        """Ouverture circulaire de rayon r."""
        y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
        t = np.zeros((N, N))
        t[(x**2 + y**2) <= r_px**2] = 1
        return t

    def ouverture_N_fentes(self, N: int, n_fentes: int,
                            d_px: int, a_px: int) -> np.ndarray:
        """N fentes régulières."""
        t = np.zeros((N, N))
        cx = N//2
        for i in range(n_fentes):
            offset = int((i - (n_fentes-1)/2) * d_px)
            x0 = cx + offset - a_px//2
            x1 = cx + offset + a_px//2
            if 0 <= x0 and x1 < N:
                t[:, x0:x1] = 1
        return t

    # --- Champ proche (Fresnel) ---
    def zones_fresnel(self, a_dist: float, b_dist: float,
                       m_max: int = 10) -> np.ndarray:
        """Rayons des zones de Fresnel."""
        m = np.arange(1, m_max+1)
        return np.sqrt(m * self.lam * a_dist * b_dist / (a_dist + b_dist))

    def diffraction_fresnel_1D(self, y_ecran: np.ndarray,
                                a_dist: float, b_dist: float,
                                a_ouv: float) -> np.ndarray:
        """Diffraction de Fresnel 1D (fente)."""
        k = self.k
        def amplitude_point(y0):
            def integrand_re(x):
                r1 = np.sqrt(a_dist**2 + x**2)
                r2 = np.sqrt(b_dist**2 + (y0-x)**2)
                return np.cos(k*(r1+r2)) / (r1*r2)
            def integrand_im(x):
                r1 = np.sqrt(a_dist**2 + x**2)
                r2 = np.sqrt(b_dist**2 + (y0-x)**2)
                return np.sin(k*(r1+r2)) / (r1*r2)
            lim = a_ouv/2
            re_val, _ = quad(integrand_re, -lim, lim,
                             limit=100, epsabs=1e-6)
            im_val, _ = quad(integrand_im, -lim, lim,
                             limit=100, epsabs=1e-6)
            return complex(re_val, im_val)

        amplitude = np.array([amplitude_point(y) for y in y_ecran])
        return np.abs(amplitude)**2

    def parametre_Fresnel(self, a_dist: float, b_dist: float,
                           a_ouv: float) -> float:
        """F = a²/(λ·L_eff) — paramètre de Fresnel."""
        L_eff = a_dist*b_dist/(a_dist+b_dist)
        return (a_ouv/2)**2 / (self.lam * L_eff)

    def diffraction_bord(self, y: np.ndarray,
                          D: float, x_bord: float = 0.0) -> np.ndarray:
        """Diffraction au bord d'un écran (Fresnel)."""
        xi = (y - x_bord) * np.sqrt(2/(self.lam*D))
        from scipy.special import fresnel
        S, C = fresnel(xi)
        I = 0.5 * ((C + 0.5)**2 + (S + 0.5)**2)
        return I / I.max()

    # -------------------------------------------------------
    # CHAPITRE 3 — POLARISATION
    # -------------------------------------------------------
    def loi_malus(self, I0: float, theta: np.ndarray) -> np.ndarray:
        """I = I₀·cos²(θ)"""
        return I0 * np.cos(np.radians(theta))**2

    def polariseur_analyseur(self, I_init: float,
                              theta_pol: float,
                              theta_ana: float,
                              type_lumiere: str = "non polarisée") -> float:
        """Intensité après polariseur + analyseur."""
        if type_lumiere == "non polarisée":
            I_apres_pol = I_init / 2
        else:
            I_apres_pol = I_init * np.cos(np.radians(theta_pol))**2

        delta_theta = theta_ana - theta_pol
        return I_apres_pol * np.cos(np.radians(delta_theta))**2

    def onde_elliptique(self, Ex: float, Ey: float,
                         delta_phi: float, t: np.ndarray,
                         omega: float = 1.0) -> tuple:
        """Composantes du champ d'une onde elliptique."""
        e_x = Ex * np.cos(omega*t)
        e_y = Ey * np.cos(omega*t + delta_phi)
        return e_x, e_y

    def type_polarisation(self, Ex: float, Ey: float,
                           delta_phi: float) -> str:
        """Identifie le type de polarisation."""
        delta = np.radians(delta_phi)
        if abs(Ex) < 1e-10 or abs(Ey) < 1e-10:
            return "Polarisation rectiligne"
        elif abs(delta) < 0.05 or abs(abs(delta)-np.pi) < 0.05:
            return "Polarisation rectiligne (déphasage 0 ou π)"
        elif abs(abs(delta)-np.pi/2) < 0.05 and abs(Ex-Ey) < 1e-10:
            return "Polarisation circulaire"
        else:
            return "Polarisation elliptique"

    def matrice_jones(self, type_elem: str,
                       theta: float = 0.0) -> np.ndarray:
        """Matrice de Jones d'un élément optique."""
        t = np.radians(theta)
        if type_elem == "Polariseur horizontal":
            return np.array([[1, 0], [0, 0]])
        elif type_elem == "Polariseur vertical":
            return np.array([[0, 0], [0, 1]])
        elif type_elem == "Polariseur θ":
            return np.array([[np.cos(t)**2, np.sin(t)*np.cos(t)],
                             [np.sin(t)*np.cos(t), np.sin(t)**2]])
        elif type_elem == "Lame λ/4 axe horizontal":
            return np.array([[1, 0], [0, 1j]])
        elif type_elem == "Lame λ/2 axe horizontal":
            return np.array([[1, 0], [0, -1]])
        elif type_elem == "Miroir":
            return np.array([[1, 0], [0, -1]])
        return np.eye(2)

    def propagation_jones(self, E_in: np.ndarray,
                           elements: list) -> np.ndarray:
        """Propagation d'un champ de Jones à travers N éléments."""
        E = E_in.copy().astype(complex)
        for M in elements:
            E = M @ E
        return E

    def stokes(self, Ex: float, Ey: float,
               delta: float) -> np.ndarray:
        """Paramètres de Stokes S0, S1, S2, S3."""
        S0 = Ex**2 + Ey**2
        S1 = Ex**2 - Ey**2
        S2 = 2*Ex*Ey*np.cos(delta)
        S3 = 2*Ex*Ey*np.sin(delta)
        return np.array([S0, S1, S2, S3])

    def degre_polarisation(self, S: np.ndarray) -> float:
        """P = √(S₁²+S₂²+S₃²)/S₀"""
        return np.sqrt(S[1]**2+S[2]**2+S[3]**2) / (S[0]+1e-15)

    def sphere_poincare(self, theta_arr: np.ndarray,
                         delta_arr: np.ndarray) -> tuple:
        """Points sur la sphère de Poincaré."""
        X = np.cos(2*theta_arr)*np.cos(delta_arr)
        Y = np.cos(2*theta_arr)*np.sin(delta_arr)
        Z = np.sin(2*theta_arr)
        return X, Y, Z

    def epaisseur_lame_onde(self, n_e: float, n_o: float,
                             fraction: str = "λ/4") -> float:
        """Épaisseur de lame quart ou demi-onde."""
        dn = abs(n_e - n_o)
        if fraction == "λ/4":
            return self.lam / (4 * dn)
        elif fraction == "λ/2":
            return self.lam / (2 * dn)
        return self.lam / dn

    def angle_brewster(self, n1: float, n2: float) -> float:
        """tan(θ_B) = n₂/n₁"""
        return np.degrees(np.arctan(n2/n1))

    def reflectivite_fresnel(self, n1: float, n2: float,
                              theta_i: np.ndarray) -> tuple:
        """Coefficients de Fresnel rs, rp."""
        theta_r = np.arcsin(np.clip(n1*np.sin(np.radians(theta_i))/n2, -1, 1))
        rs = ((n1*np.cos(np.radians(theta_i)) - n2*np.cos(theta_r)) /
              (n1*np.cos(np.radians(theta_i)) + n2*np.cos(theta_r) + 1e-15))
        rp = ((n2*np.cos(np.radians(theta_i)) - n1*np.cos(theta_r)) /
              (n2*np.cos(np.radians(theta_i)) + n1*np.cos(theta_r) + 1e-15))
        Rs = rs**2
        Rp = rp**2
        return Rs, Rp, rs, rp


# ============================================================
# PAGE PRINCIPALE — OPTIQUE ONDULATOIRE
# ============================================================
def optique_ondulatoire_page():
    st.markdown("## 🔭 Optique Ondulatoire — Vue d'ensemble")
    st.markdown("*Interférences, Diffraction (Fraunhofer & Fresnel), Polarisation*")
    st.markdown("---")

    PLOT_LAYOUT = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.92)',
        font=dict(color='#1f2937'),
    )
    AXIS_STYLE = dict(gridcolor='rgba(148,163,184,0.18)', color='#1f2937')
    colors = ['#00ccff','#7700ff','#ff00cc','#00ff88',
               '#ffcc00','#ff4400','#88ccff']

    def lam_to_color(lam_nm: float) -> str:
        if lam_nm < 450:   return '#8800ff'
        elif lam_nm < 500: return '#0044ff'
        elif lam_nm < 560: return '#00cc44'
        elif lam_nm < 590: return '#aacc00'
        elif lam_nm < 625: return '#ffaa00'
        else:               return '#ff2200'

    def layout(fig, title="", xt="", yt="", h=420):
        fig.update_layout(**PLOT_LAYOUT, title=title,
                          xaxis_title=xt, yaxis_title=yt, height=h,
                          xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
                          legend=dict(bgcolor='rgba(0,0,0,0.5)'))
        return fig

    # Sélection option
    st.markdown("### 📚 Options — Optique")
    chapitre = st.radio(
        "Sélectionnez une option",
        list(CHAPITRES_OPTIQUE.keys()),
        index=0,
        key="chapitre_optique"
    )
    ch_key = CHAPITRES_OPTIQUE[chapitre]

    # ============================================================
    # CHAPITRE 1 — INTERFÉRENCES
    # ============================================================
    if ch_key == "interferences":
        st.markdown("## 📖 Section 1 — Interférences à 2 Ondes")
        st.markdown("""
        Deux ondes cohérentes se superposent : leur intensité résultante
        dépend de leur **différence de marche** δ et de leur
        **déphasage** Δφ = 2πδ/λ.
        """)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎛️ Fentes de Young",
            "🎨 Polychrom. & Cohérence",
            "📊 N fentes",
            "🔬 Fabry-Pérot",
            "💧 Lame d'air",
            "📖 Formules"
        ])

        # ---- TAB 1 : YOUNG ----
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                lam_nm = st.slider("λ (nm)", 380, 780, 550, 5)
                lam = lam_nm * 1e-9
                d_fentes = st.slider("d (mm)", 0.01, 5.0, 0.5, 0.01) * 1e-3
                D_ecran = st.slider("D (m)", 0.1, 5.0, 1.0, 0.05)
                I1_y = st.slider("I₁ (u.a.)", 0.1, 5.0, 1.0, 0.1)
                I2_y = st.slider("I₂ (u.a.)", 0.1, 5.0, 1.0, 0.1)
                y_max = st.slider("Fenêtre ±y (mm)", 0.5, 20.0, 5.0, 0.5)
                approx_parax = st.checkbox("Approximation paraxiale", True)

                eng_y = OptiqueEngine(lam)
                i_fr = eng_y.interfranges(d_fentes, D_ecran)
                V = eng_y.visibilite(I1_y, I2_y)
                y_br, y_som = eng_y.position_franges(d_fentes, D_ecran, 5)

                st.metric("i (interfranges, mm)", f"{i_fr*1000:.4f}")
                st.metric("Visibilité V", f"{V:.4f}")
                st.metric("δ à y=i/2", f"{d_fentes*i_fr/(2*D_ecran)*1e9:.2f} nm")

                st.markdown("**Franges brillantes (mm) :**")
                st.text(", ".join([f"{y*1000:.2f}" for y in y_br[:5]]))

            with col2:
                y = np.linspace(-y_max*1e-3, y_max*1e-3, 2000)
                if approx_parax:
                    I_y = eng_y.figure_interference_young(
                        d_fentes, D_ecran, y, I1_y, I2_y)
                else:
                    delta_ex = eng_y.difference_marche_exacte(
                        d_fentes, D_ecran, y)
                    dph = 2*np.pi*delta_ex/lam
                    I_y = eng_y.intensite_2ondes(I1_y, I2_y, dph)

                # Couleur selon λ
                col_lam = lam_to_color(lam_nm)
                fig_y = go.Figure()
                fig_y.add_trace(go.Scatter(x=y*1000, y=I_y, mode='lines',
                    name=f'I(y) λ={lam_nm}nm',
                    line=dict(color=col_lam, width=2.5),
                    fill='tozeroy',
                    fillcolor=col_lam.replace('#','rgba(').replace(
                        col_lam[1:], ','.join(str(int(col_lam[i:i+2],16))
                        for i in (1,3,5)))+',0.15)' if len(col_lam)==7 else 'tozeroy'))

                # Marquer les franges brillantes
                for y_b in y_br:
                    if abs(y_b*1000) <= y_max:
                        fig_y.add_vline(x=y_b*1000,
                            line_color='rgba(255,255,0,0.25)',
                            line_dash='dot')

                layout(fig_y, f"Interférences de Young (λ={lam_nm}nm, d={d_fentes*1000:.2f}mm)",
                       "y (mm)", "I (u.a.)")
                st.plotly_chart(fig_y, use_container_width=True)

                # Image 2D de la figure d'interférence
                st.markdown("#### 🖼️ Figure 2D")
                I_2D = np.tile(I_y, (80, 1))
                fig_2D = go.Figure(go.Heatmap(
                    z=I_2D,
                    x=y*1000, y=np.linspace(0, 1, 80),
                    colorscale=[[0,'#000000'],[1,col_lam]],
                    showscale=False
                ))
                fig_2D.update_layout(**PLOT_LAYOUT,
                    xaxis=dict(**AXIS_STYLE, title="y (mm)"),
                    yaxis=dict(showticklabels=False, showgrid=False),
                    height=180, margin=dict(l=60,r=10,t=10,b=40))
                st.plotly_chart(fig_2D, use_container_width=True)

        # ---- TAB 2 : POLYCHROMATIQUE ----
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                d_pc = st.slider("d (mm)", 0.01, 5.0, 0.5, 0.01,
                                  key="d_pc") * 1e-3
                D_pc = st.slider("D (m)", 0.1, 5.0, 1.0, 0.1, key="D_pc")
                y_max_pc = st.slider("±y (mm)", 0.5, 20.0, 5.0, 0.5, key="ym_pc")
                type_src = st.selectbox("Source", [
                    "Monochromatique",
                    "Doublet (rouge+bleu)",
                    "Lumière blanche (RGB)",
                    "Quasi-monochromatique"
                ])
                lc = None
                if type_src == "Quasi-monochromatique":
                    dlam = st.slider("Δλ (nm)", 0.1, 50.0, 5.0, 0.5)
                    lam_c = st.slider("λ₀ (nm)", 400, 700, 550, 5)
                    lc = (lam_c*1e-9)**2 / (dlam*1e-9)
                    st.metric("L_cohérence (μm)", f"{lc*1e6:.2f}")

            with col2:
                y = np.linspace(-y_max_pc*1e-3, y_max_pc*1e-3, 2000)
                fig_pc = go.Figure()

                if type_src == "Monochromatique":
                    lam_sel = st.slider("λ (nm)", 380, 780, 550, 5,
                                        key="lam_mono")
                    eng_pc = OptiqueEngine(lam_sel*1e-9)
                    I_pc = eng_pc.figure_interference_young(d_pc, D_pc, y)
                    fig_pc.add_trace(go.Scatter(x=y*1000, y=I_pc, mode='lines',
                        line=dict(color=lam_to_color(lam_sel), width=2.5),
                        name=f'λ={lam_sel}nm'))

                elif type_src == "Doublet (rouge+bleu)":
                    for lam_i, nom_i in [(650e-9,"Rouge 650nm"),
                                          (450e-9,"Bleu 450nm")]:
                        eng_i = OptiqueEngine(lam_i)
                        I_i = eng_i.figure_interference_young(d_pc, D_pc, y)
                        fig_pc.add_trace(go.Scatter(x=y*1000, y=I_i,
                            mode='lines', name=nom_i,
                            line=dict(color=lam_to_color(int(lam_i*1e9)),
                                      width=2)))
                    # Somme
                    I_sum = (OptiqueEngine(650e-9).figure_interference_young(d_pc,D_pc,y)+
                             OptiqueEngine(450e-9).figure_interference_young(d_pc,D_pc,y))/2
                    fig_pc.add_trace(go.Scatter(x=y*1000, y=I_sum,
                        mode='lines', name='Somme',
                        line=dict(color='#ffffff', width=2.5, dash='dash')))

                elif type_src == "Lumière blanche (RGB)":
                    lams_rgb = [(650e-9,"R",'#ff2200'),
                                 (550e-9,"G",'#00cc44'),
                                 (450e-9,"B",'#0044ff')]
                    I_total = np.zeros_like(y)
                    for lam_i, nom_i, col_i in lams_rgb:
                        eng_i = OptiqueEngine(lam_i)
                        I_i = eng_i.figure_interference_young(d_pc, D_pc, y)
                        I_total += I_i/3
                        fig_pc.add_trace(go.Scatter(x=y*1000, y=I_i/3,
                            mode='lines', name=f'λ={int(lam_i*1e9)}nm',
                            line=dict(color=col_i, width=1.5, dash='dot')))
                    fig_pc.add_trace(go.Scatter(x=y*1000, y=I_total,
                        mode='lines', name='Blanc (somme)',
                        line=dict(color='#ffffff', width=2.5)))

                else:  # Quasi-mono
                    lam_0 = 550e-9
                    eng_qm = OptiqueEngine(lam_0)
                    I_qm = eng_qm.figure_interference_young(d_pc, D_pc, y)
                    # Enveloppe de cohérence
                    if lc:
                        env_coh = np.sinc(d_pc*y/(D_pc*lc))
                        I_qm = I_qm * env_coh**2
                    fig_pc.add_trace(go.Scatter(x=y*1000, y=I_qm,
                        mode='lines', name='Quasi-mono',
                        line=dict(color='#00ccff', width=2.5)))

                layout(fig_pc, f"Figure d'interférence — {type_src}",
                       "y (mm)", "I (u.a.)")
                st.plotly_chart(fig_pc, use_container_width=True)

        # ---- TAB 3 : N FENTES ----
        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                lam_n = st.slider("λ (nm)", 380, 780, 550, 5, key="lam_n") * 1e-9
                N_fentes = st.slider("N (fentes)", 1, 20, 5, key="N_f")
                d_n = st.slider("d (mm)", 0.01, 5.0, 1.0, 0.05, key="d_n") * 1e-3
                D_n = st.slider("D (m)", 0.1, 5.0, 1.0, 0.1, key="D_n")
                a_n = st.slider("a (largeur fente, μm)", 1.0, 500.0, 100.0, 1.0)
                y_max_n = st.slider("±y (mm)", 1.0, 30.0, 10.0, 0.5, key="ymn")
                show_enveloppe = st.checkbox("Enveloppe de diffraction", True)

                eng_n = OptiqueEngine(lam_n)
                i_n = eng_n.interfranges(d_n, D_n)
                st.metric("Interfranges (mm)", f"{i_n*1000:.4f}")
                st.metric("Largeur pic principal (mm)",
                          f"{2*i_n*1000/N_fentes:.4f}")

            with col2:
                y_n = np.linspace(-y_max_n*1e-3, y_max_n*1e-3, 3000)
                I_n = eng_n.N_fentes(d_n, D_n, y_n, N_fentes,
                                      a_n*1e-6 if show_enveloppe else None)
                I_env = eng_n.diffraction_fente_simple(
                    a_n*1e-6, np.arctan(y_n/D_n))

                fig_n = go.Figure()
                fig_n.add_trace(go.Scatter(x=y_n*1000, y=I_n, mode='lines',
                    name=f'N={N_fentes} fentes',
                    line=dict(color='#00ccff', width=2.5)))
                if show_enveloppe:
                    fig_n.add_trace(go.Scatter(x=y_n*1000, y=I_env, mode='lines',
                        name='Enveloppe (sinc²)',
                        line=dict(color='#ffcc00', width=2, dash='dash')))
                layout(fig_n, f"Réseau {N_fentes} fentes (d={d_n*1000:.2f}mm)",
                       "y (mm)", "I (u.a.)")
                st.plotly_chart(fig_n, use_container_width=True)

                # Ordres de diffraction
                df_ord = eng_n.ordres_reseau(d_n, m_max=5)
                st.markdown("#### 📐 Ordres de diffraction")
                st.dataframe(df_ord, use_container_width=True)

        # ---- TAB 4 : FABRY-PÉROT ----
        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                R_fp = st.slider("Réflectivité R", 0.1, 0.999, 0.9, 0.001)
                lam_fp = st.slider("λ (nm)", 380, 780, 550, 5, key="l_fp") * 1e-9
                e_fp = st.slider("Épaisseur e (μm)", 0.1, 1000.0, 100.0, 0.5) * 1e-6
                n_fp = st.slider("Indice n", 1.0, 3.0, 1.0, 0.01)

                eng_fp = OptiqueEngine(lam_fp)
                F_fp = 4*R_fp/(1-R_fp)**2
                finesse = np.pi*np.sqrt(R_fp)/(1-R_fp)
                delta_nu = 3e8/(2*n_fp*e_fp)
                resolution = lam_fp/(2*n_fp*e_fp/1000)

                st.metric("Coefficient F", f"{F_fp:.2f}")
                st.metric("Finesse ℱ", f"{finesse:.2f}")
                st.metric("ISL Δν (GHz)", f"{delta_nu/1e9:.3f}")

                lam_theo = 2*n_fp*e_fp/round(2*n_fp*e_fp/lam_fp)
                st.metric("λ résonance (nm)", f"{lam_theo*1e9:.4f}")

            with col2:
                delta_arr = np.linspace(0, 6*np.pi, 2000)
                I_fp = eng_fp.fabry_perot(delta_arr, R_fp)

                fig_fp = go.Figure()
                fig_fp.add_trace(go.Scatter(x=delta_arr/np.pi, y=I_fp,
                    mode='lines', name=f'R={R_fp}',
                    line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'))
                fig_fp.add_hline(y=0.5, line_color='#ffcc00', line_dash='dash',
                                  annotation_text="T=0.5 (BW)")
                layout(fig_fp, "Transmittance Fabry-Pérot",
                       "δ/π", "T(δ)")
                st.plotly_chart(fig_fp, use_container_width=True)

                # Comparaison R
                fig_fr2 = go.Figure()
                for Ri, col_i in zip([0.3, 0.5, 0.8, 0.9, 0.99], colors):
                    I_ri = eng_fp.fabry_perot(delta_arr, Ri)
                    fig_fr2.add_trace(go.Scatter(x=delta_arr/np.pi, y=I_ri,
                        mode='lines', name=f'R={Ri}',
                        line=dict(color=col_i, width=2)))
                layout(fig_fr2, "Fabry-Pérot — comparaison R",
                       "δ/π", "T(δ)")
                st.plotly_chart(fig_fr2, use_container_width=True)

        # ---- TAB 5 : LAME D'AIR ----
        with tab5:
            col1, col2 = st.columns([1, 2])
            with col1:
                lam_la = st.slider("λ (nm)", 380, 780, 550, 5, key="l_la") * 1e-9
                e_max = st.slider("e_max (μm)", 0.1, 100.0, 10.0, 0.1)
                n_la = st.slider("n (lame)", 1.0, 3.0, 1.0, 0.01)
                type_lame = st.selectbox("Géométrie", [
                    "Lame planparallèle",
                    "Coin d'air (Anneaux de Newton)"
                ])
                eng_la = OptiqueEngine(lam_la)
                st.metric("Ordre max m", f"{int(2*n_la*e_max*1e-6/lam_la)}")
                st.metric("Anneau brillant r₁ (μm)",
                          f"{np.sqrt(lam_la*1e-6)*1e6:.2f}" if type_lame=="Coin d'air (Anneaux de Newton)" else "N/A")

            with col2:
                e_arr = np.linspace(0, e_max*1e-6, 2000)
                I_la = eng_la.interference_lame_air(e_arr, n_la)

                fig_la = go.Figure()
                fig_la.add_trace(go.Scatter(x=e_arr*1e9, y=I_la, mode='lines',
                    name='I(e)', line=dict(color='#00ccff', width=2.5)))
                layout(fig_la, "Interférence lame d'air",
                       "e (nm)", "I (u.a.)")
                st.plotly_chart(fig_la, use_container_width=True)

                if type_lame == "Coin d'air (Anneaux de Newton)":
                    R_newton = st.slider("Rayon de courbure R (m)",
                                         0.1, 10.0, 1.0, 0.1)
                    r = np.linspace(0, 5e-3, 500)
                    e_r = r**2/(2*R_newton)
                    I_newton = eng_la.interference_lame_air(e_r, 1)
                    # Image 2D
                    N_im = 300
                    x_im = np.linspace(-5e-3, 5e-3, N_im)
                    X_im, Y_im = np.meshgrid(x_im, x_im)
                    r_im = np.sqrt(X_im**2 + Y_im**2)
                    e_im = r_im**2/(2*R_newton)
                    I_im = eng_la.interference_lame_air(e_im, 1)
                    fig_new = go.Figure(go.Heatmap(
                        z=I_im, x=x_im*1000, y=x_im*1000,
                        colorscale=[[0,'#000000'],[1,'#ffffff']],
                        showscale=False
                    ))
                    fig_new.update_layout(**PLOT_LAYOUT,
                        title="Anneaux de Newton",
                        xaxis=dict(**AXIS_STYLE, title="x (mm)"),
                        yaxis=dict(**AXIS_STYLE, title="y (mm)"),
                        height=380)
                    st.plotly_chart(fig_new, use_container_width=True)

        # ---- TAB 6 : FORMULES ----
        with tab6:
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_OPTIQUE.items():
                if any(k in nom for k in ["Interfér","frange","Contrast",
                                           "phase","Fabry","cohér",
                                           "marche","2 ondes"]):
                    with cols[col_idx % 2]:
                        st.markdown(f"**{nom}**")
                        safe_latex(f_latex)
                    col_idx += 1

    # ============================================================
    # CHAPITRE 2 — DIFFRACTION
    # ============================================================
    elif ch_key == "diffraction":
        st.markdown("## 📖 Section 2 — Diffraction")
        st.markdown("""
        La diffraction résulte de la nature ondulatoire de la lumière.
        Deux régimes selon le paramètre de Fresnel **F = a²/(λL)** :
        - **F ≫ 1** → Fresnel (champ proche)
        - **F ≪ 1** → Fraunhofer (champ lointain)
        """)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔭 Fraunhofer — fente",
            "🎯 Réseau & Résolution",
            "💿 TF optique 2D",
            "🌊 Fresnel (champ proche)",
            "💡 Tache d'Airy",
            "📖 Formules"
        ])

        # ---- TAB 1 : FRAUNHOFER FENTE ----
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                lam_fr = st.slider("λ (nm)", 380, 780, 550, 5,
                                    key="lam_fr") * 1e-9
                a_fr = st.slider("a (largeur fente, μm)", 1.0, 1000.0,
                                  100.0, 1.0) * 1e-6
                type_ouv = st.radio("Ouverture",
                    ["Simple fente", "Double fente"], horizontal=True)

                eng_fr = OptiqueEngine(lam_fr)
                theta_min = np.degrees(np.arcsin(lam_fr/a_fr))
                theta_max = np.degrees(np.arcsin(min(2*lam_fr/a_fr, 0.9999)))

                st.metric("1er zéro θ₁ (°)", f"{theta_min:.4f}")
                st.metric("2ème zéro θ₂ (°)", f"{theta_max:.4f}")
                st.metric("Largeur tache centrale (°)",
                          f"{2*theta_min:.4f}")

                if type_ouv == "Double fente":
                    d_fr2 = st.slider("d (écartement, mm)", 0.01, 5.0,
                                       1.0, 0.01, key="d_fr2") * 1e-3

            with col2:
                theta = np.linspace(-10, 10, 4000)
                theta_rad = np.radians(theta)

                if type_ouv == "Simple fente":
                    I_fr = eng_fr.diffraction_fente_simple(a_fr, theta_rad)
                    titre_fr = f"Diffraction Fraunhofer — fente a={a_fr*1e6:.0f}μm"
                else:
                    I_fr = eng_fr.diffraction_fente_double(a_fr, d_fr2, theta_rad)
                    I_env = eng_fr.diffraction_fente_simple(a_fr, theta_rad)
                    titre_fr = f"Double fente a={a_fr*1e6:.0f}μm, d={d_fr2*1e3:.2f}mm"

                fig_fr_d = go.Figure()
                fig_fr_d.add_trace(go.Scatter(x=theta, y=I_fr, mode='lines',
                    name='I(θ)', line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'))
                if type_ouv == "Double fente":
                    fig_fr_d.add_trace(go.Scatter(x=theta, y=I_env, mode='lines',
                        name='Enveloppe',
                        line=dict(color='#ffcc00', width=2, dash='dash')))

                # Zeros
                n_zeros = 3
                for m in range(1, n_zeros+1):
                    sin_z = m*lam_fr/a_fr
                    if sin_z <= 1:
                        theta_z = np.degrees(np.arcsin(sin_z))
                        for pm in [1, -1]:
                            fig_fr_d.add_vline(x=pm*theta_z,
                                line_color='rgba(255,100,0,0.4)',
                                line_dash='dot',
                                annotation_text=f"m={pm*m}" if pm>0 else "")

                layout(fig_fr_d, titre_fr, "θ (°)", "I(θ)/I₀")
                st.plotly_chart(fig_fr_d, use_container_width=True)

                # Vue 2D
                I_2D_fr = np.tile(I_fr, (60, 1))
                fig_2D_fr = go.Figure(go.Heatmap(
                    z=I_2D_fr, x=theta, y=np.linspace(0,1,60),
                    colorscale=[[0,'#000000'],[1,'#ffffff']],
                    showscale=False
                ))
                fig_2D_fr.update_layout(**PLOT_LAYOUT,
                    xaxis=dict(**AXIS_STYLE, title="θ (°)"),
                    yaxis=dict(showticklabels=False, showgrid=False),
                    height=150, margin=dict(l=60,r=10,t=10,b=40))
                st.plotly_chart(fig_2D_fr, use_container_width=True)

        # ---- TAB 2 : RÉSEAU ----
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                lam_re = st.slider("λ (nm)", 380, 780, 550, 5,
                                    key="lam_re") * 1e-9
                N_re = st.slider("N (fentes)", 2, 1000, 10, 1, key="N_re")
                d_re = st.slider("d (μm)", 0.5, 100.0, 2.0, 0.1,
                                  key="d_re") * 1e-6
                theta_max_r = st.slider("θ_max (°)", 5.0, 90.0, 45.0, 1.0)

                eng_re = OptiqueEngine(lam_re)
                df_ord_re = eng_re.ordres_reseau(d_re, m_max=5)
                R_res = eng_re.pouvoir_resolution_reseau(1, N_re)

                st.metric("R = mN (m=1)", f"{R_res}")
                st.metric("Δλ min résolu (nm)",
                          f"{lam_re*1e9/R_res:.4f}" if R_res > 0 else "∞")
                st.dataframe(df_ord_re, use_container_width=True)

            with col2:
                theta_r = np.linspace(-theta_max_r, theta_max_r, 5000)
                theta_r_rad = np.radians(theta_r)
                I_re = eng_re.diffraction_reseau(d_re, N_re, theta_r_rad)

                fig_re = go.Figure()
                fig_re.add_trace(go.Scatter(x=theta_r, y=I_re, mode='lines',
                    name=f'N={N_re}', line=dict(color='#00ccff', width=2.5)))

                # Marquer les ordres
                for _, row in df_ord_re.iterrows():
                    if abs(row["θ (°)"]) <= theta_max_r:
                        fig_re.add_vline(x=row["θ (°)"],
                            line_color='rgba(255,200,0,0.4)',
                            line_dash='dash',
                            annotation_text=f"m={row['m']}")

                layout(fig_re, f"Réseau N={N_re}, d={d_re*1e6:.1f}μm",
                       "θ (°)", "I(θ)/I₀")
                st.plotly_chart(fig_re, use_container_width=True)

                # Polychromatic réseau
                st.markdown("#### 🌈 Réseau — lumière blanche")
                fig_poly = go.Figure()
                for lam_i_nm, col_i in [(450,'#0044ff'),(520,'#00cc44'),
                                          (589,'#ffaa00'),(650,'#ff2200')]:
                    eng_i_r = OptiqueEngine(lam_i_nm*1e-9)
                    I_i_r = eng_i_r.diffraction_reseau(d_re, N_re, theta_r_rad)
                    fig_poly.add_trace(go.Scatter(x=theta_r, y=I_i_r,
                        mode='lines', name=f'{lam_i_nm}nm',
                        line=dict(color=col_i, width=1.5)))
                layout(fig_poly, "Réseau — lumière polychromatique",
                       "θ (°)", "I(θ)")
                st.plotly_chart(fig_poly, use_container_width=True)

        # ---- TAB 3 : TF OPTIQUE 2D ----
        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                type_ouv_2D = st.selectbox("Ouverture", [
                    "Rectangle", "Cercle",
                    "N fentes", "Réseau croisé"
                ])
                N_grid = st.slider("Résolution grille", 64, 512, 256, 64)
                eng_tf = OptiqueEngine(550e-9)

                if type_ouv_2D == "Rectangle":
                    a_px = st.slider("a (px)", 1, N_grid//4, 20)
                    b_px = st.slider("b (px)", 1, N_grid//4, 40)
                    ouv = eng_tf.ouverture_rectangulaire(N_grid, a_px, b_px)
                elif type_ouv_2D == "Cercle":
                    r_px = st.slider("r (px)", 1, N_grid//4, 20)
                    ouv = eng_tf.ouverture_circulaire(N_grid, r_px)
                elif type_ouv_2D == "N fentes":
                    n_f_2D = st.slider("N fentes", 2, 10, 5, key="nf2D")
                    d_f_2D = st.slider("d (px)", 5, N_grid//8, 20)
                    a_f_2D = st.slider("a (px)", 1, d_f_2D//2, 5)
                    ouv = eng_tf.ouverture_N_fentes(N_grid, n_f_2D,
                                                     d_f_2D, a_f_2D)
                else:  # Réseau croisé
                    r_2D = st.slider("r (px)", 1, N_grid//4, 15)
                    ouv = eng_tf.ouverture_circulaire(N_grid, r_2D)
                    # Croisé = cercle × cercle transposé
                    ouv2 = ouv.T
                    ouv = np.maximum(ouv, ouv2)

            with col2:
                I_TF = eng_tf.TF_ouverture_2D(ouv, 1.0, 1.0)
                I_TF_log = np.log10(I_TF + 1e-6)

                fig_2col = make_subplots(rows=1, cols=2,
                    subplot_titles=["Ouverture t(x,y)",
                                    "Diffraction |TF|² (log)"])
                fig_2col.add_trace(go.Heatmap(z=ouv,
                    colorscale=[[0,'#000000'],[1,'#ffffff']],
                    showscale=False), row=1, col=1)
                fig_2col.add_trace(go.Heatmap(z=I_TF_log,
                    colorscale=[[0,'#020817'],[0.5,'#7700ff'],
                                [0.8,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                    colorbar=dict(title='log I',
                                  tickfont=dict(color='#c0d0ff'))),
                    row=1, col=2)
                fig_2col.update_layout(**PLOT_LAYOUT,
                    title="Diffraction 2D (TF optique)",
                    height=420)
                fig_2col.update_xaxes(showticklabels=False, showgrid=False)
                fig_2col.update_yaxes(showticklabels=False, showgrid=False)
                st.plotly_chart(fig_2col, use_container_width=True)

                # Coupe centrale
                centre = I_TF[N_grid//2, :]
                fig_coupe = go.Figure()
                fig_coupe.add_trace(go.Scatter(
                    x=np.arange(N_grid)-N_grid//2, y=centre,
                    mode='lines', line=dict(color='#00ccff', width=2.5)))
                layout(fig_coupe, "Coupe horizontale",
                       "u (pixels fréquence)", "I (u.a.)", h=280)
                fig_coupe.update_yaxes(type='log')
                st.plotly_chart(fig_coupe, use_container_width=True)

        # ---- TAB 4 : FRESNEL ----
        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                lam_fn = st.slider("λ (nm)", 380, 780, 550, 5,
                                    key="lam_fn") * 1e-9
                a_fn = st.slider("a (demi-ouverture, mm)", 0.01, 5.0,
                                  0.5, 0.01) * 1e-3
                b_dist = st.slider("b (dist. observateur, m)",
                                    0.01, 10.0, 1.0, 0.05)
                a_src = st.slider("a (dist. source, m)",
                                   0.01, 100.0, 10.0, 0.1)

                eng_fn = OptiqueEngine(lam_fn)
                F_param = eng_fn.parametre_Fresnel(a_src, b_dist, a_fn*2)
                zones = eng_fn.zones_fresnel(a_src, b_dist, 5)

                st.metric("Paramètre de Fresnel F", f"{F_param:.3f}")
                regime = "Fraunhofer" if F_param < 0.1 else \
                         "Fresnel" if F_param < 10 else "Ombre géométrique"
                st.metric("Régime", regime)
                st.markdown("**Rayons zones de Fresnel (mm) :**")
                for i, r_z in enumerate(zones[:5]):
                    st.text(f"  r_{i+1} = {r_z*1000:.3f} mm")

            with col2:
                # Diffraction au bord
                y_bord = np.linspace(-5e-3, 5e-3, 400)
                I_bord = eng_fn.diffraction_bord(y_bord, b_dist)

                fig_fn_d = go.Figure()
                fig_fn_d.add_trace(go.Scatter(x=y_bord*1000, y=I_bord,
                    mode='lines', name='I (bord écran)',
                    line=dict(color='#00ccff', width=2.5)))
                fig_fn_d.add_vline(x=0, line_color='#ffcc00',
                                   line_dash='dash',
                                   annotation_text="Bord")
                fig_fn_d.add_hline(y=0.25,
                    line_color='rgba(255,100,0,0.4)',
                    line_dash='dot', annotation_text="I₀/4")
                layout(fig_fn_d, "Diffraction au bord (Fresnel)",
                       "y (mm)", "I/I₀")
                st.plotly_chart(fig_fn_d, use_container_width=True)

                # Passage Fresnel → Fraunhofer
                st.markdown("#### 🔄 Transition Fresnel → Fraunhofer")
                b_arr = np.logspace(-2, 2, 200)
                F_arr = np.array([eng_fn.parametre_Fresnel(a_src, b_i, a_fn*2)
                                   for b_i in b_arr])
                fig_trans = go.Figure()
                fig_trans.add_trace(go.Scatter(x=b_arr, y=F_arr, mode='lines',
                    line=dict(color='#00ccff', width=2.5), name='F(b)'))
                fig_trans.add_hline(y=1, line_color='#ffcc00', line_dash='dash',
                                    annotation_text="F=1 (transition)")
                fig_trans.add_hline(y=0.1, line_color='#00ff88', line_dash='dot',
                                    annotation_text="F=0.1 (Fraunhofer)")
                fig_trans.add_vline(x=b_dist, line_color='#ff00cc',
                                    line_dash='dot',
                                    annotation_text=f"b={b_dist}m")
                fig_trans.update_layout(**PLOT_LAYOUT,
                    xaxis=dict(**AXIS_STYLE, type='log', title="b (m)"),
                    yaxis=dict(**AXIS_STYLE, type='log', title="F"),
                    height=300,
                    title="Paramètre de Fresnel vs distance")
                st.plotly_chart(fig_trans, use_container_width=True)

        # ---- TAB 5 : AIRY ----
        with tab5:
            col1, col2 = st.columns([1, 2])
            with col1:
                lam_ay = st.slider("λ (nm)", 380, 780, 550, 5,
                                    key="lam_ay") * 1e-9
                D_ay = st.slider("D (diamètre, mm)", 0.5, 200.0, 10.0, 0.5) * 1e-3
                f_ay = st.slider("f (focale, mm)", 1.0, 1000.0, 100.0, 1.0) * 1e-3
                D_ay2 = st.slider("D₂ (2ème instrument, mm)", 0.5, 200.0,
                                   5.0, 0.5) * 1e-3

                eng_ay = OptiqueEngine(lam_ay)
                theta_min_ay = eng_ay.resolution_rayleigh(D_ay)
                r_tache = theta_min_ay * f_ay
                theta_min2 = eng_ay.resolution_rayleigh(D_ay2)

                st.metric("θ_min Rayleigh (μrad)", f"{theta_min_ay*1e6:.4f}")
                st.metric("r tache Airy (μm)", f"{r_tache*1e6:.4f}")
                st.metric("θ_min D₂ (μrad)", f"{theta_min2*1e6:.4f}")
                st.metric("Gain résolution", f"{theta_min2/theta_min_ay:.2f}×")
                st.metric("ON = n·sin(θ_max)", f"{np.sin(np.arctan(D_ay/(2*f_ay))):.4f}")

            with col2:
                theta_ay = np.linspace(0, 10*theta_min_ay, 2000)
                I_ay = eng_ay.diffraction_circulaire_airy(D_ay, theta_ay)
                I_ay2 = eng_ay.diffraction_circulaire_airy(D_ay2,
                    theta_ay + theta_min_ay)

                fig_ay = go.Figure()
                fig_ay.add_trace(go.Scatter(
                    x=theta_ay*1e6, y=I_ay, mode='lines',
                    name=f'D={D_ay*1000:.0f}mm',
                    line=dict(color='#00ccff', width=2.5)))
                fig_ay.add_trace(go.Scatter(
                    x=theta_ay*1e6, y=I_ay2, mode='lines',
                    name=f'D={D_ay2*1000:.0f}mm (décalé)',
                    line=dict(color='#7700ff', width=2, dash='dash')))
                fig_ay.add_trace(go.Scatter(
                    x=theta_ay*1e6, y=(I_ay+I_ay2)/2, mode='lines',
                    name='Somme', line=dict(color='#ffffff', width=1.5,
                    dash='dot')))
                fig_ay.add_vline(x=theta_min_ay*1e6, line_color='#ffcc00',
                                  line_dash='dash',
                                  annotation_text=f"θ_R={theta_min_ay*1e6:.1f}μrad")
                layout(fig_ay, f"Tache d'Airy & critère de Rayleigh",
                       "θ (μrad)", "I(θ)/I₀")
                st.plotly_chart(fig_ay, use_container_width=True)

                # Image 2D tache d'Airy
                N_aim = 200
                x_aim = np.linspace(-15*theta_min_ay, 15*theta_min_ay, N_aim)
                X_ai, Y_ai = np.meshgrid(x_aim, x_aim)
                R_ai = np.sqrt(X_ai**2 + Y_ai**2)
                I_aim = eng_ay.diffraction_circulaire_airy(D_ay, R_ai)
                fig_aim = go.Figure(go.Heatmap(
                    z=I_aim, x=x_aim*1e6, y=x_aim*1e6,
                    colorscale=[[0,'#000000'],[0.3,'#000080'],
                                [0.6,'#0044ff'],[1,'#ffffff']],
                    showscale=True,
                    colorbar=dict(tickfont=dict(color='#c0d0ff'))
                ))
                fig_aim.update_layout(**PLOT_LAYOUT,
                    title="Tache d'Airy 2D",
                    xaxis=dict(**AXIS_STYLE, title="x (μrad)"),
                    yaxis=dict(**AXIS_STYLE, title="y (μrad)"),
                    height=380)
                st.plotly_chart(fig_aim, use_container_width=True)

        # Les formules pour la section Diffraction ont été supprimées
        # (présentaient des erreurs côté client MathJax).

    # ============================================================
    # CHAPITRE 3 — POLARISATION & MALUS
    # ============================================================
    elif ch_key == "polarisation":
        st.markdown("## 📖 Section 3 — Polarisation & Théorème de Malus")
        st.markdown("""
        La polarisation décrit l'orientation du champ électrique **E**.
        - **Rectiligne** : E oscille dans un plan fixe
        - **Circulaire** : E tourne uniformément (|Ex|=|Ey|, δ=π/2)
        - **Elliptique** : cas général
        """)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📐 Loi de Malus",
            "🌀 État de polarisation",
            "🔵 Sphère de Poincaré",
            "🧮 Matrices de Jones",
            "🔬 Angle de Brewster",
            "📖 Formules"
        ])

        # ---- TAB 1 : MALUS ----
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                I0_mal = st.slider("I₀ (intensité incidente, W/m²)",
                                    0.1, 100.0, 10.0, 0.1)
                type_lumiere = st.selectbox("Type de lumière",
                    ["Non polarisée", "Polarisée rectiligne",
                     "Partiellement polarisée"])
                degree_pol = 1.0
                if type_lumiere == "Partiellement polarisée":
                    degree_pol = st.slider("Degré de polarisation P",
                                           0.0, 1.0, 0.5, 0.01)

                theta_pol = st.slider("θ_polariseur (°)", 0, 180, 0, 5)
                theta_ana = st.slider("θ_analyseur (°)", 0, 180, 45, 5)

                delta_theta = theta_ana - theta_pol

                if type_lumiere == "Non polarisée":
                    I_apres_pol = I0_mal / 2
                elif type_lumiere == "Polarisée rectiligne":
                    I_apres_pol = I0_mal * np.cos(np.radians(theta_pol))**2
                else:
                    I_apres_pol = I0_mal * (1 - degree_pol)/2 + \
                                  I0_mal * degree_pol * \
                                  np.cos(np.radians(theta_pol))**2

                I_finale = I_apres_pol * np.cos(np.radians(delta_theta))**2

                st.metric("I après polariseur (W/m²)", f"{I_apres_pol:.4f}")
                st.metric("I finale (W/m²)", f"{I_finale:.4f}")
                st.metric("Atténuation I/I₀", f"{I_finale/I0_mal:.4f}")
                st.metric("Atténuation (dB)",
                          f"{-10*np.log10(max(I_finale/I0_mal,1e-10)):.2f}")
                st.metric("δθ = θ_ana - θ_pol", f"{delta_theta}°")

            with col2:
                theta_sweep = np.linspace(0, 360, 1000)
                eng_mal = OptiqueEngine()

                I_mal = eng_mal.loi_malus(I_apres_pol, theta_sweep - theta_pol)

                fig_mal = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["I(θ_ana) linéaire", "I(θ_ana) polaire"],
                    specs=[[ {"type": "xy"}, {"type": "polar"} ]])

                fig_mal.add_trace(go.Scatter(x=theta_sweep, y=I_mal,
                    mode='lines', name='I(θ)',
                    line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'),
                    row=1, col=1)
                fig_mal.add_vline(x=theta_ana,
                    line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"θ_ana={theta_ana}°",
                    row=1, col=1)
                fig_mal.add_trace(go.Scatter(x=[theta_ana],
                    y=[I_finale], mode='markers', name='Point Q',
                    marker=dict(color='#ff00cc', size=12, symbol='star')),
                    row=1, col=1)

                # Polaire
                theta_rad_sw = np.radians(theta_sweep)
                r_pol = I_mal
                fig_mal.add_trace(go.Scatterpolar(
                    r=r_pol, theta=theta_sweep, mode='lines',
                    name='I(θ)', line=dict(color='#00ccff', width=2)),
                    row=1, col=2)

                fig_mal.update_layout(**PLOT_LAYOUT, height=450,
                    title="Loi de Malus — I = I₀cos²(θ)",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_mal.update_xaxes(row=1, col=1, **AXIS_STYLE,
                                     title_text="θ_analyseur (°)")
                fig_mal.update_yaxes(row=1, col=1, **AXIS_STYLE,
                                     title_text="I (W/m²)")
                st.plotly_chart(fig_mal, use_container_width=True)

                # N polariseurs en cascade
                st.markdown("#### 🔀 N polariseurs en cascade")
                n_pol = st.slider("Nombre de polariseurs N", 2, 10, 3, 1,
                                   key="n_polariseurs")
                
                st.markdown(f"**Définir les angles pour {n_pol} polariseurs :**")
                angles = []
                cols = st.columns(min(n_pol, 5))
                for i in range(n_pol):
                    with cols[i % 5]:
                        angle = st.number_input(
                            f"θ_{i+1} (°)",
                            min_value=0,
                            max_value=180,
                            value=(i+1) * (90 // n_pol) if i > 0 else 0,
                            step=5,
                            key=f"angle_pol_{i}"
                        )
                        angles.append(angle)
                
                # Calcul de l'intensité pour N polariseurs
                I_n_pol = I0_mal
                angle_details = []
                for i in range(n_pol):
                    if i == 0:
                        cos_sq = np.cos(np.radians(angles[0]))**2
                        angle_details.append(f"cos²({angles[0]}°) = {cos_sq:.4f}")
                    else:
                        delta_angle = angles[i] - angles[i-1]
                        cos_sq = np.cos(np.radians(delta_angle))**2
                        angle_details.append(f"cos²({angles[i]}° - {angles[i-1]}°) = {cos_sq:.4f}")
                    I_n_pol *= cos_sq
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric(f"I finale ({n_pol} polariseurs)", f"{I_n_pol:.6f} W/m²")
                with col2:
                    efficiency = (I_n_pol / I0_mal) * 100 if I0_mal > 0 else 0
                    st.metric("Transmission (%)", f"{efficiency:.2f}%")
                
                with st.expander("Détail du calcul"):
                    for detail in angle_details:
                        st.write(f"• {detail}")
                    safe_latex(r"I_{finale} = I_0 \times " + " \times ".join([f"\\cos^2(\\Delta\\theta_{i+1})" for i in range(n_pol)]))
                    joined_factors = " × ".join(detail.split(" = ")[1] for detail in angle_details)
                    st.write(f"I_finale = {I0_mal:.4f} × {joined_factors} = {I_n_pol:.6f} W/m²")
                
                st.info(f"💡 Sans polariseurs intermédiaires : I = {I0_mal*np.cos(np.radians(angles[-1]))**2:.4f} W/m²")

        # ---- TAB 2 : ÉTAT DE POLARISATION ----
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                Ex_pol = st.slider("Ex", 0.0, 3.0, 1.0, 0.1, key="Ex_pol")
                Ey_pol = st.slider("Ey", 0.0, 3.0, 1.0, 0.1, key="Ey_pol")
                delta_pol = st.slider("δ (°)", -180, 180, 90, 5)

                eng_pol = OptiqueEngine()
                type_pol = eng_pol.type_polarisation(Ex_pol, Ey_pol,
                                                      np.radians(delta_pol))
                S = eng_pol.stokes(Ex_pol, Ey_pol, np.radians(delta_pol))
                P_deg = eng_pol.degre_polarisation(S)

                st.metric("Type", type_pol)
                st.metric("S₀ (intensité)", f"{S[0]:.4f}")
                st.metric("S₁", f"{S[1]:.4f}")
                st.metric("S₂", f"{S[2]:.4f}")
                st.metric("S₃", f"{S[3]:.4f}")
                st.metric("Degré de polarisation P", f"{P_deg:.4f}")

            with col2:
                omega = 1.0
                t_pol = np.linspace(0, 4*np.pi, 500)
                ex, ey = eng_pol.onde_elliptique(Ex_pol, Ey_pol,
                    np.radians(delta_pol), t_pol, omega)

                # Ellipse de polarisation
                fig_ell = make_subplots(rows=1, cols=2,
                    subplot_titles=["Champ E(t)", "Ellipse de polarisation"])
                fig_ell.add_trace(go.Scatter(x=t_pol, y=ex, mode='lines',
                    name='Ex', line=dict(color='#00ccff', width=2)),
                    row=1, col=1)
                fig_ell.add_trace(go.Scatter(x=t_pol, y=ey, mode='lines',
                    name='Ey', line=dict(color='#7700ff', width=2)),
                    row=1, col=1)
                # Colorer l'ellipse par le temps
                fig_ell.add_trace(go.Scatter(x=ex, y=ey, mode='lines',
                    name='Ellipse',
                    line=dict(color='#00ccff', width=2.5)),
                    row=1, col=2)
                fig_ell.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                    marker=dict(color='#ffcc00', size=10, symbol='cross'),
                    name='Origine', showlegend=False), row=1, col=2)

                fig_ell.update_layout(**PLOT_LAYOUT, height=430,
                    title=type_pol,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_ell.update_xaxes(**AXIS_STYLE)
                fig_ell.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig_ell, use_container_width=True)

        # ---- TAB 3 : POINCARÉ ----
        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔵 Sphère de Poincaré")
                st.info("""
                Chaque état de polarisation = 1 point sur la sphère.
                - **Équateur** : polarisations rectilignes
                - **Pôle Nord** : circulaire gauche
                - **Pôle Sud** : circulaire droite
                - **Intérieur** : polarisation partielle
                """)
                psi_poinc = st.slider("ψ (angle ellipse °)", 0, 180, 45, 5)
                chi_poinc = st.slider("χ (ellipticité °)", -45, 45, 30, 5)

                psi_r = np.radians(psi_poinc)
                chi_r = np.radians(chi_poinc)
                X_p = np.cos(2*chi_r)*np.cos(2*psi_r)
                Y_p = np.cos(2*chi_r)*np.sin(2*psi_r)
                Z_p = np.sin(2*chi_r)

                st.metric("S₁ = cos2χ·cos2ψ", f"{X_p:.4f}")
                st.metric("S₂ = cos2χ·sin2ψ", f"{Y_p:.4f}")
                st.metric("S₃ = sin2χ", f"{Z_p:.4f}")

            with col2:
                u = np.linspace(0, 2*np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                xs = np.outer(np.cos(u), np.sin(v))
                ys = np.outer(np.sin(u), np.sin(v))
                zs = np.outer(np.ones_like(u), np.cos(v))

                fig_poinc = go.Figure()
                fig_poinc.add_trace(go.Surface(x=xs, y=ys, z=zs,
                    opacity=0.15, colorscale='Blues', showscale=False))

                # Points notables
                points_notables = {
                    "H (rectiligne 0°)": (1,0,0,'#ff4444'),
                    "V (rectiligne 90°)": (-1,0,0,'#4444ff'),
                    "+45°": (0,1,0,'#44ff44'),
                    "-45°": (0,-1,0,'#ffff44'),
                    "Circ. G (N)": (0,0,1,'#ffffff'),
                    "Circ. D (S)": (0,0,-1,'#aaaaaa'),
                }
                for nom_pt, (xp, yp, zp, cp) in points_notables.items():
                    fig_poinc.add_trace(go.Scatter3d(x=[xp], y=[yp], z=[zp],
                        mode='markers+text', text=[nom_pt],
                        textposition='top center',
                        marker=dict(color=cp, size=6),
                        name=nom_pt, showlegend=False))

                # Point courant
                fig_poinc.add_trace(go.Scatter3d(x=[X_p], y=[Y_p], z=[Z_p],
                    mode='markers', name='État courant',
                    marker=dict(color='#ff00cc', size=12, symbol='diamond')))

                # Ligne vers le centre
                fig_poinc.add_trace(go.Scatter3d(x=[0,X_p], y=[0,Y_p],
                    z=[0,Z_p], mode='lines',
                    line=dict(color='#ff00cc', width=3),
                    showlegend=False))

                fig_poinc.update_layout(
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff', title='S₁'),
                        yaxis=dict(color='#c0d0ff', title='S₂'),
                        zaxis=dict(color='#c0d0ff', title='S₃'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    title="Sphère de Poincaré",
                    font=dict(color='#c0d0ff'),
                    height=550,
                    margin=dict(l=0,r=0,t=40,b=0)
                )
                st.plotly_chart(fig_poinc, use_container_width=True)

        # ---- TAB 4 : JONES ----
        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🧮 Formalisme de Jones")
                type_entree = st.selectbox("Champ incident", [
                    "Polarisation H (→)",
                    "Polarisation V (↑)",
                    "Polarisation +45°",
                    "Circulaire gauche",
                    "Personnalisé"
                ])
                etats = {
                    "Polarisation H (→)":  np.array([1, 0]),
                    "Polarisation V (↑)":  np.array([0, 1]),
                    "Polarisation +45°":   np.array([1, 1])/np.sqrt(2),
                    "Circulaire gauche":   np.array([1, 1j])/np.sqrt(2),
                }
                if type_entree == "Personnalisé":
                    ex_j = st.slider("Ex (réel)", 0.0, 2.0, 1.0, 0.1)
                    ey_r = st.slider("Ey (réel)", 0.0, 2.0, 0.0, 0.1)
                    ey_i = st.slider("Ey (imag)", -1.0, 1.0, 0.0, 0.1)
                    E_in = np.array([ex_j, complex(ey_r, ey_i)])
                else:
                    E_in = etats[type_entree]

                st.markdown("**Éléments optiques :**")
                n_elem = st.slider("Nombre d'éléments", 1, 4, 2)
                elements_jones = []
                eng_jones = OptiqueEngine()

                types_elem = ["Polariseur horizontal", "Polariseur vertical",
                              "Polariseur θ", "Lame λ/4 axe horizontal",
                              "Lame λ/2 axe horizontal", "Miroir"]
                for i in range(n_elem):
                    elem_type = st.selectbox(f"Élément {i+1}",
                        types_elem, key=f"jones_el_{i}")
                    theta_j = 0
                    if "θ" in elem_type:
                        theta_j = st.slider(f"θ_{i+1} (°)", -180, 180, 45, 5,
                                            key=f"theta_jones_{i}")
                    M = eng_jones.matrice_jones(elem_type, theta_j)
                    elements_jones.append(M)
                    st.text(f"M_{i+1}:\n{np.round(M, 3)}")

            with col2:
                E_out = eng_jones.propagation_jones(E_in, elements_jones)
                I_in  = np.abs(E_in[0])**2 + np.abs(E_in[1])**2
                I_out = np.abs(E_out[0])**2 + np.abs(E_out[1])**2

                st.metric("I entrée", f"{I_in:.4f}")
                st.metric("I sortie", f"{I_out:.4f}")
                st.metric("Transmission I_s/I_e",
                          f"{I_out/max(I_in,1e-10):.4f}")
                st.markdown("**Champ de sortie E_out :**")
                st.markdown(f"Ex = {E_out[0]:.4f}")
                st.markdown(f"Ey = {E_out[1]:.4f}")

                # Ellipses entrée/sortie
                t_j = np.linspace(0, 2*np.pi, 500)
                ex_in_j = np.real(E_in[0]) * np.cos(t_j)
                ey_in_j = np.real(E_in[1]) * np.cos(t_j + np.angle(E_in[1]))
                ex_out_j = np.real(E_out[0]) * np.cos(t_j)
                ey_out_j = (np.abs(E_out[1]) *
                            np.cos(t_j + np.angle(E_out[1])))

                fig_jones = go.Figure()
                fig_jones.add_trace(go.Scatter(x=ex_in_j, y=ey_in_j,
                    mode='lines', name='Entrée',
                    line=dict(color='#00ccff', width=2.5)))
                fig_jones.add_trace(go.Scatter(x=ex_out_j, y=ey_out_j,
                    mode='lines', name='Sortie',
                    line=dict(color='#ff00cc', width=2.5, dash='dash')))
                max_v = max(I_in, I_out, 1) ** 0.5 * 1.1
                layout(fig_jones, "Ellipses de polarisation",
                       "Ex", "Ey", h=400)
                fig_jones.update_xaxes(range=[-max_v, max_v])
                fig_jones.update_yaxes(range=[-max_v, max_v],
                                       scaleanchor='x')
                st.plotly_chart(fig_jones, use_container_width=True)

        # ---- TAB 5 : BREWSTER ----
        with tab5:
            col1, col2 = st.columns([1, 2])
            with col1:
                n1_br = st.slider("n₁ (milieu incident)", 1.0, 2.5, 1.0, 0.01)
                n2_br = st.slider("n₂ (milieu réfracté)", 1.0, 3.5, 1.5, 0.01)
                theta_B = OptiqueEngine().angle_brewster(n1_br, n2_br)
                theta_lim = np.degrees(np.arcsin(min(n1_br/n2_br, 1)))

                st.metric("Angle de Brewster θ_B (°)", f"{theta_B:.4f}")
                st.metric("tan(θ_B) = n₂/n₁", f"{n2_br/n1_br:.4f}")
                if n1_br > n2_br:
                    st.metric("Angle limite θ_c (°)", f"{theta_lim:.4f}")
                else:
                    st.metric("Réflexion totale interne", "Impossible (n₁<n₂)")

                st.info(f"""
                À θ = {theta_B:.1f}° :
                - Le rayon **réfléchi** est **perpendiculaire** au réfracté
                - La composante **p** n'est pas réfléchie
                - La lumière réfléchie est **totalement polarisée** (composante s)
                """)

            with col2:
                theta_i = np.linspace(0, 90, 1000)
                eng_br = OptiqueEngine()
                Rs_br, Rp_br, _, _ = eng_br.reflectivite_fresnel(
                    n1_br, n2_br, theta_i)

                fig_br = go.Figure()
                fig_br.add_trace(go.Scatter(x=theta_i, y=Rs_br, mode='lines',
                    name='Rs (perpend.)',
                    line=dict(color='#00ccff', width=2.5)))
                fig_br.add_trace(go.Scatter(x=theta_i, y=Rp_br, mode='lines',
                    name='Rp (parallèle)',
                    line=dict(color='#7700ff', width=2.5)))
                fig_br.add_vline(x=theta_B, line_color='#ffcc00',
                                  line_dash='dash',
                                  annotation_text=f"θ_B={theta_B:.1f}°")
                if n1_br > n2_br:
                    fig_br.add_vline(x=theta_lim,
                        line_color='#ff0000', line_dash='dot',
                        annotation_text=f"θ_c={theta_lim:.1f}°")
                layout(fig_br,
                       f"Coefficients de Fresnel (n₁={n1_br}, n₂={n2_br})",
                       "θ incident (°)", "Réflectivité R")
                st.plotly_chart(fig_br, use_container_width=True)

                # Degré de polarisation de la lumière réfléchie
                P_refl = (Rs_br - Rp_br) / (Rs_br + Rp_br + 1e-15)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=theta_i, y=P_refl, mode='lines',
                    line=dict(color='#ff00cc', width=2.5),
                    name='P = (Rs-Rp)/(Rs+Rp)'))
                fig_pr.add_vline(x=theta_B, line_color='#ffcc00',
                                  line_dash='dash')
                layout(fig_pr, "Degré de polarisation du faisceau réfléchi",
                       "θ incident (°)", "P", h=300)
                st.plotly_chart(fig_pr, use_container_width=True)

        # ---- TAB 6 : FORMULES ----
        # Les formules pour la section Polarisation & Malus ont été supprimées
        # (présentaient des erreurs côté client MathJax).
