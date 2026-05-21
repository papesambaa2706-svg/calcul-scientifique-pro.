"""
Module unique d'analyse numerique regroupant integration, interpolation et equ_diff.
Ce module exporte integration_page, interpolation_page et equ_diff_page.
"""

__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import linalg, stats
from scipy.integrate import odeint, solve_ivp, quad, dblquad, tplquad, fixed_quad
from scipy.interpolate import (interp1d, CubicSpline, BarycentricInterpolator, make_interp_spline, RBFInterpolator, RegularGridInterpolator)
from scipy.optimize import curve_fit, fsolve, brentq, minimize_scalar
from numpy.polynomial.polynomial import Polynomial
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

__all__ = ["integration_page", "interpolation_page", "equ_diff_page"]

################################################################################
# Contenu de integration.py
################################################################################
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


# ============================================================
# FORMULAIRE
# ============================================================
FORMULES_INTEGRATION = {
    "Trapèzes":          r"I \approx \frac{h}{2}\left[f(a)+2\sum_{i=1}^{n-1}f(x_i)+f(b)\right],\quad h=\frac{b-a}{n}",
    "Simpson 1/3":       r"I \approx \frac{h}{3}\left[f(a)+4f(x_1)+2f(x_2)+\cdots+4f(x_{n-1})+f(b)\right]",
    "Simpson 3/8":       r"I \approx \frac{3h}{8}\left[f(a)+3f(x_1)+3f(x_2)+2f(x_3)+\cdots\right]",
    "Gauss-Legendre":    r"I \approx \frac{b-a}{2}\sum_{i=1}^n w_i\,f\!\left(\frac{b-a}{2}x_i+\frac{a+b}{2}\right)",
    "Monte Carlo":       r"I \approx (b-a)\cdot\frac{1}{N}\sum_{i=1}^N f(X_i),\quad X_i\sim\mathcal{U}(a,b)",
    "Romberg":           r"R_{n,m}=\frac{4^m R_{n,m-1}-R_{n-1,m-1}}{4^m-1}",
    "Erreur Trapèzes":   r"|E_T|\leq\frac{(b-a)^3}{12n^2}\max|f''|",
    "Erreur Simpson":    r"|E_S|\leq\frac{(b-a)^5}{180n^4}\max|f^{(4)}|",
    "Erreur Gauss(n)":   r"|E_G|\leq\frac{(b-a)^{2n+1}(n!)^4}{(2n+1)[(2n)!]^3}\max|f^{(2n)}|",
    "Intégrale double":  r"\iint_D f(x,y)\,dx\,dy\approx\sum_{i,j}w_i w_j f(x_i,y_j)",
}

FONCTIONS_PREDEF_INTEGRATION = {
    "x²":                   (lambda x: x**2,           "Polynôme",       True),
    "sin(x)":               (np.sin,                    "Trigonométrique",True),
    "exp(-x²)":             (lambda x: np.exp(-x**2),  "Gaussienne",     True),
    "cos(x)":               (np.cos,                    "Trigonométrique",True),
    "1/(1+x²)":             (lambda x: 1/(1+x**2),     "Rationnelle",    True),
    "x·sin(x)":             (lambda x: x*np.sin(x),    "Mixte",          True),
    "√(1-x²)":             (lambda x: np.sqrt(np.maximum(1-x**2, 0)),
                                                          "Semi-circulaire",True),
    "ln(x)":                (lambda x: np.where(x>0, np.log(x), 0),
                                                          "Logarithmique",  False),
    "1/√x":                 (lambda x: np.where(x>1e-10, 1/np.sqrt(x), 0),
                                                          "Singulière",     False),
    "sin(1/x)":             (lambda x: np.where(np.abs(x)>1e-8, np.sin(1/x), 0),
                                                          "Oscillante",     False),
    "exp(-x)·cos(10x)":    (lambda x: np.exp(-x)*np.cos(10*x),
                                                          "Oscillante amorties",True),
    "Heaviside(x-0.5)":    (lambda x: np.where(x>=0.5, 1.0, 0.0),
                                                          "Discontinue",    True),
}

METHODES_INFO_INTEGRATION = {
    "Trapèzes":        {"ordre": 2, "formule": "Open",   "complexité": "O(n)"},
    "Simpson 1/3":     {"ordre": 4, "formule": "Closed", "complexité": "O(n)"},
    "Simpson 3/8":     {"ordre": 4, "formule": "Closed", "complexité": "O(n)"},
    "Gauss-Legendre":  {"ordre": "2n-1","formule": "Gauss","complexité": "O(n)"},
    "Monte Carlo":     {"ordre": "½",  "formule": "Stoch.","complexité": "O(n)"},
    "Romberg":         {"ordre": "∞", "formule": "Adapt.","complexité": "O(n²)"},
    "SciPy (quad)":    {"ordre": "adapt.","formule": "QUADPACK","complexité": "adapt."},
}


# ============================================================
# SOLVEURS D'INTÉGRATION
# ============================================================
class IntegrationSolver:
    """Solveurs numériques d'intégration avancés."""

    @staticmethod
    def trapeze(f, a, b, n):
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])

    @staticmethod
    def simpson_13(f, a, b, n):
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

    @staticmethod
    def simpson_38(f, a, b, n):
        if n % 3 != 0:
            n += 3 - n % 3
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        S = y[0] + y[-1]
        for i in range(1, n):
            S += (3 if i % 3 != 0 else 2) * y[i]
        return 3*h/8 * S

    @staticmethod
    def gauss_legendre(f, a, b, n=10):
        xi, wi = np.polynomial.legendre.leggauss(n)
        x = 0.5*(xi+1)*(b-a) + a
        return np.sum(wi * f(x)) * 0.5*(b-a)

    @staticmethod
    def monte_carlo(f, a, b, n, seed=42):
        np.random.seed(seed)
        x = np.random.uniform(a, b, n)
        return (b - a) * np.mean(f(x))

    @staticmethod
    def romberg_method(f, a, b, max_order=6):
        try:
            R = np.zeros((max_order, max_order), dtype=float)
            for k in range(max_order):
                n = 2**k
                x = np.linspace(a, b, n + 1)
                y = f(x)
                R[k, 0] = safe_trapz(y, x)
                for j in range(1, k + 1):
                    R[k, j] = (4**j * R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
            return float(R[max_order - 1, max_order - 1])
        except Exception:
            return None

    @staticmethod
    def scipy_quad(f, a, b):
        try:
            val, err = quad(f, a, b, limit=200, epsabs=1e-12)
            return val, err
        except:
            return None, None

    @staticmethod
    def integrale_double(f2d, xa, xb, ya, yb, nx=20, ny=20):
        """Intégrale double par règle des trapèzes 2D."""
        x = np.linspace(xa, xb, nx)
        y = np.linspace(ya, yb, ny)
        X, Y = np.meshgrid(x, y)
        Z = f2d(X, Y)
        dx = (xb-xa)/(nx-1)
        dy = (yb-ya)/(ny-1)
        return safe_trapz(safe_trapz(Z, dx=dy), dx=dx)

    @staticmethod
    def convergence_etude(f, a, b, methode: str,
                          ref: float, n_range: np.ndarray) -> np.ndarray:
        """Erreur de convergence pour N variable."""
        solver = IntegrationSolver()
        erreurs = []
        for n in n_range:
            n = int(n)
            try:
                if methode == "Trapèzes":
                    val = solver.trapeze(f, a, b, n)
                elif methode == "Simpson 1/3":
                    val = solver.simpson_13(f, a, b, n)
                elif methode == "Gauss-Legendre":
                    val = solver.gauss_legendre(f, a, b, n)
                else:
                    val = solver.monte_carlo(f, a, b, n)
                erreurs.append(abs(val - ref))
            except:
                erreurs.append(np.nan)
        return np.array(erreurs)


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def integration_page():
    st.markdown("## ∫ Intégration Numérique Avancée")
    st.markdown("*Méthodes classiques, Gauss, Monte Carlo, Romberg, intégrale double, convergence*")
    st.markdown("---")

    solver = IntegrationSolver()

    section = st.selectbox(
        "Section",
        [
            "🔢 Intégration 1D",
            "⚔️ Benchmark méthodes",
            "📉 Convergence",
            "🌐 Intégrale double",
            "🎲 Monte Carlo avancé",
            "📖 Théorie",
        ],
        key="section_integration"
    )


    # ============================================================
    # TAB 1 : INTÉGRATION 1D
    # ============================================================
    if section == "🔢 Intégration 1D":
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Paramètres")

            func_name = st.selectbox("Fonction", list(FONCTIONS_PREDEF_INTEGRATION.keys()))
            f_info = FONCTIONS_PREDEF_INTEGRATION[func_name]
            f = f_info[0]
            st.caption(f"Catégorie : *{f_info[1]}*")

            a = st.number_input("Borne a", value=0.0, step=0.1)
            b = st.number_input("Borne b", value=1.0, step=0.1)
            if b <= a:
                st.error("b doit être > a")
                st.stop()

            n = st.slider("Discrétisation n", 4, 2000, 50)

            methodes_sel = st.multiselect("Méthodes",
                list(METHODES_INFO_INTEGRATION.keys()),
                default=["Trapèzes", "Simpson 1/3", "Gauss-Legendre", "SciPy (quad)"]
            )

            show_fill  = st.checkbox("Afficher l'aire", True)
            show_pts   = st.checkbox("Afficher les points de calcul", False)

        with col2:
            # Valeur de référence
            ref_val, ref_err = solver.scipy_quad(f, a, b)
            if ref_val is None:
                st.warning("Référence SciPy indisponible.")
                ref_val = 0.0

            # Graphique
            x_plot = np.linspace(a, b, 2000)
            try:
                y_plot = f(x_plot)
            except:
                y_plot = np.zeros_like(x_plot)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_plot, y=y_plot, mode='lines', name='f(x)',
                line=dict(color='#00ccff', width=3)
            ))
            if show_fill:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[a], x_plot, [b]]),
                    y=np.concatenate([[0], y_plot, [0]]),
                    fill='toself', fillcolor='rgba(119,0,255,0.2)',
                    mode='none', name='Aire intégrée'
                ))
            if show_pts:
                x_pts = np.linspace(a, b, n+1)
                try:
                    y_pts = f(x_pts)
                    fig.add_trace(go.Scatter(
                        x=x_pts, y=y_pts, mode='markers', name='Points n',
                        marker=dict(color='#ff00cc', size=5)
                    ))
                except:
                    pass

            fig.update_layout(
                title=f"Intégration de {func_name} sur [{a:.2f}, {b:.2f}]",
                xaxis_title='x', yaxis_title='f(x)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Calcul et résultats
            resultats = []
            for m in methodes_sel:
                try:
                    if m == "Trapèzes":
                        val = solver.trapeze(f, a, b, n)
                    elif m == "Simpson 1/3":
                        val = solver.simpson_13(f, a, b, n)
                    elif m == "Simpson 3/8":
                        val = solver.simpson_38(f, a, b, n)
                    elif m == "Gauss-Legendre":
                        val = solver.gauss_legendre(f, a, b, min(n, 100))
                    elif m == "Monte Carlo":
                        val = solver.monte_carlo(f, a, b, n*100)
                    elif m == "Romberg":
                        val = solver.romberg_method(f, a, b)
                    else:
                        val, _ = solver.scipy_quad(f, a, b)

                    err = abs(val - ref_val) if val is not None else np.nan
                    resultats.append({
                        "Méthode": m,
                        "Valeur": f"{val:.8f}" if val else "N/A",
                        "Erreur abs.": f"{err:.2e}" if not np.isnan(err) else "N/A",
                        "Erreur rel. (%)": f"{err/abs(ref_val)*100:.4f}" if ref_val and not np.isnan(err) else "N/A",
                    })
                except Exception as e:
                    resultats.append({"Méthode": m, "Valeur": "❌",
                                      "Erreur abs.": str(e)[:30], "Erreur rel. (%)": "N/A"})

            df_res = pd.DataFrame(resultats)
            st.markdown("### 📊 Résultats")
            st.dataframe(df_res, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Référence SciPy", f"{ref_val:.8f}")
            with c2: st.metric("Erreur estimée", f"{ref_err:.2e}" if ref_err else "N/A")
            with c3:
                if resultats:
                    best_idx = np.argmin([float(r["Erreur abs."].replace("N/A","inf").replace("❌","inf"))
                                          if r["Erreur abs."] not in ["N/A","❌"] else np.inf
                                          for r in resultats])
                    st.metric("Meilleure méthode", resultats[best_idx]["Méthode"])

    # ============================================================
    # TAB 2 : BENCHMARK
    # ============================================================
    elif section == "⚔️ Benchmark méthodes":
        st.markdown("### ⚔️ Benchmark toutes méthodes")
        col1, col2 = st.columns([1, 2])

        with col1:
            func_b = st.selectbox("Fonction", list(FONCTIONS_PREDEF_INTEGRATION.keys()), key="fb")
            a_b = st.number_input("a", value=0.0, key="ab")
            b_b = st.number_input("b", value=1.0, key="bb")
            n_b = st.slider("n", 10, 1000, 100, key="nb")

        with col2:
            f_b = FONCTIONS_PREDEF_INTEGRATION[func_b][0]
            ref_b, _ = solver.scipy_quad(f_b, a_b, b_b)
            if ref_b is None:
                ref_b = 0.0

            methodes_all = ["Trapèzes", "Simpson 1/3", "Simpson 3/8",
                            "Gauss-Legendre", "Monte Carlo", "Romberg"]
            bench_results = []
            for m in methodes_all:
                try:
                    if m == "Trapèzes":
                        val = solver.trapeze(f_b, a_b, b_b, n_b)
                    elif m == "Simpson 1/3":
                        val = solver.simpson_13(f_b, a_b, b_b, n_b)
                    elif m == "Simpson 3/8":
                        val = solver.simpson_38(f_b, a_b, b_b, n_b)
                    elif m == "Gauss-Legendre":
                        val = solver.gauss_legendre(f_b, a_b, b_b, min(n_b, 100))
                    elif m == "Monte Carlo":
                        val = solver.monte_carlo(f_b, a_b, b_b, n_b*100)
                    else:
                        val = solver.romberg_method(f_b, a_b, b_b)

                    err = abs(val - ref_b) if val is not None else np.nan
                    bench_results.append({
                        "Méthode": m,
                        "Valeur": round(val, 8) if val else None,
                        "Erreur abs.": err,
                        "Ordre": METHODES_INFO_INTEGRATION[m]["ordre"],
                    })
                except:
                    bench_results.append({"Méthode": m, "Valeur": None,
                                          "Erreur abs.": np.nan, "Ordre": "?"})

            df_bench = pd.DataFrame(bench_results)
            st.dataframe(df_bench[["Méthode","Valeur","Erreur abs.","Ordre"]],
                         use_container_width=True)

            # Barplot erreurs
            fig_b = go.Figure()
            for row in bench_results:
                if not np.isnan(row["Erreur abs."]) and row["Erreur abs."] > 0:
                    fig_b.add_trace(go.Bar(
                        x=[row["Méthode"]], y=[row["Erreur abs."]+1e-16],
                        marker_color='#7700ff', showlegend=False
                    ))
            fig_b.update_layout(
                title="Erreur absolue |val - ref|",
                yaxis_type='log', yaxis_title="|E| (log)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=350,
            )
            st.plotly_chart(fig_b, use_container_width=True)

    # ============================================================
    # TAB 3 : CONVERGENCE
    # ============================================================
    elif section == "📉 Convergence":
        st.markdown("### 📉 Étude de convergence")
        col1, col2 = st.columns([1, 2])

        with col1:
            func_c = st.selectbox("Fonction", list(FONCTIONS_PREDEF_INTEGRATION.keys()), key="fc")
            a_c = st.number_input("a", value=0.0, key="ac_conv")
            b_c = st.number_input("b", value=1.0, key="bc_conv")
            methodes_conv = st.multiselect("Méthodes",
                ["Trapèzes", "Simpson 1/3", "Gauss-Legendre", "Monte Carlo"],
                default=["Trapèzes", "Simpson 1/3", "Gauss-Legendre"])

        with col2:
            f_c = FONCTIONS_PREDEF_INTEGRATION[func_c][0]
            ref_c, _ = solver.scipy_quad(f_c, a_c, b_c)
            if ref_c is None:
                ref_c = 0.0

            n_range = np.logspace(1, 3, 30).astype(int)
            colors_conv = ['#00ccff','#7700ff','#ff00cc','#00ff88']

            fig_conv = go.Figure()
            for i, m in enumerate(methodes_conv):
                errs = solver.convergence_etude(f_c, a_c, b_c, m, ref_c, n_range)
                fig_conv.add_trace(go.Scatter(
                    x=n_range, y=errs+1e-16, mode='lines+markers', name=m,
                    line=dict(color=colors_conv[i%len(colors_conv)], width=2.5),
                    marker=dict(size=5)
                ))

            # Lignes de référence
            n_ref = np.array([n_range[0], n_range[-1]], dtype=float)
            fig_conv.add_trace(go.Scatter(
                x=n_ref, y=1.0/n_ref**2, mode='lines', name='O(h²)',
                line=dict(color='rgba(255,200,0,0.4)', width=2, dash='dot')
            ))
            fig_conv.add_trace(go.Scatter(
                x=n_ref, y=1.0/n_ref**4, mode='lines', name='O(h⁴)',
                line=dict(color='rgba(0,255,136,0.4)', width=2, dash='dot')
            ))
            fig_conv.add_trace(go.Scatter(
                x=n_ref, y=1.0/np.sqrt(n_ref), mode='lines', name='O(1/√N)',
                line=dict(color='rgba(255,100,0,0.4)', width=2, dash='dot')
            ))

            fig_conv.update_layout(
                title=f"Convergence — {func_c}",
                xaxis_title="n (nombre de points)", yaxis_title="|E| (log)",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=480,
            )
            st.plotly_chart(fig_conv, use_container_width=True)

    # ============================================================
    # TAB 4 : INTÉGRALE DOUBLE
    # ============================================================
    elif section == "🌐 Intégrale double":
        st.markdown("### 🌐 Intégrale double")
        col1, col2 = st.columns([1, 2])

        with col1:
            func2d_name = st.selectbox("Fonction f(x,y)", [
                "x² + y²", "sin(x)·cos(y)", "exp(-(x²+y²))",
                "x·y", "sin(x+y)"
            ])
            xa = st.slider("xa", -5.0, 0.0, 0.0, 0.1)
            xb = st.slider("xb", 0.1, 5.0, 1.0, 0.1)
            ya = st.slider("ya", -5.0, 0.0, 0.0, 0.1, key="ya2")
            yb = st.slider("yb", 0.1, 5.0, 1.0, 0.1, key="yb2")
            nx2 = st.slider("nx", 10, 80, 30)
            ny2 = st.slider("ny", 10, 80, 30)

            f2d_map = {
                "x² + y²":         lambda x, y: x**2 + y**2,
                "sin(x)·cos(y)":   lambda x, y: np.sin(x)*np.cos(y),
                "exp(-(x²+y²))":   lambda x, y: np.exp(-(x**2+y**2)),
                "x·y":             lambda x, y: x*y,
                "sin(x+y)":        lambda x, y: np.sin(x+y),
            }
            f2d = f2d_map[func2d_name]

        with col2:
            val2d = solver.integrale_double(f2d, xa, xb, ya, yb, nx2, ny2)

            # Référence SciPy
            try:
                ref2d, _ = dblquad(lambda y, x: f2d(x, y), xa, xb,
                                    lambda x: ya, lambda x: yb)
            except:
                ref2d = None

            c1, c2 = st.columns(2)
            with c1: st.metric("Valeur numérique", f"{val2d:.6f}")
            with c2: st.metric("Référence SciPy", f"{ref2d:.6f}" if ref2d else "N/A")
            if ref2d:
                st.metric("Erreur abs.", f"{abs(val2d-ref2d):.2e}")

            # Surface 3D
            x2d = np.linspace(xa, xb, 60)
            y2d = np.linspace(ya, yb, 60)
            X2d, Y2d = np.meshgrid(x2d, y2d)
            Z2d = f2d(X2d, Y2d)

            fig_2d = go.Figure(data=[go.Surface(
                z=Z2d, x=x2d, y=y2d,
                colorscale=[[0,'#020817'],[0.4,'#7700ff'],[0.7,'#00ccff'],[1,'#ffffff']],
                showscale=True,
                lighting=dict(ambient=0.5, diffuse=0.8),
            )])
            fig_2d.update_layout(
                title=f"∬ {func2d_name} dxdy = {val2d:.4f}",
                scene=dict(
                    bgcolor='rgba(5,0,20,0.9)',
                    xaxis=dict(color='#c0d0ff', title='x'),
                    yaxis=dict(color='#c0d0ff', title='y'),
                    zaxis=dict(color='#c0d0ff', title='f(x,y)'),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0d0ff'),
                height=480, margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig_2d, use_container_width=True)

    # ============================================================
    # TAB 5 : MONTE CARLO AVANCÉ
    # ============================================================
    elif section == "🎲 Monte Carlo avancé":
        st.markdown("### 🎲 Monte Carlo avancé")
        col1, col2 = st.columns([1, 2])

        with col1:
            func_mc = st.selectbox("Fonction", list(FONCTIONS_PREDEF_INTEGRATION.keys()), key="fmc")
            a_mc = st.number_input("a", value=0.0, key="amc")
            b_mc = st.number_input("b", value=1.0, key="bmc")
            N_mc = st.slider("N (points Monte Carlo)", 100, 100000, 5000, 100)
            n_runs = st.slider("Répétitions (variance)", 5, 50, 20)
            seed_mc = st.slider("Seed", 0, 999, 42)

        with col2:
            f_mc = FONCTIONS_PREDEF_INTEGRATION[func_mc][0]
            ref_mc, _ = solver.scipy_quad(f_mc, a_mc, b_mc)
            if ref_mc is None:
                ref_mc = 0.0

            # Convergence Monte Carlo
            N_range = np.geomspace(10, N_mc, 40).astype(int)
            mc_vals = []
            for n_iter in N_range:
                np.random.seed(seed_mc)
                x_s = np.random.uniform(a_mc, b_mc, n_iter)
                mc_vals.append((b_mc - a_mc) * np.mean(f_mc(x_s)))

            fig_mc = make_subplots(rows=2, cols=1,
                subplot_titles=[f"Convergence Monte Carlo — {func_mc}",
                                 "Histogramme des estimations (répétitions)"])

            fig_mc.add_trace(go.Scatter(
                x=N_range, y=mc_vals, mode='lines+markers', name='MC estimé',
                line=dict(color='#00ccff', width=2)
            ), row=1, col=1)
            fig_mc.add_hline(y=ref_mc, line_color='#ffcc00', line_dash='dash',
                             annotation_text=f"Exacte={ref_mc:.4f}", row=1, col=1)

            # Distribution des estimations sur n_runs tirages
            estimates = []
            for _ in range(n_runs):
                np.random.seed(np.random.randint(0, 9999))
                xs = np.random.uniform(a_mc, b_mc, N_mc)
                estimates.append((b_mc - a_mc) * np.mean(f_mc(xs)))

            fig_mc.add_trace(go.Histogram(
                x=estimates, nbinsx=15, name='Estimations',
                marker=dict(color='rgba(119,0,255,0.6)',
                           line=dict(color='rgba(0,204,255,0.8)', width=0.5))
            ), row=2, col=1)
            fig_mc.add_vline(x=ref_mc, line_color='#ffcc00', line_dash='dash',
                             annotation_text="Exacte", row=2, col=1)

            fig_mc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                height=550,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_mc.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_mc.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_mc.update_xaxes(type='log', row=1, col=1, title_text="N")
            st.plotly_chart(fig_mc, use_container_width=True)

            # Stats Monte Carlo
            mc_mean = np.mean(estimates)
            mc_std  = np.std(estimates)
            mc_err  = abs(mc_mean - ref_mc)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Moyenne MC", f"{mc_mean:.6f}")
            with c2: st.metric("Écart-type", f"{mc_std:.6f}")
            with c3: st.metric("Erreur moy.", f"{mc_err:.2e}")
            with c4: st.metric("1/√N théorique", f"{(b_mc-a_mc)/np.sqrt(N_mc):.2e}")

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    elif section == "📖 Théorie":
        st.markdown("### 📖 Formulaire scientifique")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES_INTEGRATION.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 📊 Comparaison des méthodes")
        df_m = pd.DataFrame([
            {"Méthode": k, "Ordre erreur": v["ordre"],
             "Type": v["formule"], "Coût": v["complexité"]}
            for k, v in METHODES_INFO_INTEGRATION.items()
        ])
        st.dataframe(df_m, use_container_width=True)

        st.markdown("---")
        st.markdown("### ⚗️ Tableau de diagnostic")
        diag_int = {
            "Problème": ["Erreur grande", "Oscillation intégrale", "Singularité",
                         "Monte Carlo lent", "Simpson invalide"],
            "Cause": ["n trop petit", "f oscillante", "Pôle en [a,b]",
                      "N insuffisant", "n impair"],
            "Solution": ["Augmenter n", "Gauss ou subdivision", "Changer bornes",
                         "Augmenter N", "Prendre n pair"]
        }
        st.dataframe(pd.DataFrame(diag_int), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Davis & Rabinowitz — *Methods of Numerical Integration* (Academic Press, 1984)",
            "Press et al. — *Numerical Recipes* (Cambridge, 2007)",
            "Golub & Welsch — *Calculation of Gauss quadrature rules* (Math. Comp., 1969)",
        ]:
            st.markdown(f"- {r}")

################################################################################
# Contenu de interpolation.py
################################################################################

# ============================================================
# FORMULAIRE SCIENTIFIQUE
# ============================================================
FORMULES_INTERPOLATION = {
    "Polynôme de Lagrange": r"P_n(x) = \sum_{i=0}^{n} y_i \prod_{j\neq i} \frac{x-x_j}{x_i-x_j}",
    "Spline cubique": r"S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3",
    "Erreur interpolation": r"|f(x) - P_n(x)| \leq \frac{M_{n+1}}{(n+1)!}\prod_{i=0}^n|x-x_i|",
    "Phénomène de Runge": r"f(x) = \frac{1}{1+25x^2} \quad \text{(oscille aux bords)}",
    "Nœuds de Chebyshev": r"x_k = \cos\left(\frac{2k+1}{2n+2}\pi\right), \quad k=0,\ldots,n",
    "Interpolation RBF": r"s(x) = \sum_{i=1}^n \lambda_i \phi(\|x - x_i\|)",
    "Régression linéaire": r"\hat{y} = \beta_0 + \beta_1 x, \quad \beta_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}",
    "R² (coefficient)": r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}",
    "RMSE": r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}",
    "MAE": r"\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|",
}

METHODES_INFO_INTERPOLATION = {
    "Linéaire":         {"ordre": 1,  "dérivées": "C⁰", "oscille": False, "complexité": "O(n)"},
    "Cubique (Spline)": {"ordre": 3,  "dérivées": "C²", "oscille": False, "complexité": "O(n)"},
    "Lagrange":         {"ordre": n,  "dérivées": "C∞", "oscille": True,  "complexité": "O(n²)"} if (n := "n-1") else {},
    "Nearest":          {"ordre": 0,  "dérivées": "C⁻¹","oscille": False, "complexité": "O(1)"},
    "Polynôme ordre N": {"ordre": "N","dérivées": "C∞", "oscille": True,  "complexité": "O(n²)"},
    "Chebyshev":        {"ordre": "N","dérivées": "C∞", "oscille": False, "complexité": "O(n²)"},
    "RBF (Gaussien)":   {"ordre": "∞","dérivées": "C∞", "oscille": False, "complexité": "O(n³)"},
    "B-Spline":         {"ordre": "k","dérivées": "Cᵏ⁻¹","oscille": False, "complexité": "O(n)"},
}

EXEMPLES_PREDIFINIS = {
    "Parabole":     ("0,1,2,3,4", "0,1,4,9,16",          "y = x²"),
    "Sinus":        ("0,0.5,1,1.5,2,2.5,3", "0,0.479,0.841,0.997,0.909,0.598,0.141", "y = sin(x)"),
    "Exponentielle":("0,0.5,1,1.5,2,2.5,3", "1,1.65,2.72,4.48,7.39,12.2,20.1",       "y = eˣ"),
    "Runge":        ("-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1",
                     "0.038,0.058,0.099,0.200,0.500,1.000,0.500,0.200,0.099,0.058,0.038",
                     "y = 1/(1+25x²)"),
    "Données bruitées": ("0,1,2,3,4,5,6", "1.1,2.9,4.2,8.8,15.9,25.1,36.4",         "données réelles"),
    "Oscillant":    ("0,1,2,3,4,5,6,7",   "0,1,-1,0,1,-1,0,1",                       "signal périodique"),
    "Chebyshev test":("-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1",
                      "0.038,0.066,0.138,0.390,1.0,0.390,0.138,0.066,0.038",
                      "nœuds de Chebyshev"),
}


# ============================================================
# MOTEUR INTERPOLATION
# ============================================================
class InterpolationEngine:
    """Moteur d'interpolation scientifique avancé."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self._valider(x_data, y_data)
        self.x = np.array(x_data, dtype=float)
        self.y = np.array(y_data, dtype=float)
        self.n = len(x_data)

    def _valider(self, x, y):
        if len(x) != len(y):
            raise ValueError("x et y doivent avoir la même longueur")
        if len(x) < 2:
            raise ValueError("Au moins 2 points requis")
        if len(np.unique(x)) != len(x):
            raise ValueError("Les valeurs de x doivent être distinctes")

    def interpoler(self, x_fin: np.ndarray, methode: str,
                   poly_order: int = 3, rbf_kernel: str = "gaussian") -> np.ndarray:
        try:
            x_sorted = self.x
            y_sorted = self.y
            idx = np.argsort(x_sorted)
            x_s = x_sorted[idx]
            y_s = y_sorted[idx]

            if methode == "Linéaire":
                f = interp1d(x_s, y_s, kind='linear', fill_value='extrapolate')
                return f(x_fin)

            elif methode == "Cubique (Spline)":
                cs = CubicSpline(x_s, y_s)
                return cs(x_fin)

            elif methode == "Lagrange":
                bi = BarycentricInterpolator(x_s, y_s)
                return bi(x_fin)

            elif methode == "Nearest":
                f = interp1d(x_s, y_s, kind='nearest', fill_value='extrapolate')
                return f(x_fin)

            elif methode == "Polynôme ordre N":
                p = Polynomial.fit(x_s, y_s, deg=poly_order)
                return p(x_fin)

            elif methode == "Chebyshev":
                c = chebyshev.chebfit(x_s, y_s, deg=min(poly_order, self.n-1))
                return chebyshev.chebval(x_fin, c)

            elif methode == "RBF (Gaussien)":
                rbf = RBFInterpolator(x_s.reshape(-1,1), y_s,
                                      kernel=rbf_kernel, epsilon=1.0)
                return rbf(x_fin.reshape(-1,1))

            elif methode == "B-Spline":
                k = min(poly_order, self.n - 1)
                bsp = make_interp_spline(x_s, y_s, k=k)
                return bsp(x_fin)

            else:
                return np.zeros_like(x_fin)

        except Exception as e:
            st.error(f"Erreur interpolation ({methode}): {e}")
            return np.zeros_like(x_fin)

    def metriques(self, x_fin: np.ndarray, y_fin: np.ndarray,
                  methode: str) -> dict:
        """Calcule les métriques d'erreur sur les points de données."""
        y_pred = np.interp(self.x, x_fin, y_fin)
        residus = self.y - y_pred
        ss_res = np.sum(residus**2)
        ss_tot = np.sum((self.y - self.y.mean())**2)
        rmse = np.sqrt(np.mean(residus**2))
        mae = np.mean(np.abs(residus))
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        max_err = np.max(np.abs(residus))
        return {
            "R²": r2, "RMSE": rmse, "MAE": mae,
            "Erreur max": max_err,
            "Erreur moy.": np.mean(np.abs(residus)),
            "Biais": np.mean(residus),
        }

    def nœuds_chebyshev(self, n: int, a: float, b: float) -> np.ndarray:
        """Génère n nœuds de Chebyshev sur [a, b]."""
        k = np.arange(n)
        x_cheb = np.cos((2*k + 1) * np.pi / (2*n))
        return 0.5 * (a + b) + 0.5 * (b - a) * x_cheb

    def regression_polynomiale(self, degre: int) -> dict:
        """Régression polynomiale avec intervalles de confiance."""
        try:
            p, cov = np.polyfit(self.x, self.y, degre, cov=True)
            poly = np.poly1d(p)
            y_pred = poly(self.x)
            residus = self.y - y_pred
            ss_res = np.sum(residus**2)
            ss_tot = np.sum((self.y - self.y.mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            n_dof = self.n - degre - 1
            sigma2 = ss_res / max(n_dof, 1)
            return {
                "coefficients": p,
                "r2": r2,
                "rmse": np.sqrt(sigma2),
                "poly": poly,
                "cov": cov
            }
        except:
            return {}

    def benchmark_methodes(self, x_fin: np.ndarray,
                           methodes: list) -> pd.DataFrame:
        """Compare toutes les méthodes sur les mêmes données."""
        resultats = []
        for m in methodes:
            try:
                y_fin = self.interpoler(x_fin, m)
                met = self.metriques(x_fin, y_fin, m)
                resultats.append({
                    "Méthode": m,
                    "R²": f"{met['R²']:.6f}",
                    "RMSE": f"{met['RMSE']:.2e}",
                    "MAE": f"{met['MAE']:.2e}",
                    "Erreur max": f"{met['Erreur max']:.2e}",
                })
            except:
                resultats.append({"Méthode": m, "R²": "❌", "RMSE": "❌",
                                   "MAE": "❌", "Erreur max": "❌"})
        return pd.DataFrame(resultats)

    def detecter_anomalies(self) -> list:
        """Détecte les problèmes potentiels dans les données."""
        issues = []
        if self.n < 3:
            issues.append("⚠️ Peu de points — interpolation peu fiable")
        if np.any(np.diff(self.x) <= 0):
            issues.append("⚠️ Points x non triés ou dupliqués")
        z_scores = np.abs(stats.zscore(self.y))
        outliers = np.where(z_scores > 2.5)[0]
        if len(outliers) > 0:
            issues.append(f"⚠️ Outliers détectés aux indices : {outliers.tolist()}")
        x_range = self.x.max() - self.x.min()
        gaps = np.diff(np.sort(self.x))
        if gaps.max() > 0.5 * x_range:
            issues.append("⚠️ Grand écart entre points — risque d'oscillation")
        if self.n > 10:
            issues.append("ℹ️ N > 10 — préférer Spline ou RBF à Lagrange")
        return issues if issues else ["✅ Données valides"]


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def interpolation_page():
    st.markdown("## 📈 Interpolation & Régression Scientifique")
    st.markdown("*Interpolation avancée, benchmark, analyse d'erreur, régression*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔢 Interpolation",
        "📊 Benchmark méthodes",
        "📉 Régression",
        "🔬 Analyse avancée",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : INTERPOLATION PRINCIPALE
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📊 Saisie des données")

            # Exemples
            ex_sel = st.selectbox("Exemple prédéfini", ["— Manuel —"] + list(EXEMPLES_PREDIFINIS.keys()))
            if ex_sel != "— Manuel —":
                x_def, y_def, desc = EXEMPLES_PREDIFINIS[ex_sel]
                st.caption(f"*{desc}*")
            else:
                x_def, y_def = "0,1,2,3,4", "0,1,4,9,16"

            x_str = st.text_input("**x** (séparés par virgule)", x_def)
            y_str = st.text_input("**y** (séparés par virgule)", y_def)

            try:
                x_data = np.array([float(v.strip()) for v in x_str.split(',')])
                y_data = np.array([float(v.strip()) for v in y_str.split(',')])
                engine = InterpolationEngine(x_data, y_data)
                st.success(f"✅ {len(x_data)} points chargés")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                st.stop()

            # Détection anomalies
            anomalies = engine.detecter_anomalies()
            for a in anomalies:
                if "✅" in a:
                    st.success(a)
                elif "⚠️" in a:
                    st.warning(a)
                else:
                    st.info(a)

            st.markdown("### ⚙️ Configuration")
            methode = st.selectbox("Méthode", [
                "Linéaire", "Cubique (Spline)", "Lagrange",
                "Nearest", "Polynôme ordre N",
                "Chebyshev", "RBF (Gaussien)", "B-Spline"
            ])

            n_points_fin = st.slider("Points interpolés", 50, 2000, 500)

            poly_order = 3
            rbf_kernel = "gaussian"
            if methode in ["Polynôme ordre N", "Chebyshev", "B-Spline"]:
                poly_order = st.slider("Ordre", 1, min(15, engine.n-1), 3)
            if methode == "RBF (Gaussien)":
                rbf_kernel = st.selectbox("Noyau RBF",
                    ["gaussian", "linear", "thin_plate_spline",
                     "cubic", "quintic", "multiquadric"])

            extrapoler = st.checkbox("Extrapolation (±20%)", False)

            # Nœuds de Chebyshev
            show_cheb = st.checkbox("Afficher nœuds de Chebyshev", False)

        with col2:
            x_min, x_max = x_data.min(), x_data.max()
            marge = 0.2 * (x_max - x_min) if extrapoler else 0.0
            x_fin = np.linspace(x_min - marge, x_max + marge, n_points_fin)

            y_fin = engine.interpoler(x_fin, methode, poly_order, rbf_kernel)

            # Graphique principal
            fig = go.Figure()

            # Courbe interpolée
            fig.add_trace(go.Scatter(
                x=x_fin, y=y_fin, mode='lines', name=methode,
                line=dict(color='#00ccff', width=3)
            ))

            # Points originaux
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                name='Points donnés',
                marker=dict(color='#ff00cc', size=12, symbol='circle',
                           line=dict(width=2, color='#ffffff'))
            ))

            # Nœuds de Chebyshev
            if show_cheb:
                x_cheb = engine.nœuds_chebyshev(engine.n, x_min, x_max)
                y_cheb = np.interp(x_cheb, x_fin, y_fin)
                fig.add_trace(go.Scatter(
                    x=x_cheb, y=y_cheb, mode='markers',
                    name='Nœuds Chebyshev',
                    marker=dict(color='#ffcc00', size=10, symbol='diamond')
                ))

            if extrapoler:
                fig.add_vrect(x0=x_min-marge, x1=x_min,
                              fillcolor='rgba(255,100,0,0.1)',
                              line_width=0, annotation_text="extrapol.")
                fig.add_vrect(x0=x_max, x1=x_max+marge,
                              fillcolor='rgba(255,100,0,0.1)',
                              line_width=0, annotation_text="extrapol.")

            fig.update_layout(
                title=f"Interpolation {methode} — {engine.n} points",
                xaxis_title="x", yaxis_title="y",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=450,
            )
            st.plotly_chart(fig, width='stretch')

            # Métriques
            met = engine.metriques(x_fin, y_fin, methode)
            cols_m = st.columns(3)
            with cols_m[0]: st.metric("R²", f"{met['R²']:.6f}")
            with cols_m[1]: st.metric("RMSE", f"{met['RMSE']:.2e}")
            with cols_m[2]: st.metric("Erreur max", f"{met['Erreur max']:.2e}")

            # Erreur aux points
            st.markdown("#### 📉 Erreur résiduelle aux points")
            y_pred_pts = np.interp(x_data, x_fin, y_fin)
            erreurs = np.abs(y_data - y_pred_pts)
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=list(range(len(x_data))), y=erreurs,
                marker=dict(
                    color=erreurs,
                    colorscale=[[0,'#7700ff'],[0.5,'#00ccff'],[1,'#ff0000']],
                    showscale=True
                ),
                name='|résidu|'
            ))
            fig_err.update_layout(
                xaxis_title="Index point", yaxis_title="|y - ŷ|",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                height=280,
            )
            st.plotly_chart(fig_err, width='stretch')

            # Export
            df_exp = pd.DataFrame({"x": x_fin, "y_interp": y_fin})
            st.download_button("💾 Export CSV", df_exp.to_csv(index=False).encode(),
                               "interpolation.csv", "text/csv")

    # ============================================================
    # TAB 2 : BENCHMARK
    # ============================================================
    with tab2:
        st.markdown("### 📊 Benchmark de toutes les méthodes")

        try:
            x_data_b = np.array([float(v.strip()) for v in x_str.split(',')])
            y_data_b = np.array([float(v.strip()) for v in y_str.split(',')])
            engine_b = InterpolationEngine(x_data_b, y_data_b)
            x_fin_b = np.linspace(x_data_b.min(), x_data_b.max(), 500)

            methodes_bench = ["Linéaire", "Cubique (Spline)", "Lagrange",
                              "Nearest", "Polynôme ordre N", "B-Spline"]

            if engine_b.n >= 3:
                methodes_bench.append("RBF (Gaussien)")

            with st.spinner("Calcul benchmark..."):
                df_bench = engine_b.benchmark_methodes(x_fin_b, methodes_bench)

            st.dataframe(df_bench, width='stretch')

            # Barplot RMSE
            fig_bench = go.Figure()
            for _, row in df_bench.iterrows():
                try:
                    rmse_val = float(row["RMSE"])
                    fig_bench.add_trace(go.Bar(
                        x=[row["Méthode"]], y=[rmse_val + 1e-16],
                        name=row["Méthode"],
                        marker_color='#7700ff'
                    ))
                except:
                    pass

            fig_bench.update_layout(
                title="RMSE par méthode",
                yaxis_type='log', yaxis_title="RMSE (log)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                showlegend=False, height=380,
            )
            st.plotly_chart(fig_bench, width='stretch')

            # Graphique comparatif toutes méthodes
            st.markdown("#### Comparaison visuelle")
            fig_comp = go.Figure()
            colors_c = ['#00ccff','#7700ff','#ff00cc','#00ff88',
                        '#ffcc00','#ff4400','#88ccff','#cc88ff']

            for i, m in enumerate(methodes_bench[:6]):
                try:
                    y_m = engine_b.interpoler(x_fin_b, m)
                    fig_comp.add_trace(go.Scatter(
                        x=x_fin_b, y=y_m, mode='lines', name=m,
                        line=dict(color=colors_c[i % len(colors_c)], width=2)
                    ))
                except:
                    pass

            fig_comp.add_trace(go.Scatter(
                x=x_data_b, y=y_data_b, mode='markers', name='Données',
                marker=dict(color='#ffffff', size=12, symbol='circle')
            ))
            fig_comp.update_layout(
                title="Comparaison visuelle des méthodes",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=420,
            )
            st.plotly_chart(fig_comp, width='stretch')

        except Exception as e:
            st.error(f"Erreur benchmark: {e}")

    # ============================================================
    # TAB 3 : RÉGRESSION
    # ============================================================
    with tab3:
        st.markdown("### 📉 Régression polynomiale & analyse")

        try:
            x_r = np.array([float(v.strip()) for v in x_str.split(',')])
            y_r = np.array([float(v.strip()) for v in y_str.split(',')])
            engine_r = InterpolationEngine(x_r, y_r)

            col1, col2 = st.columns([1, 2])
            with col1:
                deg_reg = st.slider("Degré de régression", 1, min(8, engine_r.n-1), 2)
                res_reg = engine_r.regression_polynomiale(deg_reg)

                if res_reg:
                    st.metric("R²", f"{res_reg['r2']:.6f}")
                    st.metric("RMSE", f"{res_reg['rmse']:.4e}")
                    st.markdown("**Coefficients :**")
                    for i, c in enumerate(res_reg["coefficients"]):
                        st.text(f"  a{deg_reg-i} = {c:.6e}")

            with col2:
                if res_reg:
                    x_fin_r = np.linspace(x_r.min(), x_r.max(), 500)
                    y_reg = res_reg["poly"](x_fin_r)
                    y_pred_r = res_reg["poly"](x_r)
                    residus_r = y_r - y_pred_r

                    fig_reg = make_subplots(rows=2, cols=1,
                        subplot_titles=["Régression polynomiale", "Résidus"])

                    fig_reg.add_trace(go.Scatter(
                        x=x_r, y=y_r, mode='markers', name='Données',
                        marker=dict(color='#ff00cc', size=10)
                    ), row=1, col=1)
                    fig_reg.add_trace(go.Scatter(
                        x=x_fin_r, y=y_reg, mode='lines', name=f'Deg {deg_reg}',
                        line=dict(color='#00ccff', width=3)
                    ), row=1, col=1)

                    fig_reg.add_trace(go.Bar(
                        x=list(range(len(x_r))), y=residus_r,
                        name='Résidus', marker_color='#7700ff'
                    ), row=2, col=1)
                    fig_reg.add_hline(y=0, line_color='white', line_dash='dash',
                                      row=2, col=1)

                    fig_reg.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.92)',
                        font=dict(color='#c0d0ff'),
                        height=520,
                        legend=dict(bgcolor='rgba(0,0,0,0.5)')
                    )
                    fig_reg.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                    fig_reg.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                    st.plotly_chart(fig_reg, width='stretch')

            # Comparaison degrés
            st.markdown("#### 📈 R² vs degré")
            degs = list(range(1, min(engine_r.n, 9)))
            r2_vals = []
            for d in degs:
                res_d = engine_r.regression_polynomiale(d)
                r2_vals.append(res_d.get("r2", 0) if res_d else 0)

            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(
                x=degs, y=r2_vals, mode='lines+markers',
                line=dict(color='#00ccff', width=2.5),
                marker=dict(size=8, color='#7700ff')
            ))
            fig_r2.add_hline(y=1.0, line_color='#ffcc00', line_dash='dash',
                             annotation_text="R²=1")
            fig_r2.update_layout(
                xaxis_title="Degré", yaxis_title="R²",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                          range=[0, 1.05]),
                height=300,
            )
            st.plotly_chart(fig_r2, width='stretch')

        except Exception as e:
            st.error(f"Erreur régression: {e}")

    # ============================================================
    # TAB 4 : ANALYSE AVANCÉE
    # ============================================================
    with tab4:
        st.markdown("### 🔬 Analyse avancée — Phénomène de Runge")

        col1, col2 = st.columns([1, 2])
        with col1:
            n_runge = st.slider("Nombre de points uniformes", 3, 20, 10)
            deg_runge = st.slider("Degré Lagrange", 2, 19, 9)

        with col2:
            x_true = np.linspace(-1, 1, 500)
            y_true = 1 / (1 + 25 * x_true**2)

            # Points uniformes
            x_unif = np.linspace(-1, 1, n_runge)
            y_unif = 1 / (1 + 25 * x_unif**2)

            # Points Chebyshev
            engine_r2 = InterpolationEngine(x_unif, y_unif)
            k = np.arange(n_runge)
            x_cheb_r = np.cos((2*k + 1) * np.pi / (2*n_runge))
            x_cheb_r = np.sort(x_cheb_r)
            y_cheb_r = 1 / (1 + 25 * x_cheb_r**2)

            engine_cheb = InterpolationEngine(x_cheb_r, y_cheb_r)
            y_lagr_unif = engine_r2.interpoler(x_true, "Lagrange")
            y_lagr_cheb = engine_cheb.interpoler(x_true, "Lagrange")
            y_spline = engine_r2.interpoler(x_true, "Cubique (Spline)")

            fig_runge = go.Figure()
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_true, mode='lines', name='f(x) vraie',
                line=dict(color='#ffffff', width=2)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_lagr_unif, mode='lines', name='Lagrange (uniforme)',
                line=dict(color='#ff4444', width=2.5)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_lagr_cheb, mode='lines', name='Lagrange (Chebyshev)',
                line=dict(color='#00ff88', width=2.5)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_spline, mode='lines', name='Spline cubique',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_unif, y=y_unif, mode='markers', name='Pts uniformes',
                marker=dict(color='#ff4444', size=8, symbol='circle')
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_cheb_r, y=y_cheb_r, mode='markers', name='Pts Chebyshev',
                marker=dict(color='#00ff88', size=8, symbol='diamond')
            ))
            fig_runge.update_layout(
                title=f"Phénomène de Runge — f(x)=1/(1+25x²), n={n_runge}",
                xaxis_title="x", yaxis_title="y",
                yaxis=dict(range=[-2, 2]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=480,
            )
            st.plotly_chart(fig_runge, width='stretch')

            st.info("🔑 Les nœuds de Chebyshev éliminent le phénomène de Runge en concentrant les points aux bords de l'intervalle.")

    # ============================================================
    # TAB 5 : THÉORIE
    # ============================================================
    with tab5:
        st.markdown("### 📖 Formulaire scientifique")

        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES_INTERPOLATION.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 📊 Comparaison des méthodes")
        df_comp = pd.DataFrame([
            {"Méthode": "Linéaire", "Continuité": "C⁰", "Oscillation": "Non",
             "Complexité": "O(n)", "Usage": "Données irrégulières"},
            {"Méthode": "Spline cubique", "Continuité": "C²", "Oscillation": "Rare",
             "Complexité": "O(n)", "Usage": "Données lisses ✅"},
            {"Méthode": "Lagrange", "Continuité": "C∞", "Oscillation": "Oui (Runge)",
             "Complexité": "O(n²)", "Usage": "Peu de points"},
            {"Méthode": "Chebyshev", "Continuité": "C∞", "Oscillation": "Réduite",
             "Complexité": "O(n²)", "Usage": "Nœuds optimaux ✅"},
            {"Méthode": "RBF", "Continuité": "C∞", "Oscillation": "Rare",
             "Complexité": "O(n³)", "Usage": "Multi-dimensionnel"},
            {"Méthode": "B-Spline", "Continuité": "Cᵏ⁻¹", "Oscillation": "Non",
             "Complexité": "O(n)", "Usage": "CAO, animation"},
        ])
        st.dataframe(df_comp, width='stretch')

        st.markdown("---")
        st.markdown("### 📚 Références")
        refs = [
            "Burden & Faires — *Numerical Analysis* (Cengage, 2010)",
            "Press et al. — *Numerical Recipes in Python* (Cambridge, 2007)",
            "Stoer & Bulirsch — *Introduction to Numerical Analysis* (Springer, 2002)",
            "De Boor — *A Practical Guide to Splines* (Springer, 2001)",
        ]
        for r in refs:
            st.markdown(f"- {r}")

################################################################################
# Contenu de equ_diff.py
################################################################################

# ============================================================
# FORMULAIRE
# ============================================================
FORMULES_EQU_DIFF = {
    "EDO ordre 1":           r"\frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0",
    "EDO ordre 2":           r"\frac{d^2y}{dt^2} + 2\zeta\omega_n\frac{dy}{dt} + \omega_n^2 y = 0",
    "Méthode d'Euler":       r"y_{n+1} = y_n + h\,f(t_n, y_n)",
    "Runge-Kutta 4":         r"y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)",
    "Lorenz":                r"\dot{x}=\sigma(y-x),\quad\dot{y}=x(\rho-z)-y,\quad\dot{z}=xy-\beta z",
    "Lotka-Volterra":        r"\dot{x}=\alpha x - \beta xy,\quad \dot{y}=-\gamma y + \delta xy",
    "Logistique":            r"\frac{dy}{dt} = ky\left(1 - \frac{y}{M}\right)",
    "Oscillateur Van der Pol":r"\ddot{x} - \mu(1-x^2)\dot{x} + x = 0",
    "Pendule non-linéaire":  r"\ddot{\theta} + \frac{g}{L}\sin\theta + b\dot{\theta} = 0",
    "Équation de la chaleur":r"\frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2}",
    "SIR (épidémiologie)":   r"\dot{S}=-\beta SI,\quad\dot{I}=\beta SI-\gamma I,\quad\dot{R}=\gamma I",
    "Erreur RK4":            r"|e_n| \leq C h^4 \quad \text{(ordre 4)}",
}

SYSTEMES = {
    "Décroissance exponentielle": "dy/dt = -k·y",
    "Croissance exponentielle":   "dy/dt = +k·y",
    "Oscillateur harmonique":     "ÿ + ω²y = 0",
    "Oscillateur amorti":         "ÿ + 2ζω·ẏ + ω²y = 0",
    "Oscillateur Van der Pol":    "ÿ - μ(1-y²)ẏ + y = 0",
    "Logistique":                 "dy/dt = k·y(1-y/M)",
    "Lotka-Volterra":             "ẋ=αx-βxy, ẏ=-γy+δxy",
    "Lorenz (chaos)":             "système 3D chaotique",
    "Pendule non-linéaire":       "ÿ + (g/L)sinθ + bθ̇ = 0",
    "SIR épidémique":             "S,I,R couplés",
    "Double pendule":             "système chaotique à 4D",
    "Équation de Duffing":        "ÿ + δẏ + αy + βy³ = γcos(ωt)",
}

METHODES_NUMERIQUES = {
    "Euler explicite":  {"ordre": 1, "stabilité": "Conditionnelle", "coût": "O(n)"},
    "Euler implicite":  {"ordre": 1, "stabilité": "Inconditionnelle","coût": "O(n²)"},
    "RK4":              {"ordre": 4, "stabilité": "Conditionnelle", "coût": "O(4n)"},
    "RK45 (adaptatif)": {"ordre": 5, "stabilité": "Adaptative",    "coût": "Variable"},
    "LSODA (scipy)":    {"ordre": 12,"stabilité": "Adaptative",    "coût": "Variable"},
}


# ============================================================
# SOLVEURS NUMÉRIQUES
# ============================================================
class SolveurODE:
    """Solveurs numériques pour EDO avec comparaison de méthodes."""

    @staticmethod
    def euler_explicite(f, y0, t):
        y = np.zeros((len(t), len(np.atleast_1d(y0))))
        y[0] = np.atleast_1d(y0)
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            y[i+1] = y[i] + h * np.atleast_1d(f(t[i], y[i]))
        return y

    @staticmethod
    def euler_implicite(f, y0, t, tol=1e-8):
        """Euler implicite via point fixe."""
        from scipy.optimize import fsolve
        y = np.zeros((len(t), len(np.atleast_1d(y0))))
        y[0] = np.atleast_1d(y0)
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            y_pred = y[i] + h * np.atleast_1d(f(t[i], y[i]))
            try:
                y[i+1] = fsolve(lambda yn: yn - y[i] - h*np.atleast_1d(f(t[i+1], yn)),
                                 y_pred, full_output=False)
            except:
                y[i+1] = y_pred
        return y

    @staticmethod
    def rk4(f, y0, t):
        y = np.zeros((len(t), len(np.atleast_1d(y0))))
        y[0] = np.atleast_1d(y0)
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            yi = y[i]
            k1 = np.atleast_1d(f(t[i],         yi))
            k2 = np.atleast_1d(f(t[i]+h/2,     yi+h/2*k1))
            k3 = np.atleast_1d(f(t[i]+h/2,     yi+h/2*k2))
            k4 = np.atleast_1d(f(t[i]+h,       yi+h*k3))
            y[i+1] = yi + h/6*(k1 + 2*k2 + 2*k3 + k4)
        return y

    @staticmethod
    def rk45_scipy(f_ivp, y0, t):
        sol = solve_ivp(f_ivp, [t[0], t[-1]], np.atleast_1d(y0),
                        t_eval=t, method='RK45', rtol=1e-8, atol=1e-10)
        return sol.y.T

    @staticmethod
    def lsoda_scipy(f_odeint, y0, t):
        return odeint(f_odeint, np.atleast_1d(y0), t, rtol=1e-8, atol=1e-10)


# ============================================================
# MOTEUR EDO
# ============================================================
class EquDiffEngine:
    """Moteur complet de résolution et analyse d'EDO."""

    def __init__(self, eq_type: str, params: dict):
        self.eq_type = eq_type
        self.params = params
        self.solveur = SolveurODE()

    def definir_systeme(self):
        """Retourne (f_odeint, f_ivp, y0, n_dim, labels)."""
        p = self.params

        if self.eq_type == "Décroissance exponentielle":
            f = lambda y, t: -p["k"] * y
            f_ivp = lambda t, y: [-p["k"] * y[0]]
            y0 = [p["y0"]]
            labels = ["y(t)"]

        elif self.eq_type == "Croissance exponentielle":
            f = lambda y, t: p["k"] * y
            f_ivp = lambda t, y: [p["k"] * y[0]]
            y0 = [p["y0"]]
            labels = ["y(t)"]

        elif self.eq_type == "Oscillateur harmonique":
            omega = p["k"]
            f = lambda Y, t: [Y[1], -omega**2 * Y[0]]
            f_ivp = lambda t, Y: [Y[1], -omega**2 * Y[0]]
            y0 = [p["y0"], 0]
            labels = ["y(t)", "ẏ(t)"]

        elif self.eq_type == "Oscillateur amorti":
            omega, zeta = p["k"], p.get("zeta", 0.3)
            f = lambda Y, t: [Y[1], -2*zeta*omega*Y[1] - omega**2*Y[0]]
            f_ivp = lambda t, Y: [Y[1], -2*zeta*omega*Y[1] - omega**2*Y[0]]
            y0 = [p["y0"], 0]
            labels = ["y(t)", "ẏ(t)"]

        elif self.eq_type == "Oscillateur Van der Pol":
            mu = p.get("mu", 1.0)
            f = lambda Y, t: [Y[1], mu*(1-Y[0]**2)*Y[1] - Y[0]]
            f_ivp = lambda t, Y: [Y[1], mu*(1-Y[0]**2)*Y[1] - Y[0]]
            y0 = [p["y0"], 0]
            labels = ["y(t)", "ẏ(t)"]

        elif self.eq_type == "Logistique":
            k, M = p["k"], p.get("M", 10.0)
            f = lambda y, t: k * y * (1 - y/M)
            f_ivp = lambda t, y: [k * y[0] * (1 - y[0]/M)]
            y0 = [p["y0"]]
            labels = ["N(t)"]

        elif self.eq_type == "Lotka-Volterra":
            alpha, beta = p.get("alpha", 1.0), p.get("beta", 0.1)
            gamma, delta = p.get("gamma", 0.075), p.get("delta", 1.5)
            f = lambda Y, t: [alpha*Y[0] - beta*Y[0]*Y[1],
                               -gamma*Y[1] + delta*beta*Y[0]*Y[1]]
            f_ivp = lambda t, Y: [alpha*Y[0] - beta*Y[0]*Y[1],
                                   -gamma*Y[1] + delta*beta*Y[0]*Y[1]]
            y0 = [p["y0"], p.get("y0_pred", 1.0)]
            labels = ["Proies", "Prédateurs"]

        elif self.eq_type == "Lorenz (chaos)":
            sigma = p.get("sigma", 10.0)
            rho = p.get("rho", 28.0)
            beta = p.get("beta_l", 8/3)
            f = lambda Y, t: [sigma*(Y[1]-Y[0]),
                               Y[0]*(rho-Y[2])-Y[1],
                               Y[0]*Y[1]-beta*Y[2]]
            f_ivp = lambda t, Y: [sigma*(Y[1]-Y[0]),
                                   Y[0]*(rho-Y[2])-Y[1],
                                   Y[0]*Y[1]-beta*Y[2]]
            y0 = [1.0, 1.0, 1.0]
            labels = ["x", "y", "z"]

        elif self.eq_type == "Pendule non-linéaire":
            g, L, b = 9.81, p.get("L", 1.0), p.get("b", 0.1)
            f = lambda Y, t: [Y[1], -(g/L)*np.sin(Y[0]) - b*Y[1]]
            f_ivp = lambda t, Y: [Y[1], -(g/L)*np.sin(Y[0]) - b*Y[1]]
            y0 = [p["y0"], 0]
            labels = ["θ(t)", "ω(t)"]

        elif self.eq_type == "SIR épidémique":
            N = p.get("N_pop", 1000)
            beta_s = p.get("beta_sir", 0.3)
            gamma_s = p.get("gamma_sir", 0.05)
            f = lambda Y, t: [-beta_s*Y[0]*Y[1]/N,
                               beta_s*Y[0]*Y[1]/N - gamma_s*Y[1],
                               gamma_s*Y[1]]
            f_ivp = lambda t, Y: [-beta_s*Y[0]*Y[1]/N,
                                   beta_s*Y[0]*Y[1]/N - gamma_s*Y[1],
                                   gamma_s*Y[1]]
            y0 = [N - 1, 1.0, 0.0]
            labels = ["S (Susceptibles)", "I (Infectés)", "R (Guéris)"]

        elif self.eq_type == "Équation de Duffing":
            delta = p.get("delta", 0.2)
            alpha_d = p.get("alpha_d", -1.0)
            beta_d = p.get("beta_d", 1.0)
            gamma_d = p.get("gamma_d", 0.3)
            omega_d = p.get("omega_d", 1.2)
            f = lambda Y, t: [Y[1], gamma_d*np.cos(omega_d*t) - delta*Y[1] - alpha_d*Y[0] - beta_d*Y[0]**3]
            f_ivp = lambda t, Y: [Y[1], gamma_d*np.cos(omega_d*t) - delta*Y[1] - alpha_d*Y[0] - beta_d*Y[0]**3]
            y0 = [0.1, 0.0]
            labels = ["x(t)", "ẋ(t)"]

        else:
            f = lambda y, t: -p["k"] * y
            f_ivp = lambda t, y: [-p["k"] * y[0]]
            y0 = [p["y0"]]
            labels = ["y(t)"]

        return f, f_ivp, y0, labels

    def resoudre(self, methode: str, t: np.ndarray) -> np.ndarray:
        f, f_ivp, y0, labels = self.definir_systeme()
        try:
            if methode == "Euler explicite":
                return self.solveur.euler_explicite(lambda t, y: np.array(f(y, t)), y0, t), labels
            elif methode == "Euler implicite":
                return self.solveur.euler_implicite(lambda t, y: np.array(f(y, t)), y0, t), labels
            elif methode == "RK4":
                return self.solveur.rk4(lambda t, y: np.array(f(y, t)), y0, t), labels
            elif methode == "RK45 (adaptatif)":
                return self.solveur.rk45_scipy(f_ivp, y0, t), labels
            else:  # LSODA
                return self.solveur.lsoda_scipy(f, y0, t), labels
        except Exception as e:
            return np.zeros((len(t), len(y0))), labels

    def solution_analytique(self, t: np.ndarray):
        p = self.params
        if self.eq_type == "Décroissance exponentielle":
            return p["y0"] * np.exp(-p["k"] * t)
        elif self.eq_type == "Croissance exponentielle":
            return p["y0"] * np.exp(p["k"] * t)
        elif self.eq_type == "Oscillateur harmonique":
            return p["y0"] * np.cos(p["k"] * t)
        return None

    def erreur_methodes(self, t: np.ndarray) -> pd.DataFrame:
        """Erreur de chaque méthode vs solution analytique."""
        y_exact = self.solution_analytique(t)
        if y_exact is None:
            return pd.DataFrame()

        methodes = ["Euler explicite", "RK4", "RK45 (adaptatif)", "LSODA (scipy)"]
        resultats = []
        for m in methodes:
            sol, _ = self.resoudre(m, t)
            err = np.abs(sol[:, 0] - y_exact)
            resultats.append({
                "Méthode": m,
                "Erreur max": f"{err.max():.2e}",
                "Erreur moy.": f"{err.mean():.2e}",
                "Erreur finale": f"{err[-1]:.2e}",
            })
        return pd.DataFrame(resultats)


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def equ_diff_page():
    st.markdown("## 🧮 Équations Différentielles Avancées")
    st.markdown("*Solveurs numériques, systèmes dynamiques, chaos, benchmark*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔢 Résolution",
        "🌐 Portrait de phase",
        "⚔️ Benchmark méthodes",
        "🔬 Systèmes spéciaux",
        "📖 Théorie"
    ])

    # Config partagée
    with st.sidebar.expander("⚙️ Équation différentielle", expanded=True):
        eq_type = st.selectbox("Système", list(SYSTEMES.keys()))
        st.caption(SYSTEMES[eq_type])

        y0_val = st.slider("y(0) / x(0)", 0.1, 20.0, 1.0, 0.1)
        k_val = st.slider("k / ω", 0.01, 5.0, 0.5, 0.01)
        t_max = st.slider("Temps max (s)", 5.0, 200.0, 30.0, 1.0)
        n_pts = st.slider("Points", 200, 5000, 1000, 100)

        params = {"y0": y0_val, "k": k_val}

        # Paramètres spécifiques
        if eq_type == "Oscillateur amorti":
            params["zeta"] = st.slider("ζ (amortissement)", 0.0, 3.0, 0.3, 0.05)
        elif eq_type == "Oscillateur Van der Pol":
            params["mu"] = st.slider("μ", 0.1, 5.0, 1.0, 0.1)
        elif eq_type == "Logistique":
            params["M"] = st.slider("Capacité M", 1.0, 100.0, 10.0, 1.0)
        elif eq_type == "Lotka-Volterra":
            params["alpha"] = st.slider("α (croissance proies)", 0.1, 3.0, 1.0, 0.1)
            params["beta"] = st.slider("β (prédation)", 0.01, 1.0, 0.1, 0.01)
            params["gamma"] = st.slider("γ (mort prédateurs)", 0.01, 1.0, 0.075, 0.001)
            params["delta"] = st.slider("δ (assimilation)", 0.1, 3.0, 1.5, 0.1)
            params["y0_pred"] = st.slider("Prédateurs(0)", 0.1, 20.0, 1.0, 0.1)
        elif eq_type == "Lorenz (chaos)":
            params["sigma"] = st.slider("σ", 1.0, 20.0, 10.0, 0.5)
            params["rho"] = st.slider("ρ", 1.0, 50.0, 28.0, 0.5)
            params["beta_l"] = st.slider("β", 0.1, 5.0, 8/3, 0.1)
        elif eq_type == "Pendule non-linéaire":
            params["L"] = st.slider("Longueur L (m)", 0.1, 5.0, 1.0, 0.1)
            params["b"] = st.slider("Amortissement b", 0.0, 2.0, 0.1, 0.01)
        elif eq_type == "SIR épidémique":
            params["N_pop"] = st.slider("Population N", 100, 10000, 1000, 100)
            params["beta_sir"] = st.slider("β (contagion)", 0.01, 1.0, 0.3, 0.01)
            params["gamma_sir"] = st.slider("γ (guérison)", 0.01, 0.5, 0.05, 0.01)
        elif eq_type == "Équation de Duffing":
            params["delta"] = st.slider("δ (amortissement)", 0.0, 2.0, 0.2, 0.01)
            params["alpha_d"] = st.slider("α", -2.0, 2.0, -1.0, 0.1)
            params["beta_d"] = st.slider("β", 0.1, 5.0, 1.0, 0.1)
            params["gamma_d"] = st.slider("γ (force)", 0.0, 1.0, 0.3, 0.01)
            params["omega_d"] = st.slider("ω_d", 0.1, 5.0, 1.2, 0.05)

    engine = EquDiffEngine(eq_type, params)
    t = np.linspace(0, t_max, n_pts)

    # ============================================================
    # TAB 1 : RÉSOLUTION
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Solveur")
            methode = st.selectbox("Méthode numérique", list(METHODES_NUMERIQUES.keys()))
            info_m = METHODES_NUMERIQUES[methode]
            st.markdown(f"""
            - **Ordre :** {info_m['ordre']}
            - **Stabilité :** {info_m['stabilité']}
            - **Coût :** {info_m['coût']}
            """)
            show_exact = st.checkbox("Afficher solution exacte (si disponible)", True)
            show_energie = st.checkbox("Afficher énergie", False)

        with col2:
            sol, labels = engine.resoudre(methode, t)

            colors_sol = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88']
            fig = go.Figure()

            # Solution numérique
            for i, lab in enumerate(labels):
                fig.add_trace(go.Scatter(
                    x=t, y=sol[:, i], mode='lines', name=f'{lab} ({methode})',
                    line=dict(color=colors_sol[i % len(colors_sol)], width=2.5)
                ))

            # Solution analytique
            if show_exact:
                y_exact = engine.solution_analytique(t)
                if y_exact is not None:
                    fig.add_trace(go.Scatter(
                        x=t, y=y_exact, mode='lines', name='Solution exacte',
                        line=dict(color='#ffffff', width=2, dash='dash')
                    ))

            fig.update_layout(
                title=f"{eq_type} — Méthode : {methode}",
                xaxis_title="Temps (s)", yaxis_title="État",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Métriques
            c1, c2, c3 = st.columns(3)
            with c1: st.metric(f"{labels[0]} final", f"{sol[-1, 0]:.4f}")
            with c2: st.metric(f"Max {labels[0]}", f"{sol[:, 0].max():.4f}")
            with c3: st.metric(f"Min {labels[0]}", f"{sol[:, 0].min():.4f}")

            # Énergie oscillateur
            if show_energie and sol.shape[1] >= 2 and eq_type in [
                "Oscillateur harmonique", "Oscillateur amorti", "Oscillateur Van der Pol"
            ]:
                E_cin = 0.5 * sol[:, 1]**2
                E_pot = 0.5 * params["k"]**2 * sol[:, 0]**2
                E_tot = E_cin + E_pot
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=t, y=E_cin, name='E cinétique',
                                           line=dict(color='#00ccff', width=2)))
                fig_e.add_trace(go.Scatter(x=t, y=E_pot, name='E potentielle',
                                           line=dict(color='#7700ff', width=2)))
                fig_e.add_trace(go.Scatter(x=t, y=E_tot, name='E totale',
                                           line=dict(color='#ffffff', width=2.5, dash='dash')))
                fig_e.update_layout(
                    title="Énergie du système",
                    xaxis_title="t (s)", yaxis_title="Énergie (u.a.)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320,
                )
                st.plotly_chart(fig_e, use_container_width=True)

            # Export
            df_exp = pd.DataFrame({"t": t, **{lab: sol[:, i]
                                               for i, lab in enumerate(labels)}})
            st.download_button("💾 Export CSV", df_exp.to_csv(index=False).encode(),
                               "ode_solution.csv", "text/csv")

    # ============================================================
    # TAB 2 : PORTRAIT DE PHASE
    # ============================================================
    with tab2:
        st.markdown("### 🌐 Portrait de phase")

        sol_ph, labels_ph = engine.resoudre("RK45 (adaptatif)", t)

        if sol_ph.shape[1] >= 2:
            if sol_ph.shape[1] >= 3:
                fig_ph = go.Figure(data=go.Scatter3d(
                    x=sol_ph[:, 0], y=sol_ph[:, 1], z=sol_ph[:, 2],
                    mode='lines',
                    line=dict(color=t, colorscale=[[0,'#7700ff'],[0.5,'#00ccff'],[1,'#ffffff']],
                             width=2),
                    name='Trajectoire 3D'
                ))
                fig_title = "Attracteur de Lorenz" if eq_type == "Lorenz (chaos)" else f"Portrait de phase 3D — {eq_type}"
                fig_ph.update_layout(
                    title=fig_title,
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff'), yaxis=dict(color='#c0d0ff'),
                        zaxis=dict(color='#c0d0ff')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c0d0ff'),
                    height=550, margin=dict(l=0, r=0, t=40, b=0)
                )
            else:
                fig_ph = go.Figure()
                fig_ph.add_trace(go.Scatter(
                    x=sol_ph[:, 0], y=sol_ph[:, 1], mode='lines',
                    name=f'{labels_ph[0]} vs {labels_ph[1]}',
                    line=dict(color='#00ccff', width=2),
                ))
                # Point initial
                fig_ph.add_trace(go.Scatter(
                    x=[sol_ph[0, 0]], y=[sol_ph[0, 1]], mode='markers',
                    name='Départ', marker=dict(color='#ffcc00', size=14, symbol='star')
                ))
                fig_ph.update_layout(
                    title=f"Portrait de phase — {eq_type}",
                    xaxis_title=labels_ph[0], yaxis_title=labels_ph[1],
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=500,
                )
            st.plotly_chart(fig_ph, use_container_width=True)

            # Champ de vecteurs
            if sol_ph.shape[1] == 2 and eq_type not in ["Lorenz (chaos)"]:
                st.markdown("#### 🧭 Champ de vecteurs")
                x_range = np.linspace(sol_ph[:, 0].min()*1.2, sol_ph[:, 0].max()*1.2, 15)
                y_range = np.linspace(sol_ph[:, 1].min()*1.2, sol_ph[:, 1].max()*1.2, 15)
                X_g, Y_g = np.meshgrid(x_range, y_range)
                _, f_ivp, _, _ = engine.definir_systeme()

                U_g = np.zeros_like(X_g)
                V_g = np.zeros_like(Y_g)
                for i in range(X_g.shape[0]):
                    for j in range(X_g.shape[1]):
                        try:
                            dxy = f_ivp(0, [X_g[i,j], Y_g[i,j]])
                            norm = np.sqrt(dxy[0]**2 + dxy[1]**2) + 1e-8
                            U_g[i,j] = dxy[0] / norm
                            V_g[i,j] = dxy[1] / norm
                        except:
                            pass

                fig_vf = go.Figure()
                fig_vf.add_trace(go.Scatter(
                    x=sol_ph[:, 0], y=sol_ph[:, 1], mode='lines', name='Trajectoire',
                    line=dict(color='#00ccff', width=2.5)
                ))
                # Flèches
                for i in range(0, X_g.shape[0], 2):
                    for j in range(0, X_g.shape[1], 2):
                        fig_vf.add_annotation(
                            x=X_g[i,j] + U_g[i,j]*0.2,
                            y=Y_g[i,j] + V_g[i,j]*0.2,
                            ax=X_g[i,j], ay=Y_g[i,j],
                            xref='x', yref='y', axref='x', ayref='y',
                            arrowhead=2, arrowsize=1,
                            arrowcolor='rgba(119,0,255,0.5)',
                            arrowwidth=1.5, showarrow=True
                        )
                fig_vf.update_layout(
                    title="Champ de vecteurs",
                    xaxis_title=labels_ph[0], yaxis_title=labels_ph[1],
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    height=430,
                )
                st.plotly_chart(fig_vf, use_container_width=True)
        else:
            st.info("Portrait de phase disponible pour systèmes d'ordre ≥ 2.")

    # ============================================================
    # TAB 3 : BENCHMARK
    # ============================================================
    with tab3:
        st.markdown("### ⚔️ Benchmark des méthodes numériques")

        if eq_type in ["Décroissance exponentielle", "Croissance exponentielle",
                        "Oscillateur harmonique"]:
            df_err = engine.erreur_methodes(t)
            if not df_err.empty:
                st.dataframe(df_err, use_container_width=True)

                # Courbes d'erreur
                fig_bench = go.Figure()
                methodes_b = ["Euler explicite", "RK4", "RK45 (adaptatif)"]
                colors_b = ['#ff4444', '#7700ff', '#00ccff']
                y_exact = engine.solution_analytique(t)

                for m, col in zip(methodes_b, colors_b):
                    sol_b, _ = engine.resoudre(m, t)
                    err = np.abs(sol_b[:, 0] - y_exact)
                    fig_bench.add_trace(go.Scatter(
                        x=t, y=err, mode='lines', name=m,
                        line=dict(color=col, width=2)
                    ))
                fig_bench.update_layout(
                    title="Erreur |numérique - exact| par méthode",
                    xaxis_title="t (s)", yaxis_title="Erreur absolue",
                    yaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
                )
                st.plotly_chart(fig_bench, use_container_width=True)

            # Effet du pas h
            st.markdown("#### 📉 Convergence vs pas h")
            y_exact_fin = engine.solution_analytique(np.array([t_max]))[0]
            h_vals = np.logspace(-3, 0, 20)
            errs_euler, errs_rk4 = [], []
            for h in h_vals:
                t_h = np.arange(0, t_max + h, h)
                sol_e, _ = engine.resoudre("Euler explicite", t_h)
                sol_r, _ = engine.resoudre("RK4", t_h)
                errs_euler.append(abs(sol_e[-1, 0] - y_exact_fin))
                errs_rk4.append(abs(sol_r[-1, 0] - y_exact_fin))

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=h_vals, y=errs_euler, mode='lines+markers',
                name='Euler (ordre 1)', line=dict(color='#ff4444', width=2)))
            fig_conv.add_trace(go.Scatter(x=h_vals, y=errs_rk4, mode='lines+markers',
                name='RK4 (ordre 4)', line=dict(color='#00ccff', width=2)))
            # Lignes de référence
            fig_conv.add_trace(go.Scatter(x=h_vals, y=h_vals**1 * errs_euler[0]/h_vals[0],
                mode='lines', name='O(h)', line=dict(color='rgba(255,100,0,0.5)', dash='dot')))
            fig_conv.add_trace(go.Scatter(x=h_vals, y=h_vals**4 * errs_rk4[0]/h_vals[0]**4,
                mode='lines', name='O(h⁴)', line=dict(color='rgba(0,200,255,0.5)', dash='dot')))

            fig_conv.update_layout(
                title="Ordre de convergence des méthodes",
                xaxis_title="Pas h", yaxis_title="Erreur |y(T)-y_exact(T)|",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("Le benchmark avec solution exacte est disponible pour les systèmes à solution analytique connue (décroissance, croissance, oscillateur harmonique).")

        st.markdown("#### 📊 Comparaison des méthodes")
        df_meth = pd.DataFrame([
            {"Méthode": k, **v} for k, v in METHODES_NUMERIQUES.items()
        ])
        st.dataframe(df_meth, use_container_width=True)

    # ============================================================
    # TAB 4 : SYSTÈMES SPÉCIAUX
    # ============================================================
    with tab4:
        st.markdown("### 🔬 Systèmes spéciaux")

        special = st.radio("Système", [
            "Sensibilité aux conditions initiales (Lorenz)",
            "Équation de la chaleur 1D (EDP)",
            "SIR — R₀ et seuil épidémique",
        ], horizontal=False)

        if special == "Sensibilité aux conditions initiales (Lorenz)":
            st.markdown("*Effet papillon : deux trajectoires proches divergent exponentiellement*")
            perturbation = st.slider("Perturbation ε", 1e-10, 1e-2, 1e-5, format="%.0e")
            t_lor = np.linspace(0, 40, 5000)

            def lorenz_sys(Y, t):
                return [10*(Y[1]-Y[0]), Y[0]*(28-Y[2])-Y[1], Y[0]*Y[1]-(8/3)*Y[2]]

            sol1 = odeint(lorenz_sys, [1.0, 1.0, 1.0], t_lor)
            sol2 = odeint(lorenz_sys, [1.0+perturbation, 1.0, 1.0], t_lor)
            dist = np.sqrt(np.sum((sol1 - sol2)**2, axis=1))

            fig_s = make_subplots(rows=2, cols=1,
                subplot_titles=["x(t) — deux trajectoires", "Distance ||Δy(t)||"])
            fig_s.add_trace(go.Scatter(x=t_lor, y=sol1[:,0], name='Trajectoire 1',
                line=dict(color='#00ccff', width=1.5)), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=t_lor, y=sol2[:,0], name=f'Trajectoire 2 (ε={perturbation:.0e})',
                line=dict(color='#ff00cc', width=1.5, dash='dot')), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=t_lor, y=dist, name='Distance',
                line=dict(color='#ffcc00', width=2)), row=2, col=1)
            fig_s.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'), height=520,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_s.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_s.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_s, use_container_width=True)

        elif special == "Équation de la chaleur 1D (EDP)":
            st.markdown("*Discrétisation par différences finies*")
            alpha_ch = st.slider("Diffusivité α", 0.01, 1.0, 0.1, 0.01)
            Nx = st.slider("Nx (points spatiaux)", 20, 100, 40)
            Nt = st.slider("Nt (pas temporels)", 100, 2000, 500)

            L_ch = 1.0
            T_max_ch = 1.0
            dx = L_ch / (Nx - 1)
            dt = T_max_ch / Nt
            r = alpha_ch * dt / dx**2

            if r > 0.5:
                st.warning(f"⚠️ r = {r:.3f} > 0.5 → schéma instable! Réduire α ou dt.")

            x_ch = np.linspace(0, L_ch, Nx)
            u = np.sin(np.pi * x_ch)  # CI
            u_exact = lambda t: np.sin(np.pi * x_ch) * np.exp(-alpha_ch * np.pi**2 * t)

            snapshots = {0.0: u.copy()}
            for n in range(Nt):
                u_new = u.copy()
                u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
                u_new[0] = u_new[-1] = 0
                u = u_new
                if n in [Nt//4, Nt//2, 3*Nt//4, Nt-1]:
                    snapshots[n*dt] = u.copy()

            fig_ch = go.Figure()
            colors_ch = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88', '#ffcc00']
            for i, (t_snap, u_snap) in enumerate(snapshots.items()):
                fig_ch.add_trace(go.Scatter(
                    x=x_ch, y=u_snap, mode='lines',
                    name=f't={t_snap:.2f}s (num.)',
                    line=dict(color=colors_ch[i%len(colors_ch)], width=2)
                ))
                u_ex = u_exact(t_snap)
                fig_ch.add_trace(go.Scatter(
                    x=x_ch, y=u_ex, mode='lines',
                    name=f't={t_snap:.2f}s (exact)',
                    line=dict(color=colors_ch[i%len(colors_ch)], width=1.5, dash='dot')
                ))
            fig_ch.update_layout(
                title=f"Équation de la chaleur 1D — α={alpha_ch}, r={r:.3f}",
                xaxis_title="x", yaxis_title="u(x,t)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig_ch, use_container_width=True)

        elif special == "SIR — R₀ et seuil épidémique":
            st.markdown("*Nombre de reproduction de base R₀ et dynamique épidémique*")
            beta_sir = st.slider("β (taux de contact)", 0.05, 1.0, 0.3, 0.01)
            gamma_sir = st.slider("γ (taux de guérison)", 0.01, 0.5, 0.1, 0.01)
            N_sir = st.slider("Population N", 100, 100000, 10000, 100)
            I0_sir = st.slider("Infectés initiaux I₀", 1, 100, 10)

            R0 = beta_sir / gamma_sir
            st.metric("Nombre de reproduction R₀", f"{R0:.3f}")
            st.metric("Statut", "🔴 Épidémie" if R0 > 1 else "🟢 Extinction")
            st.metric("Seuil immunité collective (%)", f"{(1 - 1/R0)*100:.1f}" if R0 > 1 else "N/A")

            t_sir = np.linspace(0, 200, 5000)
            def sir_model(Y, t):
                S, I, R = Y
                dS = -beta_sir * S * I / N_sir
                dI = beta_sir * S * I / N_sir - gamma_sir * I
                dR = gamma_sir * I
                return [dS, dI, dR]

            sol_sir = odeint(sir_model, [N_sir - I0_sir, I0_sir, 0], t_sir)
            fig_sir = go.Figure()
            for i, (lab, col) in enumerate([("S", '#00ccff'), ("I", '#ff4444'), ("R", '#00ff88')]):
                fig_sir.add_trace(go.Scatter(x=t_sir, y=sol_sir[:,i], mode='lines',
                    name=lab, line=dict(color=col, width=2.5)))
            fig_sir.update_layout(
                title=f"Modèle SIR — R₀={R0:.2f}",
                xaxis_title="Jours", yaxis_title="Population",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=450,
            )
            st.plotly_chart(fig_sir, use_container_width=True)

    # ============================================================
    # TAB 5 : THÉORIE
    # ============================================================
    with tab5:
        st.markdown("### 📖 Formulaire scientifique ODE/EDP")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES_EQU_DIFF.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 📊 Tableau des méthodes numériques")
        df_meth = pd.DataFrame([{"Méthode": k, **v} for k, v in METHODES_NUMERIQUES.items()])
        st.dataframe(df_meth, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Hairer & Wanner — *Solving ODEs I & II* (Springer, 2008)",
            "Strogatz — *Nonlinear Dynamics and Chaos* (Westview, 2014)",
            "Press et al. — *Numerical Recipes* (Cambridge, 2007)",
            "Murray — *Mathematical Biology I & II* (Springer, 2002)",
        ]:
            st.markdown(f"- {r}")
