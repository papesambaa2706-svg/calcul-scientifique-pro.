__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from scipy.optimize import (minimize, differential_evolution,
                             dual_annealing, basinhopping,
                             linprog, milp, LinearConstraint,
                             Bounds, root)
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE — OPTIMISATION CLASSIQUE
# ============================================================
FORMULES = {
    "Gradient":           r"\nabla f(x) = 0 \quad \text{(condition nécessaire)}",
    "Hessienne":          r"H = \nabla^2 f(x) \succ 0 \quad \text{(minimum local)}",
    "Newton":             r"x_{k+1} = x_k - H^{-1}\nabla f(x_k)",
    "Gradient conjugué":  r"x_{k+1} = x_k - \alpha_k \nabla f(x_k)",
    "Lagrangien":         r"\mathcal{L}(x,\lambda) = f(x) - \lambda g(x)",
    "KKT":                r"\nabla f(x^*) = \lambda \nabla g(x^*)",
    "Rastrigin":          r"f(x) = 10n + \sum_{i=1}^n [x_i^2 - 10\cos(2\pi x_i)]",
    "Rosenbrock":         r"f(x,y) = (1-x)^2 + 100(y-x^2)^2",
    "Ackley":             r"f(x) = -20e^{-0.2\sqrt{\frac{1}{n}\sum x_i^2}} - e^{\frac{1}{n}\sum\cos(2\pi x_i)} + e + 20",
}

METHODES_INFO = {
    "Nelder-Mead":           {"type": "Sans gradient", "ordre": 0, "convergence": "O(n²)", "use_case": "Fonctions bruitées"},
    "BFGS":                  {"type": "Quasi-Newton", "ordre": 2, "convergence": "Super-linéaire", "use_case": "Fonctions lisses"},
    "L-BFGS-B":              {"type": "Quasi-Newton limité", "ordre": 2, "convergence": "Super-linéaire", "use_case": "Grande dimension"},
    "Powell":                {"type": "Directions conjuguées", "ordre": 0, "convergence": "Super-linéaire", "use_case": "Général"},
    "Évolution différentielle":{"type": "Évolutionnaire global", "ordre": 0, "convergence": "Stochastique", "use_case": "Non-convexe"},
    "Recuit simulé":         {"type": "Métaheuristique", "ordre": 0, "convergence": "Probabiliste", "use_case": "Optimisation globale"},
}

# ============================================================
# FORMULAIRE — NOUVEAUX CHAPITRES
# ============================================================
FORMULES_PL = {
    "Forme standard PL":         r"\min_{x} c^T x \quad \text{s.c.} \quad Ax \leq b,\; x \geq 0",
    "Forme canonique":           r"\min c^T x,\quad Ax = b,\quad x \geq 0",
    "Dualité (problème dual)":   r"\max_{y} b^T y \quad \text{s.c.} \quad A^T y \leq c,\; y \geq 0",
    "Condition optimalité PL":   r"c^T x^* = b^T y^* \quad \text{(dualité forte)}",
    "Théorème simplexe":         r"x^* \text{ est un sommet du polytope } \{x : Ax=b, x\geq 0\}",
    "Complémentarité":           r"x_i^*(c_i - A_i^T y^*) = 0 \quad \forall i",
    "Écart de dualité":          r"\text{Gap} = c^T x - b^T y \geq 0",
    "Valeur fonction obj.":      r"z^* = c^T x^* = \min\{c^T x : Ax \leq b, x \geq 0\}",
}

FORMULES_CONVEXE = {
    "Ensemble convexe":          r"x, y \in C \Rightarrow \lambda x + (1-\lambda)y \in C,\; \forall\lambda\in[0,1]",
    "Fonction convexe":          r"f(\lambda x+(1-\lambda)y) \leq \lambda f(x)+(1-\lambda)f(y)",
    "Strictement convexe":       r"f(\lambda x+(1-\lambda)y) < \lambda f(x)+(1-\lambda)f(y),\; x\neq y",
    "Condition 1er ordre":       r"f(y) \geq f(x) + \nabla f(x)^T(y-x) \quad \forall x,y",
    "Condition 2ème ordre":      r"\nabla^2 f(x) \succeq 0 \quad \forall x \in \text{dom}(f)",
    "Minimum global conv.":      r"\nabla f(x^*) = 0 \Rightarrow x^* \text{ minimum global}",
    "Inégalité de Jensen":       r"f\!\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i),\; \sum\lambda_i=1",
    "Conjuguée convexe":         r"f^*(y) = \sup_x \{y^T x - f(x)\}",
    "Problème convexe":          r"\min f(x) \text{ s.c. } g_i(x)\leq 0,\; h_j(x)=0 \text{ (f,g convexes, h affine)}",
    "Condition KKT convexe":     r"\nabla f(x^*)+\sum\mu_i\nabla g_i(x^*)+\sum\lambda_j\nabla h_j(x^*)=0,\;\mu_i\geq 0",
    "Épigraphe":                 r"\text{epi}(f) = \{(x,t) : f(x) \leq t\}",
    "Niveau inférieur":          r"C_\alpha = \{x : f(x) \leq \alpha\} \quad \text{(convexe si f convexe)}",
}

FORMULES_OPT_CONV = {
    "Problème opt. convexe":     r"\min_{x \in C} f(x),\; f \text{ convexe}, C \text{ convexe}",
    "Gradient projeté":          r"x_{k+1} = P_C(x_k - \alpha_k \nabla f(x_k))",
    "Descente de gradient":      r"x_{k+1} = x_k - \alpha \nabla f(x_k),\; \alpha = \frac{1}{L}",
    "Convergence gradient":      r"f(x_k) - f^* \leq \frac{L\|x_0-x^*\|^2}{2k}",
    "Gradient accéléré (Nesterov)": r"f(x_k) - f^* \leq \frac{2L\|x_0-x^*\|^2}{(k+1)^2}",
    "Sous-gradient":             r"f(y) \geq f(x) + g^T(y-x),\; g \in \partial f(x)",
    "LASSO (L1)":                r"\min_\beta \|y-X\beta\|^2 + \lambda\|\beta\|_1",
    "Ridge (L2)":                r"\min_\beta \|y-X\beta\|^2 + \lambda\|\beta\|_2^2",
    "Proximal":                  r"\text{prox}_f(v) = \arg\min_x \{f(x) + \frac{1}{2}\|x-v\|^2\}",
    "ADMM":                      r"x^{k+1}=\arg\min_x \mathcal{L}_\rho,\; z^{k+1}=\arg\min_z \mathcal{L}_\rho",
}


# ============================================================
# CLASSE MOTEUR — OPTIMISATION CLASSIQUE (existante)
# ============================================================
class FunctionLibrary:
    @staticmethod
    def rosenbrock(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    @staticmethod
    def rastrigin(x):
        n = len(x)
        return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)
    @staticmethod
    def ackley(x):
        n = len(x)
        s1 = np.sqrt(0.5*sum(xi**2 for xi in x))
        s2 = 0.5*sum(np.cos(2*np.pi*xi) for xi in x)
        return -20*np.exp(-0.2*s1) - np.exp(s2) + np.e + 20
    @staticmethod
    def sphere(x):
        return sum(xi**2 for xi in x)
    @staticmethod
    def himmelblau(x):
        return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
    @staticmethod
    def beale(x):
        return ((1.5-x[0]+x[0]*x[1])**2 +
                (2.25-x[0]+x[0]*x[1]**2)**2 +
                (2.625-x[0]+x[0]*x[1]**3)**2)
    @staticmethod
    def booth(x):
        return (x[0]+2*x[1]-7)**2 + (2*x[0]+x[1]-5)**2
    @staticmethod
    def matyas(x):
        return 0.26*(x[0]**2+x[1]**2) - 0.48*x[0]*x[1]
    @staticmethod
    def sinusoide(x):
        return np.sin(x[0]) + 0.1*x[0]**2

    MINIMA_CONNUS = {
        "Rosenbrock":  {"x*": [1.0, 1.0], "f*": 0.0},
        "Rastrigin":   {"x*": [0.0, 0.0], "f*": 0.0},
        "Ackley":      {"x*": [0.0, 0.0], "f*": 0.0},
        "Sphère":      {"x*": [0.0, 0.0], "f*": 0.0},
        "Himmelblau":  {"x*": [3.0, 2.0], "f*": 0.0},
        "Beale":       {"x*": [3.0, 0.5], "f*": 0.0},
        "Booth":       {"x*": [1.0, 3.0], "f*": 0.0},
        "Matyas":      {"x*": [0.0, 0.0], "f*": 0.0},
    }


class OptimisationEngine:
    def __init__(self):
        self.historique = []
        self.n_iterations = 0

    def callback(self, xk):
        self.historique.append(xk.copy() if hasattr(xk,'copy') else xk)
        self.n_iterations += 1

    def resoudre(self, func, x0, methode, bounds=None):
        self.historique = []
        self.n_iterations = 0
        try:
            if methode == "Évolution différentielle":
                b = bounds or [(-5,5)]*len(x0)
                res = differential_evolution(func, b,
                    callback=lambda xk,conv: self.historique.append(xk.copy()),
                    maxiter=1000, seed=42)
            elif methode == "Recuit simulé":
                b = bounds or [(-5,5)]*len(x0)
                res = dual_annealing(func, b,
                    callback=lambda x,f,ctx: self.historique.append(x.copy()),
                    maxiter=1000, seed=42)
            else:
                res = minimize(func, x0, method=methode,
                               callback=self.callback,
                               options={"maxiter":5000})
            return res
        except:
            return None

    def gradient_numerique(self, func, x, h=1e-6):
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += h; xm[i] -= h
            grad[i] = (func(xp)-func(xm))/(2*h)
        return grad

    def hessienne_numerique(self, func, x, h=1e-5):
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                xpp=x.copy(); xpp[i]+=h; xpp[j]+=h
                xpm=x.copy(); xpm[i]+=h; xpm[j]-=h
                xmp=x.copy(); xmp[i]-=h; xmp[j]+=h
                xmm=x.copy(); xmm[i]-=h; xmm[j]-=h
                H[i,j]=(func(xpp)-func(xpm)-func(xmp)+func(xmm))/(4*h**2)
        return H

    def construire_fonction_personnalisee(self, expr: str, var_names):
        if isinstance(var_names, str):
            var_names = [v.strip() for v in var_names.split(',') if v.strip()]
        if len(var_names) == 0:
            raise ValueError("Au moins une variable doit être définie.")
        invalid = [v for v in var_names if not v.isidentifier()]
        if invalid:
            raise ValueError(f"Noms de variables invalides : {', '.join(invalid)}")
        safe_env = {
            'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'sqrt': np.sqrt, 'log': np.log,
            'abs': np.abs, 'pi': np.pi, 'e': np.e,
            'floor': np.floor, 'ceil': np.ceil,
            'pow': np.power,
        }
        compiled = compile(expr, '<string>', 'eval')
        def f(x):
            x = np.asarray(x, dtype=float)
            if x.ndim != 1 or x.shape[0] != len(var_names):
                raise ValueError(f"Vecteur de dimension {len(var_names)} attendu")
            local = {name: float(val) for name, val in zip(var_names, x)}
            return float(eval(compiled, {'__builtins__': None}, {**safe_env, **local}))
        f.__name__ = 'fonction_personnalisee'
        f.var_names = tuple(var_names)
        return f

    def analyser_point(self, func, x_opt):
        grad = self.gradient_numerique(func, x_opt)
        H = self.hessienne_numerique(func, x_opt)
        try:
            eigenvalues, eigenvectors = np.linalg.eig(H)
            cond = np.linalg.cond(H)
            positive = np.sum(eigenvalues.real > 1e-6)
            negative = np.sum(eigenvalues.real < -1e-6)
            zero = len(eigenvalues) - positive - negative
            if zero > 0:
                type_point = "Point critique dégénéré"
            elif negative == 0:
                type_point = "Minimum local strict"
            elif positive == 0:
                type_point = "Maximum local strict"
            else:
                type_point = "Point selle"
        except Exception:
            eigenvalues = np.array([np.nan])
            eigenvectors = np.full((len(x_opt), len(x_opt)), np.nan)
            cond = np.nan
            type_point = "Indéterminé"
        return {"gradient_norme": float(np.linalg.norm(grad)),
                "hessienne": H, "valeurs_propres": eigenvalues,
                "vecteurs_propres": eigenvectors,
                "conditionnement": cond, "type_point": type_point}

    def trouver_points_critiques(self, func, grad_func=None, x0_guesses=None, tol: float = 1e-6):
        """Trouve des points critiques en résolvant ∇f(x)=0 depuis plusieurs amorces."""
        if grad_func is None:
            grad_func = lambda x: self.gradient_numerique(func, x)
        solutions = []
        for x0 in x0_guesses or []:
            x0_arr = np.atleast_1d(np.asarray(x0, dtype=float)).ravel()
            if x0_arr.size == 0:
                continue
            x_opt = None
            method_used = None

            try:
                root_res = root(grad_func, x0_arr, method='hybr', tol=tol)
                if root_res.success and np.linalg.norm(root_res.fun) <= 1e-4:
                    x_opt = np.atleast_1d(root_res.x)
                    method_used = 'root'
            except Exception:
                root_res = None

            if x_opt is None:
                try:
                    res = minimize(lambda x: np.sum(np.asarray(grad_func(x))**2),
                                   x0_arr, method="BFGS",
                                   options={"gtol": 1e-8, "maxiter": 400})
                    if res.success and np.linalg.norm(grad_func(res.x)) <= 1e-4:
                        x_opt = np.atleast_1d(res.x)
                        method_used = 'minimize'
                        root_res = res
                except Exception:
                    root_res = None

            if x_opt is None:
                continue
            if np.linalg.norm(grad_func(x_opt)) > 1e-4:
                continue
            already = any(np.allclose(x_opt, sol["x"], atol=1e-4, rtol=1e-6)
                          for sol in solutions)
            if not already:
                solutions.append({"x": x_opt, "method": method_used, "result": root_res})
        return solutions

    def benchmark_methodes(self, func, x0, methodes):
        resultats = []
        for m in methodes:
            engine = OptimisationEngine()
            res = engine.resoudre(func, x0.copy(), m)
            if res is not None:
                resultats.append({"Méthode": m,
                    "f(x*)": f"{res.fun:.6e}",
                    "Itérations": engine.n_iterations,
                    "Succès": "✅" if res.success else "❌",
                    "x*": str(np.round(res.x, 4))})
        return pd.DataFrame(resultats)


# ============================================================
# NOUVELLE CLASSE — PROGRAMMATION LINÉAIRE
# ============================================================
class ProgrammationLineaireEngine:
    """Moteur de résolution de problèmes de PL en variables continues."""

    def resoudre_pl(self, c: np.ndarray, A_ub: np.ndarray,
                    b_ub: np.ndarray, A_eq: np.ndarray = None,
                    b_eq: np.ndarray = None,
                    bounds=None) -> dict:
        """
        Résout min c^T x sous Ax <= b, x >= 0.
        Retourne solution, valeur optimale, statut.
        """
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds,
                          method='highs')
            return {
                "x_opt": res.x,
                "z_opt": res.fun,
                "statut": res.message,
                "succes": res.success,
                "iterations": res.nit if hasattr(res,'nit') else 0,
            }
        except Exception as e:
            return {"erreur": str(e), "succes": False}

    def probleme_dual(self, c: np.ndarray,
                      A: np.ndarray, b: np.ndarray) -> dict:
        """
        Dual du problème primal min c^T x, Ax >= b, x >= 0.
        Dual : max b^T y, A^T y <= c, y >= 0.
        """
        try:
            m = len(b)
            res_dual = linprog(-b, A_ub=A.T, b_ub=c,
                               bounds=[(0, None)]*m, method='highs')
            return {
                "y_opt": res_dual.x,
                "z_dual": -res_dual.fun if res_dual.success else None,
                "succes": res_dual.success,
            }
        except Exception as e:
            return {"erreur": str(e)}

    def ecart_dualite(self, c, x_primal, b, y_dual) -> float:
        """Gap de dualité = c^T x - b^T y."""
        return float(c @ x_primal - b @ y_dual)

    def domaine_faisable_2D(self, A: np.ndarray,
                             b: np.ndarray,
                             x_range=(-1, 10),
                             y_range=(-1, 10),
                             n: int = 200) -> np.ndarray:
        """Masque du domaine faisable 2D."""
        x = np.linspace(*x_range, n)
        y = np.linspace(*y_range, n)
        X, Y = np.meshgrid(x, y)
        faisable = np.ones_like(X, dtype=bool)
        for i in range(len(A)):
            faisable &= (A[i, 0]*X + A[i, 1]*Y <= b[i])
        faisable &= (X >= 0) & (Y >= 0)
        return X, Y, faisable.astype(float)

    def sommets_polytope_2D(self, A: np.ndarray,
                             b: np.ndarray,
                             x_lim: float = 20) -> np.ndarray:
        """Calcule les sommets du polytope {Ax<=b, x>=0} en 2D."""
        from itertools import combinations
        n_contraintes = len(A)
        A_ext = np.vstack([A, -np.eye(2)])
        b_ext = np.hstack([b, np.zeros(2)])
        n_tot = len(A_ext)
        sommets = []

        for idx in combinations(range(n_tot), 2):
            A_sys = A_ext[list(idx)]
            b_sys = b_ext[list(idx)]
            try:
                det = np.linalg.det(A_sys)
                if abs(det) > 1e-10:
                    pt = np.linalg.solve(A_sys, b_sys)
                    if (np.all(A_ext @ pt <= b_ext + 1e-8) and
                            np.all(pt >= -1e-8) and
                            np.all(pt <= x_lim)):
                        sommets.append(pt)
            except:
                pass
        return np.array(sommets) if sommets else np.array([]).reshape(0, 2)

    def modeles_types(self) -> dict:
        """Problèmes classiques de PL."""
        return {
            "Problème de transport": {
                "description": "Minimiser coût transport entre usines et dépôts",
                "c": np.array([2, 3, 1, 4]),
                "A_ub": np.array([
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                    [-1, 0, -1, 0],
                    [0, -1, 0, -1],
                ]),
                "b_ub": np.array([100, 80, -60, -70]),
                "x_labels": ["x11","x12","x21","x22"],
            },
            "Mélange optimal": {
                "description": "Maximiser profit : mélange de 2 produits",
                "c": np.array([-5, -4]),
                "A_ub": np.array([[6, 4], [1, 2], [0, 1]]),
                "b_ub": np.array([24, 6, 4]),
                "x_labels": ["x1","x2"],
            },
            "Régime alimentaire": {
                "description": "Minimiser coût nutritionnel",
                "c": np.array([1.5, 2.0, 0.8]),
                "A_ub": np.array([
                    [-2, -3, -1],
                    [-1, -0.5, -2],
                ]),
                "b_ub": np.array([-10, -8]),
                "x_labels": ["Aliment A","Aliment B","Aliment C"],
            },
        }

    def analyse_sensibilite(self, c: np.ndarray,
                             A_ub: np.ndarray,
                             b_ub: np.ndarray,
                             idx_b: int = 0,
                             b_range: np.ndarray = None) -> pd.DataFrame:
        """Analyse de sensibilité sur b_i."""
        if b_range is None:
            b_range = np.linspace(max(0.1, b_ub[idx_b]*0.5),
                                   b_ub[idx_b]*2, 30)
        resultats = []
        for b_val in b_range:
            b_mod = b_ub.copy()
            b_mod[idx_b] = b_val
            res = self.resoudre_pl(c, A_ub, b_mod)
            if res.get("succes"):
                resultats.append({"b_{}".format(idx_b): b_val,
                                   "z*": res["z_opt"],
                                   "x1*": res["x_opt"][0] if len(res["x_opt"])>0 else np.nan,
                                   "x2*": res["x_opt"][1] if len(res["x_opt"])>1 else np.nan})
        return pd.DataFrame(resultats)


# ============================================================
# NOUVELLE CLASSE — ENSEMBLES & FONCTIONS CONVEXES
# ============================================================
class ConvexiteEngine:
    """Moteur d'analyse de convexité des ensembles et fonctions."""

    def est_convexe_numerique(self, f, x_range: tuple,
                               n: int = 100,
                               tol: float = 1e-6) -> tuple:
        """
        Vérifie numériquement la convexité d'une fonction 1D.
        Utilise la condition f((x+y)/2) <= (f(x)+f(y))/2.
        """
        x_arr = np.linspace(*x_range, n)
        try:
            f_arr = np.array([f(xi) for xi in x_arr])
        except:
            return False, np.nan

        violations = 0
        max_violation = 0.0
        for i in range(len(x_arr)-1):
            x_mid = (x_arr[i]+x_arr[i+1])/2
            f_mid = f(x_mid)
            f_comb = (f_arr[i]+f_arr[i+1])/2
            viol = f_mid - f_comb
            if viol > tol:
                violations += 1
                max_violation = max(max_violation, viol)

        est_conv = violations == 0
        return est_conv, max_violation

    def hessienne_1D(self, f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """f''(x) numérique — signe positif ↔ convexe."""
        return (np.array([f(xi+h) for xi in x]) -
                2*np.array([f(xi) for xi in x]) +
                np.array([f(xi-h) for xi in x])) / h**2

    def projection_convexe(self, point: np.ndarray,
                            type_ensemble: str,
                            **kwargs) -> np.ndarray:
        """
        Projection d'un point sur un ensemble convexe fermé.
        Types : 'boule', 'hyperplan', 'simplexe', 'cone_positif'.
        """
        if type_ensemble == "boule":
            centre = kwargs.get("centre", np.zeros_like(point))
            rayon = kwargs.get("rayon", 1.0)
            d = point - centre
            norme = np.linalg.norm(d)
            if norme <= rayon:
                return point
            return centre + rayon * d / norme

        elif type_ensemble == "hyperplan":
            a = kwargs.get("a", np.ones_like(point))
            b_val = kwargs.get("b", 0.0)
            return point - ((a@point - b_val)/np.dot(a,a)) * a

        elif type_ensemble == "simplexe":
            n = len(point)
            u = np.sort(point)[::-1]
            cssv = np.cumsum(u)
            rho = np.where(u > (cssv-1)/np.arange(1,n+1))[0][-1]
            theta = (cssv[rho]-1)/(rho+1)
            return np.maximum(point - theta, 0)

        elif type_ensemble == "cone_positif":
            return np.maximum(point, 0)

        return point

    def enveloppe_convexe_2D(self, points: np.ndarray) -> np.ndarray:
        """Enveloppe convexe de points 2D (algorithme de Graham)."""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            idx = np.append(hull.vertices, hull.vertices[0])
            return points[idx]
        except:
            return points

    def verifier_jensen(self, f, x_vals: np.ndarray,
                         lambdas: np.ndarray) -> dict:
        """
        Vérifie l'inégalité de Jensen :
        f(Σλᵢxᵢ) ≤ Σλᵢf(xᵢ).
        """
        lambdas = lambdas / lambdas.sum()
        x_bar = lambdas @ x_vals
        lhs = f(x_bar)
        rhs = lambdas @ np.array([f(xi) for xi in x_vals])
        return {"f(x̄)": lhs, "Σλᵢf(xᵢ)": rhs,
                "Jensen vérifié": lhs <= rhs + 1e-10,
                "Écart": rhs - lhs}

    def conjuguee_convexe(self, f, y_range: tuple,
                           x_range: tuple = (-10, 10),
                           n: int = 500) -> tuple:
        """
        Conjuguée de Fenchel : f*(y) = sup_x {yx - f(x)}.
        """
        x = np.linspace(*x_range, n)
        y_arr = np.linspace(*y_range, n)
        f_arr = np.array([f(xi) for xi in x])
        f_star = np.array([np.max(yi*x - f_arr) for yi in y_arr])
        return y_arr, f_star

    def fonctions_convexes_catalogue(self) -> dict:
        """Catalogue de fonctions convexes standards."""
        return {
            "Quadratique x²":         (lambda x: x**2,      True,  "∀x"),
            "Valeur absolue |x|":     (lambda x: abs(x),    True,  "∀x"),
            "Exponentielle eˣ":       (lambda x: np.exp(x), True,  "∀x"),
            "-log(x)":                (lambda x: -np.log(max(x,1e-10)), True, "x>0"),
            "x·log(x)":              (lambda x: x*np.log(max(x,1e-10)) if x>0 else 0, True, "x≥0"),
            "Norme ‖x‖² (affine)":   (lambda x: (2*x-3)**2, True, "∀x"),
            "Max(0,x) (ReLU)":        (lambda x: max(0,x),  True,  "∀x"),
            "Non convexe: sin(x)":    (lambda x: np.sin(x), False, "∀x"),
            "Non convexe: x³":        (lambda x: x**3,      False, "∀x"),
            "Non convexe: -x²":       (lambda x: -x**2,     False, "∀x"),
        }

    def niveau_sous_ensemble(self, f, alpha: float,
                              x_range: tuple,
                              y_range: tuple,
                              n: int = 200) -> tuple:
        """
        Ensemble de niveau Cα = {x : f(x) ≤ α} en 2D.
        """
        x = np.linspace(*x_range, n)
        y = np.linspace(*y_range, n)
        X, Y = np.meshgrid(x, y)
        try:
            Z = np.vectorize(lambda xi, yi: f([xi, yi]))(X, Y)
        except:
            Z = X**2 + Y**2
        masque = (Z <= alpha).astype(float)
        return X, Y, Z, masque


# ============================================================
# NOUVELLE CLASSE — OPTIMISATION CONVEXE
# ============================================================
class OptimisationConvexeEngine:
    """Moteur d'optimisation convexe : gradients, proximal, LASSO."""

    def descente_gradient(self, f, grad_f,
                           x0: np.ndarray,
                           alpha: float = 0.01,
                           n_iter: int = 200,
                           tol: float = 1e-8) -> dict:
        """
        Descente de gradient à pas fixe.
        x_{k+1} = x_k - α∇f(x_k)
        """
        x = x0.copy().astype(float)
        historique = [x.copy()]
        valeurs = [f(x)]
        gradients = []

        # If no analytic gradient provided, use central finite differences
        if grad_f is None:
            def _numeric_grad(x_pt):
                x_pt = np.asarray(x_pt, dtype=float)
                g = np.zeros_like(x_pt)
                h = 1e-6
                for i in range(len(x_pt)):
                    xp = x_pt.copy(); xp[i] += h
                    xm = x_pt.copy(); xm[i] -= h
                    g[i] = (f(xp) - f(xm)) / (2*h)
                return g

            grad_f = _numeric_grad

        for k in range(n_iter):
            g = grad_f(x)
            gradients.append(np.linalg.norm(g))
            x_new = x - alpha * g
            historique.append(x_new.copy())
            valeurs.append(f(x_new))
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        return {"x_opt": x, "f_opt": f(x),
                "historique": np.array(historique),
                "valeurs": np.array(valeurs),
                "gradients": np.array(gradients),
                "n_iter": len(historique)-1}

    def descente_gradient_armijo(self, f, grad_f,
                                  x0: np.ndarray,
                                  beta: float = 0.5,
                                  sigma: float = 0.01,
                                  n_iter: int = 200) -> dict:
        """Descente de gradient avec recherche linéaire (Armijo)."""
        x = x0.copy().astype(float)
        historique = [x.copy()]
        valeurs = [f(x)]

        # fallback numeric gradient if none provided
        if grad_f is None:
            def _numeric_grad(x_pt):
                x_pt = np.asarray(x_pt, dtype=float)
                g = np.zeros_like(x_pt)
                h = 1e-6
                for i in range(len(x_pt)):
                    xp = x_pt.copy(); xp[i] += h
                    xm = x_pt.copy(); xm[i] -= h
                    g[i] = (f(xp) - f(xm)) / (2*h)
                return g

            grad_f = _numeric_grad

        for k in range(n_iter):
            g = grad_f(x)
            if np.linalg.norm(g) < 1e-8:
                break
            d = -g
            alpha = 1.0
            for _ in range(50):
                if f(x+alpha*d) <= f(x) + sigma*alpha*(g@d):
                    break
                alpha *= beta

            x = x + alpha*d
            historique.append(x.copy())
            valeurs.append(f(x))

        return {"x_opt": x, "f_opt": f(x),
                "historique": np.array(historique),
                "valeurs": np.array(valeurs)}

    def gradient_nesterov(self, f, grad_f,
                           x0: np.ndarray,
                           L: float = 1.0,
                           n_iter: int = 200) -> dict:
        """Méthode de Nesterov (gradient accéléré)."""
        x = x0.copy().astype(float)
        y = x.copy()
        t = 1.0
        historique = [x.copy()]
        valeurs = [f(x)]

        # fallback numeric gradient if none provided
        if grad_f is None:
            def _numeric_grad(x_pt):
                x_pt = np.asarray(x_pt, dtype=float)
                g = np.zeros_like(x_pt)
                h = 1e-6
                for i in range(len(x_pt)):
                    xp = x_pt.copy(); xp[i] += h
                    xm = x_pt.copy(); xm[i] -= h
                    g[i] = (f(xp) - f(xm)) / (2*h)
                return g

            grad_f = _numeric_grad

        for k in range(n_iter):
            x_new = y - (1/L)*grad_f(y)
            t_new = (1 + np.sqrt(1 + 4*t**2))/2
            y = x_new + ((t-1)/t_new)*(x_new - x)
            x = x_new
            t = t_new
            historique.append(x.copy())
            valeurs.append(f(x))
            if np.linalg.norm(grad_f(x)) < 1e-8:
                break

        return {"x_opt": x, "f_opt": f(x),
                "historique": np.array(historique),
                "valeurs": np.array(valeurs)}

    def gradient_projete(self, f, grad_f,
                          x0: np.ndarray,
                          projection,
                          alpha: float = 0.01,
                          n_iter: int = 300) -> dict:
        """Gradient projeté sur un ensemble convexe."""
        x = x0.copy().astype(float)
        historique = [x.copy()]
        valeurs = [f(x)]
        # fallback numeric gradient if none provided
        if grad_f is None:
            def _numeric_grad(x_pt):
                x_pt = np.asarray(x_pt, dtype=float)
                g = np.zeros_like(x_pt)
                h = 1e-6
                for i in range(len(x_pt)):
                    xp = x_pt.copy(); xp[i] += h
                    xm = x_pt.copy(); xm[i] -= h
                    g[i] = (f(xp) - f(xm)) / (2*h)
                return g

            grad_f = _numeric_grad

        for k in range(n_iter):
            g = grad_f(x)
            x_new = projection(x - alpha*g)
            if x_new is None:
                x_new = x.copy()
            else:
                x_new = np.asarray(x_new, dtype=float)
                if x_new.shape != x.shape:
                    x_new = x_new.reshape(x.shape)
            historique.append(x_new.copy())
            valeurs.append(f(x_new))
            if np.linalg.norm(x_new - x) < 1e-8:
                x = x_new
                break
            x = x_new

        return {"x_opt": x, "f_opt": f(x),
                "historique": np.array(historique),
                "valeurs": np.array(valeurs)}

    def lasso(self, X: np.ndarray, y: np.ndarray,
              lambda_reg: float = 1.0,
              n_iter: int = 1000) -> dict:
        """
        LASSO : min ‖y - Xβ‖² + λ‖β‖₁
        Résolu par descente de coordonnées.
        """
        n, p = X.shape
        beta = np.zeros(p)
        residus_hist = []

        for iteration in range(n_iter):
            for j in range(p):
                r = y - X @ beta + X[:, j]*beta[j]
                z = X[:, j] @ r
                norm_xj = np.dot(X[:, j], X[:, j])
                if norm_xj < 1e-10:
                    beta[j] = 0
                else:
                    beta[j] = np.sign(z)*max(0, abs(z)-lambda_reg)/(norm_xj)
            residus_hist.append(np.linalg.norm(y - X@beta)**2)

        y_pred = X @ beta
        r2 = 1 - np.sum((y-y_pred)**2)/np.sum((y-np.mean(y))**2)
        return {"beta": beta, "y_pred": y_pred,
                "residus_hist": residus_hist,
                "r2": r2, "n_non_nuls": np.sum(np.abs(beta) > 1e-6)}

    def ridge(self, X: np.ndarray, y: np.ndarray,
              lambda_reg: float = 1.0) -> dict:
        """Ridge : β = (X^T X + λI)⁻¹ X^T y."""
        n, p = X.shape
        A = X.T @ X + lambda_reg * np.eye(p)
        beta = np.linalg.solve(A, X.T @ y)
        y_pred = X @ beta
        r2 = 1 - np.sum((y-y_pred)**2)/np.sum((y-np.mean(y))**2)
        return {"beta": beta, "y_pred": y_pred, "r2": r2}

    def chemin_regularisation(self, X: np.ndarray,
                               y: np.ndarray,
                               lambdas: np.ndarray,
                               methode: str = "LASSO") -> pd.DataFrame:
        """Chemin de régularisation pour différents λ."""
        rows = []
        for lam in lambdas:
            if methode == "LASSO":
                res = self.lasso(X, y, lam)
            else:
                res = self.ridge(X, y, lam)
            row = {"λ": lam, "R²": res["r2"]}
            for j, bj in enumerate(res["beta"]):
                row[f"β{j+1}"] = bj
            rows.append(row)
        return pd.DataFrame(rows)

    def taux_convergence(self, valeurs: np.ndarray,
                          f_star: float = 0.0) -> dict:
        """Calcule le taux de convergence numérique."""
        errs = np.maximum(valeurs - f_star, 1e-15)
        if len(errs) < 3:
            return {"taux": np.nan, "ordre": np.nan}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratios = errs[1:] / errs[:-1]
        taux_moyen = np.mean(ratios[ratios > 0][:10])
        return {"taux": taux_moyen, "erreurs": errs,
                "iterations": np.arange(len(errs))}

    def comparaison_methodes_convexes(self, f, grad_f,
                                       x0: np.ndarray,
                                       L: float = 1.0) -> dict:
        """Compare GD, Nesterov et GD-Armijo."""
        alpha = 1/L
        gd = self.descente_gradient(f, grad_f, x0, alpha)
        nest = self.gradient_nesterov(f, grad_f, x0, L)
        arm = self.descente_gradient_armijo(f, grad_f, x0)
        return {"GD (pas fixe)": gd,
                "Nesterov": nest,
                "GD Armijo": arm}


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def optimisation_page():
    st.markdown("## 🎯 Optimisation Scientifique Avancée")
    st.markdown("*Optimisation classique · PL · Convexité · Optimisation convexe*")
    st.markdown("---")

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

    lib = FunctionLibrary()
    engine = OptimisationEngine()
    pl_engine = ProgrammationLineaireEngine()
    conv_engine = ConvexiteEngine()
    opt_engine = OptimisationEngine()
    opt_conv_engine = OptimisationConvexeEngine()

    section = st.selectbox(
        "Section",
        [
            "🔍 Optimisation Classique",
            "📐 Programmation Linéaire",
            "🔷 Convexité",
            "⚡ Optimisation Convexe",
        ],
        key="section_optimisation"
    )

    # ============================================================
    # ONGLETS PRINCIPAUX
    # ============================================================

    # ============================================================
    # TAB 1 — OPTIMISATION CLASSIQUE (code existant préservé)
    # ============================================================
    if section == "🔍 Optimisation Classique":
        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
            "🔍 Optimisation 1D/2D",
            "🌐 Paysage 2D",
            "⚔️ Benchmark",
            "📐 Analyse mathématique",
            "📖 Théorie"
        ])

        fonctions_2d = {
            "Rosenbrock": lib.rosenbrock,
            "Rastrigin":  lib.rastrigin,
            "Ackley":     lib.ackley,
            "Sphère":     lib.sphere,
            "Himmelblau": lib.himmelblau,
            "Beale":      lib.beale,
            "Booth":      lib.booth,
            "Matyas":     lib.matyas,
        }

        with subtab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### ⚙️ Configuration")
                mode = st.radio("Dimension", ["1D", "2D"], horizontal=True)
                if mode == "2D":
                    func_name = st.selectbox("Fonction 2D",
                                              list(fonctions_2d.keys()))
                    func = fonctions_2d[func_name]
                    if func_name in lib.MINIMA_CONNUS:
                        info = lib.MINIMA_CONNUS[func_name]
                        st.info(f"**Min global :** x*={info['x*']}, f*={info['f*']}")
                    c1, c2 = st.columns(2)
                    with c1: x0_x = st.slider("x₀", -4.0, 4.0, 2.0, 0.1)
                    with c2: x0_y = st.slider("y₀", -4.0, 4.0, 2.0, 0.1)
                    x0 = np.array([x0_x, x0_y])
                else:
                    a = st.slider("a", -5.0, 5.0, 1.0, 0.1)
                    b_q = st.slider("b", -5.0, 5.0, -2.0, 0.1)
                    c_q = st.slider("c", -5.0, 5.0, 3.0, 0.1)
                    func = lambda x: a*x[0]**2 + b_q*x[0] + c_q
                    x_min_ana = -b_q/(2*a) if a != 0 else 0
                    st.metric("Minimum analytique", f"{x_min_ana:.4f}")
                    x0 = np.array([2.0])

                methode = st.selectbox("Méthode", [
                    "Nelder-Mead","BFGS","L-BFGS-B","Powell",
                    "Évolution différentielle","Recuit simulé"])
                lancer = st.button("🚀 Lancer", use_container_width=True)

            with col2:
                if lancer:
                    with st.spinner("Optimisation..."):
                        res = engine.resoudre(func, x0.copy(), methode)
                    if res is not None:
                        c1,c2,c3 = st.columns(3)
                        with c1: st.metric("f(x*)", f"{res.fun:.6e}")
                        with c2: st.metric("Itérations", engine.n_iterations)
                        with c3: st.metric("Statut",
                                           "✅" if res.success else "⚠️")

                        if mode == "1D":
                            x_p = np.linspace(-10, 10, 1000)
                            y_p = [func([xi]) for xi in x_p]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x_p, y=y_p, mode='lines',
                                line=dict(color='#00ccff', width=3), name='f(x)'))
                            fig.add_trace(go.Scatter(x=[res.x[0]], y=[res.fun],
                                mode='markers', name='Minimum',
                                marker=dict(color='#00ff88', size=14, symbol='star')))
                            layout(fig, "Optimisation 1D", "x", "f(x)")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            res_viz = 80
                            x_r = np.linspace(-5, 5, res_viz)
                            y_r = np.linspace(-5, 5, res_viz)
                            Z = np.array([[func([xi,yi]) for xi in x_r] for yi in y_r])
                            Z_c = np.clip(Z, np.percentile(Z,1), np.percentile(Z,98))
                            fig = go.Figure()
                            fig.add_trace(go.Contour(z=Z_c, x=x_r, y=y_r,
                                colorscale=[[0,'#020817'],[0.3,'#7700ff'],
                                            [0.6,'#00ccff'],[1,'#ffffff']],
                                showscale=True))
                            if len(engine.historique) > 1:
                                traj = np.array(engine.historique)
                                if traj.ndim==2 and traj.shape[1]>=2:
                                    fig.add_trace(go.Scatter(
                                        x=traj[:,0], y=traj[:,1],
                                        mode='lines+markers', name='Trajectoire',
                                        line=dict(color='#ffcc00', width=2),
                                        marker=dict(size=4)))
                            fig.add_trace(go.Scatter(x=[res.x[0]], y=[res.x[1]],
                                mode='markers', name='Optimum',
                                marker=dict(color='#00ff88', size=16, symbol='star')))
                            layout(fig, f"Paysage {func_name}", "x", "y")
                            st.plotly_chart(fig, use_container_width=True)

        with subtab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                func_viz_name = st.selectbox("Fonction", list(fonctions_2d.keys()),
                                              key="viz3d")
                func_viz = fonctions_2d[func_viz_name]
                range_viz = st.slider("Plage", 1.0, 6.0, 3.0, 0.5)
                mode_viz = st.radio("Vue", ["Surface 3D","Contour 2D"],
                                    horizontal=True)
            with col2:
                x_v = np.linspace(-range_viz, range_viz, 60)
                y_v = np.linspace(-range_viz, range_viz, 60)
                Z_v = np.array([[func_viz([xi,yi]) for xi in x_v] for yi in y_v])
                Z_clip = np.clip(Z_v, np.percentile(Z_v,1), np.percentile(Z_v,99))
                if mode_viz == "Surface 3D":
                    fig3d = go.Figure(go.Surface(z=Z_clip, x=x_v, y=y_v,
                        colorscale=[[0,'#020817'],[0.3,'#7700ff'],
                                    [0.6,'#00ccff'],[1,'#ffffff']]))
                    fig3d.update_layout(scene=dict(bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff'),
                        yaxis=dict(color='#c0d0ff'),
                        zaxis=dict(color='#c0d0ff')),
                        **PLOT_LAYOUT, height=520)
                else:
                    fig3d = go.Figure(go.Contour(z=Z_clip, x=x_v, y=y_v,
                        colorscale=[[0,'#020817'],[0.3,'#7700ff'],
                                    [0.6,'#00ccff'],[1,'#ffffff']],
                        contours=dict(coloring='heatmap', showlabels=True)))
                    layout(fig3d, f"Paysage {func_viz_name}", "x", "y", h=520)
                st.plotly_chart(fig3d, use_container_width=True)

        with subtab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                func_b_name = st.selectbox("Fonction", list(fonctions_2d.keys()),
                                            key="bench_f")
                func_b = fonctions_2d[func_b_name]
                met_sel = st.multiselect("Méthodes", [
                    "Nelder-Mead","BFGS","L-BFGS-B","Powell",
                    "Évolution différentielle","Recuit simulé"],
                    default=["Nelder-Mead","BFGS","L-BFGS-B"])
                bx = st.slider("x₀", -4.0, 4.0, 2.0, 0.1, key="bx")
                by = st.slider("y₀", -4.0, 4.0, 2.0, 0.1, key="by")
                if st.button("⚔️ Benchmark", use_container_width=True) and met_sel:
                    with st.spinner("..."):
                        df_b = engine.benchmark_methodes(func_b,
                            np.array([bx,by]), met_sel)
                    st.dataframe(df_b, use_container_width=True)

        with subtab4:
            df_m = pd.DataFrame([{"Méthode": k, "Type": v["type"],
                "Convergence": v["convergence"],
                "Usage": v["use_case"]}
                for k, v in METHODES_INFO.items()])
            st.dataframe(df_m, use_container_width=True)
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f_latex)
                col_idx += 1

        with subtab5:
            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f_latex)
                col_idx += 1

    # ============================================================
    # TAB 2 — PROGRAMMATION LINÉAIRE
    # ============================================================
    elif section == "📐 Programmation Linéaire":
        st.markdown("## 📐 Programmation Linéaire en Variables Continues")
        st.markdown("*Modélisation, résolution (simplexe/HiGHS), dualité, sensibilité*")

        sub1, sub2, sub3, sub4, sub5 = st.tabs([
            "🏗️ Modélisation & Résolution",
            "🗺️ Domaine faisable 2D",
            "🔄 Dualité",
            "📊 Analyse de sensibilité",
            "📖 Théorie & Formules"
        ])

        # ---- SUB 1 : MODÉLISATION ----
        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🏗️ Configuration du problème")
                mode_pl = st.radio("Mode", ["Problème personnalisé",
                                             "Exemple prédéfini"],
                                    horizontal=True)

                if mode_pl == "Exemple prédéfini":
                    modeles = pl_engine.modeles_types()
                    ex_sel = st.selectbox("Exemple", list(modeles.keys()))
                    mod = modeles[ex_sel]
                    st.info(mod["description"])
                    c_pl = mod["c"]
                    A_pl = mod["A_ub"]
                    b_pl = mod["b_ub"]
                    labels_x = mod["x_labels"]

                    st.markdown("**Coefficients objectif c :**")
                    st.write(dict(zip(labels_x, c_pl)))
                    st.markdown("**Matrice A (contraintes) :**")
                    df_A = pd.DataFrame(A_pl, columns=labels_x)
                    df_A["≤ b"] = b_pl
                    st.dataframe(df_A, use_container_width=True)

                else:
                    n_var = st.slider("Nombre de variables", 2, 5, 2)
                    n_con = st.slider("Nombre de contraintes", 1, 6, 3)

                    st.markdown("**Coefficients objectif c (minimiser c^T x) :**")
                    c_vals = []
                    cols_c = st.columns(n_var)
                    for i, col in enumerate(cols_c):
                        with col:
                            c_vals.append(st.number_input(f"c{i+1}",
                                value=float(-[5,4,3,2,1][i%5]),
                                key=f"c_{i}"))
                    c_pl = np.array(c_vals)

                    st.markdown("**Contraintes Ax ≤ b :**")
                    A_rows, b_rows = [], []
                    for j in range(n_con):
                        cols_a = st.columns(n_var + 1)
                        row = []
                        for i in range(n_var):
                            with cols_a[i]:
                                row.append(st.number_input(
                                    f"a{j+1}{i+1}",
                                    value=1.0, key=f"a_{j}_{i}"))
                        with cols_a[n_var]:
                            b_rows.append(st.number_input(
                                f"b{j+1}", value=float([10,8,6,4,2][j%5]),
                                key=f"b_{j}"))
                        A_rows.append(row)
                    A_pl = np.array(A_rows)
                    b_pl = np.array(b_rows)
                    labels_x = [f"x{i+1}" for i in range(n_var)]

                if st.button("🚀 Résoudre le PL", use_container_width=True):
                    st.session_state["pl_c"] = c_pl
                    st.session_state["pl_A"] = A_pl
                    st.session_state["pl_b"] = b_pl
                    st.session_state["pl_labels"] = labels_x

                    res_pl = pl_engine.resoudre_pl(c_pl, A_pl, b_pl)
                    st.session_state["pl_res"] = res_pl

            with col2:
                if "pl_res" in st.session_state:
                    res_pl = st.session_state["pl_res"]
                    c_pl = st.session_state["pl_c"]
                    A_pl = st.session_state["pl_A"]
                    b_pl = st.session_state["pl_b"]
                    labels_x = st.session_state["pl_labels"]

                    if res_pl.get("succes"):
                        st.success(f"✅ {res_pl['statut']}")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("z* (objectif)", f"{res_pl['z_opt']:.4f}")
                        with c2: st.metric("Sens", "Minimisation")
                        with c3: st.metric("Statut", "Optimal")

                        st.markdown("#### 📊 Solution optimale x*")
                        df_sol = pd.DataFrame({
                            "Variable": labels_x,
                            "x*": [f"{xi:.4f}" for xi in res_pl["x_opt"]],
                            "c_i": c_pl,
                            "c_i·x_i*": [f"{ci*xi:.4f}"
                                          for ci, xi in zip(c_pl, res_pl["x_opt"])]
                        })
                        st.dataframe(df_sol, use_container_width=True)

                        # Contraintes actives
                        st.markdown("#### 🔍 Saturation des contraintes")
                        Ax = A_pl @ res_pl["x_opt"]
                        df_con = pd.DataFrame({
                            "Contrainte": [f"C{j+1}" for j in range(len(b_pl))],
                            "Ax": [f"{v:.4f}" for v in Ax],
                            "b": b_pl,
                            "Écart": [f"{b_pl[j]-Ax[j]:.4f}" for j in range(len(b_pl))],
                            "Active": ["✅ Oui" if abs(Ax[j]-b_pl[j])<1e-6
                                       else "❌ Non" for j in range(len(b_pl))]
                        })
                        st.dataframe(df_con, use_container_width=True)

                        # Graphique barres
                        fig_sol = go.Figure(go.Bar(
                            x=labels_x, y=res_pl["x_opt"],
                            marker=dict(color=res_pl["x_opt"],
                                colorscale=[[0,'#7700ff'],[1,'#00ccff']],
                                showscale=True),
                            text=[f"{v:.3f}" for v in res_pl["x_opt"]],
                            textposition='outside'
                        ))
                        layout(fig_sol, "Solution optimale x*", "Variable", "Valeur", h=350)
                        st.plotly_chart(fig_sol, use_container_width=True)
                    else:
                        st.error(f"❌ Infaisable ou non borné : {res_pl.get('statut','')}")

        # ---- SUB 2 : DOMAINE FAISABLE 2D ----
        with sub2:
            st.markdown("### 🗺️ Visualisation du domaine faisable (2D)")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Contraintes :**")
                n_con2 = st.slider("Nb contraintes", 2, 5, 3, key="nc2d")
                A_2d, b_2d = [], []
                for j in range(n_con2):
                    c1, c2_col, c3 = st.columns(3)
                    with c1: a1 = st.number_input(f"a{j+1}1",
                        value=[1.0,0.0,1.0,2.0,1.0][j], key=f"a2d_{j}0")
                    with c2_col: a2 = st.number_input(f"a{j+1}2",
                        value=[0.0,1.0,1.0,1.0,3.0][j], key=f"a2d_{j}1")
                    with c3: bj = st.number_input(f"b{j+1}",
                        value=[4.0,3.0,5.0,6.0,7.0][j], key=f"b2d_{j}")
                    A_2d.append([a1, a2])
                    b_2d.append(bj)
                A_2d = np.array(A_2d)
                b_2d = np.array(b_2d)

                c1_obj = st.number_input("c1 (objectif)", value=-3.0, key="c1_2d")
                c2_obj = st.number_input("c2 (objectif)", value=-2.0, key="c2_2d")
                c_2d = np.array([c1_obj, c2_obj])

            with col2:
                X2, Y2, fais = pl_engine.domaine_faisable_2D(
                    A_2d, b_2d, (-0.5, 8), (-0.5, 8))
                sommets = pl_engine.sommets_polytope_2D(A_2d, b_2d)
                res_2d = pl_engine.resoudre_pl(c_2d, A_2d, b_2d,
                    bounds=[(0,None),(0,None)])

                fig_fais = go.Figure()
                fig_fais.add_trace(go.Contour(z=fais, x=X2[0], y=Y2[:,0],
                    colorscale=[[0,'rgba(0,0,0,0)'],
                                [0.5,'rgba(119,0,255,0.2)'],
                                [1,'rgba(0,204,255,0.3)']],
                    showscale=False, contours=dict(coloring='fill')))

                # Contraintes
                x_line = np.linspace(0, 8, 200)
                for j, (a_row, bj) in enumerate(zip(A_2d, b_2d)):
                    if abs(a_row[1]) > 1e-8:
                        y_line = (bj - a_row[0]*x_line) / a_row[1]
                        fig_fais.add_trace(go.Scatter(
                            x=x_line, y=y_line, mode='lines',
                            name=f"C{j+1}: {a_row[0]:.1f}x₁+{a_row[1]:.1f}x₂≤{bj:.1f}",
                            line=dict(color=colors[j], width=2)))

                # Sommets
                if len(sommets) > 0:
                    fig_fais.add_trace(go.Scatter(x=sommets[:,0], y=sommets[:,1],
                        mode='markers', name='Sommets',
                        marker=dict(color='#ffcc00', size=12, symbol='circle')))

                # Optimum
                if res_2d.get("succes") and res_2d["x_opt"] is not None:
                    xo = res_2d["x_opt"]
                    fig_fais.add_trace(go.Scatter(x=[xo[0]], y=[xo[1]],
                        mode='markers', name=f'Optimum z*={res_2d["z_opt"]:.3f}',
                        marker=dict(color='#ff00cc', size=16, symbol='star')))
                    st.metric("z* optimal", f"{res_2d['z_opt']:.4f}")
                    st.metric("x1*", f"{xo[0]:.4f}")
                    st.metric("x2*", f"{xo[1]:.4f}")

                # Lignes iso-coût
                z_vals = np.linspace(-20, 20, 8)
                for z_iso in z_vals:
                    if abs(c_2d[1]) > 1e-8:
                        y_iso = (z_iso - c_2d[0]*x_line) / c_2d[1]
                        fig_fais.add_trace(go.Scatter(
                            x=x_line, y=y_iso, mode='lines',
                            line=dict(color='rgba(255,255,255,0.15)', width=1,
                                      dash='dot'), showlegend=False))

                layout(fig_fais, "Domaine faisable & optimum PL",
                       "x₁", "x₂")
                fig_fais.update_xaxes(range=[-0.5, 8])
                fig_fais.update_yaxes(range=[-0.5, 8])
                st.plotly_chart(fig_fais, use_container_width=True)

                # Tableau des sommets
                if len(sommets) > 0:
                    df_som = pd.DataFrame(sommets, columns=["x₁","x₂"])
                    df_som["z = c^T x"] = df_som["x₁"]*c_2d[0] + df_som["x₂"]*c_2d[1]
                    df_som = df_som.round(4)
                    st.dataframe(df_som, use_container_width=True)

        # ---- SUB 3 : DUALITÉ ----
        with sub3:
            st.markdown("### 🔄 Dualité en Programmation Linéaire")
            st.markdown("""
            **Théorème de dualité forte :** Si le primal et le dual ont
            des solutions réalisables, alors leurs valeurs optimales
            sont **égales** : z* = w*.

            | Primal | Dual |
            |--------|------|
            | min c^T x | max b^T y |
            | Ax ≥ b | A^T y ≤ c |
            | x ≥ 0 | y ≥ 0 |
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Primal : min c^T x, Ax ≥ b, x ≥ 0")
                n_d = st.slider("Variables primal", 2, 4, 2, key="nd")
                m_d = st.slider("Contraintes", 1, 3, 2, key="md")

                c_d = np.array([st.number_input(f"c{i+1}", value=float(i+2),
                                key=f"cd_{i}") for i in range(n_d)])
                A_d, b_d = [], []
                for j in range(m_d):
                    row_d = [st.number_input(f"A{j+1}{i+1}",
                        value=float((j+i+1)%3+1), key=f"Ad_{j}_{i}")
                        for i in range(n_d)]
                    b_d.append(st.number_input(f"b{j+1}",
                                value=float((j+1)*3), key=f"bd_{j}"))
                    A_d.append(row_d)
                A_d = np.array(A_d)
                b_d = np.array(b_d)

                if st.button("🔄 Résoudre Primal & Dual", use_container_width=True):
                    # Primal : min c^T x, Ax >= b → -Ax <= -b
                    res_prim = pl_engine.resoudre_pl(c_d, -A_d, -b_d,
                                                      bounds=[(0,None)]*n_d)
                    res_dual = pl_engine.probleme_dual(c_d, A_d, b_d)

                    st.markdown("**Solution Primal :**")
                    if res_prim.get("succes"):
                        st.metric("z* primal", f"{res_prim['z_opt']:.4f}")
                        for i, xi in enumerate(res_prim["x_opt"]):
                            st.metric(f"x{i+1}*", f"{xi:.4f}")

                    with col2:
                        st.markdown("**Solution Dual :**")
                        if res_dual.get("succes"):
                            st.metric("w* dual", f"{res_dual['z_dual']:.4f}")
                            for j, yj in enumerate(res_dual["y_opt"]):
                                st.metric(f"y{j+1}*", f"{yj:.4f}")

                        if (res_prim.get("succes") and res_dual.get("succes")):
                            gap = abs(res_prim["z_opt"] - res_dual["z_dual"])
                            st.metric("Écart de dualité", f"{gap:.6e}")
                            st.success("✅ Dualité forte vérifiée" if gap < 1e-4
                                       else "⚠️ Écart non nul")

                            # Graphique comparatif
                            fig_dual = go.Figure()
                            fig_dual.add_trace(go.Bar(
                                x=["z* Primal", "w* Dual"],
                                y=[res_prim["z_opt"], res_dual["z_dual"]],
                                marker_color=['#00ccff', '#7700ff'],
                                text=[f"{res_prim['z_opt']:.4f}",
                                      f"{res_dual['z_dual']:.4f}"],
                                textposition='outside'
                            ))
                            layout(fig_dual, "Dualité Primal-Dual",
                                   "", "Valeur optimale", h=320)
                            st.plotly_chart(fig_dual, use_container_width=True)

        # ---- SUB 4 : SENSIBILITÉ ----
        with sub4:
            st.markdown("### 📊 Analyse de sensibilité")
            st.markdown("""
            *Comment évolue z* quand les données du problème varient ?*
            """)
            col1, col2 = st.columns([1, 2])
            with col1:
                if "pl_A" in st.session_state and "pl_c" in st.session_state:
                    A_sens = st.session_state["pl_A"]
                    b_sens = st.session_state["pl_b"]
                    c_sens = st.session_state["pl_c"]
                    idx_b = st.slider("Contrainte à varier (b_i)",
                                       0, len(b_sens)-1, 0)
                    b_lo = st.slider("b min", 0.1,
                                      float(b_sens[idx_b])*2, 0.5)
                    b_hi = st.slider("b max", float(b_sens[idx_b]),
                                      float(b_sens[idx_b])*5,
                                      float(b_sens[idx_b])*3)
                    n_pts_s = st.slider("Points", 10, 100, 30)
                else:
                    st.info("Résolvez d'abord un PL dans l'onglet 'Modélisation'")
                    A_sens = np.array([[1,0],[0,1],[1,1]])
                    b_sens = np.array([4.0, 3.0, 5.0])
                    c_sens = np.array([-3.0, -2.0])
                    idx_b, b_lo, b_hi, n_pts_s = 0, 1.0, 10.0, 30

            with col2:
                b_range = np.linspace(b_lo, b_hi, n_pts_s)
                df_sens = pl_engine.analyse_sensibilite(
                    c_sens, A_sens, b_sens, idx_b, b_range)

                if not df_sens.empty:
                    fig_sens = go.Figure()
                    col_b = df_sens.columns[0]
                    fig_sens.add_trace(go.Scatter(
                        x=df_sens[col_b], y=df_sens["z*"], mode='lines+markers',
                        name="z*(b)", line=dict(color='#00ccff', width=2.5)))
                    layout(fig_sens,
                           f"Sensibilité de z* à b_{idx_b}",
                           f"b_{idx_b}", "z* optimal")
                    st.plotly_chart(fig_sens, use_container_width=True)

                    if "x1*" in df_sens.columns and "x2*" in df_sens.columns:
                        fig_s2 = go.Figure()
                        fig_s2.add_trace(go.Scatter(x=df_sens[col_b],
                            y=df_sens["x1*"], mode='lines', name='x1*',
                            line=dict(color='#7700ff', width=2)))
                        fig_s2.add_trace(go.Scatter(x=df_sens[col_b],
                            y=df_sens["x2*"], mode='lines', name='x2*',
                            line=dict(color='#ff00cc', width=2)))
                        layout(fig_s2, "Évolution de la solution",
                               f"b_{idx_b}", "Valeur x*", h=300)
                        st.plotly_chart(fig_s2, use_container_width=True)

                    st.dataframe(df_sens.round(4), use_container_width=True)
                    st.download_button("💾 Export CSV",
                        df_sens.to_csv(index=False).encode(),
                        "sensibilite_pl.csv", "text/csv")

        # ---- SUB 5 : THÉORIE ----
        with sub5:
            st.markdown("### 📖 Théorie — Programmation Linéaire")
            st.markdown("""
            #### Définition
            Un **problème de PL** est un problème d'optimisation où
            la fonction objectif et toutes les contraintes sont
            **linéaires** en les variables de décision.

            #### Propriétés fondamentales
            - Le domaine faisable est un **polytope** convexe
            - Si une solution optimale existe, elle se trouve
              en un **sommet** du polytope
            - L'algorithme du **simplexe** parcourt les sommets
            - La méthode des **points intérieurs** traverse l'intérieur
            """)

            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_PL.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f_latex)
                col_idx += 1

            st.markdown("---")
            st.markdown("### 📚 Références")
            for r in [
                "Bertsimas & Tsitsiklis — *Introduction to Linear Optimization* (Athena, 1997)",
                "Dantzig — *Linear Programming and Extensions* (Princeton, 1963)",
                "Nocedal & Wright — *Numerical Optimization* (Springer, 2006)",
            ]:
                st.markdown(f"- {r}")

    # ============================================================
    # TAB 3 — CONVEXITÉ
    # ============================================================
    elif section == "🔷 Convexité":
        st.markdown("## 🔷 Ensembles Convexes & Fonctions Convexes")
        st.markdown("*Définitions, propriétés, vérification numérique, Jensen, conjuguée*")

        sub1, sub2, sub3, sub4, sub5 = st.tabs([
            "🔷 Ensembles convexes",
            "📈 Fonctions convexes",
            "⚡ Inégalité de Jensen",
            "🔀 Conjuguée de Fenchel",
            "📖 Théorie"
        ])

        # ---- SUB 1 : ENSEMBLES ----
        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔷 Types d'ensembles convexes")
                type_ens = st.selectbox("Ensemble", [
                    "Boule (convexe ✅)",
                    "Polytope (convexe ✅)",
                    "Enveloppe convexe",
                    "Intersection de convexes"
                ])
                rayon_b = st.slider("Rayon", 0.5, 3.0, 1.5, 0.1)

                st.markdown("### 📐 Projection sur un convexe")
                px = st.slider("Point x", -4.0, 4.0, 2.5, 0.1)
                py = st.slider("Point y", -4.0, 4.0, 2.0, 0.1)
                type_proj = st.selectbox("Projeter sur",
                    ["boule", "hyperplan", "cone_positif", "simplexe"])

                point = np.array([px, py])
                kwargs_p = {}
                if type_proj == "boule":
                    kwargs_p = {"centre": np.zeros(2), "rayon": rayon_b}
                elif type_proj == "hyperplan":
                    kwargs_p = {"a": np.array([1.0, 1.0]), "b": 1.0}

                proj = conv_engine.projection_convexe(point, type_proj, **kwargs_p)
                st.metric("P(point) → x", f"{proj[0]:.4f}")
                st.metric("P(point) → y", f"{proj[1]:.4f}")
                st.metric("Distance", f"{np.linalg.norm(point-proj):.4f}")

            with col2:
                fig_ens = go.Figure()
                theta_c = np.linspace(0, 2*np.pi, 200)

                if "Boule" in type_ens:
                    xc = rayon_b*np.cos(theta_c)
                    yc = rayon_b*np.sin(theta_c)
                    fig_ens.add_trace(go.Scatter(x=xc, y=yc, mode='lines',
                        name='Boule B(0,r)',
                        line=dict(color='#00ccff', width=3),
                        fill='toself', fillcolor='rgba(0,204,255,0.1)'))
                    fig_ens.add_annotation(text=f"r={rayon_b}",
                        x=rayon_b/2, y=0, font=dict(color='#ffcc00'))

                elif "Polytope" in type_ens:
                    sommets_p = np.array([[0,0],[3,0],[2.5,2],[1,3],[-0.5,1.5]])
                    env_p = conv_engine.enveloppe_convexe_2D(sommets_p)
                    fig_ens.add_trace(go.Scatter(
                        x=env_p[:,0], y=env_p[:,1], mode='lines',
                        fill='toself', fillcolor='rgba(119,0,255,0.2)',
                        line=dict(color='#7700ff', width=3), name='Polytope'))
                    fig_ens.add_trace(go.Scatter(
                        x=sommets_p[:,0], y=sommets_p[:,1],
                        mode='markers', name='Sommets',
                        marker=dict(color='#ffcc00', size=10)))

                elif "Enveloppe" in type_ens:
                    np.random.seed(42)
                    pts = np.random.randn(15, 2) * 2
                    fig_ens.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1],
                        mode='markers', name='Points',
                        marker=dict(color='#7700ff', size=8)))
                    env = conv_engine.enveloppe_convexe_2D(pts)
                    fig_ens.add_trace(go.Scatter(x=env[:,0], y=env[:,1],
                        mode='lines', fill='toself',
                        fillcolor='rgba(0,204,255,0.1)',
                        line=dict(color='#00ccff', width=2.5),
                        name='Enveloppe convexe'))

                else:  # Intersection
                    xc1 = 1.5*np.cos(theta_c)
                    yc1 = 1.5*np.sin(theta_c)
                    xc2 = 1.5*np.cos(theta_c) + 1
                    yc2 = 1.5*np.sin(theta_c) + 0.5
                    fig_ens.add_trace(go.Scatter(x=xc1, y=yc1, mode='lines',
                        name='C₁', line=dict(color='#00ccff', width=2.5)))
                    fig_ens.add_trace(go.Scatter(x=xc2, y=yc2, mode='lines',
                        name='C₂', line=dict(color='#7700ff', width=2.5)))

                # Point et projection
                fig_ens.add_trace(go.Scatter(x=[px], y=[py], mode='markers',
                    name='Point', marker=dict(color='#ff00cc', size=14,
                    symbol='circle')))
                fig_ens.add_trace(go.Scatter(x=[proj[0]], y=[proj[1]],
                    mode='markers', name='Projeté',
                    marker=dict(color='#00ff88', size=14, symbol='star')))
                fig_ens.add_trace(go.Scatter(
                    x=[px, proj[0]], y=[py, proj[1]],
                    mode='lines', showlegend=False,
                    line=dict(color='#ffcc00', width=2, dash='dash')))

                layout(fig_ens, "Ensemble convexe & Projection",
                       "x", "y")
                fig_ens.update_xaxes(range=[-5, 5])
                fig_ens.update_yaxes(range=[-5, 5], scaleanchor='x')
                st.plotly_chart(fig_ens, use_container_width=True)

                # Vérification combinaison convexe
                st.markdown("#### ✅ Vérification combinaison convexe")
                lam_val = st.slider("λ ∈ [0,1]", 0.0, 1.0, 0.4, 0.01)
                x_a = np.array([st.slider("xA", -3.0, 3.0, -2.0, 0.1),
                                 st.slider("yA", -3.0, 3.0, -1.0, 0.1)])
                x_b = np.array([st.slider("xB", -3.0, 3.0, 2.0, 0.1),
                                 st.slider("yB", -3.0, 3.0, 1.5, 0.1)])
                x_combo = lam_val*x_a + (1-lam_val)*x_b
                st.metric("λxA + (1-λ)xB", f"({x_combo[0]:.2f}, {x_combo[1]:.2f})")

        # ---- SUB 2 : FONCTIONS CONVEXES ----
        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                catalogue = conv_engine.fonctions_convexes_catalogue()
                func_sel = st.selectbox("Fonction", list(catalogue.keys()))
                f_sel, conv_theo, domaine = catalogue[func_sel]
                x_range_f = st.slider("Plage [a, b]", -5.0, 5.0, (-3.0, 3.0))
                x_arr_f = np.linspace(x_range_f[0], x_range_f[1], 500)

                est_conv_num, max_viol = conv_engine.est_convexe_numerique(
                    f_sel, x_range_f)
                f2_arr = conv_engine.hessienne_1D(f_sel, x_arr_f)

                st.metric("Convexe (théorie)", "✅ Oui" if conv_theo else "❌ Non")
                st.metric("Convexe (numérique)", "✅ Oui" if est_conv_num else "❌ Non")
                st.metric("Violation max", f"{max_viol:.2e}")
                st.metric("f'' ≥ 0 partout",
                          "✅ Oui" if np.all(f2_arr >= -0.01) else "❌ Non")
                st.metric("Domaine", domaine)

            with col2:
                try:
                    f_arr_f = np.array([f_sel(xi) for xi in x_arr_f])
                    fig_cv = make_subplots(rows=2, cols=1,
                        subplot_titles=["f(x)", "f''(x) (convexe si ≥ 0)"])
                    fig_cv.add_trace(go.Scatter(x=x_arr_f, y=f_arr_f,
                        mode='lines', name='f(x)',
                        line=dict(color='#00ccff', width=3)), row=1, col=1)

                    # Tangentes (vérification condition 1er ordre)
                    x_tan = st.slider("Point tangente x₀",
                                       x_range_f[0], x_range_f[1], 0.0, 0.1)
                    f_tan = f_sel(x_tan)
                    h_tan = 1e-5
                    f_prime = (f_sel(x_tan+h_tan)-f_sel(x_tan-h_tan))/(2*h_tan)
                    y_tan = f_tan + f_prime*(x_arr_f - x_tan)
                    fig_cv.add_trace(go.Scatter(x=x_arr_f, y=y_tan,
                        mode='lines', name=f'Tangente en x₀={x_tan:.2f}',
                        line=dict(color='#ffcc00', width=2, dash='dash')),
                        row=1, col=1)
                    fig_cv.add_trace(go.Scatter(x=[x_tan], y=[f_tan],
                        mode='markers', marker=dict(color='#ff00cc', size=10),
                        name='x₀', showlegend=False), row=1, col=1)

                    fig_cv.add_trace(go.Scatter(x=x_arr_f, y=f2_arr,
                        mode='lines', name="f''(x)",
                        line=dict(color='#7700ff', width=2.5)), row=2, col=1)
                    fig_cv.add_hline(y=0, line_color='rgba(255,255,255,0.3)',
                                     line_dash='dash', row=2, col=1)

                    fig_cv.update_layout(**PLOT_LAYOUT, height=550,
                        title=f"Analyse convexité — {func_sel}",
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                    fig_cv.update_xaxes(**AXIS)
                    fig_cv.update_yaxes(**AXIS)
                    st.plotly_chart(fig_cv, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur : {e}")

        # ---- SUB 3 : JENSEN ----
        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### ⚡ Inégalité de Jensen")
                st.latex(r"f\!\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i)")
                func_j = st.selectbox("Fonction", [
                    "x²", "eˣ", "-log(x)", "|x|", "x·log(x)"])
                fj_map = {"x²": lambda x: x**2, "eˣ": np.exp,
                          "-log(x)": lambda x: -np.log(max(x,1e-10)),
                          "|x|": abs,
                          "x·log(x)": lambda x: x*np.log(max(x,1e-10))}
                f_j = fj_map[func_j]
                n_pts_j = st.slider("Nombre de points", 2, 6, 3)
                x_pts_j = np.array([st.slider(f"x{i+1}", -3.0, 3.0,
                    float(i-n_pts_j//2), 0.1, key=f"xj_{i}")
                    for i in range(n_pts_j)])
                lam_j = np.ones(n_pts_j)/n_pts_j
                result_j = conv_engine.verifier_jensen(f_j, x_pts_j, lam_j)
                st.metric("f(x̄)", f"{result_j['f(x̄)']:.6f}")
                st.metric("Σλᵢf(xᵢ)", f"{result_j['Σλᵢf(xᵢ)']:.6f}")
                st.metric("Jensen vérifié",
                          "✅ Oui" if result_j['Jensen vérifié'] else "❌ Non")
                st.metric("Écart (≥0 si convexe)", f"{result_j['Écart']:.6f}")

            with col2:
                x_plot_j = np.linspace(-4, 4, 500)
                try:
                    y_plot_j = np.array([f_j(xi) for xi in x_plot_j])
                    fig_j = go.Figure()
                    fig_j.add_trace(go.Scatter(x=x_plot_j, y=y_plot_j,
                        mode='lines', name='f(x)',
                        line=dict(color='#00ccff', width=3)))
                    y_pts = np.array([f_j(xi) for xi in x_pts_j])
                    fig_j.add_trace(go.Scatter(x=x_pts_j, y=y_pts,
                        mode='markers+text', name='Points xᵢ',
                        text=[f"x{i+1}" for i in range(n_pts_j)],
                        textposition='top center',
                        marker=dict(color='#ffcc00', size=12)))
                    x_bar = np.mean(x_pts_j)
                    f_xbar = f_j(x_bar)
                    f_mean = np.mean(y_pts)
                    fig_j.add_trace(go.Scatter(x=[x_bar], y=[f_xbar],
                        mode='markers', name=f'f(x̄)={f_xbar:.4f}',
                        marker=dict(color='#00ff88', size=14, symbol='star')))
                    fig_j.add_trace(go.Scatter(x=[x_bar], y=[f_mean],
                        mode='markers', name=f'Σλᵢf(xᵢ)={f_mean:.4f}',
                        marker=dict(color='#ff00cc', size=14, symbol='diamond')))
                    fig_j.add_trace(go.Scatter(x=x_pts_j, y=y_pts,
                        mode='lines', showlegend=False,
                        line=dict(color='rgba(255,200,0,0.4)', width=2,
                                  dash='dot')))
                    layout(fig_j, f"Inégalité de Jensen — {func_j}",
                           "x", "f(x)")
                    st.plotly_chart(fig_j, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur : {e}")

        # ---- SUB 4 : CONJUGUÉE ----
        with sub4:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔀 Conjuguée de Fenchel")
                st.latex(r"f^*(y) = \sup_x \{yx - f(x)\}")
                func_c = st.selectbox("Fonction f(x)", [
                    "x²/2", "eˣ", "|x|", "x⁴/4", "x²"])
                fc_map = {"x²/2": lambda x: x**2/2,
                          "eˣ": np.exp,
                          "|x|": abs,
                          "x⁴/4": lambda x: x**4/4,
                          "x²": lambda x: x**2}
                f_c = fc_map[func_c]

                conj_theoriques = {
                    "x²/2": "y²/2", "eˣ": "y·log(y)-y",
                    "|x|": "0 si |y|≤1, sinon ∞",
                    "x⁴/4": "3y^(4/3)/4", "x²": "y²/4"
                }
                st.info(f"**f*(y) théorique :** {conj_theoriques[func_c]}")

            with col2:
                y_range_c = (-3.0, 3.0)
                y_arr_c, f_star_c = conv_engine.conjuguee_convexe(
                    f_c, y_range_c, x_range=(-5, 5))
                x_arr_c2 = np.linspace(-4, 4, 500)
                f_arr_c2 = np.array([f_c(xi) for xi in x_arr_c2])

                fig_conj = make_subplots(rows=1, cols=2,
                    subplot_titles=["f(x)", "f*(y) (conjuguée)"])
                fig_conj.add_trace(go.Scatter(x=x_arr_c2, y=f_arr_c2,
                    mode='lines', name='f(x)',
                    line=dict(color='#00ccff', width=2.5)), row=1, col=1)
                fig_conj.add_trace(go.Scatter(x=y_arr_c, y=f_star_c,
                    mode='lines', name='f*(y)',
                    line=dict(color='#ff00cc', width=2.5)), row=1, col=2)
                fig_conj.update_layout(**PLOT_LAYOUT, height=400,
                    title=f"Conjuguée de Fenchel — f(x) = {func_c}",
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_conj.update_xaxes(**AXIS)
                fig_conj.update_yaxes(**AXIS)
                st.plotly_chart(fig_conj, use_container_width=True)

        # ---- SUB 5 : THÉORIE CONVEXITÉ ----
        with sub5:
            st.markdown("### 📖 Théorie — Ensembles et Fonctions Convexes")
            st.markdown("""
            #### Définition — Ensemble convexe
            Un ensemble C est **convexe** si pour tout x, y ∈ C et λ ∈ [0,1] :
            λx + (1-λ)y ∈ C.

            *Exemples :* boules, polytopes, hyperplans, cône positif, espace affine.

            #### Définition — Fonction convexe
            f est **convexe** sur C si pour tout x, y ∈ C et λ ∈ [0,1] :
            f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y).

            #### Stabilité par opérations
            - Somme de fonctions convexes → convexe
            - Max de fonctions convexes → convexe
            - Composition affine f(Ax+b) → convexe si f convexe
            - Inf-convolution → convexe
            """)

            cols = st.columns(2)
            col_idx = 0
            for nom, f_latex in FORMULES_CONVEXE.items():
                with cols[col_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{nom}**")
                        st.latex(f_latex)
                col_idx += 1

    # ============================================================
    # TAB 4 — OPTIMISATION CONVEXE
    # ============================================================
    elif section == "⚡ Optimisation Convexe":
        st.markdown("## ⚡ Optimisation Convexe")
        st.markdown("*Descente de gradient, Nesterov, gradient projeté, LASSO, Ridge*")

        sub1, sub2, sub3, sub4, sub5 = st.tabs([
            "📉 Descente de gradient",
            "🚀 Méthodes avancées",
            "🎯 Gradient projeté",
            "🔗 LASSO & Ridge",
            "📖 Théorie"
        ])

        # ---- SUB 1 : DESCENTE GRADIENT ----
        with sub1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 📉 Descente de gradient")
                use_custom_gd = st.checkbox("Fonction personnalisée", value=False)
                f_gd = None
                grad_gd = None
                alpha_def = 0.1
                x_star = None
                x_dim = 2
                if use_custom_gd:
                    var_list = st.text_input(
                        "Variables (séparées par des virgules)",
                        value="x0, x1",
                        help="Entrez les variables, par exemple x0, x1 ou x, y.",
                        key="gd_var_names"
                    )
                    var_names = [v.strip() for v in var_list.split(',') if v.strip()]
                    x_dim = len(var_names)
                    default_expr = ""
                    if x_dim == 1:
                        default_expr = f"{var_names[0]}**2"
                    elif x_dim == 2:
                        default_expr = f"{var_names[0]}**2 + 2*{var_names[1]}**2"
                    else:
                        default_expr = " + ".join(f"{name}**2" for name in var_names)

                    expr = st.text_area(
                        "Expression de la fonction f(x)",
                        value=default_expr,
                        help="Utilisez ** pour la puissance, np.sin, np.exp, abs, etc.",
                        key="gd_expr"
                    )
                    expr = expr.replace('^', '**')

                    if len(var_names) == 0:
                        st.error("Veuillez définir au moins une variable.")
                    else:
                        bad_tokens = ['__', 'import', 'os.', 'sys.', 'eval(', 'exec(']
                        if any(tok in expr for tok in bad_tokens):
                            st.error("Expression invalide — token interdit détecté.")
                        else:
                            invalid_vars = [v for v in var_names if not v.isidentifier()]
                            if invalid_vars:
                                st.error(f"Noms de variables invalides : {', '.join(invalid_vars)}")
                            else:
                                try:
                                    f_gd = engine.construire_fonction_personnalisee(expr, var_names)
                                    grad_gd = None
                                    if x_dim == 1:
                                        x_star = None
                                    alpha_def = 0.1
                                    st.success("Fonction personnalisée prête à l'utilisation.")
                                except Exception as e:
                                    st.error(f"Erreur lors de la construction de la fonction : {e}")
                else:
                    func_gd = st.selectbox("Fonction", [
                        "Quadratique f(x)=x²+2y²",
                        "Rosenbrock",
                        "f(x,y)=(x-2)²+(y+1)²",
                        "f(x)=x⁴-3x²+x"
                    ])
                    fgd_map = {
                        "Quadratique f(x)=x²+2y²":   (lambda x: x[0]**2 + 2*x[1]**2,
                            lambda x: np.array([2*x[0], 4*x[1]]), 8.0, [0,0]),
                        "Rosenbrock":                  (lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2,
                            lambda x: np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2),
                                                 200*(x[1]-x[0]**2)]), 0.001, [1,1]),
                        "f(x,y)=(x-2)²+(y+1)²":      (lambda x: (x[0]-2)**2+(x[1]+1)**2,
                            lambda x: np.array([2*(x[0]-2), 2*(x[1]+1)]), 0.3, [2,-1]),
                        "f(x)=x⁴-3x²+x":             (lambda x: x[0]**4-3*x[0]**2+x[0],
                            lambda x: np.array([4*x[0]**3-6*x[0]+1]), 0.05, [1.5]),
                    }
                    f_gd, grad_gd, alpha_def, x_star = fgd_map[func_gd]
                    x_dim = 2 if "x)=x⁴" not in func_gd else 1

                x0_gd1 = st.slider("x₀", -4.0, 4.0, 3.0, 0.1, key="x0_gd")
                if x_dim == 1:
                    x0_gd = np.array([x0_gd1])
                else:
                    x0_gd2 = st.slider("y₀", -4.0, 4.0, 3.0, 0.1, key="y0_gd")
                    x0_gd = np.array([x0_gd1, x0_gd2])

                alpha_gd = st.slider("Pas α", 0.0001, 1.0,
                                      min(alpha_def, 0.5), 0.0001,
                                      format="%.4f")
                n_iter_gd = st.slider("Itérations", 10, 1000, 200, 10)

                if f_gd is not None:
                    res_gd = opt_conv_engine.descente_gradient(
                        f_gd, grad_gd, x0_gd, alpha_gd, n_iter_gd)
                    if x_star is not None:
                        conv_info = opt_conv_engine.taux_convergence(
                            res_gd["valeurs"], f_gd(np.array(x_star)))
                    else:
                        conv_info = {}
                else:
                    res_gd = {"f_opt": np.nan, "valeurs": np.array([]),
                              "historique": np.zeros((0, x_dim)), "gradients": np.array([]),
                              "n_iter": 0}
                    conv_info = {}

                st.metric("f(x*) trouvé", f"{res_gd['f_opt']:.6f}")
                st.metric("f(x*) théorique",
                          f"{f_gd(np.array(x_star)):.6f}" if x_star is not None else "N/A")
                st.metric("Itérations", res_gd["n_iter"])
                st.metric("‖∇f‖ final",
                          f"{res_gd['gradients'][-1]:.2e}" if len(res_gd['gradients'])>0 else "N/A")
                if not np.isnan(conv_info.get("taux", np.nan)):
                    st.metric("Taux convergence", f"{conv_info['taux']:.4f}")

            with col2:
                hist_gd = res_gd["historique"]
                vals_gd = res_gd["valeurs"]

                # Courbe de convergence
                fig_gd = make_subplots(rows=2, cols=1,
                    subplot_titles=["f(xₖ) — convergence", "‖∇f(xₖ)‖"])
                fig_gd.add_trace(go.Scatter(y=vals_gd, mode='lines',
                    name='f(xₖ)', line=dict(color='#00ccff', width=2.5)),
                    row=1, col=1)
                if len(res_gd["gradients"]) > 0:
                    fig_gd.add_trace(go.Scatter(y=res_gd["gradients"],
                        mode='lines', name='‖∇f‖',
                        line=dict(color='#ff00cc', width=2)), row=2, col=1)
                fig_gd.update_layout(**PLOT_LAYOUT, height=480,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                fig_gd.update_xaxes(**AXIS, title_text="Itération k")
                fig_gd.update_yaxes(**AXIS)
                fig_gd.update_yaxes(type='log', row=1, col=1)
                st.plotly_chart(fig_gd, use_container_width=True)

                # Trajectoire 2D
                if hist_gd.shape[1] >= 2:
                    x_c = np.linspace(-5, 5, 80)
                    y_c = np.linspace(-5, 5, 80)
                    Xc, Yc = np.meshgrid(x_c, y_c)
                    Zc = np.vectorize(lambda xi,yi: f_gd([xi,yi]))(Xc, Yc)
                    fig_traj = go.Figure()
                    fig_traj.add_trace(go.Contour(z=np.clip(Zc,0,np.percentile(Zc,95)),
                        x=x_c, y=y_c,
                        colorscale=[[0,'#020817'],[0.4,'#7700ff'],
                                    [0.7,'#00ccff'],[1,'#ffffff']],
                        showscale=False))
                    fig_traj.add_trace(go.Scatter(x=hist_gd[:,0], y=hist_gd[:,1],
                        mode='lines+markers', name='Trajectoire GD',
                        line=dict(color='#ffcc00', width=2),
                        marker=dict(size=4)))
                    if x_star is not None:
                        x_star_arr = np.asarray(x_star)
                        if x_star_arr.size >= 2:
                            fig_traj.add_trace(go.Scatter(x=[float(x_star_arr.ravel()[0])],
                                y=[float(x_star_arr.ravel()[1])],
                                mode='markers', name='Optimum théorique',
                                marker=dict(color='#ff0000', size=14, symbol='x')))
                    layout(fig_traj, "Trajectoire de descente", "x", "y")
                    st.plotly_chart(fig_traj, use_container_width=True)

        # ---- SUB 2 : MÉTHODES AVANCÉES ----
        with sub2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🚀 Comparaison GD / Nesterov / Armijo")
                L_cmp = st.slider("L (Lipschitz gradient)", 0.1, 20.0, 2.0, 0.1)
                x0_cmp = np.array([st.slider("x₀", -5.0, 5.0, 3.0, 0.1, key="x0c"),
                                    st.slider("y₀", -5.0, 5.0, 3.0, 0.1, key="y0c")])
                f_cmp = lambda x: x[0]**2/L_cmp + x[1]**2
                grad_cmp = lambda x: np.array([2*x[0]/L_cmp, 2*x[1]])
                f_star_cmp = 0.0

                comparaison = opt_conv_engine.comparaison_methodes_convexes(
                    f_cmp, grad_cmp, x0_cmp, L_cmp)

                for nom_m, res_m in comparaison.items():
                    c1, c2 = st.columns(2)
                    with c1: st.metric(f"{nom_m} — f*", f"{res_m['f_opt']:.6f}")
                    with c2: st.metric("Itérations", len(res_m["valeurs"])-1)

                st.info("""
                **Nesterov** converge en O(1/k²) contre O(1/k) pour GD.
                """)
                st.latex(r"f(x_k)-f^*\leq\frac{2L\|x_0-x^*\|^2}{(k+1)^2}")

                with st.expander("🔎 Analyse des points critiques", expanded=True):
                    st.markdown("*Outil hybride : root-finding + minimisation du carré du gradient pour trouver des points critiques robustes.*")
                    use_preset = st.checkbox("Utiliser une fonction prédéfinie", value=True)
                    if use_preset:
                        points_choices = {
                            "Quadratique f(x)=x²+2y²": (
                                lambda x: x[0]**2 + 2*x[1]**2,
                                lambda x: np.array([2*x[0], 4*x[1]]),
                                [np.array([a, b]) for a in [-3, 0, 3] for b in [-3, 0, 3]]),
                            "f(x,y)=(x-2)²+(y+1)²": (
                                lambda x: (x[0]-2)**2 + (x[1]+1)**2,
                                lambda x: np.array([2*(x[0]-2), 2*(x[1]+1)]),
                                [np.array([a, b]) for a in [-4, 0, 4] for b in [-4, 0, 4]]),
                            "Rosenbrock": (
                                lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
                                lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                                                     200*(x[1]-x[0]**2)]),
                                [np.array([a, b]) for a in [-2, 0, 2] for b in [-1, 1, 3]]),
                            "f(x)=x⁴-3x²+x": (
                                lambda x: x[0]**4 - 3*x[0]**2 + x[0],
                                lambda x: np.array([4*x[0]**3 - 6*x[0] + 1]),
                                [np.array([a]) for a in [-3, -1, 0, 1, 3]]),
                        }
                        crit_choice = st.selectbox("Fonction à analyser",
                                                  list(points_choices.keys()), key="crit_choice")
                        f_crit, grad_crit, seeds = points_choices[crit_choice]
                    else:
                        var_list = st.text_input(
                            "Variables (séparées par des virgules)",
                            value="x0, x1",
                            help="Entrez les noms de variables à utiliser dans l'expression, par exemple x0, x1 ou x, y.",
                            key="crit_vars")
                        var_names = [v.strip() for v in var_list.split(',') if v.strip()]
                        if len(var_names) == 0:
                            st.error("Veuillez saisir au moins une variable.")
                        N = len(var_names)
                        st.markdown(f"Dimension détectée : **{N}** variable(s).")

                        if N == 1:
                            default_expr = f"{var_names[0]}**4 - 3*{var_names[0]}**2 + {var_names[0]}"
                        elif N == 2:
                            default_expr = f"{var_names[0]}**2 + 2*{var_names[1]}**2"
                        else:
                            default_expr = " + ".join(f"{name}**2" for name in var_names)

                        expr = st.text_area(
                            "Expression de la fonction f(x)",
                            default_expr,
                            help="Utiliser les variables définies ci-dessus ; utilisez ** pour la puissance.",
                            key="crit_expr")
                        expr = expr.replace('^', '**')

                        x_min = st.number_input("Min (valeur commune)", -10.0, 10.0, -3.0, 0.1, key="crit_x_min")
                        x_max = st.number_input("Max (valeur commune)", -10.0, 10.0, 3.0, 0.1, key="crit_x_max")
                        if x_min >= x_max:
                            st.error("Le minimum doit être inférieur au maximum.")

                        n_seeds = st.slider("Nombre de points d'amorçage", 1, 200, 9, key="crit_n_seeds")

                        if N == 1:
                            seeds = [np.array([x0]) for x0 in np.linspace(x_min, x_max, max(2, n_seeds))]
                        elif N == 2 and n_seeds <= 100:
                            grid = int(np.ceil(np.sqrt(n_seeds)))
                            x0_vals = np.linspace(x_min, x_max, grid)
                            x1_vals = np.linspace(x_min, x_max, grid)
                            seeds = [np.array([x0, x1]) for x0 in x0_vals for x1 in x1_vals]
                        else:
                            seeds = [np.random.uniform(x_min, x_max, size=N) for _ in range(n_seeds)]

                        # Validations simples sur l'expression fournie
                        f_crit = None
                        grad_crit = None
                        bad_tokens = ['__', 'import', 'os.', 'sys.', 'eval(', 'exec(']
                        if any(tok in expr for tok in bad_tokens):
                            st.error("Expression invalide — token interdit détecté.")
                            f_crit = None
                        else:
                            invalid_vars = [v for v in var_names if not v.isidentifier()]
                            if invalid_vars:
                                st.error(f"Noms de variables invalides : {', '.join(invalid_vars)}")
                                f_crit = None
                            else:
                                used_vars = re.findall(r'([A-Za-z_]\\w*)', expr)
                                if not any(v in used_vars for v in var_names):
                                    st.error("Aucune des variables définies n'est utilisée dans l'expression.")
                                    f_crit = None
                                else:
                                    try:
                                        f_crit = engine.construire_fonction_personnalisee(expr, var_names)
                                        grad_crit = None
                                    except Exception as e:
                                        st.error(f"Erreur lors de la construction de la fonction : {e}")
                                        f_crit = None
                                        grad_crit = None
                    if st.button("Calculer les points critiques", key="calc_crit"):
                        if 'f_crit' in locals() and f_crit is not None:
                            solutions = engine.trouver_points_critiques(
                                f_crit, grad_crit, seeds)
                            if len(solutions) == 0:
                                st.warning("Aucun point critique trouvé avec les amorces fournies.")
                            else:
                                rows = []
                                for sol in solutions:
                                    x_opt = sol["x"]
                                    analysis = engine.analyser_point(f_crit, x_opt)
                                    coord = ", ".join(f"{xi:.4f}" for xi in x_opt)
                                    rows.append({
                                        "Coordonnées": f"({coord})",
                                        "Méthode": sol.get("method", "N/A"),
                                        "Nature": analysis["type_point"],
                                        "‖∇f‖": f"{analysis['gradient_norme']:.3e}",
                                        "Conditionnement H": f"{analysis['conditionnement']:.3e}"
                                    })
                                    st.markdown(f"#### Point critique : ({coord})")
                                    st.markdown("**Matrice Hessienne**")
                                    st.write(np.round(analysis["hessienne"], 4))
                                    st.markdown("**Valeurs propres**")
                                    st.write(np.round(analysis["valeurs_propres"].real, 6))
                                    st.markdown("**Vecteurs propres (colonnes)**")
                                    st.write(np.round(analysis["vecteurs_propres"].real, 4))
                                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            with col2:
                fig_cmp = go.Figure()
                style_map = {"GD (pas fixe)": ('#00ccff', 'solid'),
                              "Nesterov": ('#ff00cc', 'dash'),
                              "GD Armijo": ('#00ff88', 'dot')}
                for nom_m, res_m in comparaison.items():
                    errs = np.maximum(res_m["valeurs"] - f_star_cmp, 1e-15)
                    col_m, dash_m = style_map[nom_m]
                    fig_cmp.add_trace(go.Scatter(y=errs, mode='lines',
                        name=nom_m,
                        line=dict(color=col_m, width=2.5, dash=dash_m)))

                # Courbes théoriques
                k_th = np.arange(1, max(len(v["valeurs"]) for v in comparaison.values())+1)
                R2 = np.linalg.norm(x0_cmp)**2
                gd_th = L_cmp*R2/(2*k_th)
                nest_th = 2*L_cmp*R2/(k_th+1)**2
                fig_cmp.add_trace(go.Scatter(y=gd_th[:len(k_th)], mode='lines',
                    name='O(1/k) théorique',
                    line=dict(color='rgba(0,204,255,0.3)', width=1.5, dash='dot')))
                fig_cmp.add_trace(go.Scatter(y=nest_th[:len(k_th)], mode='lines',
                    name='O(1/k²) théorique',
                    line=dict(color='rgba(255,0,204,0.3)', width=1.5, dash='dot')))

                fig_cmp.update_layout(**PLOT_LAYOUT,
                    title="Comparaison convergence f(xₖ)-f*",
                    yaxis=dict(**AXIS, type='log', title="f(xₖ)-f*"),
                    xaxis=dict(**AXIS, title="Itération k"),
                    height=450, legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_cmp, use_container_width=True)

        # ---- SUB 3 : GRADIENT PROJETÉ ----
        with sub3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🎯 Gradient projeté")
                st.latex(r"x_{k+1} = P_C(x_k - \alpha \nabla f(x_k))")
                type_c_proj = st.selectbox("Ensemble de contraintes C", [
                    "Boule unitaire", "Orthant positif (x≥0)",
                    "Simplexe standard", "Intervalle [a,b]"
                ])
                alpha_proj = st.slider("α", 0.001, 0.5, 0.1, 0.001,
                                       format="%.3f", key="alpha_proj")
                x0_p1 = st.slider("x₀", -3.0, 3.0, 2.0, 0.1, key="x0_p")
                x0_p2 = st.slider("y₀", -3.0, 3.0, 2.0, 0.1, key="y0_p")
                x0_proj = np.array([x0_p1, x0_p2])

                f_proj = lambda x: (x[0]-1.5)**2 + 2*(x[1]+0.5)**2
                grad_proj = lambda x: np.array([2*(x[0]-1.5), 4*(x[1]+0.5)])

                conv_eng_loc = ConvexiteEngine()
                proj_map = {
                    "Boule unitaire": lambda x: conv_eng_loc.projection_convexe(x,"boule",centre=np.zeros(2),rayon=1.0),
                    "Orthant positif (x≥0)": lambda x: conv_eng_loc.projection_convexe(x,"cone_positif"),
                    "Simplexe standard": lambda x: conv_eng_loc.projection_convexe(x,"simplexe"),
                    "Intervalle [a,b]": lambda x: np.clip(x, -1, 1),
                }
                proj_fn = proj_map[type_c_proj]

                res_proj = opt_conv_engine.gradient_projete(
                    f_proj, grad_proj, x0_proj, proj_fn, alpha_proj)

                x_opt_proj = res_proj.get("x_opt")
                if x_opt_proj is None or len(x_opt_proj) < 2:
                    x_opt_display = "N/A"
                else:
                    x_opt_display = f"({x_opt_proj[0]:.4f}, {x_opt_proj[1]:.4f})"

                st.metric("f(x*)", f"{res_proj.get('f_opt', np.nan):.6f}")
                st.metric("x*", x_opt_display)
                st.metric("Itérations", len(res_proj.get("valeurs", []))-1)

            with col2:
                hist_p = res_proj["historique"]
                x_c2 = np.linspace(-3, 3, 80)
                y_c2 = np.linspace(-3, 3, 80)
                Xp, Yp = np.meshgrid(x_c2, y_c2)
                Zp = (Xp-1.5)**2 + 2*(Yp+0.5)**2

                fig_proj = go.Figure()
                fig_proj.add_trace(go.Contour(z=Zp, x=x_c2, y=y_c2,
                    colorscale=[[0,'#020817'],[0.4,'#7700ff'],
                                [0.7,'#00ccff'],[1,'#ffffff']],
                    showscale=False))

                # Visualiser l'ensemble C
                theta_p = np.linspace(0, 2*np.pi, 200)
                if type_c_proj == "Boule unitaire":
                    fig_proj.add_trace(go.Scatter(
                        x=np.cos(theta_p), y=np.sin(theta_p),
                        mode='lines', name='C (boule)',
                        line=dict(color='#ffcc00', width=2.5, dash='dash')))
                elif type_c_proj == "Orthant positif (x≥0)":
                    fig_proj.add_vrect(x0=0, x1=3,
                        fillcolor='rgba(0,255,136,0.05)',
                        line_width=0)
                    fig_proj.add_vline(x=0, line_color='#ffcc00', line_dash='dash')
                    fig_proj.add_hline(y=0, line_color='#ffcc00', line_dash='dash')
                elif type_c_proj == "Intervalle [a,b]":
                    fig_proj.add_vrect(x0=-1, x1=1,
                        fillcolor='rgba(0,255,136,0.05)', line_width=0)
                    for xb in [-1, 1]:
                        fig_proj.add_vline(x=xb, line_color='#ffcc00',
                                           line_dash='dash')

                if len(hist_p) > 1:
                    fig_proj.add_trace(go.Scatter(
                        x=hist_p[:,0], y=hist_p[:,1],
                        mode='lines+markers', name='Trajectoire',
                        line=dict(color='#ff00cc', width=2),
                        marker=dict(size=5, color='#ff00cc')))
                if x_opt_proj is not None and len(x_opt_proj) >= 2:
                    fig_proj.add_trace(go.Scatter(
                        x=[x_opt_proj[0]], y=[x_opt_proj[1]],
                        mode='markers', name='Optimum',
                        marker=dict(color='#00ff88', size=14, symbol='star')))
                fig_proj.add_trace(go.Scatter(x=[x0_p1], y=[x0_p2],
                    mode='markers', name='Départ',
                    marker=dict(color='#ffcc00', size=12, symbol='circle')))
                layout(fig_proj, f"Gradient projeté — {type_c_proj}",
                       "x", "y")
                st.plotly_chart(fig_proj, use_container_width=True)

        # ---- SUB 4 : LASSO & RIDGE ----
        with sub4:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔗 LASSO & Ridge — Régression régularisée")
                st.markdown("**Génération des données**")
                n_obs = st.slider("n observations", 20, 500, 100, 10)
                n_feat = st.slider("p variables", 2, 20, 8)
                sparsity = st.slider("Sparsité (% vrais coeff.≠0)", 10, 100, 40)
                sigma_n = st.slider("Bruit σ", 0.0, 2.0, 0.5, 0.05)
                lambda_reg = st.slider("λ (régularisation)", 0.001, 10.0, 1.0, 0.01)
                methode_reg = st.selectbox("Méthode", ["LASSO", "Ridge", "Comparer"])
                np.random.seed(42)

                X_dat = np.random.randn(n_obs, n_feat)
                n_true = max(1, int(n_feat * sparsity/100))
                beta_true = np.zeros(n_feat)
                beta_true[:n_true] = np.random.randn(n_true) * 3
                np.random.shuffle(beta_true)
                y_dat = X_dat @ beta_true + np.random.randn(n_obs)*sigma_n
                X_dat = (X_dat - X_dat.mean(0)) / (X_dat.std(0)+1e-8)
                y_dat = (y_dat - y_dat.mean())

            with col2:
                if methode_reg == "LASSO":
                    res_reg = opt_conv_engine.lasso(X_dat, y_dat, lambda_reg)
                    beta_est = res_reg["beta"]
                    r2_val = res_reg["r2"]
                    n_nz = res_reg["n_non_nuls"]
                    label_m = "LASSO"

                elif methode_reg == "Ridge":
                    res_reg = opt_conv_engine.ridge(X_dat, y_dat, lambda_reg)
                    beta_est = res_reg["beta"]
                    r2_val = res_reg["r2"]
                    n_nz = np.sum(np.abs(beta_est) > 0.01)
                    label_m = "Ridge"

                else:  # Comparer
                    res_l = opt_conv_engine.lasso(X_dat, y_dat, lambda_reg)
                    res_r = opt_conv_engine.ridge(X_dat, y_dat, lambda_reg)
                    fig_comp_r = go.Figure()
                    x_feat = [f"β{i+1}" for i in range(n_feat)]
                    fig_comp_r.add_trace(go.Bar(x=x_feat, y=beta_true,
                        name='Vrais β', marker_color='#ffcc00'))
                    fig_comp_r.add_trace(go.Bar(x=x_feat, y=res_l["beta"],
                        name=f'LASSO (R²={res_l["r2"]:.3f})',
                        marker_color='#00ccff'))
                    fig_comp_r.add_trace(go.Bar(x=x_feat, y=res_r["beta"],
                        name=f'Ridge (R²={res_r["r2"]:.3f})',
                        marker_color='#ff00cc'))
                    fig_comp_r.update_layout(**PLOT_LAYOUT,
                        title="Comparaison LASSO vs Ridge",
                        barmode='group', height=400,
                        xaxis=AXIS, yaxis=AXIS,
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                    st.plotly_chart(fig_comp_r, use_container_width=True)
                    beta_est = res_l["beta"]
                    r2_val = res_l["r2"]
                    n_nz = res_l["n_non_nuls"]
                    label_m = "LASSO"

                if methode_reg != "Comparer":
                    c1,c2,c3 = st.columns(3)
                    with c1: st.metric("R²", f"{r2_val:.4f}")
                    with c2: st.metric("β ≠ 0 détectés", n_nz)
                    with c3: st.metric("β ≠ 0 vrais", n_true)

                    fig_reg = go.Figure()
                    fig_reg.add_trace(go.Bar(
                        x=[f"β{i+1}" for i in range(n_feat)],
                        y=beta_true, name='Vrais β',
                        marker_color='#ffcc00'))
                    fig_reg.add_trace(go.Bar(
                        x=[f"β{i+1}" for i in range(n_feat)],
                        y=beta_est, name=f'{label_m} estimé',
                        marker_color='#00ccff'))
                    fig_reg.update_layout(**PLOT_LAYOUT,
                        title=f"{label_m} — Vrais vs Estimés (λ={lambda_reg})",
                        barmode='group', height=380,
                        xaxis=AXIS, yaxis=dict(**AXIS, title="β"),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                    st.plotly_chart(fig_reg, use_container_width=True)

                # Chemin de régularisation
                st.markdown("#### 📈 Chemin de régularisation")
                lambdas_path = np.logspace(-3, 2, 50)
                methode_path = st.radio("Méthode", ["LASSO","Ridge"],
                                         horizontal=True, key="path_meth")
                df_path = opt_conv_engine.chemin_regularisation(
                    X_dat, y_dat, lambdas_path, methode_path)

                fig_path = go.Figure()
                beta_cols = [c for c in df_path.columns if c.startswith("β")]
                for j, bc in enumerate(beta_cols[:min(8, len(beta_cols))]):
                    fig_path.add_trace(go.Scatter(x=df_path["λ"], y=df_path[bc],
                        mode='lines', name=bc,
                        line=dict(color=colors[j%len(colors)], width=2)))
                fig_path.update_layout(**PLOT_LAYOUT,
                    title=f"Chemin de régularisation {methode_path}",
                    xaxis=dict(**AXIS, type='log', title="λ"),
                    yaxis=dict(**AXIS, title="Valeur β"),
                    height=380, legend=dict(bgcolor='rgba(0,0,0,0.5)'))
                st.plotly_chart(fig_path, use_container_width=True)

        # ---- SUB 5 : THÉORIE OPT. CONVEXE ----
        with sub5:
            st.markdown("### 📖 Théorie — Optimisation Convexe")
            st.markdown("""
            #### Propriété fondamentale
            Tout **minimum local** d'un problème convexe est un
            **minimum global**. La condition ∇f(x*)=0 est
            **nécessaire et suffisante**.

            #### Descente de gradient (convergence)
            Pour f L-lisse et μ-fortement convexe :
            - Pas optimal : α = 1/L
            - Nombre de conditionnement : κ = L/μ
            - Convergence : O((1-1/κ)ᵏ) = convergence géométrique

            #### Méthode de Nesterov (optimale)
            Converge en O(1/k²) pour f convexe, L-lisse.
            C'est la **borne inférieure** théorique.
            """)
            for nom, f_latex in FORMULES_OPT_CONV.items():
                st.markdown(f"**{nom}**")
                st.latex(f_latex)

            st.markdown("---")
            for r in [
                "Boyd & Vandenberghe — *Convex Optimization* (Cambridge, 2004)",
                "Nesterov — *Introductory Lectures on Stochastic Optimization* (2004)",
                "Bubeck — *Convex Optimization: Algorithms and Complexity* (2015)",
            ]:
                st.markdown(f"- {r}")
