__lpn_ms_signature__ = 'Papa Samba Fall - LPN-MS'
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, signal, integrate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBRegressor   # pip install xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import datetime
import joblib
import io
import warnings
warnings.filterwarnings('ignore')


def _limit_points(arr, max_points=500):
    if len(arr) <= max_points:
        return arr
    indices = np.linspace(0, len(arr) - 1, max_points, dtype=int)
    return arr[indices]


def _select_corr_columns(df, candidate_cols, max_cols=12):
    if len(candidate_cols) <= max_cols:
        return candidate_cols
    variances = []
    for c in candidate_cols:
        try:
            v = float(df[c].var())
        except Exception:
            v = -np.inf
        variances.append((c, v if np.isfinite(v) else -np.inf))
    selected = [c for c, _ in sorted(variances, key=lambda x: x[1], reverse=True)[:max_cols]]
    return selected


# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
CONSTANTES_ENERGIE = {
    "Pouvoir calorifique gaz naturel": (10.55,  "kWh/m³"),
    "Pouvoir calorifique fioul":       (10.00,  "kWh/L"),
    "Pouvoir calorifique charbon":     (8.14,   "kWh/kg"),
    "Facteur émission électricité FR": (0.052,  "kgCO₂/kWh"),
    "Facteur émission gaz naturel":    (0.234,  "kgCO₂/kWh"),
    "1 tep (tonne équivalent pétrole)":(11628,  "kWh"),
    "1 kWh":                           (3.6e6,  "J"),
}

FORMULES_ENERGIE = {
    "Puissance électrique":    r"P = U \cdot I = \frac{U^2}{R} = R \cdot I^2 \quad \text{(W)}",
    "Énergie thermique":       r"Q = m \cdot c_p \cdot \Delta T \quad \text{(J)}",
    "Rendement":               r"\eta = \frac{E_{utile}}{E_{totale}} \times 100\%",
    "Loi de Joule":            r"P_{Joule} = R \cdot I^2 \quad \text{(W)}",
    "Énergie cinétique":       r"E_c = \frac{1}{2}mv^2 \quad \text{(J)}",
    "Énergie potentielle":     r"E_p = mgh \quad \text{(J)}",
    "Puissance mécanique":     r"P = F \cdot v = \tau \cdot \omega \quad \text{(W)}",
    "Bilan énergétique":       r"E_{in} = E_{utile} + E_{pertes}",
    "Facteur de puissance":    r"\cos\phi = \frac{P}{S} = \frac{P}{\sqrt{P^2+Q^2}}",
    "Coefficient performance": r"\text{COP} = \frac{E_{utile}}{W_{electrique}}",
    "Degré-jour":              r"DJ = \sum_{j} \max(T_{ref} - T_j, 0)",
    "Intensité énergétique":   r"IE = \frac{\text{Consommation (kWh)}}{\text{Surface (m}^2)}",
}

INDICATEURS_BATIMENT = {
    "Classe A": (0,    50),
    "Classe B": (51,   90),
    "Classe C": (91,  150),
    "Classe D": (151, 230),
    "Classe E": (231, 330),
    "Classe F": (331, 450),
    "Classe G": (451, 9999),
}


# ============================================================
# MOTEUR ÉNERGIE
# ============================================================
class EnergyEngine:
    """Moteur d'analyse énergétique scientifique."""

    def __init__(self, df: pd.DataFrame):
        # Work on a copy to avoid mutating the caller's dataframe
        self.df = df.copy()

        # Robustly coerce columns that look numeric but were read as object/string
        for col in list(self.df.columns):
            if self.df[col].dtype == object:
                s = self.df[col].astype(str).str.strip()
                # Remove common thousands separators (space, non-breaking space, apostrophe)
                s = s.str.replace("[ \u00A0']", "", regex=True)
                # Remove percent sign and remember to keep numeric (50% -> 50)
                s = s.str.replace('%', '', regex=False)
                # Replace decimal comma with dot
                s = s.str.replace(',', '.', regex=False)
                # Convert to numeric, coerce errors to NaN
                num = pd.to_numeric(s, errors='coerce')
                # If a majority of values converted successfully, adopt the numeric column
                if num.notna().sum() / max(1, len(num)) > 0.6:
                    self.df[col] = num

        # Store numeric columns (after coercion)
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)

    # ----------------------------------------------------------
    # Méthodes existantes
    # ----------------------------------------------------------
    def profil_energetique(self, col: str) -> dict:
        """Analyse statistique complète d'une série énergétique."""
        s = self.df[col].dropna()
        return {
            "Moyenne": s.mean(),
            "Médiane": s.median(),
            "Std": s.std(),
            "Min": s.min(),
            "Max": s.max(),
            "Somme": s.sum(),
            "CV (%)": s.std() / s.mean() * 100 if s.mean() != 0 else 0,
            "Skewness": float(stats.skew(s)),
            "Kurtosis": float(stats.kurtosis(s)),
            "P10": s.quantile(0.10),
            "P90": s.quantile(0.90),
            "Outliers IQR": int(((s < s.quantile(0.25)-1.5*(s.quantile(0.75)-s.quantile(0.25))) |
                                 (s > s.quantile(0.75)+1.5*(s.quantile(0.75)-s.quantile(0.25)))).sum()),
        }

    def decomposition_tendance(self, col: str) -> dict:
        """Décomposition Tendance + Saisonnalité (Fourier) + Résidu."""
        s = self.df[col].dropna().values
        N = len(s)
        t = np.arange(N)

        p = np.polyfit(t, s, 2)
        tendance = np.polyval(p, t)
        residu = s - tendance

        fft_vals = np.fft.rfft(residu)
        freqs = np.fft.rfftfreq(N)
        magnitude = np.abs(fft_vals)

        idx_dom = np.argmax(magnitude[1:]) + 1
        f_dom = freqs[idx_dom]

        return {
            "t": t, "original": s, "tendance": tendance,
            "residu": residu, "freqs": freqs, "magnitude": magnitude,
            "f_dom": f_dom, "periode_dom": 1/f_dom if f_dom > 0 else np.inf,
        }

    def bilan_energetique(self, E_in: float, eta: float) -> dict:
        """Calcul du bilan énergétique simple."""
        E_utile = E_in * eta / 100
        E_pertes = E_in - E_utile
        return {
            "E_in (kWh)": E_in,
            "E_utile (kWh)": E_utile,
            "E_pertes (kWh)": E_pertes,
            "Rendement (%)": eta,
            "CO₂ évité (kgCO₂)": E_utile * 0.052,
        }

    def intensite_energetique(self, col: str, surface: float) -> pd.Series:
        return self.df[col] / surface

    def modele_predictif(self, target: str, features: list,
                          modele: str = "Ridge", test_size: float = 0.2) -> dict:
        """Modèle prédictif de consommation énergétique."""
        X = self.df[features].dropna()
        y = self.df.loc[X.index, target]
        if len(X) < 10:
            return {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        models = {
            "Régression Linéaire": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        if modele == "XGBoost":
            if XGBOOST_AVAILABLE:
                model = XGBRegressor(n_estimators=100, learning_rate=0.1,
                                     random_state=42, verbosity=0)
            else:
                model = Ridge(alpha=1.0)
        else:
            model = models.get(modele, Ridge())
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        residus = y_test.values - y_pred

        importance = None
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        return {
            "model": model, "scaler": scaler,
            "y_test": y_test, "y_pred": y_pred,
            "r2": r2, "rmse": rmse, "mae": mae, "residus": residus,
            "importance": importance, "features": features
        }

    def classe_dpe(self, consommation_kwh_m2: float) -> str:
        """Classe DPE selon consommation en kWh/m²/an."""
        for classe, (mini, maxi) in INDICATEURS_BATIMENT.items():
            if mini <= consommation_kwh_m2 <= maxi:
                return classe
        return "G"

    def detecter_pointes(self, col: str, seuil_sigma: float = 2.0) -> np.ndarray:
        """Détecte les pics de consommation (pointes)."""
        s = pd.to_numeric(self.df[col], errors='coerce').dropna()
        if len(s) == 0:
            return np.array([], dtype=bool)
        z = np.abs(stats.zscore(s))
        return z > seuil_sigma

    # ----------------------------------------------------------
    # MÉTHODES MSD — Prévision Centrale Énergétique
    # ----------------------------------------------------------
    @staticmethod
    def clean_time(x) -> float:
        """Convertit une valeur temporelle (time, str HH:MM:SS, int) en minutes depuis minuit."""
        if isinstance(x, datetime.time):
            return x.hour * 60 + x.minute + x.second / 60
        elif isinstance(x, str):
            try:
                h, m, s = map(float, x.split(':'))
                return h * 60 + m + s / 60
            except Exception:
                return 0.0
        return float(x) if isinstance(x, (int, float, np.integer, np.floating)) else 0.0

    def preparer_donnees_msd(self, col_features: list, col_cible: str,
                              col_heure: str = None, test_size: float = 0.2) -> dict:
        """
        Nettoyage complet et préparation des données pour le pipeline MSD :
          - Suppression des valeurs aberrantes (Z-score)
          - Conversion de la colonne Heure si présente
          - Calcul automatique de l'Energy_Gap si Ch 6 et Ch 7 sont présents
          - Split train/test
        Retourne un dict avec X_train, X_test, y_train, y_test et le df nettoyé.
        """
        D = self.df.copy()

        # Conversion colonne Heure
        if col_heure and col_heure in D.columns:
            D[col_heure] = D[col_heure].apply(self.clean_time)

        # Conversion numérique de toutes les colonnes sauf Date/Heure texte
        for col in D.columns:
            if col not in [col_heure, 'Date']:
                D[col] = pd.to_numeric(D[col], errors='coerce')

        # Suppression valeurs aberrantes (Z-score > 3)
        num_cols = D.select_dtypes(include=[float, int]).columns
        if len(num_cols) > 0:
            z = np.abs(stats.zscore(D[num_cols].fillna(0)))
            D = D[(z < 3).all(axis=1)]

        D = D.dropna(subset=col_features + [col_cible])

        # Nettoyage final X
        X = D[col_features].apply(pd.to_numeric, errors='coerce')
        # fill with column means where possible, else with 0
        X = X.fillna(X.mean()).fillna(0)
        y = pd.to_numeric(D[col_cible], errors='coerce').dropna()

        # Align X and y indices (in case coercion removed some y values)
        X = X.loc[y.index]

        n_samples = len(X)
        # If no samples, return safe empty structures to avoid downstream crashes
        if n_samples == 0:
            empty_X = X.iloc[0:0]
            empty_y = y.iloc[0:0]
            return {
                "df_clean": D,
                "X": X, "y": y,
                "X_train": empty_X, "X_test": empty_X,
                "y_train": empty_y, "y_test": empty_y,
            }

        # Compute integer test size to avoid train_test_split ValueError when
        # floating fraction yields empty splits for very small datasets.
        n_test = int(round(n_samples * test_size))
        # Ensure at least one test sample and at least one train sample when possible
        n_test = min(max(1, n_test), max(1, n_samples - 1)) if n_samples > 1 else 0

        if n_samples == 1:
            # Only one sample — put it into train set and leave test empty
            X_train = X
            X_test = X.iloc[0:0]
            y_train = y
            y_test = y.iloc[0:0]
        else:
            # Use integer `test_size` to guarantee non-empty splits
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=n_test, random_state=42
            )

        return {
            "df_clean": D,
            "X": X, "y": y,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }

    def entrainer_modeles_msd(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Entraîne les 4 modèles MSD (LinearRegression, DecisionTree,
        RandomForest, XGBoost) et retourne métriques + prédictions.
        """
        modeles = {
            "Régression Linéaire": LinearRegression(),
            "Decision Tree":       DecisionTreeRegressor(random_state=42),
            "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
        }
        if XGBOOST_AVAILABLE:
            modeles["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1,
                                               random_state=42, verbosity=0)

        resultats = {}
        for nom, modele in modeles.items():
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_test)
            resultats[nom] = {
                "model":  modele,
                "y_pred": y_pred,
                "r2":     r2_score(y_test, y_pred),
                "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae":    mean_absolute_error(y_test, y_pred),
            }

        # Meilleur modèle selon R²
        meilleur = max(resultats, key=lambda k: resultats[k]["r2"])
        resultats["__meilleur__"] = meilleur

        return resultats


# ============================================================
# ANALYSE AUTOMATIQUE AVANCÉE
# ============================================================

def _label_qualite_r2(r2: float) -> str:
    if r2 >= 0.95: return "excellente (quasi-parfaite)"
    if r2 >= 0.85: return "très bonne"
    if r2 >= 0.70: return "bonne"
    if r2 >= 0.50: return "moyenne"
    return "faible — à améliorer"


def _interprete_correlation(val: float) -> str:
    a = abs(val)
    signe = "positive" if val > 0 else "négative"
    if a >= 0.9:  return f"corrélation {signe} très forte ({val:.2f})"
    if a >= 0.7:  return f"corrélation {signe} forte ({val:.2f})"
    if a >= 0.5:  return f"corrélation {signe} modérée ({val:.2f})"
    if a >= 0.3:  return f"corrélation {signe} faible ({val:.2f})"
    return f"corrélation quasi-nulle ({val:.2f})"


def _interprete_skewness(skew: float) -> str:
    if abs(skew) < 0.5:
        return "distribution quasi symétrique"
    if skew > 0:
        return "distribution asymétrique à droite (valeurs extrêmes élevées)"
    return "distribution asymétrique à gauche (valeurs extrêmes basses)"


def _interprete_kurtosis(kurt: float) -> str:
    if abs(kurt) < 1:
        return "queue de distribution proche de la normale"
    if kurt > 1:
        return "queue épaisse — risque d'outliers fréquents"
    return "queue légère — données concentrées autour de la moyenne"


def generer_analyse_msd(info: dict) -> str:
    """Génère une analyse textuelle avancée et automatique de la section MSD."""
    lignes = []

    # ── Données ──────────────────────────────────────────────
    n_raw   = info.get("n_raw", 0)
    n_clean = info.get("n_clean", 0)
    n_train = info.get("n_train", 0)
    n_test  = info.get("n_test", 0)
    pct_suppr = round((1 - n_clean / n_raw) * 100, 1) if n_raw > 0 else 0

    lignes.append("## 📊 Analyse automatique — Pipeline MSD")
    lignes.append(f"### 1. Qualité des données")
    lignes.append(
        f"Le jeu de données brut contient **{n_raw} observations**. Après suppression des valeurs "
        f"aberrantes (Z-score > 3), **{n_clean} lignes** sont conservées, soit une élimination de "
        f"**{pct_suppr}%** des données. "
        + ("Ce taux est acceptable et n'affecte pas la représentativité du dataset." if pct_suppr < 10
           else "Ce taux élevé suggère une forte présence d'outliers dans les données brutes — "
                "une vérification du capteur ou de la source est recommandée." if pct_suppr < 25
           else "⚠️ Plus d'un quart des données ont été supprimées — la qualité du signal brut est à investiguer.")
    )
    lignes.append(
        f"Le split train/test retenu donne **{n_train} échantillons d'entraînement** "
        f"et **{n_test} échantillons de test** (ratio test ≈ {round(n_test/(n_train+n_test)*100) if (n_train+n_test)>0 else 0}%)."
    )

    # ── Energy Gap ───────────────────────────────────────────
    if info.get("has_gap"):
        eg_min  = info.get("eg_min",  0)
        eg_max  = info.get("eg_max",  0)
        eg_mean = info.get("eg_mean", 0)
        eg_std  = info.get("eg_std",  0)
        lignes.append("### 2. Energy Gap (Ch 7 − Ch 6)")
        lignes.append(
            f"L'**Energy Gap** mesuré varie entre **{eg_min:.2f}** et **{eg_max:.2f}** "
            f"avec une moyenne de **{eg_mean:.2f}** et un écart-type de **{eg_std:.2f}**. "
        )
        if eg_mean > 0:
            lignes.append(
                "La moyenne positive indique que Ch 7 fournit globalement plus d'énergie que Ch 6 — "
                "ce différentiel peut représenter un surplus de production ou une perte mesurable sur le réseau."
            )
        elif eg_mean < 0:
            lignes.append(
                "La moyenne négative indique une consommation nette supérieure à la production sur Ch 7, "
                "signalant potentiellement un déséquilibre dans le bilan central."
            )
        else:
            lignes.append("La moyenne proche de zéro indique un équilibre entre les deux canaux sur la période.")
        cv_gap = abs(eg_std / eg_mean * 100) if eg_mean != 0 else 0
        lignes.append(
            f"Le coefficient de variation de l'Energy Gap est de **{cv_gap:.1f}%**, "
            + ("ce qui traduit une grande stabilité du différentiel." if cv_gap < 20
               else "ce qui indique une variabilité significative à surveiller." if cv_gap < 50
               else "ce qui révèle une forte instabilité — des pics ponctuels de production/consommation sont présents.")
        )

    # ── Corrélations ─────────────────────────────────────────
    corr_cible = info.get("corr_cible", {})
    if corr_cible:
        lignes.append("### 3. Analyse des corrélations")
        top_corr = sorted(corr_cible.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        lignes.append(
            f"Les trois variables les plus corrélées à **{info.get('col_cible', 'la cible')}** sont : "
            + ", ".join([f"**{k}** ({_interprete_correlation(v)})" for k, v in top_corr]) + "."
        )
        if top_corr and abs(top_corr[0][1]) > 0.85:
            lignes.append(
                f"La corrélation très forte avec **{top_corr[0][0]}** suggère une relation quasi-linéaire — "
                "un modèle de régression linéaire devrait suffire pour cette variable."
            )
        elif top_corr and abs(top_corr[0][1]) < 0.4:
            lignes.append(
                "Les faibles corrélations linéaires observées suggèrent que des relations non-linéaires "
                "dominent — les modèles ensemblistes (Random Forest, XGBoost) seront plus adaptés."
            )

    # ── Modèles ───────────────────────────────────────────────
    metriques = info.get("metriques", {})
    best_nom  = info.get("best_model", "")
    if metriques and best_nom:
        lignes.append("### 4. Performance des modèles")
        for nom, m in metriques.items():
            badge = "🏆 " if nom == best_nom else ""
            lignes.append(
                f"- **{badge}{nom}** : R²={m['r2']:.4f} | RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} "
                f"→ qualité {_label_qualite_r2(m['r2'])}"
            )

        best_m = metriques[best_nom]
        lignes.append(
            f"\nLe modèle **{best_nom}** est sélectionné comme meilleur avec un **R² = {best_m['r2']:.4f}**, "
            f"un RMSE de **{best_m['rmse']:.4f}** et un MAE de **{best_m['mae']:.4f}**."
        )

        # Analyse comparative
        r2_vals = [m['r2'] for m in metriques.values()]
        r2_spread = max(r2_vals) - min(r2_vals)
        if r2_spread < 0.05:
            lignes.append(
                "L'écart entre les modèles est très faible (< 5 points de R²) — "
                "le problème est probablement bien structuré et linéairement séparable."
            )
        elif r2_spread > 0.20:
            lignes.append(
                f"L'écart important entre les modèles ({r2_spread:.2f} en R²) confirme que "
                "les relations non-linéaires sont déterminantes — les modèles ensemblistes surpassent clairement la régression."
            )

        # Analyse RMSE
        rmse_best = best_m['rmse']
        y_std = info.get("y_std", 1)
        ratio_rmse = rmse_best / y_std if y_std > 0 else 0
        lignes.append(
            f"\nLe RMSE du meilleur modèle représente **{ratio_rmse*100:.1f}%** de l'écart-type de la variable cible "
            + ("— une erreur très faible, le modèle est fiable." if ratio_rmse < 0.15
               else "— une erreur raisonnable pour ce type de données." if ratio_rmse < 0.30
               else "— l'erreur résiduelle est notable ; un enrichissement des features ou un tuning hyperparamétrique est conseillé.")
        )

    # ── Importance des variables ──────────────────────────────
    feat_imp = info.get("feat_importance", {})
    if feat_imp:
        lignes.append("### 5. Variables les plus influentes")
        top_feat = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:3]
        lignes.append(
            "Les variables les plus déterminantes pour la prédiction sont : "
            + ", ".join([f"**{k}** ({v*100:.1f}%)" for k, v in top_feat]) + "."
        )
        lignes.append(
            "Cette hiérarchie d'importance peut guider la réduction de dimensionnalité "
            "ou orienter les mesures terrain prioritaires."
        )

    return "\n\n".join(lignes)


def generer_conclusion_msd(info: dict) -> str:
    """Génère une conclusion scientifique avancée du pipeline MSD."""
    best_nom = info.get("best_model", "inconnu")
    best_m   = info.get("metriques", {}).get(best_nom, {})
    r2       = best_m.get("r2", 0)
    n_clean  = info.get("n_clean", 0)
    has_gap  = info.get("has_gap", False)
    y_std    = info.get("y_std", 1)
    rmse     = best_m.get("rmse", 0)

    qualite = _label_qualite_r2(r2)
    lignes  = ["## ✅ Conclusion scientifique — Prévision Centrale MSD"]

    lignes.append(
        f"Le pipeline MSD a été exécuté sur **{n_clean} observations** nettoyées. "
        f"Le modèle retenu est **{best_nom}** avec une performance **{qualite}** (R² = {r2:.4f})."
    )

    if has_gap:
        lignes.append(
            "Le calcul de l'**Energy Gap** entre les canaux Ch 6 et Ch 7 constitue un indicateur clé "
            "de l'équilibre énergétique de la centrale. Sa variabilité temporelle doit être surveillée "
            "en continu pour détecter des dérives de rendement."
        )

    if r2 >= 0.85:
        lignes.append(
            f"Avec un R² de **{r2:.4f}**, le modèle capture l'essentiel de la variance de la variable cible. "
            "Il peut être déployé en production pour générer des prédictions fiables et servir de base "
            "à un système d'alerte précoce ou d'optimisation de la production."
        )
    elif r2 >= 0.60:
        lignes.append(
            f"Le modèle explique **{r2*100:.1f}%** de la variance — une performance correcte mais perfectible. "
            "Des pistes d'amélioration : ajout de variables exogènes (météo, charge réseau), "
            "ingénierie de features (fenêtres temporelles glissantes, lags), ou tuning hyperparamétrique (GridSearch/Optuna)."
        )
    else:
        lignes.append(
            f"La performance modeste (R² = {r2:.4f}) indique que le modèle peine à capturer "
            "la dynamique du système. Cela peut résulter d'un manque de features explicatives, "
            "d'un signal très bruité ou d'une non-stationnarité des données."
        )

    ratio = rmse / y_std if y_std > 0 else 0
    lignes.append(
        f"L'erreur résiduelle (RMSE = {rmse:.4f}, soit **{ratio*100:.1f}%** de σ_cible) "
        "est à interpréter dans le contexte opérationnel : "
        + ("cette précision est suffisante pour piloter les décisions énergétiques en temps réel." if ratio < 0.20
           else "une marge d'erreur à prendre en compte dans le dimensionnement des alertes." if ratio < 0.40
           else "des actions correctives sur la qualité des données ou l'architecture du modèle sont prioritaires.")
    )

    lignes.append(
        "**Recommandations opérationnelles :**\n"
        f"1. Surveiller l'Energy Gap en continu et définir des seuils d'alerte.\n"
        f"2. Réentraîner le modèle **{best_nom}** périodiquement avec les nouvelles données.\n"
        "3. Intégrer des variables temporelles (heure, jour de semaine) pour capturer la saisonnalité.\n"
        "4. Automatiser le pipeline de nettoyage Z-score pour garantir la robustesse en production."
    )

    return "\n\n".join(lignes)


def generer_analyse_automatique(section: str, info: dict) -> str:
    if section == "🏭 Prévision Centrale MSD":
        return generer_analyse_msd(info)
    lignes = [
        f"**Section :** {section}",
        f"**Données :** {info.get('n_rows', 0)} lignes, {info.get('n_numeric', 0)} variables numériques.",
    ]
    if section == "📋 Profil":
        col  = info.get('col_prof', 'N/A')
        mean = info.get('mean', 0)
        std  = info.get('std', 0)
        out  = info.get('outliers', 0)
        skew = info.get('skew', 0)
        kurt = info.get('kurtosis', 0)
        cv   = abs(std / mean * 100) if mean != 0 else 0
        quartile_gap = info.get('q90', 0) - info.get('q10', 0)
        lignes.append(
            f"La variable **{col}** présente une moyenne de **{mean:.2f}**, un écart-type de **{std:.2f}** "
            f"(CV = {cv:.1f}%) et un écart inter-quantile P10-P90 de **{quartile_gap:.2f}**."
        )
        lignes.append(
            f"L'analyse de la distribution révèle une {_interprete_skewness(skew)} "
            f"et {_interprete_kurtosis(kurt)}."
        )
        out_msg = "vérifiez les mesures et capteurs" if out > 0 else "pas d'anomalies majeures identifiées"
        lignes.append(f"{out} valeurs aberrantes IQR détectées — {out_msg}.")
        if cv > 50:
            lignes.append("La forte dispersion relative (CV > 50%) indique une variabilité marquée, possiblement liée à des cycles de charge/décharge ou des événements ponctuels.")
        elif cv < 10:
            lignes.append("La consommation est remarquablement stable, ce qui facilite la planification énergétique.")
    elif section == "📊 Visualisation":
        graph_type = info.get('graph_type', 'N/A')
        lignes.append(f"Graphique sélectionné : **{graph_type}**.")
        if graph_type == "Scatter + régression" and info.get('corr_coef') is not None:
            lignes.append(
                f"La relation entre **{info.get('x_col', '?')}** et **{info.get('y_col', '?')}** est {_interprete_correlation(info['corr_coef'])}."
            )
            lignes.append(
                f"La pente du régressif est {info.get('slope', 0):.3f}, suggérant {'une augmentation' if info.get('slope', 0) > 0 else 'une diminution'} de la variable Y lorsque X évolue."
            )
        elif graph_type == "Corrélation matricielle" and info.get('top_corr'):
            top_corr = info['top_corr']
            lignes.append(
                f"Les variables les plus corrélées sont : {', '.join([f'**{k}** ({_interprete_correlation(v)})' for k, v in top_corr])}."
            )
    elif section == "📈 Décomposition":
        p = info.get('periode_dom', np.inf)
        resid_ratio = info.get('residu_ratio', None)
        lignes.append(
            f"Période dominante détectée : **{p:.1f} points**." if p < np.inf
            else "Aucune période dominante claire — le signal est soit apériodique soit fortement bruité."
        )
        if resid_ratio is not None:
            lignes.append(
                f"Le composant résiduel représente **{resid_ratio:.1f}%** de la variance totale, ce qui indique {'un signal relativement stable' if resid_ratio < 40 else 'un niveau de bruit important'}."
            )
    elif section == "🤖 Prédiction":
        if info.get('model_name'):
            r2 = info.get('r2', 0)
            lignes.append(
                f"Modèle **{info['model_name']}** avec {info.get('n_features', 0)} variables explicatives : R² = {r2:.3f} — qualité **{_label_qualite_r2(r2)}**."
            )
            lignes.append(
                f"RMSE = {info.get('rmse', 0):.3f}, MAE = {info.get('mae', 0):.3f}."
            )
            if r2 < 0.5:
                lignes.append("Une amélioration des features ou un modèle plus flexible est conseillé.")
    elif section == "⚡ Bilan & DPE":
        c = info.get('dpe_class', 'N/A')
        consommation = info.get('conso_m2', 0)
        lignes.append(
            f"Consommation estimée : **{consommation:.1f} kWh/m²/an** → DPE **{c}**."
        )
        if info.get('dpe_range'):
            low, high = info['dpe_range']
            lignes.append(
                f"Cette consommation se situe {'dans la partie basse' if consommation < (low+high)/2 else 'dans la partie haute'} de l'intervalle {low:.0f}-{high:.0f}."
            )
    elif section == "🔍 Détection pointes":
        lignes.append(
            f"**{info.get('n_pointes', 0)} pointes** détectées représentant **{info.get('taux_temps', 0):.1f}%** du temps et **{info.get('energie_pointe_pct', 0):.1f}%** de l'énergie totale."
        )
        if info.get('pointes_moyennes') is not None:
            lignes.append(
                f"La valeur moyenne des pointes est **{info['pointes_moyennes']:.2f}**, avec un maximum mesuré à **{info['pointes_max']:.2f}**."
            )
    return "\n\n".join(lignes)


def generer_conclusion_automatique(section: str, info: dict) -> str:
    if section == "🏭 Prévision Centrale MSD":
        return generer_conclusion_msd(info)
    if section == "📋 Profil":
        cv = abs(info.get('std', 0) / info.get('mean', 1) * 100) if info.get('mean', 0) != 0 else 0
        skew = info.get('skew', 0)
        conclusion = (
            f"Le profil est {'stable' if cv < 20 else 'variable'} (CV = {cv:.1f}%). "
            f"La distribution est {_interprete_skewness(skew)}. "
        )
        return conclusion + "Poursuivre par une analyse de corrélation ou de modélisation selon les objectifs."
    if section == "📊 Visualisation":
        if info.get('graph_type') == "Scatter + régression" and info.get('corr_coef') is not None:
            return (
                f"La relation linéaire estimée (R² ≈ {info.get('corr_coef', 0)**2:.3f}) permet de prioriser les variables explicatives fiables."
            )
        if info.get('graph_type') == "Corrélation matricielle" and info.get('top_corr'):
            return (
                "Les corrélations fortes identifiées indiquent les variables à surveiller ou à intégrer dans un modèle prédictif."
            )
        return "Les visualisations facilitent la détection des tendances et des structures qui orientent les actions énergétiques."
    if section == "📈 Décomposition":
        p = info.get('periode_dom', np.inf)
        if p < np.inf:
            return (
                f"Une périodicité de {p:.1f} points a été détectée — utile pour la planification et la détection de cycles."
            )
        return "Aucune périodicité marquée détectée, ce qui invite à surveiller la variabilité stationnaire ou les anomalies temporelles."
    if section == "🤖 Prédiction":
        r2 = info.get('r2', 0)
        if r2 > 0.85:
            return "Le modèle présente une qualité robuste et peut être considéré pour un premier déploiement supervisé."
        if r2 > 0.6:
            return "Le modèle est prometteur, mais un tuning ou des variables supplémentaires amélioreront la fiabilité."
        return "Le modèle nécessite un enrichissement des données et une meilleure sélection des features avant usage opérationnel."
    if section == "⚡ Bilan & DPE":
        c = info.get('dpe_class', 'N/A')
        if c in ["A", "B", "C"]:
            return "Le bâtiment est dans une classe performante ; poursuivre les bonnes pratiques d'efficacité énergétique."
        return "Le DPE indique un potentiel d'amélioration important — prioriser l'isolation, le chauffage et la régulation."
    if section == "🔍 Détection pointes":
        return (
            f"La détection automatique des pointes identifie les périodes critiques ({info.get('n_pointes', 0)} occurrences). "
            "Ces événements doivent être analysés pour réduire les coûts et les surcharges."
        )
    return "Analyse automatique complète sur les données importées."


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def energy_page():
    summary_context = {}
    st.markdown("## ⚡ Analyse Énergétique Avancée")
    st.markdown("*Profil énergétique, décomposition, prédiction, bilan, DPE, prévision centrale MSD*")
    st.markdown("---")

    uploaded = st.file_uploader("📁 Importer CSV ou Excel", type=["csv", "xlsx"])

    if uploaded is None:
        st.info("👆 Importez vos données ou utilisez le mode démo ci-dessous.")
        if st.button("🎲 Générer données démo"):
            np.random.seed(42)
            n = 365
            t_demo = np.arange(n)
            consommation = (
                200 + 100 * np.cos(2*np.pi*t_demo/365) +
                np.random.normal(0, 20, n) +
                0.1 * t_demo
            )
            temperature = 15 - 10 * np.cos(2*np.pi*t_demo/365) + np.random.normal(0, 3, n)
            df_demo = pd.DataFrame({
                "Consommation_kWh": np.clip(consommation, 50, 500),
                "Temperature_C":    temperature,
                "Jour":             t_demo,
                "Irradiation_Wh_m2": np.clip(
                    300 + 200*np.sin(2*np.pi*t_demo/365) + np.random.normal(0, 50, n), 0, 800)
            })
            st.session_state["df_energy"] = df_demo
            st.success("✅ Données démo générées (365 jours)")
            st.dataframe(df_demo, use_container_width=True)
        return

    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
             else pd.read_excel(uploaded)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        st.success(f"✅ {df.shape[0]} lignes × {df.shape[1]} colonnes")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        return

    engine = EnergyEngine(df)
    numeric_cols = engine.numeric_cols

    if not numeric_cols:
        st.error("❌ Aucune colonne numérique.")
        return

    section = st.selectbox(
        "Section",
        [
            "📋 Profil",
            "📊 Visualisation",
            "📈 Décomposition",
            "🤖 Prédiction",
            "⚡ Bilan & DPE",
            "🔍 Détection pointes",
            "🏭 Prévision Centrale MSD",
            "📖 Théorie",
        ],
        key="section_energy"
    )


    # ============================================================
    # TAB 1 : PROFIL
    # ============================================================
    if section == "📋 Profil":
        st.markdown("### 📋 Profil énergétique complet")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Lignes", df.shape[0])
        with c2: st.metric("Colonnes", df.shape[1])
        with c3: st.metric("Valeurs nulles", int(df.isnull().sum().sum()))
        with c4: st.metric("Variables num.", len(numeric_cols))

        col_prof = st.selectbox("Variable à analyser", numeric_cols)
        profil = engine.profil_energetique(col_prof)

        cols_p = st.columns(4)
        items = list(profil.items())
        for i, (k, v) in enumerate(items):
            with cols_p[i % 4]:
                st.metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))

        # Force numeric series for statistics and plotting
        s_p = pd.to_numeric(df[col_prof], errors='coerce').dropna()
        s_p_vals = s_p.values if len(s_p) > 0 else np.array([])
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(
            x=s_p, nbinsx=40, name='Distribution',
            marker=dict(color='rgba(119,0,255,0.6)',
                       line=dict(color='rgba(0,204,255,0.8)', width=0.5))
        ))
        # Compute KDE only when there are enough samples
        if len(s_p_vals) >= 2 and np.isfinite(s_p_vals).all():
            try:
                kde = stats.gaussian_kde(s_p_vals)
                x_kde = np.linspace(s_p_vals.min(), s_p_vals.max(), 300)
                # scale to histogram approximate height (nbins 40)
                scale = len(s_p_vals) * (s_p_vals.max() - s_p_vals.min()) / 40 if s_p_vals.max() != s_p_vals.min() else len(s_p_vals)
                fig_d.add_trace(go.Scatter(
                    x=x_kde, y=kde(x_kde) * scale, mode='lines', name='KDE',
                    line=dict(color='#00ccff', width=3)
                ))
            except Exception:
                # silently skip KDE if scipy fails for numeric reasons
                pass
        if len(s_p_vals) > 0:
            fig_d.add_vline(x=float(s_p.mean()), line_color='#ffcc00', line_dash='dash',
                            annotation_text=f"Moy={float(s_p.mean()):.1f}")
            fig_d.add_vline(x=float(s_p.median()), line_color='#00ff88', line_dash='dot',
                            annotation_text=f"Méd={float(s_p.median()):.1f}")
        fig_d.update_layout(
            title=f"Distribution — {col_prof}",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
            font=dict(color='#c0d0ff'),
            xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
            yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
        )
        st.plotly_chart(fig_d, use_container_width=True)

        summary_context.update({
            "n_rows": df.shape[0],
            "n_numeric": len(numeric_cols),
            "col_prof": col_prof,
            "mean": float(profil['Moyenne']),
            "std": float(profil['Std']),
            "skew": float(profil['Skewness']),
            "kurtosis": float(profil['Kurtosis']),
            "q10": float(profil['P10']),
            "q90": float(profil['P90']),
            "outliers": int(profil['Outliers IQR']),
        })


    # ============================================================
    # TAB 2 : VISUALISATION
    # ============================================================
    elif section == "📊 Visualisation":
        st.markdown("### 📊 Visualisation énergétique")

        col1, col2 = st.columns([1, 2])
        with col1:
            graph_type = st.selectbox("Type", [
                "Série temporelle", "Scatter + régression",
                "Boxplot comparatif", "Corrélation matricielle",
                "Violin plot"
            ])
            sel_col = st.selectbox("Variable principale", numeric_cols, key="viz_col")

        with col2:
            corr_coef = None
            slope = None
            top_corr = None
            x_col = None
            y_col = None
            if graph_type == "Série temporelle":
                fig = go.Figure()
                series = pd.to_numeric(df[sel_col], errors='coerce')
                fig.add_trace(go.Scatter(
                    x=list(range(len(series))), y=series,
                    mode='lines', name=sel_col,
                    line=dict(color='#00ccff', width=2)
                ))
                ma = series.rolling(7, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=list(range(len(series))), y=ma, mode='lines',
                    name='Moyenne mobile (7)',
                    line=dict(color='#ffcc00', width=2.5, dash='dash')
                ))

            elif graph_type == "Scatter + régression":
                x_col = st.selectbox("X", numeric_cols, key="sc_x")
                y_col = st.selectbox("Y", numeric_cols, key="sc_y",
                                      index=min(1, len(numeric_cols)-1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[y_col], mode='markers',
                    marker=dict(color='#7700ff', size=6, opacity=0.7)
                ))
                mask = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
                if len(mask) > 2:
                    slope, intercept, r, *_ = stats.linregress(mask[x_col], mask[y_col])
                    corr_coef = r
                    x_r = np.linspace(mask[x_col].min(), mask[x_col].max(), 200)
                    fig.add_trace(go.Scatter(
                        x=x_r, y=slope*x_r+intercept, mode='lines',
                        name=f'y={slope:.3f}x+{intercept:.3f} (R²={r**2:.3f})',
                        line=dict(color='#ffcc00', width=2.5)
                    ))

            elif graph_type == "Boxplot comparatif":
                fig = go.Figure()
                for col in numeric_cols[:5]:
                    s = pd.to_numeric(df[col], errors='coerce').dropna()
                    if s.size == 0:
                        continue
                    s_norm = (s - s.mean()) / s.std()
                    fig.add_trace(go.Box(y=s_norm, name=col, boxmean='sd'))

            elif graph_type == "Corrélation matricielle":
                numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                numeric_df = numeric_df.loc[:, numeric_df.notna().any()]
                if numeric_df.shape[1] < 2:
                    st.warning("Pas assez de colonnes numériques valides pour générer une matrice de corrélation.")
                    fig = go.Figure()
                else:
                    corr = numeric_df.corr()
                    fig = go.Figure(go.Heatmap(
                        z=corr.values, x=corr.columns, y=corr.columns,
                        colorscale=[[0,'#020817'],[0.5,'#7700ff'],[1,'#00ccff']],
                        zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
                        colorbar=dict(tickfont=dict(color='#c0d0ff'))
                    ))
                    flat = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    pairs = flat.stack().sort_values(key=lambda x: x.abs(), ascending=False)
                    top_corr = [(f"{idx[0]} vs {idx[1]}", float(val)) for idx, val in pairs.head(3).items()]

            else:  # Violin
                fig = go.Figure()
                for col in numeric_cols[:4]:
                    s = pd.to_numeric(df[col], errors='coerce').dropna()
                    if s.size == 0:
                        continue
                    fig.add_trace(go.Violin(y=s, name=col,
                                           box_visible=True, meanline_visible=True))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            summary_context.update({
                "n_rows": df.shape[0],
                "n_numeric": len(numeric_cols),
                "section": section,
                "graph_type": graph_type,
                "x_col": x_col if graph_type == 'Scatter + régression' else None,
                "y_col": y_col if graph_type == 'Scatter + régression' else None,
                "corr_coef": float(corr_coef) if corr_coef is not None else None,
                "slope": float(slope) if slope is not None else None,
                "top_corr": top_corr,
            })


    # ============================================================
    # TAB 3 : DÉCOMPOSITION
    # ============================================================
    elif section == "📈 Décomposition":
        st.markdown("### 📈 Décomposition de série temporelle")
        col_dec = st.selectbox("Variable", numeric_cols, key="dec_col")

        decomp = engine.decomposition_tendance(col_dec)
        fig_dec = make_subplots(rows=3, cols=1,
            subplot_titles=["Signal original + Tendance",
                            "Résidu (détrend)", "Spectre FFT"])

        fig_dec.add_trace(go.Scatter(
            x=decomp["t"], y=decomp["original"], mode='lines', name='Original',
            line=dict(color='rgba(0,204,255,0.5)', width=1.5)
        ), row=1, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp["t"], y=decomp["tendance"], mode='lines', name='Tendance',
            line=dict(color='#ffcc00', width=2.5)
        ), row=1, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp["t"], y=decomp["residu"], mode='lines', name='Résidu',
            line=dict(color='#7700ff', width=1.5)
        ), row=2, col=1)
        fig_dec.add_hline(y=0, line_color='rgba(255,255,255,0.3)', row=2, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp["freqs"][1:len(decomp["freqs"])//2],
            y=decomp["magnitude"][1:len(decomp["freqs"])//2],
            mode='lines', fill='tozeroy', fillcolor='rgba(119,0,255,0.2)',
            line=dict(color='#7700ff', width=2), name='|FFT|'
        ), row=3, col=1)

        f_dom  = decomp.get("f_dom", 0)
        periode = decomp.get("periode_dom", 0)
        if f_dom > 0:
            fig_dec.add_vline(x=f_dom, line_color='#ffcc00', line_dash='dash',
                              annotation_text=f"f={f_dom:.3f}", row=3, col=1)

        fig_dec.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
            font=dict(color='#c0d0ff'), height=620,
            legend=dict(bgcolor='rgba(0,0,0,0.5)')
        )
        fig_dec.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
        fig_dec.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
        st.plotly_chart(fig_dec, use_container_width=True)

        if periode < np.inf:
            st.metric("Période dominante (points)", f"{periode:.1f}")

        variance_original = np.var(decomp['original']) if len(decomp['original']) > 0 else 0
        variance_residu = np.var(decomp['residu']) if len(decomp['residu']) > 0 else 0
        resid_ratio = (variance_residu / variance_original * 100) if variance_original > 0 else None
        summary_context.update({
            "n_rows": df.shape[0],
            "n_numeric": len(numeric_cols),
            "section": section,
            "col_dec": col_dec,
            "periode_dom": float(periode) if periode < np.inf else np.inf,
            "f_dom": float(decomp.get('f_dom', 0)),
            "residu_ratio": float(resid_ratio) if resid_ratio is not None else None,
        })


    # ============================================================
    # TAB 4 : PRÉDICTION
    # ============================================================
    elif section == "🤖 Prédiction":
        st.markdown("### 🤖 Modèle prédictif de consommation")

        col1, col2 = st.columns([1, 2])
        with col1:
            target_e   = st.selectbox("Variable cible", numeric_cols)
            features_e = st.multiselect(
                "Variables explicatives",
                [c for c in numeric_cols if c != target_e],
                default=[c for c in numeric_cols if c != target_e][:2]
            )
            modele_e = st.selectbox("Modèle", ["Ridge", "Régression Linéaire", "Gradient Boosting"])
            test_sz  = st.slider("Test (%)", 10, 40, 20) / 100

        with col2:
            if st.button("🚀 Entraîner", use_container_width=True) and features_e:
                res_e = engine.modele_predictif(target_e, features_e, modele_e, test_sz)
                if res_e:
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("R²", f"{res_e['r2']:.4f}")
                    with c2: st.metric("RMSE", f"{res_e['rmse']:.4f}")
                    with c3: st.metric("Qualité", "✅ Bon" if res_e['r2'] > 0.8
                                       else "⚠️ Moyen" if res_e['r2'] > 0.5 else "❌ Faible")

                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=res_e['y_test'], y=res_e['y_pred'], mode='markers',
                        marker=dict(color='#00ccff', size=7, opacity=0.7), name='Prédictions'
                    ))
                    lim = [min(res_e['y_test'].min(), res_e['y_pred'].min()),
                           max(res_e['y_test'].max(), res_e['y_pred'].max())]
                    fig_pr.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
                        name='Parfait', line=dict(color='#ffcc00', dash='dash')))
                    fig_pr.update_layout(
                        title="Réel vs Prédit",
                        xaxis_title="Consommation réelle",
                        yaxis_title="Consommation prédite",
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)

                    if res_e['importance'] is not None:
                        imp_df = pd.DataFrame({
                            "Variable": features_e,
                            "Importance": res_e['importance']
                        }).sort_values("Importance", ascending=True)
                        fig_imp = go.Figure(go.Bar(
                            x=imp_df["Importance"], y=imp_df["Variable"],
                            orientation='h',
                            marker=dict(color=imp_df["Importance"],
                                       colorscale=[[0,'#7700ff'],[1,'#00ccff']])
                        ))
                        fig_imp.update_layout(
                            title="Importance des variables",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                            font=dict(color='#c0d0ff'),
                            xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                            yaxis=dict(color='#c0d0ff'), height=280,
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                    summary_context.update({
                        "n_rows": df.shape[0],
                        "n_numeric": len(numeric_cols),
                        "section": section,
                        "model_name": modele_e,
                        "n_features": len(features_e),
                        "r2": float(res_e['r2']),
                        "rmse": float(res_e['rmse']),
                        "mae": float(res_e['mae']),
                    })


    # ============================================================
    # TAB 5 : BILAN & DPE
    # ============================================================
    elif section == "⚡ Bilan & DPE":
        st.markdown("### ⚡ Bilan énergétique & DPE")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔋 Bilan simplifié")
            E_in = st.slider("Énergie entrante (kWh)", 100.0, 100000.0, 10000.0, 100.0)
            eta  = st.slider("Rendement η (%)", 1.0, 100.0, 80.0, 0.5)
            bilan = engine.bilan_energetique(E_in, eta)

            for k, v in bilan.items():
                st.metric(k, f"{v:.2f}")

            fig_san = go.Figure(go.Funnel(
                y=["Énergie entrante", "Énergie utile", "Pertes"],
                x=[E_in, bilan["E_utile (kWh)"], bilan["E_pertes (kWh)"]],
                marker=dict(color=['#00ccff', '#00ff88', '#ff4444'])
            ))
            fig_san.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0d0ff'), height=300,
            )
            st.plotly_chart(fig_san, use_container_width=True)

        with col2:
            st.markdown("#### 🏠 Diagnostic de Performance Énergétique (DPE)")
            surface   = st.slider("Surface habitable (m²)", 10, 1000, 100)
            col_conso = st.selectbox("Variable de consommation", numeric_cols, key="dpe_col")
            conso_series = pd.to_numeric(df[col_conso], errors='coerce').dropna()
            conso_totale = conso_series.sum()
            conso_m2     = conso_totale / surface if surface != 0 else 0.0

            classe = engine.classe_dpe(conso_m2)
            couleurs_dpe = {
                "A": "#00cc00", "B": "#66cc00", "C": "#cccc00",
                "D": "#ffaa00", "E": "#ff6600", "F": "#ff3300", "G": "#cc0000"
            }
            st.markdown(f"""
            <div style='text-align:center; padding:20px; border-radius:12px;
                        background:rgba(10,0,40,0.6); border:2px solid {couleurs_dpe.get(classe, '#ffffff')};'>
                <div style='font-size:3rem; color:{couleurs_dpe.get(classe, "#ffffff")}'>
                    {classe}
                </div>
                <div style='color:#c0d0ff; font-size:1.2rem'>
                    {conso_m2:.0f} kWh/m²/an
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Barème DPE")
            df_dpe = pd.DataFrame([
                {"Classe": k, "Min (kWh/m²)": v[0], "Max (kWh/m²)": v[1]}
                for k, v in INDICATEURS_BATIMENT.items()
            ])
            st.dataframe(df_dpe, use_container_width=True)

            summary_context.update({
                "n_rows": df.shape[0],
                "n_numeric": len(numeric_cols),
                "section": section,
                "conso_m2": float(conso_m2),
                "dpe_class": classe,
                "dpe_range": INDICATEURS_BATIMENT.get(classe, (0, 0)),
            })


    # ============================================================
    # TAB 6 : DÉTECTION POINTES
    # ============================================================
    elif section == "🔍 Détection pointes":
        st.markdown("### 🔍 Détection des pointes de consommation")

        col1, col2 = st.columns([1, 2])
        with col1:
            col_pointe = st.selectbox("Variable", numeric_cols, key="pointe_col")
            seuil      = st.slider("Seuil (σ)", 1.0, 4.0, 2.0, 0.1)

        with col2:
            pointes_mask = engine.detecter_pointes(col_pointe, seuil)
            s_p = pd.to_numeric(df[col_pointe], errors='coerce').dropna()
            if len(pointes_mask) != len(s_p):
                pointes_mask = pointes_mask[:len(s_p)]
            idx = list(range(len(s_p)))
            n_pointes = int(pointes_mask.sum()) if len(s_p) > 0 else 0
            taux_temps = (n_pointes / len(s_p) * 100) if len(s_p) > 0 else 0.0
            seuil_value = float(s_p.mean() + seuil*s_p.std()) if len(s_p) > 0 else 0.0
            seuil_line = float(s_p.mean() + seuil*s_p.std()) if len(s_p) > 0 else 0.0

            st.metric("Pointes détectées", n_pointes)
            st.metric("% du temps", f"{taux_temps:.1f}%")
            st.metric("Valeur seuil", f"{seuil_value:.2f}")

            fig_pt = go.Figure()
            fig_pt.add_trace(go.Scatter(
                x=idx, y=s_p.values, mode='lines', name='Consommation',
                line=dict(color='rgba(0,204,255,0.4)', width=1.5)
            ))
            if len(s_p) > 0:
                fig_pt.add_trace(go.Scatter(
                    x=[i for i, m in enumerate(pointes_mask) if m],
                    y=[s_p.iloc[i] for i, m in enumerate(pointes_mask) if m],
                    mode='markers', name='Pointes',
                    marker=dict(color='#ff4444', size=8, symbol='circle')
                ))
                fig_pt.add_hline(
                    y=seuil_line,
                    line_color='#ffcc00', line_dash='dash',
                    annotation_text=f"Seuil {seuil}σ"
                )
            fig_pt.update_layout(
                title=f"Détection de pointes — {col_pointe}",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.92)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=420,
            )
            st.plotly_chart(fig_pt, use_container_width=True)

            energy_peaks = float(s_p[pointes_mask].sum()) if n_pointes > 0 else 0.0
            total_energy = float(s_p.sum()) if len(s_p) > 0 else 0.0
            summary_context.update({
                "n_rows": df.shape[0],
                "n_numeric": len(numeric_cols),
                "section": section,
                "n_pointes": n_pointes,
                "taux_temps": float(taux_temps),
                "seuil": float(seuil),
                "pointes_moyennes": float(s_p[pointes_mask].mean()) if n_pointes > 0 else None,
                "pointes_max": float(s_p[pointes_mask].max()) if n_pointes > 0 else None,
                "energie_pointe_pct": (energy_peaks / total_energy * 100) if total_energy > 0 else 0.0,
            })


    # ============================================================

    # ============================================================
    # TAB 7 : PRÉVISION CENTRALE MSD
    # ============================================================
    elif section == "🏭 Prévision Centrale MSD":
        st.markdown("## 🏭 Prévision Énergétique — Centrale MSD")
        st.markdown(
            "*Pipeline complet : aperçu, nettoyage, Energy Gap, corrélations, "
            "entraînement 4 modèles, courbes Réel vs Prédit, comparaison finale et export.*"
        )
        st.markdown("---")

        # ── Configuration (expander fermé par défaut) ─────────────
        with st.expander("⚙️ Configuration des variables (optionnel)", expanded=False):
            cfg1, cfg2 = st.columns(2)
            with cfg1:
                col_heure_msd = st.selectbox(
                    "Colonne Heure",
                    ["— Aucune —"] + list(df.columns),
                    key="msd_heure"
                )
                col_cible_msd = st.selectbox(
                    "Variable cible (y)",
                    numeric_cols,
                    index=numeric_cols.index("Ch 7") if "Ch 7" in numeric_cols else 0,
                    key="msd_cible"
                )
            with cfg2:
                _feat_default_key = st.session_state.get("msd_cible",
                    numeric_cols[numeric_cols.index("Ch 7") if "Ch 7" in numeric_cols else 0])
                col_features_msd = st.multiselect(
                    "Variables explicatives (X)",
                    [c for c in numeric_cols if c != _feat_default_key],
                    default=[c for c in numeric_cols if c != _feat_default_key][:6],
                    key="msd_features"
                )

        # Valeurs par défaut robustes
        _cible_key = st.session_state.get("msd_cible",
            numeric_cols[numeric_cols.index("Ch 7") if "Ch 7" in numeric_cols else 0])
        col_cible_msd    = _cible_key
        col_features_msd = st.session_state.get("msd_features",
            [c for c in numeric_cols if c != col_cible_msd][:6]) or \
            [c for c in numeric_cols if c != col_cible_msd][:6]
        _heure_raw       = st.session_state.get("msd_heure", "— Aucune —")
        col_heure_msd    = None if _heure_raw == "— Aucune —" else _heure_raw

        # ── 1. APERÇU DES DONNÉES ──────────────────────────────────
        st.markdown("### 📂 1. Aperçu des données importées")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Lignes brutes", df.shape[0])
        a2.metric("Colonnes", df.shape[1])
        a3.metric("Valeurs nulles", int(df.isnull().sum().sum()))
        a4.metric("Colonnes numériques", len(numeric_cols))
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("**Statistiques descriptives**")
        st.dataframe(df[numeric_cols].describe().T.style.format("{:.3f}"),
                     use_container_width=True)
        st.markdown("---")

        # ── 2. NETTOYAGE & ENERGY GAP ──────────────────────────────
        st.markdown("### 🧹 2. Nettoyage & Calcul Energy Gap")

        ch6_candidates = [c for c in df.columns if "ch 6" in c.lower() or "ch6" in c.lower()]
        ch7_candidates = [c for c in df.columns if "ch 7" in c.lower() or "ch7" in c.lower()]
        auto_gap = bool(ch6_candidates and ch7_candidates)

        df_msd_input = df.copy()
        eg_stats = {}
        if auto_gap:
            c6, c7 = ch6_candidates[0], ch7_candidates[0]
            df_msd_input[c6] = pd.to_numeric(df_msd_input[c6], errors='coerce').fillna(0)
            df_msd_input[c7] = pd.to_numeric(df_msd_input[c7], errors='coerce').fillna(0)
            df_msd_input["Energy_Gap"] = df_msd_input[c7] - df_msd_input[c6]
            eg = df_msd_input["Energy_Gap"]
            eg_stats = {"eg_min": eg.min(), "eg_max": eg.max(),
                        "eg_mean": eg.mean(), "eg_std": eg.std()}
            st.success(
                f"✅ **Energy Gap** = `{c7}` − `{c6}` "
                f"| min={eg.min():.2f} | max={eg.max():.2f} "
                f"| moy={eg.mean():.2f} | σ={eg.std():.2f}"
            )
        else:
            st.info("ℹ️ Colonnes Ch 6 / Ch 7 non détectées — Energy Gap non calculé.")

        engine_msd = EnergyEngine(df_msd_input)
        prep = engine_msd.preparer_donnees_msd(
            col_features=col_features_msd,
            col_cible=col_cible_msd,
            col_heure=col_heure_msd,
            test_size=0.20,
        )
        df_clean = prep["df_clean"]
        X_train  = prep["X_train"]
        X_test   = prep["X_test"]
        y_train  = prep["y_train"]
        y_test   = prep["y_test"]

        b1, b2, b3 = st.columns(3)
        b1.metric("Lignes après nettoyage (Z<3)", len(df_clean))
        b2.metric("Train", len(X_train))
        b3.metric("Test",  len(X_test))

        # Séries temporelles après nettoyage
        st.markdown("#### 📉 Séries temporelles après nettoyage")
        colors_ch = ["#00ccff","#7700ff","#ffcc00","#00ff88","#ff4444","#ff88ff","#aaffaa"]
        cols_to_plot = [c for c in col_features_msd if c in df_clean.columns][:7]
        fig_series = go.Figure()
        for i_ch, col_ch in enumerate(cols_to_plot):
            s_ch = pd.to_numeric(df_clean[col_ch], errors='coerce').dropna()
            if len(s_ch) > 600:
                sampled = _limit_points(s_ch.values, max_points=600)
                x_vals = list(range(len(sampled)))
                y_vals = sampled
            else:
                x_vals = list(range(len(s_ch)))
                y_vals = s_ch.values
            fig_series.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="lines", name=col_ch,
                line=dict(color=colors_ch[i_ch % len(colors_ch)], width=1.8)
            ))
        fig_series.update_layout(
            title="Évolution temporelle des canaux (données nettoyées)",
            xaxis_title="Index", yaxis_title="Valeur",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.92)",
            font=dict(color="#c0d0ff"),
            xaxis=dict(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)"),
            yaxis=dict(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)"), height=400,
        )
        st.plotly_chart(fig_series, use_container_width=True)

        # Courbe Energy Gap
        if auto_gap and "Energy_Gap" in df_clean.columns:
            st.markdown("#### ⚡ Courbe Energy Gap (Ch 7 − Ch 6)")
            eg_s = pd.to_numeric(df_clean["Energy_Gap"], errors='coerce')
            if len(eg_s) > 800:
                eg_s = pd.Series(_limit_points(eg_s.values, max_points=800))
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Scatter(
                x=list(range(len(eg_s))), y=eg_s,
                mode="lines", fill="tozeroy",
                fillcolor="rgba(255,68,68,0.12)",
                line=dict(color="#ff4444", width=2.2),
                name="Energy Gap"
            ))
            fig_gap.add_hline(y=float(eg_s.mean()),
                              line_color="#ffcc00", line_dash="dash",
                              annotation_text=f"Moy={eg_s.mean():.2f}")
            fig_gap.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_dash="dot")
            fig_gap.update_layout(
                title="Energy Gap = Ch 7 − Ch 6",
                xaxis_title="Index", yaxis_title="Énergie (unité données)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.92)",
                font=dict(color="#c0d0ff"),
                xaxis=dict(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)"),
                yaxis=dict(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)"),
                height=310,
            )
            st.plotly_chart(fig_gap, use_container_width=True)
        st.markdown("---")

        # ── 3. MATRICE DE CORRÉLATION ───────────────────────────────
        st.markdown("### 🔗 3. Matrice de Corrélation")
        cols_corr = list(dict.fromkeys(
            [c for c in col_features_msd if c in df_clean.columns] + [col_cible_msd]
            + (["Energy_Gap"] if auto_gap and "Energy_Gap" in df_clean.columns else [])
        ))
        cols_corr = _select_corr_columns(df_clean, cols_corr, max_cols=12)
        corr_msd = df_clean[cols_corr].corr()
        corr_cible = {
            c: float(corr_msd.loc[c, col_cible_msd])
            for c in cols_corr
            if c != col_cible_msd and col_cible_msd in corr_msd.columns
        }

        fig_corr = go.Figure(go.Heatmap(
            z=corr_msd.values,
            x=corr_msd.columns.tolist(),
            y=corr_msd.columns.tolist(),
            colorscale=[[0,"#020817"],[0.5,"#7700ff"],[1,"#00ccff"]],
            zmid=0,
            text=np.round(corr_msd.values, 2),
            texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color="#c0d0ff")),
        ))
        fig_corr.update_layout(
            title="Matrice de Corrélation — Variables d'énergie",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.92)",
            font=dict(color="#c0d0ff"), height=480,
            xaxis=dict(color="#c0d0ff"), yaxis=dict(color="#c0d0ff"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("---")

        # ── 4. ENTRAÎNEMENT & RÉSULTATS ─────────────────────────────
        st.markdown("### 🤖 4. Entraînement & Résultats des Modèles")

        if len(X_test) == 0:
            st.warning("⚠️ Jeu de test vide — augmentez le dataset.")
            summary_context.update({
                "n_raw": df.shape[0], "n_clean": len(df_clean),
                "n_train": len(X_train), "n_test": 0,
                "has_gap": False, "corr_cible": {}, "metriques": {},
                "best_model": "", "y_std": 1, "feat_importance": {},
                "col_cible": col_cible_msd,
            })
        else:
            with st.spinner("Entraînement des 4 modèles en cours…"):
                resultats_msd = engine_msd.entrainer_modeles_msd(
                    X_train, X_test, y_train, y_test
                )
            meilleur_nom = resultats_msd.pop("__meilleur__")

            # Tableau métriques
            df_metrics = pd.DataFrame([
                {"Modèle": nom,
                 "R²": round(res["r2"], 4),
                 "RMSE": round(res["rmse"], 4),
                 "MAE": round(res["mae"], 4),
                 "🏆": "✅" if nom == meilleur_nom else ""}
                for nom, res in resultats_msd.items()
            ])
            st.dataframe(df_metrics, use_container_width=True)
            st.success(
                f"🏆 Meilleur modèle : **{meilleur_nom}** "
                f"| R² = {resultats_msd[meilleur_nom]['r2']:.4f} "
                f"| RMSE = {resultats_msd[meilleur_nom]['rmse']:.4f}"
            )

            # Barres comparatives
            st.markdown("#### 📊 Comparaison visuelle des métriques")
            noms_m       = list(resultats_msd.keys())
            colors_bar_m = ["#00ccff" if n == meilleur_nom else "#7700ff" for n in noms_m]
            fig_m = make_subplots(rows=1, cols=3, subplot_titles=["R²", "RMSE", "MAE"])
            for _col_idx, (_key, _col_pos) in enumerate([("r2",1),("rmse",2),("mae",3)]):
                fig_m.add_trace(go.Bar(
                    x=noms_m,
                    y=[resultats_msd[n][_key] for n in noms_m],
                    marker_color=colors_bar_m, showlegend=False,
                    text=[f"{resultats_msd[n][_key]:.3f}" for n in noms_m],
                    textposition="outside",
                ), row=1, col=_col_pos)
            fig_m.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.92)",
                font=dict(color="#c0d0ff"), height=370,
            )
            fig_m.update_xaxes(color="#c0d0ff", tickangle=12)
            fig_m.update_yaxes(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)")
            st.plotly_chart(fig_m, use_container_width=True)

            # Scatter Réel vs Prédit — grille 2×2
            st.markdown("#### 🔬 Réel vs Prédit — 4 modèles")
            colors_map = {
                "Régression Linéaire": "#4c72b0",
                "Decision Tree":       "#8172b3",
                "Random Forest":       "#55a868",
                "XGBoost":             "#dd8452",
            }
            n_models = len(resultats_msd)
            n_rows = (n_models + 1) // 2
            fig_grid = make_subplots(
                rows=n_rows, cols=2,
                subplot_titles=list(resultats_msd.keys()),
                horizontal_spacing=0.08, vertical_spacing=0.14,
            )
            for i_m, (nom_m, res_m) in enumerate(resultats_msd.items()):
                row = i_m // 2 + 1
                col = i_m % 2 + 1
                y_pred = np.asarray(res_m["y_pred"])
                y_real = np.asarray(y_test.values)
                if len(y_real) > 500:
                    sel = np.linspace(0, len(y_real) - 1, 500, dtype=int)
                    y_real = y_real[sel]
                    y_pred = y_pred[sel]
                min_val = min(float(y_real.min()), float(y_pred.min()))
                max_val = max(float(y_real.max()), float(y_pred.max()))
                fig_grid.add_trace(go.Scatter(
                    x=y_real, y=y_pred, mode="markers",
                    marker=dict(color=colors_map.get(nom_m, "#00ccff"), size=5, opacity=0.55),
                    name=f"Prédit ({nom_m})",
                    showlegend=False,
                ), row=row, col=col)
                fig_grid.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val], mode="lines",
                    line=dict(color="red", dash="dash", width=2),
                    name="Réel = Prédit",
                    showlegend=False,
                ), row=row, col=col)
                fig_grid.update_xaxes(title_text="Réel", row=row, col=col)
                fig_grid.update_yaxes(title_text="Prédit", row=row, col=col)
            fig_grid.update_layout(
                title="Réel vs Prédit — Comparaison des modèles",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.92)",
                font=dict(color="#c0d0ff"), height=420 * n_rows,
                legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            )
            st.plotly_chart(fig_grid, use_container_width=True)

            # Comparaison overlay finale
            st.markdown("#### 🎯 Comparaison finale — tous modèles superposés")
            markers_sym = ["x", "circle", "square", "triangle-up"]
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(
                x=y_test.values, y=y_test.values,
                mode="lines", name="Prédiction parfaite (Réel = Prédit)",
                line=dict(color="red", dash="dash", width=3)
            ))
            for i_ov, (nom_ov, res_ov) in enumerate(resultats_msd.items()):
                y_pred = np.asarray(res_ov["y_pred"])
                y_real = np.asarray(y_test.values)
                if len(y_real) > 500:
                    sel = np.linspace(0, len(y_real) - 1, 500, dtype=int)
                    y_real = y_real[sel]
                    y_pred = y_pred[sel]
                fig_ov.add_trace(go.Scatter(
                    x=y_real, y=y_pred,
                    mode="markers", name=nom_ov,
                    marker=dict(
                        color=list(colors_map.values())[i_ov % len(colors_map)],
                        size=6, opacity=0.5,
                        symbol=markers_sym[i_ov % len(markers_sym)]
                    ),
                ))
            fig_ov.update_layout(
                title="Comparaison des modèles : Prédiction vs Réel",
                xaxis_title=f"{col_cible_msd} réel",
                yaxis_title=f"{col_cible_msd} prédit",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.92)",
                font=dict(color="#c0d0ff"),
                xaxis=dict(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)"),
                yaxis=dict(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)"),
                legend=dict(bgcolor="rgba(0,0,0,0.5)"), height=520,
            )
            st.plotly_chart(fig_ov, use_container_width=True)

            # Importance des variables
            best_res_msd  = resultats_msd[meilleur_nom]
            feat_imp_dict = {}
            if hasattr(best_res_msd["model"], "feature_importances_"):
                st.markdown(f"#### 🔑 Importance des variables — {meilleur_nom}")
                imp_vals   = best_res_msd["model"].feature_importances_
                feat_names = col_features_msd[:len(imp_vals)]
                feat_imp_dict = dict(zip(feat_names, imp_vals.tolist()))
                imp_df = (
                    pd.DataFrame({"Variable": feat_names, "Importance": imp_vals})
                    .sort_values("Importance", ascending=True)
                )
                fig_imp = go.Figure(go.Bar(
                    x=imp_df["Importance"], y=imp_df["Variable"],
                    orientation="h",
                    marker=dict(color=imp_df["Importance"].tolist(),
                                colorscale=[[0,"#7700ff"],[1,"#00ccff"]],
                                showscale=True),
                    text=[f"{v:.4f}" for v in imp_df["Importance"]],
                    textposition="outside",
                ))
                fig_imp.update_layout(
                    title=f"Importance des variables — {meilleur_nom}",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.92)",
                    font=dict(color="#c0d0ff"),
                    xaxis=dict(color="#c0d0ff", gridcolor="rgba(100,0,255,0.2)"),
                    yaxis=dict(color="#c0d0ff"),
                    height=max(300, len(feat_names) * 38),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("---")

            # Export
            st.markdown("### 💾 5. Export")
            ex1, ex2 = st.columns(2)
            with ex1:
                buf_pkl = io.BytesIO()
                joblib.dump(best_res_msd["model"], buf_pkl)
                buf_pkl.seek(0)
                st.download_button(
                    label=f"⬇️ Meilleur modèle ({meilleur_nom}) — .pkl",
                    data=buf_pkl,
                    file_name="meilleur_modele_energie_msd.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            with ex2:
                st.download_button(
                    label="⬇️ Données nettoyées — .csv",
                    data=df_clean.to_csv(index=False).encode(),
                    file_name="donnees_msd_propres.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Alimenter le contexte pour l'analyse avancée
            y_std_v = float(y_test.std()) if len(y_test) > 1 else 1.0
            summary_context.update({
                "n_raw": df.shape[0],
                "n_clean": len(df_clean),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "has_gap": auto_gap and "Energy_Gap" in df_clean.columns,
                "corr_cible": corr_cible,
                "col_cible": col_cible_msd,
                "metriques": {n: {"r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"]}
                              for n, r in resultats_msd.items()},
                "best_model": meilleur_nom,
                "y_std": y_std_v,
                "feat_importance": feat_imp_dict,
                **eg_stats,
            })
            resultats_msd["__meilleur__"] = meilleur_nom


    # TAB 8 : THÉORIE
    # ============================================================
    elif section == "📖 Théorie":
        st.markdown("### 📖 Formulaire scientifique énergétique")
        cols = st.columns(2)
        col_idx = 0
        for nom, formule in FORMULES_ENERGIE.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 🔬 Constantes énergétiques")
        df_cst = pd.DataFrame([
            {"Constante": k, "Valeur": v[0], "Unité": v[1]}
            for k, v in CONSTANTES_ENERGIE.items()
        ])
        st.dataframe(df_cst, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "IEA — *World Energy Outlook* (2023)",
            "ADEME — *Guide de l'efficacité énergétique* (2022)",
            "Pérez-Lombard et al. — *A review on buildings energy consumption* (Energy & Buildings, 2008)",
            "MacKay — *Sustainable Energy Without the Hot Air* (UIT Cambridge, 2009)",
        ]:
            st.markdown(f"- {r}")

    # ── Analyse & Conclusion automatique ─────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Analyse automatique")
    st.markdown(generer_analyse_automatique(section, summary_context))
    st.markdown("---")
    st.markdown("### ✅ Conclusion automatique")
    st.markdown(generer_conclusion_automatique(section, summary_context))

    # ── Export global CSV ────────────────────────────────────────
    st.markdown("---")
    st.download_button("💾 Export CSV", df.to_csv(index=False).encode(),
                       "energy_export.csv", "text/csv")


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="Analyse Énergétique",
        page_icon="⚡",
        layout="wide"
    )
    energy_page()
