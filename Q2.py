import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# Question 2 : Lookback Put Option par Monte-Carlo
# ============================================================
# Paramètres alignés avec la Question 1 :
# S0 = 110, r = 0.04, sigma = 0.15, T = 0.5
# ============================================================


# ============================================================
# 1. Paramètres globaux
# ============================================================

S0 = 110.0
K_EUROPEAN = 110.0
r = 0.04
sigma = 0.15
T = 0.5

N_STEPS_WEEKLY = 26
N_STEPS_DAILY = 180
N_STEPS_LIST = [12, 26, 52, 180, 360, 720]
N_SIM_LIST = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
PAS_LIST = {
    f"{n_steps}_pas": {
        "n_steps": n_steps,
        "label": f"{n_steps} pas"
    }
    for n_steps in N_STEPS_LIST
}
N_FIXED = 100_000
N_REP_STABILITY = 30
OUTPUT_DIR = Path("Lookback")

# ============================================================
# Style global des figures, aligne sur analyse_exploratoire.py
# ============================================================
PICTURE_WIDTH_CM = 15.0
HW_RATIO = 3.0 / 4.0
FONT_SIZE = 10
LINE_WIDTH = 1.0
MARKER_SIZE = 6
DPI = 300
CM_TO_INCH = 1.0 / 2.54

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE * 1.10,
    "axes.titleweight": "bold",
    "xtick.labelsize": FONT_SIZE * 0.9,
    "ytick.labelsize": FONT_SIZE * 0.9,
    "legend.fontsize": FONT_SIZE * 0.9,
    "axes.linewidth": LINE_WIDTH * 0.75,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 100,
})
# ============================================================
# 2. Fonctions utilitaires pour les figures
# ============================================================
def make_figure(width_cm=PICTURE_WIDTH_CM, ratio=HW_RATIO, scale=1.0):
    """Crée une figure Matplotlib.
    Args:
        width_cm (float): Largeur de la figure en centimetres.
        ratio (float): Ratio hauteur / largeur.
        scale (float): Facteur d'agrandissement.
    Returns:
        tuple: Figure et axe Matplotlib.
    """
    width_in = width_cm * CM_TO_INCH * scale
    height_in = width_in * ratio
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    return fig, ax
def save_figure(fig, path, dpi=DPI):
    """Sauvegarde une figure dans le dossier de sortie.
    Args:
        fig: Figure Matplotlib à sauvegarder.
        path (str | Path): Chemin du fichier de sortie.
        dpi (int): Resolution de sauvegarde.
    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [sauve] {path}")
# ============================================================
# 3. Formules fermées
# ============================================================
def european_put_black_scholes(S0, K, r, sigma, T):
    """Calcule le prix Black-Scholes d'un put européen.
    Args:
        S0 (float): Prix initial du sous-jacent.
        K (float): Strike du put européen.
        r (float): Taux sans risque continu.
        sigma (float): Volatilité du sous-jacent.
        T (float): Maturité en années.
    Returns:
        dict: Prix du put européen, d1 et d2.
    """
    d1 = (np.log(S0 / K)+ (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    return {"price": price,"d1": d1,"d2": d2}

def lookback_put_formula(S0, r, sigma, T):
    """Calcule le prix théorique d'une lookback put flottante continue.
    La formule correspond au payoff :
        max_{0 <= t <= T} S_t - S_T.
    Args:
        S0 (float): Prix initial du sous-jacent.
        r (float): Taux sans risque continu.
        sigma (float): Volatilité du sous-jacent.
        T (float): Maturité en années.
    Returns:
        dict: Prix théorique de la lookback put, c1 et c2.
    """
    c1 = ((r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    c2 = ((r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    discount = np.exp(-r * T)
    price = (
        S0 * (discount * norm.cdf(-c2) - norm.cdf(-c1))
        + (S0 * sigma**2 / (2.0 * r))
        * (norm.cdf(c1) - discount * norm.cdf(-c2))
    )
    return {"price": price,"c1": c1,"c2": c2}
# ============================================================
# 4. Simulation Monte-Carlo du lookback put
# ============================================================

def estimate_lookback_put_mc(S0,r,sigma,T,n_steps,n_sim,seed=12345, chunk_size=200_000):
    """Estime le prix d'une lookback put par Monte-Carlo.
    Le sous-jacent est simulé avec la solution exacte du mouvement
    brownien géométrique sous la mesure risque-neutre :
        S_{t+dt} = S_t exp((r - sigma^2/2) dt + sigma sqrt(dt) Z).
    Le maximum utilisé est le maximum observé sur la grille de simulation.
    Aucune correction Brownian bridge n'est ajoutée.
    Args:
        S0 (float): Prix initial du sous-jacent.
        r (float): Taux sans risque continu.
        sigma (float): Volatilité du sous-jacent.
        T (float): Maturité en années.
        n_steps (int): Nombre de pas temporels.
        n_sim (int): Nombre de simulations Monte-Carlo.
        seed (int): Graine aléatoire pour la reproductibilité.
        chunk_size (int): Taille maximale des blocs de simulation.
    Returns:
        dict: Prix estimé, erreur-type et intervalle de confiance à 95%.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * T)
    sum_payoff = 0.0
    sum_payoff_squared = 0.0
    count = 0
    for start in range(0, n_sim, chunk_size):
        m = min(chunk_size, n_sim - start)
        S = np.full(m, S0, dtype=float)
        running_max = np.full(m, S0, dtype=float)
        for _ in range(n_steps):
            Z = rng.standard_normal(m)
            S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            running_max = np.maximum(running_max, S)
        payoff = np.maximum(running_max - S, 0.0)
        discounted_payoff = discount * payoff
        sum_payoff += discounted_payoff.sum()
        sum_payoff_squared += np.sum(discounted_payoff**2)
        count += m
    price = sum_payoff / count
    sample_variance = (sum_payoff_squared - count * price**2) / (count - 1)
    std_error = np.sqrt(sample_variance / count)
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error
    return {"price": price,"std_error": std_error,"ci_low": ci_low,"ci_high": ci_high}
# ============================================================
# 5. Convergence et stabilité
# ============================================================
def print_progress_bar_convergence(completed,total,n_steps,
    n_sim, prefix="Convergence Monte-Carlo",bar_length=30):
    """Affiche une barre de progression pour l'étude de convergence.
    Args:
        completed (int): Nombre de calculs déjà terminés.
        total (int): Nombre total de calculs à effectuer.
        n_steps (int): Nombre de pas temporels actuellement testé.
        n_sim (int): Nombre de simulations actuellement testé.
        prefix (str): Texte affiché avant la barre.
        bar_length (int): Longueur de la barre de progression.
    Returns:
        None.
    """
    progress = completed / total
    percent = 100.0 * progress
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    n_sim_display = f"{int(n_sim):,}".replace(",", " ")
    message = (
        f"\r{prefix} |{bar}| "
        f"{percent:5.1f}% | "
        f"{completed}/{total} | "
        f"pas = {n_steps} | "
        f"N = {n_sim_display}"
    )
    print(message, end="", flush=True)
    if completed == total:
        print()

def convergence_by_simulations(S0, r, sigma, T, pas_list, n_sim_list,
    seed=20260430, show_progress=True):
    """Étudie la convergence selon le nombre de simulations.
    Args:
        S0 (float): Prix initial du sous-jacent.
        r (float): Taux sans risque continu.
        sigma (float): Volatilité du sous-jacent.
        T (float): Maturité en années.
        pas_list (dict): Dictionnaire des pas temporels à tester.
        n_sim_list (list[int]): Liste des nombres de simulations.
        seed (int): Graine aléatoire de base.
        show_progress (bool): Affiche une barre de progression si True.
    Returns:
        pandas.DataFrame: Tableau de convergence.
    """
    theoretical = lookback_put_formula(S0, r, sigma, T)["price"]
    rows = []
    total_tasks = len(pas_list) * len(n_sim_list)
    completed_tasks = 0
    for pas_key, pas_info in pas_list.items():
        n_steps = pas_info["n_steps"]
        label = pas_info["label"]
        for i, n_sim in enumerate(n_sim_list):
            result = estimate_lookback_put_mc(S0=S0,r=r,
                sigma=sigma, T=T, n_steps=n_steps,
                n_sim=n_sim, seed=seed + 1000 * i + n_steps
            )
            rows.append({
                "pas": pas_key,"description_pas": label,
                "n_steps": n_steps, "dt": T / n_steps,
                "n_sim": n_sim, "price_mc": result["price"],
                "std_error": result["std_error"],
                "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                "theoretical_price": theoretical,
                "abs_error_vs_theory": abs(result["price"] - theoretical),
                "rel_error_vs_theory_pct": (
                    100.0 * abs(result["price"] - theoretical) / theoretical
                )
            })
            completed_tasks += 1
            if show_progress:
                print_progress_bar_convergence(
                    completed=completed_tasks, total=total_tasks,
                    n_steps=n_steps,n_sim=n_sim)
    return pd.DataFrame(rows)
def print_progress_bar(completed,total, n_steps, rep, n_rep,
    prefix="Stabilité Monte-Carlo",bar_length=30):
    """Affiche une barre de progression dans le terminal.
    Args:
        completed (int): Nombre de calculs déjà terminés.
        total (int): Nombre total de calculs à effectuer.
        n_steps (int): Nombre de pas temporels actuellement testé.
        rep (int): Numéro de répétition courante.
        n_rep (int): Nombre total de répétitions par pas temporel.
        prefix (str): Texte affiché avant la barre.
        bar_length (int): Longueur de la barre de progression.
    Returns:
        None.
    """
    progress = completed / total
    percent = 100.0 * progress
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    message = (
        f"\r{prefix} |{bar}| "
        f"{percent:5.1f}% | "
        f"{completed}/{total} | "
        f"pas = {n_steps} | "
        f"rep = {rep}/{n_rep}"
    )
    print(message, end="", flush=True)
    if completed == total:
        print()

def stability_replications(S0,r,sigma,T,
    pas_list,n_sim,
    n_rep=30,seed=777,
    show_progress=True
):
    """Répète l'estimation Monte-Carlo pour étudier la stabilité.
    Args:
        S0 (float): Prix initial du sous-jacent.
        r (float): Taux sans risque continu.
        sigma (float): Volatilité du sous-jacent.
        T (float): Maturité en années.
        pas_list (dict): Dictionnaire des pas temporels à tester.
        n_sim (int): Nombre de simulations fixé.
        n_rep (int): Nombre de répétitions indépendantes.
        seed (int): Graine aléatoire de base.
        show_progress (bool): Affiche une barre de progression si True.
    Returns:
        pandas.DataFrame: Tableau des estimations répétées.
    """
    rows = []
    total_tasks = len(pas_list) * n_rep
    completed_tasks = 0
    for pas_key, pas_info in pas_list.items():
        n_steps = pas_info["n_steps"]
        label = pas_info["label"]
        for rep in range(n_rep):
            result = estimate_lookback_put_mc(S0=S0,r=r,
                sigma=sigma, T=T, n_steps=n_steps,
                n_sim=n_sim, seed=seed + 100 * rep + n_steps
            )
            rows.append({
                "pas": pas_key, "description_pas": label,
                "n_steps": n_steps, "n_sim": n_sim,
                "replication": rep + 1, "price_mc": result["price"],
                "std_error": result["std_error"]
            })
            completed_tasks += 1
            if show_progress:
                print_progress_bar( completed=completed_tasks, total=total_tasks,
                    n_steps=n_steps, rep=rep + 1,n_rep=n_rep)
    return pd.DataFrame(rows)
# ============================================================
# 6. Graphiques
# ============================================================
def plot_convergence_prices(df, output_dir):
    """Trace le prix estimé selon le nombre de simulations.
    Args:
        df (pandas.DataFrame): Tableau de convergence.
        output_dir (str | Path): Dossier de sauvegarde.
    Returns:
        None.
    """
    output_dir = Path(output_dir)
    fig, ax = make_figure()
    for pas_key in df["pas"].unique():
        sub = df[df["pas"] == pas_key]
        label = sub["description_pas"].iloc[0]
        ax.plot(
            sub["n_sim"],
            sub["price_mc"],
            marker="o",
            linewidth=2,
            label=label
        )
    theoretical = df["theoretical_price"].iloc[0]
    ax.axhline(
        theoretical,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Prix théorique continu"
    )
    n_sim_ticks = sorted(df["n_sim"].unique())
    ax.set_xscale("log")
    ax.set_xticks(n_sim_ticks)
    ax.set_xticklabels([f"{int(x):,}".replace(",", " ") for x in n_sim_ticks])
    ax.set_xlabel("Nombre de simulations")
    ax.set_ylabel("Prix estimé")
    ax.set_title("Convergence du prix Monte-Carlo")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    save_figure(fig, output_dir / "01_convergence_prix.png")


def plot_convergence_errors(df, output_dir):
    """Trace l'erreur relative selon le nombre de simulations.
    Args:
        df (pandas.DataFrame): Tableau de convergence.
        output_dir (str | Path): Dossier de sauvegarde.
    Returns:
        None.
    """
    output_dir = Path(output_dir)
    fig, ax = make_figure()
    for pas_key in df["pas"].unique():
        sub = df[df["pas"] == pas_key]
        label = sub["description_pas"].iloc[0]
        ax.loglog(
            sub["n_sim"],
            sub["rel_error_vs_theory_pct"],
            marker="o",
            linewidth=2,
            label=label
        )
    ax.set_xlabel("Nombre de simulations")
    ax.set_ylabel("Erreur relative vs formule continue (%)")
    ax.set_title("Erreur relative Monte-Carlo")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    save_figure(fig, output_dir / "02_convergence_erreur_relative.png")


def plot_time_step_comparison(df, output_dir, n_fixed):
    """Compare les pas temporels pour un nombre de simulations fixé.
    Args:
        df (pandas.DataFrame): Tableau de convergence.
        output_dir (str | Path): Dossier de sauvegarde.
        n_fixed (int): Nombre de simulations fixé.
    Returns:
        None.
    """
    output_dir = Path(output_dir)
    sub = df[df["n_sim"] == n_fixed].copy()
    sub = sub.sort_values("n_steps")
    fig, ax = make_figure()
    ax.plot(
        sub["n_steps"],
        sub["price_mc"],
        marker="o",
        linewidth=2,
        label="Prix Monte-Carlo"
    )
    ax.axhline(
        sub["theoretical_price"].iloc[0],
        color="black",linestyle="--",
        linewidth=2,label="Prix théorique continu"
    )
    ax.set_xticks(sub["n_steps"])
    ax.set_xlabel("Nombre de pas temporels")
    ax.set_ylabel("Prix estimé")
    ax.set_title(f"Impact du pas temporel, N = {n_fixed}")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    save_figure(fig, output_dir / "03_impact_pas_temporel.png")


def plot_stability_boxplot(df_stability, output_dir):
    """Trace un boxplot des estimations répétées.
    Args:
        df_stability (pandas.DataFrame): Tableau des répétitions.
        output_dir (str | Path): Dossier de sauvegarde.
    Returns:
        None.
    """
    output_dir = Path(output_dir)
    labels = []
    data = []
    for pas_key in df_stability["pas"].unique():
        sub = df_stability[df_stability["pas"] == pas_key]
        labels.append(sub["description_pas"].iloc[0])
        data.append(sub["price_mc"].values)
    fig, ax = make_figure()
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Prix estimé")
    ax.set_title("Stabilité des estimations Monte-Carlo")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, linestyle=":", alpha=0.7)
    save_figure(fig, output_dir / "04_stabilite_boxplot.png")


def plot_option_comparison(
    european_price,lookback_mc_price,
    lookback_theoretical_price,output_dir):
    """Compare graphiquement le put européen et la lookback put.
    Args:
        european_price (float): Prix Black-Scholes du put européen.
        lookback_mc_price (float): Prix Monte-Carlo de la lookback put.
        lookback_theoretical_price (float): Prix théorique de la lookback put.
        output_dir (str | Path): Dossier de sauvegarde.
    Returns:
        None.
    """
    output_dir = Path(output_dir)
    labels = ["European Put BS","Lookback Put MC","Lookback Put théorique"]
    values = [european_price,lookback_mc_price,lookback_theoretical_price]
    fig, ax = make_figure()
    ax.bar(labels, values)
    ax.set_ylabel("Prix")
    ax.set_title("Comparaison European Put / Lookback Put")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom")
    save_figure(fig, output_dir / "05_comparaison_options.png")

def plot_example_paths_with_maximum(S0,r,
    sigma,T,n_steps,
    n_paths,output_dir,seed=123):
    """Trace des trajectoires simulées et marque le maximum utilisé.
    Pour chaque trajectoire, l'étoile indique le maximum discret :
        max(S_{t_0}, S_{t_1}, ..., S_{t_n}).
    C'est ce maximum discret qui est utilisé dans le payoff Monte-Carlo :
        payoff = max_i S_{t_i} - S_T.
    Args:
        S0 (float): Prix initial du sous-jacent.
        r (float): Taux sans risque continu.
        sigma (float): Volatilité du sous-jacent.
        T (float): Maturité en années.
        n_steps (int): Nombre de pas temporels.
        n_paths (int): Nombre de trajectoires à tracer.
        output_dir (str | Path): Dossier de sauvegarde.
        seed (int): Graine aléatoire.

    Returns:
        None.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    time_grid = np.linspace(0.0, T, n_steps + 1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )
    fig, ax = make_figure()
    for i in range(n_paths):
        max_index = np.argmax(paths[i])
        max_time = time_grid[max_index]
        max_value = paths[i, max_index]
        ax.plot(time_grid, paths[i], linewidth=1)
        ax.scatter(max_time, max_value,
            marker="*", s=120,
            color="black", zorder=5,
            label="Maximum utilisé dans le payoff" if i == 0 else None
        )
    ax.set_xlabel("Temps")
    ax.set_ylabel("S_t")
    ax.set_title("Trajectoires simulées du sous-jacent et maximum utilisé")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    save_figure(fig, Path(output_dir) / "06_trajectoires_sous_jacent_maximum.png")

def print_convergence_table(df):
    """Affiche proprement le tableau de convergence dans le terminal.
    Args:
        df (pandas.DataFrame): Tableau complet de convergence.
    Returns:
        None.
    """
    display_df = df.copy()
    display_df["dt"] = display_df["dt"].map(lambda x: f"{x:.6f}")
    display_df["n_sim"] = display_df["n_sim"].map(lambda x: f"{int(x):,}".replace(",", " "))
    display_df["price_mc"] = display_df["price_mc"].map(lambda x: f"{x:.6f}")
    display_df["std_error"] = display_df["std_error"].map(lambda x: f"{x:.6f}")
    display_df["ci_95"] = display_df.apply(
        lambda row: f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]",
        axis=1
    )
    display_df["abs_error_vs_theory"] = display_df["abs_error_vs_theory"].map(lambda x: f"{x:.6f}")
    display_df["rel_error_vs_theory_pct"] = display_df["rel_error_vs_theory_pct"].map(lambda x: f"{x:.3f}%")
    display_df = display_df[
        [
            "description_pas",
            "n_steps",
            "dt",
            "n_sim",
            "price_mc",
            "std_error",
            "ci_95",
            "abs_error_vs_theory",
            "rel_error_vs_theory_pct"
        ]
    ]

    display_df = display_df.rename(columns={
        "description_pas": "Pas temporel",
        "n_steps": "n_steps",
        "dt": "dt",
        "n_sim": "N simulations",
        "price_mc": "Prix MC",
        "std_error": "Erreur-type",
        "ci_95": "IC 95%",
        "abs_error_vs_theory": "Erreur abs.",
        "rel_error_vs_theory_pct": "Erreur rel."
    })
    print()
    print("=" * 120)
    print("TABLEAU DE CONVERGENCE MONTE-CARLO")
    print("=" * 120)
    for pas in display_df["Pas temporel"].unique():
        sub = display_df[display_df["Pas temporel"] == pas]
        print()
        print(f"--- {pas} ---")
        print(sub.drop(columns=["Pas temporel"]).to_string(index=False))
    print("=" * 120)
    print()

def print_stability_table(stability_summary):
    """Affiche proprement le résumé de stabilité dans le terminal.
    Args:
        stability_summary (pandas.DataFrame): Résumé statistique des répétitions.
    Returns:
        None.
    """
    display_df = stability_summary.copy()
    display_df["mean_price"] = display_df["mean_price"].map(lambda x: f"{x:.6f}")
    display_df["std_price"] = display_df["std_price"].map(lambda x: f"{x:.6f}")
    display_df["min_price"] = display_df["min_price"].map(lambda x: f"{x:.6f}")
    display_df["max_price"] = display_df["max_price"].map(lambda x: f"{x:.6f}")
    display_df = display_df[
        [
            "description_pas",
            "mean_price",
            "std_price",
            "min_price",
            "max_price"
        ]
    ]
    display_df = display_df.rename(columns={
        "description_pas": "Pas temporel",
        "mean_price": "Prix moyen",
        "std_price": "Écart-type",
        "min_price": "Prix min.",
        "max_price": "Prix max."
    })
    print()
    print("=" * 90)
    print("RÉSUMÉ DE STABILITÉ MONTE-CARLO")
    print("=" * 90)
    print(display_df.to_string(index=False))
    print("=" * 90)
    print()

# ============================================================
# 7. Programme principal : réponses numériques Q2a à Q2e
# ============================================================

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("QUESTION 2 - LOOKBACK PUT OPTION")
    print("============================================================")
    print(f"S0 = {S0}")
    print(f"r = {r}")
    print(f"sigma = {sigma}")
    print(f"T = {T}")
    print(f"Pas temporels testés = {N_STEPS_LIST}")
    print(f"Nombres de simulations testés = {N_SIM_LIST}")
    print()

    # --------------------------------------------------------
    # Q2a : Estimation Monte-Carlo aujourd'hui
    # --------------------------------------------------------
    print("Q2a) Prix Monte-Carlo de la Lookback Put")
    print("------------------------------------------------------------")

    result_q2a = estimate_lookback_put_mc(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=N_STEPS_WEEKLY,
        n_sim=N_FIXED,
        seed=12345
    )

    print(f"Nombre de simulations = {N_FIXED}")
    print(f"Pas temporel = hebdomadaire ({N_STEPS_WEEKLY} pas)")
    print(f"Prix Monte-Carlo = {result_q2a['price']:.8f}")
    print(f"Erreur-type = {result_q2a['std_error']:.8f}")
    print()

    # --------------------------------------------------------
    # Q2b : Stabilité et convergence
    # --------------------------------------------------------
    print("Q2b) Stabilité et convergence")
    print("------------------------------------------------------------")

    df_convergence = convergence_by_simulations(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        pas_list=PAS_LIST,
        n_sim_list=N_SIM_LIST,
        seed=20260430
    )
    print_convergence_table(df_convergence)
    df_stability = stability_replications(
        S0=S0,r=r,
        sigma=sigma,T=T,
        pas_list=PAS_LIST,
        n_sim=N_FIXED,
        n_rep=N_REP_STABILITY,
        seed=777
    )
    stability_summary = (
        df_stability
        .groupby(["pas", "description_pas"])
        .agg(
            mean_price=("price_mc", "mean"),
            std_price=("price_mc", "std"),
            min_price=("price_mc", "min"),
            max_price=("price_mc", "max")
        )
        .reset_index()
    )
    print_stability_table(stability_summary)
    # --------------------------------------------------------
    # Q2c : Intervalle de confiance à 95%
    # --------------------------------------------------------
    print("Q2c) Intervalle de confiance à 95%")
    print("------------------------------------------------------------")
    result_q2c = estimate_lookback_put_mc(S0=S0,r=r,
        sigma=sigma,T=T,
        n_steps=N_STEPS_DAILY,
        n_sim=N_FIXED,seed=54321
    )
    print(f"Nombre de simulations retenu = {N_FIXED}")
    print(f"Pas temporel retenu = journalier ({N_STEPS_DAILY} pas)")
    print(f"Prix estimé = {result_q2c['price']:.8f}")
    print(f"Erreur-type = {result_q2c['std_error']:.8f}")
    print(f"IC 95% = [{result_q2c['ci_low']:.8f}, {result_q2c['ci_high']:.8f}]")
    print()
    # --------------------------------------------------------
    # Q2d : Prix théorique type Black-Scholes
    # --------------------------------------------------------
    print("Q2d) Prix théorique de la Lookback Put")
    print("------------------------------------------------------------")
    theoretical_lookback = lookback_put_formula(S0=S0,r=r,sigma=sigma,T=T)
    print(f"c1 = {theoretical_lookback['c1']:.8f}")
    print(f"c2 = {theoretical_lookback['c2']:.8f}")
    print(f"Prix théorique Lookback Put = {theoretical_lookback['price']:.8f}")
    print(f"Écart MC journalier - formule = {result_q2c['price'] - theoretical_lookback['price']:.8f}")
    print()
    # --------------------------------------------------------
    # Q2e : Comparaison European Put / Lookback Put
    # --------------------------------------------------------
    print("Q2e) Comparaison European Put / Lookback Put")
    print("------------------------------------------------------------")
    european = european_put_black_scholes(S0=S0,K=K_EUROPEAN,r=r,sigma=sigma,T=T)
    print(f"Prix European Put Black-Scholes = {european['price']:.8f}")
    print(f"Prix Lookback Put MC journalier = {result_q2c['price']:.8f}")
    print(f"Prix Lookback Put théorique = {theoretical_lookback['price']:.8f}")
    print()
    # --------------------------------------------------------
    # Sauvegardes CSV et figures
    # --------------------------------------------------------
    df_convergence.to_csv(OUTPUT_DIR / "convergence_lookback.csv", index=False)
    df_stability.to_csv(OUTPUT_DIR / "stabilite_lookback.csv", index=False)
    stability_summary.to_csv(OUTPUT_DIR / "resume_stabilite_lookback.csv", index=False)
    plot_convergence_prices(df_convergence, OUTPUT_DIR)
    plot_convergence_errors(df_convergence, OUTPUT_DIR)
    plot_time_step_comparison(df_convergence, OUTPUT_DIR, N_FIXED)
    plot_stability_boxplot(df_stability, OUTPUT_DIR)
    plot_option_comparison(
        european_price=european["price"],lookback_mc_price=result_q2c["price"],
        lookback_theoretical_price=theoretical_lookback["price"],output_dir=OUTPUT_DIR)
    plot_example_paths_with_maximum(S0=S0,r=r,sigma=sigma,T=T,
        n_steps=N_STEPS_WEEKLY,n_paths=20,output_dir=OUTPUT_DIR,seed=123)
    print(f"Fichiers CSV et figures sauvegardés dans : {OUTPUT_DIR}/")
