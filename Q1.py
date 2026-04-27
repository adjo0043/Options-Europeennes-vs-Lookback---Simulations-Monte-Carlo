# ============================================================================
# Question 1a : Prix d'un Put européen par Monte-Carlo (pas hebdomadaire)
# Modèle de Black-Scholes : dS = r S dt + sigma S dW
# Paramètres : S0=110, r=0.04, sigma=0.15, K=110, T=0.5 an
# Pas de temps : 1 semaine → 26 pas sur 6 mois
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Paramètres fixes
# ---------------------------
S0      = 110      # Prix initial
K       = 110      # Strike
r       = 0.04     # Taux sans risque
sigma   = 0.15     # Volatilité
T       = 0.5      # Maturité (6 mois = 0.5 an)
n_pas   = 26       # Nombre de pas de temps (hebdomadaire)
dt      = T / n_pas # Pas de temps = 0.5/26 ≈ 0.01923 an

# Nombre de simulations (pour les analyses d'impact)
N_sim   = 50000

# Création d'un dossier pour les graphiques (dans le répertoire du projet)
dir_graph = "graphiques"
if not os.path.exists(dir_graph):
    os.makedirs(dir_graph)

# ---------------------------
# 2. Simulation et calcul du prix
#    Discrétisation exacte : S(t+dt) = S(t) * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
# ---------------------------
np.random.seed(123)  # Pour reproductibilité

# Matrice des prix : chaque ligne = une simulation, chaque colonne = un pas
S = np.zeros((N_sim, n_pas + 1))
S[:, 0] = S0

for i in range(n_pas):
    Z = np.random.normal(0, 1, N_sim)
    S[:, i+1] = S[:, i] * np.exp((r - sigma**2/2) * dt + sigma * np.sqrt(dt) * Z)

# Payoffs et prix estimé
ST = S[:, -1]
payoff = np.maximum(K - ST, 0)
prix_simule = np.mean(payoff) * np.exp(-r * T)
print(f"Prix estimé du put européen (Monte-Carlo, pas hebdo) : {prix_simule:.4f}")

# ---------------------------
# 3. Analyse de l'impact des paramètres avec sauvegarde JPEG
# ---------------------------

# 3a. Impact de la volatilité sigma
sigma_test = np.arange(0.05, 0.55, 0.05)
prix_sigma = []

for sig in sigma_test:
    S_temp = np.zeros((N_sim, n_pas + 1))
    S_temp[:, 0] = S0
    for i in range(n_pas):
        Z = np.random.normal(0, 1, N_sim)
        S_temp[:, i+1] = S_temp[:, i] * np.exp((r - sig**2/2) * dt + sig * np.sqrt(dt) * Z)
    payoff_temp = np.maximum(K - S_temp[:, -1], 0)
    prix_temp = np.mean(payoff_temp) * np.exp(-r * T)
    prix_sigma.append(prix_temp)

# Sauvegarde en JPEG
plt.figure(figsize=(8,6))
plt.plot(sigma_test, prix_sigma, 'bo-', linewidth=2)
plt.xlabel("Volatilité sigma")
plt.ylabel("Prix du put")
plt.title("Impact de la volatilité sur le prix du put européen")
plt.grid(True)
plt.savefig(os.path.join(dir_graph, "impact_sigma.jpeg"), dpi=90, bbox_inches='tight')
plt.close()

# 3b. Impact du taux r
r_test = np.arange(0.01, 0.11, 0.01)
prix_r = []

for rt in r_test:
    S_temp = np.zeros((N_sim, n_pas + 1))
    S_temp[:, 0] = S0
    for i in range(n_pas):
        Z = np.random.normal(0, 1, N_sim)
        S_temp[:, i+1] = S_temp[:, i] * np.exp((rt - sigma**2/2) * dt + sigma * np.sqrt(dt) * Z)
    payoff_temp = np.maximum(K - S_temp[:, -1], 0)
    prix_temp = np.mean(payoff_temp) * np.exp(-rt * T)
    prix_r.append(prix_temp)

plt.figure(figsize=(8,6))
plt.plot(r_test, prix_r, 'ro-', linewidth=2)
plt.xlabel("Taux sans risque r")
plt.ylabel("Prix du put")
plt.title("Impact du taux r sur le prix du put européen")
plt.grid(True)
plt.savefig(os.path.join(dir_graph, "impact_r.jpeg"), dpi=90, bbox_inches='tight')
plt.close()

# 3c. Impact du prix initial S0
S0_test = np.arange(80, 145, 5)
prix_S0 = []

for s0 in S0_test:
    S_temp = np.zeros((N_sim, n_pas + 1))
    S_temp[:, 0] = s0
    for i in range(n_pas):
        Z = np.random.normal(0, 1, N_sim)
        S_temp[:, i+1] = S_temp[:, i] * np.exp((r - sigma**2/2) * dt + sigma * np.sqrt(dt) * Z)
    payoff_temp = np.maximum(K - S_temp[:, -1], 0)
    prix_temp = np.mean(payoff_temp) * np.exp(-r * T)
    prix_S0.append(prix_temp)

plt.figure(figsize=(8,6))
plt.plot(S0_test, prix_S0, 'go-', linewidth=2, color='darkgreen')
plt.xlabel("Prix initial S(0)")
plt.ylabel("Prix du put")
plt.title("Impact de S(0) sur le prix du put européen")
plt.grid(True)
plt.savefig(os.path.join(dir_graph, "impact_S0.jpeg"), dpi=90, bbox_inches='tight')
plt.close()

# ---------------------------
# 4. Message de confirmation
# ---------------------------
print(f"Les graphiques ont été enregistrés dans le dossier : {dir_graph}")
print("Fichiers : impact_sigma.jpeg, impact_r.jpeg, impact_S0.jpeg")

# ============================================================================
# Question 1b : Prix théorique (Black-Scholes) et comparaison avec Monte-Carlo
# ============================================================================


from scipy.stats import norm


# ---------------------------
# 1. Calcul du prix Black-Scholes
# ---------------------------
d1 = (np.log(S0/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
P_bs = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# ---------------------------
# 2. Recalcul du prix Monte-Carlo (pas hebdomadaire, 50000 simulations)
#    (identique à la question 1a)
# ---------------------------
np.random.seed(123)
n_pas = 26
dt = T / n_pas
N_sim = 50000

# Simulation exacte du GBM
S = np.zeros((N_sim, n_pas + 1))
S[:, 0] = S0
for i in range(n_pas):
    Z = np.random.normal(0, 1, N_sim)
    S[:, i+1] = S[:, i] * np.exp((r - sigma**2/2) * dt + sigma * np.sqrt(dt) * Z)

ST = S[:, -1]
payoff = np.maximum(K - ST, 0)
P_mc = np.mean(payoff) * np.exp(-r * T)

# ---------------------------
# 3. Création du dossier "graphiques" 
# ---------------------------
dir_graph = "graphiques"
if not os.path.exists(dir_graph):
    os.makedirs(dir_graph)

# ==================== GRAPHIQUE EN POINTS ====================
# Ouvrir une figure et sauvegarder en JPEG
plt.figure(figsize=(8, 6))

# Tracer les deux prix sous forme de points
x_pos = [1, 2]
y_vals = [P_mc, P_bs]
plt.plot(x_pos[0], y_vals[0], 'o', color='blue', markersize=10)
plt.plot(x_pos[1], y_vals[1], 'o', color='red', markersize=10)

# Personnaliser l'axe des x
plt.xticks(x_pos, ['Monte-Carlo', 'Black-Scholes'])

# Zoom sur l'écart (petite marge)
y_min = min(P_mc, P_bs) - 0.05
y_max = max(P_mc, P_bs) + 0.05
plt.ylim(y_min, y_max)

# Tracer une ligne entre les deux points pour visualiser l'écart
plt.plot(x_pos, y_vals, '--', color='gray', linewidth=1)

# Ajouter les valeurs numériques au-dessus des points
plt.text(1, P_mc + 0.02, f"{P_mc:.4f}", ha='center', va='bottom', fontsize=10)
plt.text(2, P_bs + 0.02, f"{P_bs:.4f}", ha='center', va='bottom', fontsize=10)

# Labels et titre
plt.xlabel("Méthode")
plt.ylabel("Prix (euros)")
plt.title("Comparaison Monte-Carlo / Black-Scholes")
plt.grid(True, linestyle=':', alpha=0.7)

# Sauvegarde
plt.savefig(os.path.join(dir_graph, "comparaison_put_points.jpeg"),
            dpi=90, bbox_inches='tight')
plt.close()
# ===========================================================================

# ---------------------------
# 4. Affichage des résultats numériques dans la console
# ---------------------------
print(f"Prix Monte-Carlo (pas hebdo) : {P_mc:.4f} euros")
print(f"Prix Black-Scholes            : {P_bs:.4f} euros")
print(f"Écart absolu : {P_mc - P_bs:.4f} euros")
print(f"\nGraphique enregistré dans : {dir_graph}/comparaison_put_points.jpeg")


# ============================================================================
# Question 1c : Convergence du prix Monte-Carlo (put européen)
# Influence du nombre de simulations et du pas de temps (hebdo vs journalier)
# ============================================================================


# Prix théorique (Black-Scholes) - référence
d1 = (np.log(S0/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
P_bs = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

print(f"Prix théorique (Black-Scholes) : {P_bs:.4f} euros")

# ---------------------------
# 2. Choix des paramètres d'étude
# ---------------------------
N_sim_list = [1000, 5000, 10000, 50000, 100000, 500000]

# Pas de temps : hebdomadaire (26 pas) et journalier (180 pas sur 6 mois)
pas_list = {
    "hebdo": {"n_pas": 26, "nom": "Hebdomadaire (26 pas)"},
    "jour":  {"n_pas": 180, "nom": "Journalier (180 pas)"}
}

# ---------------------------
# 3. Fonction pour estimer le prix par Monte-Carlo
#    Discrétisation exacte du GBM
# ---------------------------
def estimer_put_mc(N_sim, n_pas, S0, K, r, sigma, T):
    dt = T / n_pas
    S = np.zeros((N_sim, n_pas + 1))
    S[:, 0] = S0
    for i in range(n_pas):
        Z = np.random.normal(0, 1, N_sim)
        S[:, i+1] = S[:, i] * np.exp((r - sigma**2/2) * dt + sigma * np.sqrt(dt) * Z)
    ST = S[:, -1]
    payoff = np.maximum(K - ST, 0)
    prix = np.mean(payoff) * np.exp(-r * T)
    return prix

# ---------------------------
# 4. Boucles de simulation pour chaque pas de temps et chaque N_sim
#    On stocke les prix estimés et on mesure l'erreur relative
# ---------------------------
np.random.seed(123)  # Pour reproductibilité globale

resultats = {}  # Dictionnaire pour stocker les dataframes (simulé avec listes)

for pas_nom, pas_info in pas_list.items():
    n_pas = pas_info["n_pas"]
    print(f"\nSimulation avec pas {pas_info['nom']} - n_pas = {n_pas}")
    
    prix_mc = []
    erreur_rel = []
    
    for N in N_sim_list:
        prix = estimer_put_mc(N, n_pas, S0, K, r, sigma, T)
        err = abs(prix - P_bs) / P_bs * 100
        prix_mc.append(prix)
        erreur_rel.append(err)
        print(f"  N = {N} -> Prix = {prix:.4f}  Erreur rel. = {err:.4f} %")
    
    resultats[pas_nom] = {
        "N_sim": np.array(N_sim_list),
        "prix_mc": np.array(prix_mc),
        "erreur_rel": np.array(erreur_rel)
    }

# ---------------------------
# 5. Création du dossier graphiques (s'il n'existe pas)
# ---------------------------
dir_graph = "graphiques"
if not os.path.exists(dir_graph):
    os.makedirs(dir_graph)

# ---------------------------
# 6. Graphiques de convergence
# ---------------------------

# 6a. Erreur relative en fonction de N_sim (échelle log-log)
plt.figure(figsize=(9, 6))
couleurs = {"hebdo": "blue", "jour": "red"}

# Tracer les deux courbes
for pas_nom, pas_info in pas_list.items():
    N_vals = resultats[pas_nom]["N_sim"]
    err_vals = resultats[pas_nom]["erreur_rel"]
    plt.loglog(N_vals, err_vals, 'o-', color=couleurs[pas_nom], linewidth=2,
               markersize=8, label=pas_info["nom"])

# Droite théorique de pente -1/2 (c / sqrt(N))
N_vals_th = np.array(N_sim_list)
ref_err = 1 / np.sqrt(N_vals_th)
# Ajustement du facteur en utilisant les valeurs du pas hebdo
facteur = np.mean(resultats["hebdo"]["erreur_rel"] / ref_err)
N_th = np.logspace(np.log10(N_sim_list[0]), np.log10(N_sim_list[-1]), 100)
err_th = facteur / np.sqrt(N_th)
plt.loglog(N_th, err_th, 'k--', linewidth=2, label="Théorie : c/√N")

plt.xlabel("Nombre de simulations N (échelle log)")
plt.ylabel("Erreur relative absolue (%) - échelle log")
plt.title("Convergence du prix Monte-Carlo vers Black-Scholes")
plt.legend()
plt.grid(True, which="both", linestyle=":", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(dir_graph, "convergence_erreur_relative.jpeg"), dpi=90)
plt.close()

# 6b. Prix estimé en fonction de N_sim (échelle log sur x, linéaire sur y)
plt.figure(figsize=(9, 6))
for pas_nom, pas_info in pas_list.items():
    N_vals = resultats[pas_nom]["N_sim"]
    prix_vals = resultats[pas_nom]["prix_mc"]
    plt.semilogx(N_vals, prix_vals, 'o-', color=couleurs[pas_nom], linewidth=2,
                 markersize=8, label=pas_info["nom"])

# Ligne horizontale au prix BS
plt.axhline(y=P_bs, color='black', linestyle='-', linewidth=2, label="Prix BS")
plt.xlim(min(N_sim_list), max(N_sim_list))
plt.ylim(P_bs - 0.2, P_bs + 0.2)
plt.xlabel("Nombre de simulations (échelle log)")
plt.ylabel("Prix estimé (euros)")
plt.title("Estimation Monte-Carlo du put européen vs nombre de simulations")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(dir_graph, "convergence_prix_estime.jpeg"), dpi=90)
plt.close()

# 6c. Graphique supplémentaire "convergence_visible" (superposition)
plt.figure(figsize=(9, 6))
# Points pour les deux pas
plt.loglog(resultats["hebdo"]["N_sim"], resultats["hebdo"]["erreur_rel"],
           'o', color='blue', markersize=8, label="Pas hebdomadaire (26)")
plt.loglog(resultats["jour"]["N_sim"], resultats["jour"]["erreur_rel"],
           's', color='red', markersize=8, label="Pas journalier (180)")
# Droite théorique
plt.loglog(N_th, err_th, 'k-', linewidth=2, label="Théorie : c/√N")
plt.xlabel("Nombre de simulations N (échelle log)")
plt.ylabel("Erreur relative absolue (%) - échelle log")
plt.title("Convergence Monte-Carlo du put européen\n(les deux pas de temps se superposent)")
plt.legend()
plt.grid(True, which="both", linestyle=":", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(dir_graph, "convergence_visible.jpeg"), dpi=90)
plt.close()

# ---------------------------
# 7. Boxplot pour N fixe (N = 100 000) avec répétitions
# ---------------------------
def repeter_estimation(N_sim, n_pas, n_rep=30):
    prix = np.zeros(n_rep)
    for k in range(n_rep):
        prix[k] = estimer_put_mc(N_sim, n_pas, S0, K, r, sigma, T)
    return prix

N_fixe = 100000
n_rep = 50
np.random.seed(456)
prix_rep_hebdo = repeter_estimation(N_fixe, pas_list["hebdo"]["n_pas"], n_rep)
prix_rep_jour  = repeter_estimation(N_fixe, pas_list["jour"]["n_pas"], n_rep)

plt.figure(figsize=(8, 6))
plt.boxplot([prix_rep_hebdo, prix_rep_jour], labels=["Pas hebdo", "Pas jour"])
plt.ylabel("Prix estimé (euros)")
plt.title(f"Distribution du prix estimé (N = {N_fixe} simulations)")
plt.axhline(y=P_bs, color='darkgreen', linestyle='--', linewidth=2, label="Prix BS")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(dir_graph, "boxplot_convergence.jpeg"), dpi=90)
plt.close()

# ---------------------------
# 8. Conclusion textuelle affichée dans la console
# ---------------------------
print("\n========== Analyse de convergence ==========")
print("Pour un put européen, la simulation exacte du GBM (discrétisation logarithmique)")
print("donne la loi parfaite de S(T) quel que soit le pas de temps. Par conséquent,")
print("le biais de discrétisation est nul. L'erreur provient uniquement de la variance")
print("statistique : elle décroît comme 1/sqrt(N). Les deux courbes (hebdo et jour)")
print("se superposent donc aux fluctuations d'échantillonnage près.")
print("On observe bien que l'erreur relative diminue quand N augmente, et que")
print("l'augmentation du pas de temps (de 26 à 180) ne change pas la précision.")
print("Graphiques sauvegardés :")
print("- convergence_visible.jpeg : montre que l'erreur relative suit la loi 1/sqrt(N)")
print("  et que les deux pas de temps donnent les mêmes performances (superposition).")
print("- boxplot_convergence.jpeg : pour N=100000, les deux distributions sont centrées")
print("  autour du prix BS (ligne pointillée) et ont la même variance.")
