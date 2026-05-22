import os
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats as scipy_stats

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTES DE ESTILO
# ─────────────────────────────────────────────────────────────────────
COLOR_BEST   = "#2ecc71"   # verde — mejor individuo
COLOR_MEAN   = "#3498db"   # azul  — media
COLOR_WORST  = "#e74c3c"   # rojo  — peor individuo
COLOR_STD    = "#9b59b6"   # morado— desv. estándar
COLOR_ALPHA  = "#f39c12"   # naranja — líder α
COLOR_BG     = "#1a1a2e"   # fondo oscuro
COLOR_PANEL  = "#16213e"
COLOR_GRID   = "#0f3460"

STYLE_PARAMS = dict(
    facecolor=COLOR_BG, edgecolor="none"
)

# ─────────────────────────────────────────────────────────────────────
# 1. CARGA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────

def preparar_datos(path: str = "data/Dry_Bean_Dataset.xlsx"):
    """
    Carga el dataset, codifica etiquetas y genera tres particiones

    La división es ESTRATIFICADA para mantener la distribución de clases.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró '{path}'. "
            "Coloca el archivo Dry_Bean_Dataset.xlsx dentro de la carpeta data/."
        )

    print(f"[DATA] Cargando dataset: {path} …")
    df = pd.read_excel(path)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_clases = len(le.classes_)
    print(f"[DATA] {X.shape[0]:,} registros | {X.shape[1]} descriptores | {n_clases} clases")
    print(f"[DATA] Clases: {list(le.classes_)}\n")

    # 80 % entrenamiento completo / 20 % prueba
    X_full, X_te, y_full, y_te = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    # 85 % entrenamiento interno / 15 % validación (sobre el 80 %)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_full, y_full, test_size=0.15, random_state=42, stratify=y_full
    )

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_te_s  = scaler.transform(X_te)
    X_full_s = scaler.fit_transform(X_full)   # re-entrena scaler sobre todo el split

    print(f"[DATA] Particiones → Entrenamiento: {len(X_tr):,} | "
          f"Validación: {len(X_va):,} | Prueba: {len(X_te):,}\n")

    return X_tr_s, X_va_s, X_te_s, X_full_s, y_tr, y_va, y_te, y_full, le


# ─────────────────────────────────────────────────────────────────────
# 2. DECODIFICADOR DE HIPERPARÁMETROS (espacio continuo [-1, 1]⁵)
# ─────────────────────────────────────────────────────────────────────

ACTIVACIONES = ["tanh", "relu", "logistic"]

def decodificar(pos: np.ndarray) -> dict:
    """
    Mapea un vector continuo de 5 dimensiones en [-1, 1]
    a hiperparámetros concretos del MLPClassifier.
    """
    n1 = max(10, int(10 + (pos[0] + 1) / 2 * 140))
    n2 = int((pos[1] + 1) / 2 * 80)
    capas = (n1, n2) if n2 >= 10 else (n1,)

    alpha = float(10 ** (-5 + (pos[2] + 1) / 2 * 4))
    lr    = float(10 ** (-4 + (pos[3] + 1) / 2 * 3))

    idx_act = int(np.clip((pos[4] + 1) / 2 * (len(ACTIVACIONES) - 1), 0, len(ACTIVACIONES) - 1))
    activacion = ACTIVACIONES[idx_act]

    return {
        "hidden_layer_sizes": capas,
        "alpha":              alpha,
        "learning_rate_init": lr,
        "activation":         activacion,
        "solver":             "adam",
        "max_iter":           200,
        "early_stopping":     True,
        "n_iter_no_change":   15,
        "random_state":       42,
    }


def evaluar_individuo(pos: np.ndarray, X_tr, y_tr, X_va, y_va) -> float:
    """Entrena un MLP con los hiperparámetros dados y devuelve accuracy en validación."""
    try:
        params = decodificar(pos)
        mlp = MLPClassifier(**params)
        mlp.fit(X_tr, y_tr)
        return accuracy_score(y_va, mlp.predict(X_va))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────
# 3. ALGORITMO HÍBRIDO AG-GWO
# ─────────────────────────────────────────────────────────────────────

class HibridoAGGWO:
    """
    Optimizador híbrido que combina:
    GWO  (Grey Wolf Optimizer)
    AG  (Algoritmo Genético)
    """

    def __init__(
        self,
        tam_pob:   int   = 15,
        max_iter:  int   = 20,
        pc:        float = 0.6,
        pm:        float = 0.20,
        sigma_mut: float = 0.12,
        dim:       int   = 5,
    ):
        self.tam_pob   = tam_pob
        self.max_iter  = max_iter
        self.pc        = pc
        self.pm        = pm
        self.sigma_mut = sigma_mut
        self.dim       = dim

    # ── Selección por torneo ──────────────────────────────────────────
    def _torneo(self, pob: np.ndarray, fits: list, k: int = 3) -> np.ndarray:
        idx = np.random.choice(len(pob), k, replace=False)
        ganador = idx[np.argmax([fits[i] for i in idx])]
        return pob[ganador].copy()

    # ── Cruce aritmético ─────────────────────────────────────────────
    def _cruce(self, p1: np.ndarray, p2: np.ndarray) -> tuple:
        if np.random.rand() < self.pc:
            alpha = np.random.rand(self.dim)
            h1 = alpha * p1 + (1 - alpha) * p2
            h2 = alpha * p2 + (1 - alpha) * p1
            return h1, h2
        return p1.copy(), p2.copy()

    # ── Mutación gaussiana ───────────────────────────────────────────
    def _mutar(self, ind: np.ndarray, t: int) -> np.ndarray:
        """Mutación adaptativa: σ decrece con las iteraciones."""
        sigma = self.sigma_mut * (1 - t / self.max_iter) + 0.02
        mascara = np.random.rand(self.dim) < self.pm
        ind[mascara] += np.random.normal(0, sigma, mascara.sum())
        return np.clip(ind, -1, 1)

    # ── Bucle principal ───────────────────────────────────────────────
    def ejecutar(self, X_tr, X_va, y_tr, y_va) -> tuple:
        """
        Devuelve:
          mejor_pos  : vector de posición del mejor individuo encontrado
          historial  : dict con series temporales para graficar
        """
        # Población inicial aleatoria
        pob = np.random.uniform(-1, 1, (self.tam_pob, self.dim))

        # Líderes GWO
        a_pos = np.zeros(self.dim); a_fit = -1.0   # Alpha
        b_pos = np.zeros(self.dim); b_fit = -1.0   # Beta
        d_pos = np.zeros(self.dim); d_fit = -1.0   # Delta

        historial = {
            "mejor":      [],   # fitness Alpha por iteración
            "media":      [],   # media de la población
            "mediana":    [],   # mediana
            "peor":       [],   # peor individuo
            "std":        [],   # desviación estándar
            "q1":         [],   # percentil 25
            "q3":         [],   # percentil 75
            "pob_fit":    [],   # lista completa de fits por iteración
            "diversidad": [],   # diversidad geométrica media de la población
            "tiempo_iter":[],   # tiempo de cada iteración
        }

        print("=" * 60)
        print(f"  AG-GWO | Población: {self.tam_pob} | Iteraciones: {self.max_iter}")
        print("=" * 60)

        for t in range(self.max_iter):
            t0 = time.time()

            # ── Evaluación de la población ────────────────────────────
            fits = [
                evaluar_individuo(pob[i], X_tr, y_tr, X_va, y_va)
                for i in range(self.tam_pob)
            ]

            # ── Actualizar líderes GWO ────────────────────────────────
            orden = np.argsort(fits)[::-1]  # mayor → menor
            if fits[orden[0]] > a_fit:
                a_fit, a_pos = fits[orden[0]], pob[orden[0]].copy()
            if len(orden) > 1 and fits[orden[1]] > b_fit:
                b_fit, b_pos = fits[orden[1]], pob[orden[1]].copy()
            if len(orden) > 2 and fits[orden[2]] > d_fit:
                d_fit, d_pos = fits[orden[2]], pob[orden[2]].copy()

            # ── Estadísticas de la iteración ──────────────────────────
            arr = np.array(fits)
            historial["mejor"].append(a_fit)
            historial["media"].append(arr.mean())
            historial["mediana"].append(np.median(arr))
            historial["peor"].append(arr.min())
            historial["std"].append(arr.std())
            historial["q1"].append(np.percentile(arr, 25))
            historial["q3"].append(np.percentile(arr, 75))
            historial["pob_fit"].append(fits.copy())
            historial["diversidad"].append(np.mean(np.std(pob, axis=0)))
            historial["tiempo_iter"].append(time.time() - t0)

            print(
                f"  Iter {t+1:>3}/{self.max_iter} | "
                f"α={a_fit:.4f} | Media={arr.mean():.4f} | "
                f"Std={arr.std():.4f} | Div={historial['diversidad'][-1]:.4f} | "
                f"T={historial['tiempo_iter'][-1]:.1f}s"
            )

            # ── Movimiento GWO ────────────────────────────────────────
            a_lin = 2 - 2 * (t / self.max_iter)   # decrece linealmente 2→0
            nueva_pob = np.empty_like(pob)

            for i in range(self.tam_pob):
                pos_nueva = np.empty(self.dim)
                for j in range(self.dim):
                    # --- Contribución de Alpha ---
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a_lin * r1 - a_lin
                    C1 = 2 * r2
                    D_a = abs(C1 * a_pos[j] - pob[i, j])
                    X1  = a_pos[j] - A1 * D_a

                    # --- Contribución de Beta ---
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a_lin * r1 - a_lin
                    C2 = 2 * r2
                    D_b = abs(C2 * b_pos[j] - pob[i, j])
                    X2  = b_pos[j] - A2 * D_b

                    # --- Contribución de Delta ---
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a_lin * r1 - a_lin
                    C3 = 2 * r2
                    D_d = abs(C3 * d_pos[j] - pob[i, j])
                    X3  = d_pos[j] - A3 * D_d

                    pos_nueva[j] = (X1 + X2 + X3) / 3.0

                nueva_pob[i] = np.clip(pos_nueva, -1, 1)

            # ── Operadores Genéticos ──────────────────────────────────
            # Cruce: generar descendencia entre pares seleccionados por torneo
            hijos = []
            while len(hijos) < self.tam_pob:
                p1 = self._torneo(nueva_pob, fits)
                p2 = self._torneo(nueva_pob, fits)
                h1, h2 = self._cruce(p1, p2)
                hijos.extend([h1, h2])
            hijos = np.array(hijos[: self.tam_pob])

            # Mutación adaptativa
            for i in range(self.tam_pob):
                hijos[i] = self._mutar(hijos[i], t)

            # Elitismo: preservar al Alpha en la nueva población
            pob = hijos
            pob[0] = a_pos.copy()

        print("=" * 60)
        print(f"  ÓPTIMO ENCONTRADO → Fitness α = {a_fit:.4f}")
        print("=" * 60)
        return a_pos, historial


# ─────────────────────────────────────────────────────────────────────
# 4. GENERACIÓN DE GRÁFICAS
# ─────────────────────────────────────────────────────────────────────

def _dark_axes(ax, title="", xlabel="", ylabel=""):
    """Aplica estilo oscuro homogéneo a un eje matplotlib."""
    ax.set_facecolor(COLOR_PANEL)
    ax.spines[:].set_color(COLOR_GRID)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.5, alpha=0.7)


def grafica_evolucion(historial: dict, out: str):
    """
    Figura 1 – Evolución del fitness por iteración
    """
    iters = list(range(1, len(historial["mejor"]) + 1))

    fig = plt.figure(figsize=(14, 5), facecolor=COLOR_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel izquierdo: evolución de fitness ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _dark_axes(ax1, "Evolución del Fitness por Iteración",
               "Iteración", "Accuracy (validación)")

    ax1.fill_between(iters, historial["q1"], historial["q3"],
                     color=COLOR_MEAN, alpha=0.15, label="IQR (Q1–Q3)")
    ax1.plot(iters, historial["mejor"],  color=COLOR_BEST,  lw=2,   marker="o",
             ms=5, label=f"Mejor (α)  final={historial['mejor'][-1]:.4f}")
    ax1.plot(iters, historial["media"],  color=COLOR_MEAN,  lw=1.5, marker="s",
             ms=4, linestyle="--", label=f"Media       final={historial['media'][-1]:.4f}")
    ax1.plot(iters, historial["mediana"],color=COLOR_ALPHA, lw=1.5, marker="^",
             ms=4, linestyle="-.", label=f"Mediana     final={historial['mediana'][-1]:.4f}")
    ax1.plot(iters, historial["peor"],   color=COLOR_WORST, lw=1,   marker="v",
             ms=4, linestyle=":", label=f"Peor        final={historial['peor'][-1]:.4f}")

    ax1.legend(loc="lower right", framealpha=0.3, labelcolor="white",
               facecolor=COLOR_PANEL, edgecolor=COLOR_GRID, fontsize=8)
    ax1.set_xticks(iters)

    # ── Panel derecho: std y diversidad ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _dark_axes(ax2, "Dispersión y Diversidad", "Iteración", "Valor")

    ax2.bar(iters, historial["std"], color=COLOR_STD, alpha=0.8,
            label="Std fitness", width=0.5)
    ax2_r = ax2.twinx()
    ax2_r.plot(iters, historial["diversidad"], color=COLOR_ALPHA,
               lw=2, marker="D", ms=4, label="Diversidad geom.")
    ax2_r.tick_params(colors="white", labelsize=8)
    ax2_r.yaxis.label.set_color("white")
    ax2_r.set_ylabel("Diversidad", fontsize=8, color="white")

    # Leyendas combinadas
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, framealpha=0.3, labelcolor="white",
               facecolor=COLOR_PANEL, edgecolor=COLOR_GRID, fontsize=7,
               loc="upper right")
    ax2.set_xticks(iters)

    plt.suptitle("AG-GWO: Ajuste de Hiperparámetros · Dry Bean Dataset",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"[PLOT] Guardada: {out}")


def grafica_boxplot(historial: dict, out: str):
    """
    Figura 2 – Distribución estadística de fitness por iteración (boxplot)
    Incluye: media, mediana, puntos individuales, y anotaciones de tendencia.
    """
    n_iter = len(historial["pob_fit"])
    iters  = list(range(1, n_iter + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLOR_BG)
    fig.subplots_adjust(wspace=0.35)

    # ── Boxplot con puntos superpuestos ───────────────────────────────
    ax = axes[0]
    _dark_axes(ax, "Distribución de Fitness por Iteración (Boxplot)",
               "Iteración", "Accuracy (validación)")

    bp = ax.boxplot(
        historial["pob_fit"],
        patch_artist=True,
        positions=iters,
        widths=0.6,
        medianprops=dict(color=COLOR_ALPHA, lw=2),
        whiskerprops=dict(color="white", lw=1),
        capprops=dict(color="white", lw=1.5),
        flierprops=dict(marker="x", color=COLOR_WORST, markersize=4),
    )
    for patch in bp["boxes"]:
        patch.set(facecolor=COLOR_MEAN, alpha=0.4, edgecolor="white", lw=0.8)

    # Puntos individuales con jitter
    for i, fits in enumerate(historial["pob_fit"]):
        jitter = np.random.normal(0, 0.08, len(fits))
        ax.scatter([iters[i]] * len(fits) + jitter, fits,
                   color="white", s=8, alpha=0.4, zorder=3)

    # Línea de medias
    ax.plot(iters, historial["media"], color=COLOR_MEAN, lw=1.5,
            linestyle="--", label="Media", zorder=5)
    ax.plot(iters, historial["mejor"], color=COLOR_BEST, lw=2,
            marker="*", ms=8, label="Mejor (α)", zorder=6)

    ax.legend(framealpha=0.3, labelcolor="white",
              facecolor=COLOR_PANEL, edgecolor=COLOR_GRID, fontsize=8)
    ax.set_xticks(iters)

    # ── Violin plot ───────────────────────────────────────────────────
    ax2 = axes[1]
    _dark_axes(ax2, "Violin Plot: Distribución Completa por Iteración",
               "Iteración", "Accuracy (validación)")

    vparts = ax2.violinplot(
        historial["pob_fit"],
        positions=iters,
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )
    for pc in vparts["bodies"]:
        pc.set_facecolor(COLOR_MEAN)
        pc.set_alpha(0.5)
        pc.set_edgecolor("white")
    for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if part in vparts:
            vparts[part].set_color("white")
            vparts[part].set_linewidth(1)
    vparts["cmeans"].set_color(COLOR_MEAN)
    vparts["cmedians"].set_color(COLOR_ALPHA)

    ax2.plot(iters, historial["mejor"], color=COLOR_BEST, lw=2,
             marker="*", ms=8, label="Mejor (α)", zorder=6)
    ax2.legend(framealpha=0.3, labelcolor="white",
               facecolor=COLOR_PANEL, edgecolor=COLOR_GRID, fontsize=8)
    ax2.set_xticks(iters)

    plt.suptitle("Comportamiento Estadístico de la Población · AG-GWO",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"[PLOT] Guardada: {out}")


def grafica_convergencia(historial: dict, out: str):
    """
    Figura 3 – Análisis de convergencia y tiempo de ejecución.
    Incluye: tasa de mejora por iteración, tiempo acumulado, heatmap de fitness.
    """
    n_iter = len(historial["mejor"])
    iters  = np.arange(1, n_iter + 1)

    # Tasa de mejora (∆ entre iteraciones consecutivas)
    mejoras = np.diff([0] + historial["mejor"])

    fig = plt.figure(figsize=(14, 8), facecolor=COLOR_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Tasa de mejora ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_axes(ax1, "Tasa de Mejora por Iteración", "Iteración", "∆ Accuracy")
    bars = ax1.bar(iters, mejoras, color=[
        COLOR_BEST if m > 0 else COLOR_WORST for m in mejoras
    ], alpha=0.85, edgecolor=COLOR_BG, linewidth=0.5)
    ax1.axhline(0, color="white", lw=0.8, linestyle="--")
    ax1.set_xticks(iters)

    # ── Panel 2: Tiempo por iteración ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_axes(ax2, "Tiempo de Ejecución por Iteración", "Iteración", "Segundos")
    ax2.fill_between(iters, 0, historial["tiempo_iter"],
                     color=COLOR_ALPHA, alpha=0.3)
    ax2.plot(iters, historial["tiempo_iter"], color=COLOR_ALPHA,
             lw=2, marker="o", ms=5)
    tiempo_acum = np.cumsum(historial["tiempo_iter"])
    ax2r = ax2.twinx()
    ax2r.plot(iters, tiempo_acum, color=COLOR_MEAN, lw=1.5,
              linestyle="--", label="Acumulado")
    ax2r.tick_params(colors="white", labelsize=8)
    ax2r.yaxis.label.set_color("white")
    ax2r.set_ylabel("Tiempo acumulado (s)", color="white", fontsize=8)
    ax2.set_xticks(iters)

    # ── Panel 3: Heatmap de fitness ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    _dark_axes(ax3, "Mapa de Calor: Fitness de cada Individuo por Iteración",
               "Iteración", "Individuo #")

    # Pad o recortar a matriz regular
    n_ind = max(len(f) for f in historial["pob_fit"])
    mat = np.full((n_ind, n_iter), np.nan)
    for j, fits in enumerate(historial["pob_fit"]):
        mat[:len(fits), j] = fits

    im = ax3.imshow(
        mat, aspect="auto", cmap="RdYlGn",
        origin="lower", vmin=0, vmax=1,
        extent=[0.5, n_iter + 0.5, 0.5, n_ind + 0.5]
    )
    cbar = fig.colorbar(im, ax=ax3, fraction=0.02, pad=0.01)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
    cbar.ax.set_ylabel("Accuracy", color="white", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax3.set_xticks(iters)
    ax3.set_yticks(range(1, n_ind + 1))

    plt.suptitle("Análisis de Convergencia y Rendimiento · AG-GWO",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"[PLOT] Guardada: {out}")


def grafica_tendencia_central(historial: dict, out: str):
    """
    Figura 4 – Análisis estadístico profundo de tendencia central.
    Panel de 6 métricas: media, mediana, std, IQR, asimetría, curtosis.
    """
    n_iter = len(historial["pob_fit"])
    iters  = list(range(1, n_iter + 1))

    # Calcular momentos estadísticos por iteración
    asimetria = [scipy_stats.skew(f)     for f in historial["pob_fit"]]
    curtosis   = [scipy_stats.kurtosis(f) for f in historial["pob_fit"]]
    iqr        = [q3 - q1 for q3, q1 in zip(historial["q3"], historial["q1"])]

    metrics = [
        ("Media",     historial["media"],   COLOR_MEAN,  "Accuracy promedio de la población"),
        ("Mediana",   historial["mediana"], COLOR_ALPHA, "Valor central resistente a outliers"),
        ("Std",       historial["std"],     COLOR_STD,   "Dispersión de la población"),
        ("IQR",       iqr,                  COLOR_BEST,  "Rango intercuartílico (Q3-Q1)"),
        ("Asimetría", asimetria,            COLOR_WORST, "Sesgo de la distribución"),
        ("Curtosis",  curtosis,             "#1abc9c",   "Apuntamiento de la distribución"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=COLOR_BG)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for ax, (nombre, datos, color, desc) in zip(axes.flatten(), metrics):
        _dark_axes(ax, nombre, "Iteración", nombre)
        ax.set_title(f"{nombre}\n{desc}", color="white", fontsize=9,
                     fontweight="bold", pad=4)

        ax.fill_between(iters, 0, datos, color=color, alpha=0.2)
        ax.plot(iters, datos, color=color, lw=2, marker="o", ms=5)
        ax.set_xticks(iters)

        # Línea de regresión lineal
        z = np.polyfit(iters, datos, 1)
        p = np.poly1d(z)
        ax.plot(iters, p(iters), color="white", lw=1, linestyle="--",
                alpha=0.5, label=f"Tendencia (slope={z[0]:.4f})")
        ax.legend(framealpha=0.3, labelcolor="white",
                  facecolor=COLOR_PANEL, edgecolor=COLOR_GRID, fontsize=7)

    plt.suptitle("Estadísticas de Tendencia Central · Población AG-GWO",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"[PLOT] Guardada: {out}")


def grafica_matriz_confusion(y_te, y_pred, clases, out: str):
    """Figura 5 – Matriz de confusión del modelo final."""
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_PANEL)

    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=clases)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")

    disp.im_.set_clim(0, cm.max())
    ax.set_title("Matriz de Confusión — Modelo Final (conjunto de prueba)",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white")
    plt.setp(ax.get_yticklabels(), color="white")

    cbar = disp.im_.colorbar
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"[PLOT] Guardada: {out}")


def grafica_resumen_hiperparams(historial: dict, mejor_params: dict, out: str):
    """
    Figura 6 – Tabla resumen de los mejores hiperparámetros y
               tabla de estadísticas finales del proceso.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=COLOR_BG)

    for ax in axes:
        ax.set_facecolor(COLOR_PANEL)
        ax.axis("off")

    # ── Tabla de hiperparámetros ──────────────────────────────────────
    hp = mejor_params
    capas = str(hp["hidden_layer_sizes"])
    datos_hp = [
        ["Parámetro",            "Valor"],
        ["Arquitectura (capas)", capas],
        ["Alpha (L2)",           f"{hp['alpha']:.2e}"],
        ["Learning rate",        f"{hp['learning_rate_init']:.2e}"],
        ["Activación",           hp["activation"]],
        ["Solver",               hp["solver"]],
        ["Max iter.",            str(hp["max_iter"])],
        ["Early stopping",       str(hp["early_stopping"])],
    ]

    tab1 = axes[0].table(
        cellText=datos_hp[1:], colLabels=datos_hp[0],
        loc="center", cellLoc="center"
    )
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(10)
    tab1.scale(1, 1.6)
    for (r, c), cell in tab1.get_celld().items():
        cell.set_facecolor(COLOR_BG if r == 0 else COLOR_PANEL)
        cell.set_text_props(color="white")
        cell.set_edgecolor(COLOR_GRID)
    axes[0].set_title("Mejores Hiperparámetros Encontrados",
                      color="white", fontsize=11, fontweight="bold", pad=10)

    # ── Tabla de estadísticas del proceso ────────────────────────────
    n_eval = len(historial["pob_fit"]) * len(historial["pob_fit"][0])
    datos_est = [
        ["Métrica",                  "Valor"],
        ["Mejor fitness (α)",        f"{historial['mejor'][-1]:.4f}"],
        ["Media final población",    f"{historial['media'][-1]:.4f}"],
        ["Mediana final población",  f"{historial['mediana'][-1]:.4f}"],
        ["Std final población",      f"{historial['std'][-1]:.4f}"],
        ["IQR final",                f"{historial['q3'][-1]-historial['q1'][-1]:.4f}"],
        ["Total evaluaciones MLP",   str(n_eval)],
        ["Tiempo total (s)",         f"{sum(historial['tiempo_iter']):.1f}"],
    ]

    tab2 = axes[1].table(
        cellText=datos_est[1:], colLabels=datos_est[0],
        loc="center", cellLoc="center"
    )
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(10)
    tab2.scale(1, 1.6)
    for (r, c), cell in tab2.get_celld().items():
        cell.set_facecolor(COLOR_BG if r == 0 else COLOR_PANEL)
        cell.set_text_props(color="white")
        cell.set_edgecolor(COLOR_GRID)
    axes[1].set_title("Estadísticas del Proceso de Optimización",
                      color="white", fontsize=11, fontweight="bold", pad=10)

    plt.suptitle("Resumen Final · AG-GWO sobre Dry Bean Dataset",
                 color="white", fontsize=13, fontweight="bold", y=1.04)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"[PLOT] Guardada: {out}")


# ─────────────────────────────────────────────────────────────────────
# 5. EXPORTACIÓN DE DATOS EN BRUTO A CSV
# ─────────────────────────────────────────────────────────────────────

def exportar_csv(historial: dict, mejor_params: dict,
                 y_te, y_pred, clases, acc_final: float,
                 out_dir: str):
    """
    Genera tres archivos CSV con todos los datos en bruto usados en las gráficas:
    """
    from scipy import stats as scipy_stats

    # ── CSV 1: serie temporal por iteración ──────────────────────────
    n_iter = len(historial["mejor"])
    iters  = list(range(1, n_iter + 1))

    asimetria  = [scipy_stats.skew(f)     for f in historial["pob_fit"]]
    curtosis   = [scipy_stats.kurtosis(f) for f in historial["pob_fit"]]
    iqr        = [q3 - q1 for q3, q1 in zip(historial["q3"], historial["q1"])]
    mejoras    = list(np.diff([0.0] + historial["mejor"]))
    tiempo_ac  = list(np.cumsum(historial["tiempo_iter"]))

    df_iter = pd.DataFrame({
        "iteracion":          iters,
        "fitness_mejor_alpha":historial["mejor"],
        "fitness_media":      historial["media"],
        "fitness_mediana":    historial["mediana"],
        "fitness_peor":       historial["peor"],
        "fitness_std":        historial["std"],
        "percentil_25_q1":    historial["q1"],
        "percentil_75_q3":    historial["q3"],
        "iqr_q3_menos_q1":    iqr,
        "asimetria_skew":     asimetria,
        "curtosis_kurtosis":  curtosis,
        "diversidad_geom":    historial["diversidad"],
        "tiempo_iter_s":      historial["tiempo_iter"],
        "tiempo_acumulado_s": tiempo_ac,
        "delta_mejora_alpha": mejoras,
    })
    ruta1 = f"{out_dir}/07_serie_iteraciones.csv"
    df_iter.to_csv(ruta1, index=False, float_format="%.6f")
    print(f"[CSV]  Guardado: {ruta1}  ({len(df_iter)} filas × {len(df_iter.columns)} cols)")

    # ── CSV 2: fitness de cada individuo por iteración ────────────────
    registros = []
    for i, fits in enumerate(historial["pob_fit"], start=1):
        for j, f in enumerate(fits, start=1):
            registros.append({"iteracion": i, "individuo": j, "fitness": f})

    df_ind = pd.DataFrame(registros)
    ruta2 = f"{out_dir}/08_fitness_individuos.csv"
    df_ind.to_csv(ruta2, index=False, float_format="%.6f")
    print(f"[CSV]  Guardado: {ruta2}  ({len(df_ind)} filas × {len(df_ind.columns)} cols)")

    # ── CSV 3: hiperparámetros óptimos + reporte por clase ────────────
    # Bloque 1: hiperparámetros
    hp = mejor_params
    filas_hp = [
        {"seccion": "hiperparametros", "campo": "arquitectura_capas",
         "valor": str(hp["hidden_layer_sizes"])},
        {"seccion": "hiperparametros", "campo": "alpha_regularizacion",
         "valor": f"{hp['alpha']:.8f}"},
        {"seccion": "hiperparametros", "campo": "learning_rate_init",
         "valor": f"{hp['learning_rate_init']:.8f}"},
        {"seccion": "hiperparametros", "campo": "activacion",
         "valor": hp["activation"]},
        {"seccion": "hiperparametros", "campo": "solver",
         "valor": hp["solver"]},
        {"seccion": "hiperparametros", "campo": "max_iter",
         "valor": str(hp["max_iter"])},
        {"seccion": "hiperparametros", "campo": "early_stopping",
         "valor": str(hp["early_stopping"])},
        {"seccion": "proceso",         "campo": "fitness_alpha_final",
         "valor": f"{historial['mejor'][-1]:.6f}"},
        {"seccion": "proceso",         "campo": "media_final_poblacion",
         "valor": f"{historial['media'][-1]:.6f}"},
        {"seccion": "proceso",         "campo": "std_final_poblacion",
         "valor": f"{historial['std'][-1]:.6f}"},
        {"seccion": "proceso",         "campo": "total_evaluaciones_mlp",
         "valor": str(n_iter * len(historial["pob_fit"][0]))},
        {"seccion": "proceso",         "campo": "tiempo_total_s",
         "valor": f"{sum(historial['tiempo_iter']):.2f}"},
        {"seccion": "evaluacion_final","campo": "accuracy_prueba",
         "valor": f"{acc_final:.6f}"},
    ]

    # Bloque 2: métricas por clase (precision, recall, f1, support)
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_te, y_pred, labels=range(len(clases))
    )
    for idx, cls in enumerate(clases):
        filas_hp.append({
            "seccion": "reporte_por_clase",
            "campo":   f"{cls}_precision",
            "valor":   f"{prec[idx]:.6f}",
        })
        filas_hp.append({
            "seccion": "reporte_por_clase",
            "campo":   f"{cls}_recall",
            "valor":   f"{rec[idx]:.6f}",
        })
        filas_hp.append({
            "seccion": "reporte_por_clase",
            "campo":   f"{cls}_f1_score",
            "valor":   f"{f1[idx]:.6f}",
        })
        filas_hp.append({
            "seccion": "reporte_por_clase",
            "campo":   f"{cls}_support",
            "valor":   str(int(sup[idx])),
        })

    df_res = pd.DataFrame(filas_hp)
    ruta3 = f"{out_dir}/09_resultados_finales.csv"
    df_res.to_csv(ruta3, index=False)
    print(f"[CSV]  Guardado: {ruta3}  ({len(df_res)} filas × {len(df_res.columns)} cols)")


# ─────────────────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Paso 1: Datos ─────────────────────────────────────────────────
    (X_tr, X_va, X_te, X_full,
     y_tr, y_va, y_te, y_full, le) = preparar_datos()

    # ── Paso 2: Optimización ──────────────────────────────────────────
    print("\n[OPT] Iniciando optimización AG-GWO …\n")
    optimizador = HibridoAGGWO(
        tam_pob   = 15,
        max_iter  = 20,
        pc        = 0.60,
        pm        = 0.20,
        sigma_mut = 0.12,
    )
    mejor_pos, historial = optimizador.ejecutar(X_tr, X_va, y_tr, y_va)

    # ── Paso 3: Modelo final ──────────────────────────────────────────
    mejores_params = decodificar(mejor_pos)
    print(f"\n[MODEL] Hiperparámetros óptimos: {mejores_params}")

    print("[MODEL] Entrenando modelo final sobre conjunto completo de entrenamiento …")
    mlp_final = MLPClassifier(**mejores_params)
    mlp_final.fit(X_full, y_full)
    y_pred = mlp_final.predict(X_te)

    print("\n" + "=" * 60)
    print("  REPORTE FINAL DE CLASIFICACIÓN")
    print("=" * 60)
    print(classification_report(y_te, y_pred, target_names=le.classes_))
    acc_final = accuracy_score(y_te, y_pred)
    print(f"  Accuracy en prueba: {acc_final:.4f}")
    print("=" * 60)

    # ── Paso 4: Gráficas ──────────────────────────────────────────────
    print("\n[PLOT] Generando gráficas …\n")

    grafica_evolucion(historial,
        f"{OUT_DIR}/01_evolucion_fitness.png")

    grafica_boxplot(historial,
        f"{OUT_DIR}/02_distribucion_boxplot.png")

    grafica_convergencia(historial,
        f"{OUT_DIR}/03_convergencia_heatmap.png")

    grafica_tendencia_central(historial,
        f"{OUT_DIR}/04_tendencia_central.png")

    grafica_matriz_confusion(y_te, y_pred, le.classes_,
        f"{OUT_DIR}/05_matriz_confusion.png")

    grafica_resumen_hiperparams(historial, mejores_params,
        f"{OUT_DIR}/06_resumen_final.png")

    # ── Paso 5: Exportar CSV ──────────────────────────────────────────
    print("\n[CSV] Exportando datos en bruto ...\n")
    exportar_csv(historial, mejores_params, y_te, y_pred,
                 le.classes_, acc_final, OUT_DIR)

    print(f"\n[DONE] Graficas y CSVs guardados en '{OUT_DIR}/'")
    print(f"[DONE] Accuracy final en prueba: {acc_final:.4f}")