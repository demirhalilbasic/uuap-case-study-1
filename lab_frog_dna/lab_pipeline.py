"""
Case Study I: Primijenjena analiza podataka o bioloskim sekvencama
Klasifikacija DNA sekvenci s k-merima (skup podataka o zabama)

Predmet: Uvod u analizu podataka 2025/2026

Vrste:
    - Bombina bombina (Crveno-trbusna mukacica) — porodica Bombinatoridae
    - Bombina variegata (Zuto-trbusna mukacica) — porodica Bombinatoridae
    - Rana temporaria (Europska smedja zaba) — porodica Ranidae

Hipoteza:
    Dvije sestrinske vrste (B. bombina i B. variegata) trebaju formirati zajednicki
    klaster u PCA prostoru zahvaljujuci visokoj genomskoj slicnosti, dok R. temporaria
    (druga porodica) treba biti jasno odvojena. K-mer frekvencijska analiza moze
    uhvatiti filogenetski signal cak i bez poravnanja sekvenci.
"""

# ============================================================================
# 1. IMPORTS I PODESAVANJA
# ============================================================================
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ne-interaktivni backend za cuvanje slika
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Putanja do podataka (relativno od lab_frog_dna/ direktorija)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapiranje naziva fajlova na citljive labels (bosanski)
SPECIES_LABELS = {
    "bombina_bombina":   "B. bombina\n(Crveno-trbusna)",
    "bombina_variegata": "B. variegata\n(Zuto-trbusna)",
    "rana_temporaria":   "R. temporaria\n(Europska smedja)",
}

# Kratke labels za classification report (bez newline)
SHORT_LABELS = {
    "bombina_bombina":   "B. bombina",
    "bombina_variegata": "B. variegata",
    "rana_temporaria":   "R. temporaria",
}


# ============================================================================
# 2. UCITAVANJE FASTA PODATAKA
# ============================================================================
def load_fasta(file_path):
    """Ucitava sequences iz FASTA fajla. Vraca listu sekvenci (stringova)."""
    sequences = []
    with open(file_path, "r") as f:
        seq = ""
        for line in f:
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                seq = ""
            else:
                seq += line.strip().upper()
        if seq:
            sequences.append(seq)
    return sequences


def fragment_sequence(sequence, window_size=1000, step=500):
    """
    Dijeli dugacku sekvencu na preklapajuce fragmente.
    Ovo je potrebno jer svaki FASTA fajl sadrzi samo jednu sekvencu
    (kompletni mitohondrijalni genom ~17kb), a za analizu i klasifikaciju
    trebamo vise uzoraka po vrsti.
    """
    fragments = []
    for i in range(0, len(sequence) - window_size + 1, step):
        fragment = sequence[i:i + window_size]
        # Preskoci fragmente koji sadrze nevalidne karaktere
        if all(c in "ACGT" for c in fragment):
            fragments.append(fragment)
    return fragments


# ============================================================================
# 3. UCITAVANJE DATASETA I OZNAKA (LABELA)
# ============================================================================
print("=" * 65)
print("  UCITAVANJE PODATAKA")
print("=" * 65)

sequences = []
labels = []

for filename in sorted(os.listdir(DATA_DIR)):
    if filename.endswith(".fasta"):
        species_name = filename.replace(".fasta", "")
        current_file_path = os.path.join(DATA_DIR, filename)

        loaded_sequences = load_fasta(current_file_path)
        print(f"  Fajl: {filename}")
        print(f"    Originalne sekvence: {len(loaded_sequences)}")
        print(f"    Duzina genoma: {len(loaded_sequences[0])} bp")

        # Fragmentacija genoma na manje dijelove
        all_fragments = []
        for seq in loaded_sequences:
            fragments = fragment_sequence(seq, window_size=1000, step=500)
            all_fragments.extend(fragments)

        print(f"    Generisani fragmenti: {len(all_fragments)}")
        sequences.extend(all_fragments)
        labels.extend([species_name] * len(all_fragments))

print(f"\nUkupno ucitano {len(sequences)} fragmenata iz {len(set(labels))} vrste")
print()


# ============================================================================
# 4. EKSTRAKCIJA K-MER ZNACAJKI (FEATURE ENGINEERING)
# ============================================================================
def kmer_frequency(sequence, k):
    """Broji sve k-mere u sekvenci i vraca Counter objekat."""
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    counts = Counter(kmers)
    return counts


def build_kmer_matrix(sequences, k):
    """
    Gradi matricu znacajki: svaki red je jedan fragment,
    svaki stupac je jedan k-mer, vrijednost je frekvencija.
    """
    all_kmers = set()
    kmer_counts = []

    for seq in sequences:
        counts = kmer_frequency(seq, k)
        kmer_counts.append(counts)
        all_kmers.update(counts.keys())

    all_kmers = sorted(all_kmers)
    X = np.zeros((len(sequences), len(all_kmers)))

    for i, counts in enumerate(kmer_counts):
        for j, kmer in enumerate(all_kmers):
            X[i, j] = counts.get(kmer, 0)

    return X


# ============================================================================
# 5. OSNOVNI PIPELINE (k=4)
# ============================================================================
print("=" * 65)
print("  OSNOVNI PIPELINE (k=4)")
print("=" * 65)

k = 4
X = build_kmer_matrix(sequences, k)
print(f"  Matrica znacajki: {X.shape[0]} uzoraka x {X.shape[1]} k-mera (k={k})")

# Standardizacija (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA redukcija na 2 komponente
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"  PCA objasnjena varijansa: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}")

# --- PCA vizualizacija ---
plt.figure(figsize=(9, 7))
unique_labels = sorted(set(labels))
colors = ["#e74c3c", "#f39c12", "#2ecc71"]  # crvena, zuta, zelena

for color_idx, lab in enumerate(unique_labels):
    indices = [i for i, l in enumerate(labels) if l == lab]
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                label=SPECIES_LABELS.get(lab, lab),
                color=colors[color_idx], alpha=0.7, edgecolors="black", linewidths=0.3, s=50)

plt.xlabel(f"Prva glavna komponenta (PC1) — {pca.explained_variance_ratio_[0]:.1%} varijanse",
           fontsize=11)
plt.ylabel(f"Druga glavna komponenta (PC2) — {pca.explained_variance_ratio_[1]:.1%} varijanse",
           fontsize=11)
plt.title("PCA vizualizacija k-mer znacajki DNK sekvenci zaba (k=4)", fontsize=13, fontweight="bold")
plt.legend(fontsize=10, title="Vrsta", title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_k4_osnovni.png"), dpi=150)
plt.close()
print("  Sacuvan grafikon: output/pca_k4_osnovni.png")

# --- Klasifikacija (Logisticka regresija) ---
label_map = {lab: i for i, lab in enumerate(unique_labels)}
y = np.array([label_map[l] for l in labels])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

class_names = [SHORT_LABELS.get(lab, lab) for lab in unique_labels]
print("\n  Rezultati klasifikacije (Logisticka regresija, k=4):")
print(classification_report(y_test, y_pred, target_names=class_names))

# Matrica konfuzije
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title("Matrica konfuzije — Logisticka regresija (k=4)", fontsize=12, fontweight="bold")
ax.set_xlabel("Predvidjena klasa", fontsize=11)
ax.set_ylabel("Stvarna klasa", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "matrica_konfuzije_lr_k4.png"), dpi=150)
plt.close()
print("  Sacuvan grafikon: output/matrica_konfuzije_lr_k4.png")


# ============================================================================
# 6. ZADATAK 1: Promjena k od 3 do 6 — poredenje PCA separacije i accuracies
# ============================================================================
print("\n" + "=" * 65)
print("  ZADATAK 1: Poredenje razlicitih vrijednosti k (3, 4, 5, 6)")
print("=" * 65)

fig_pca, axes_pca = plt.subplots(2, 2, figsize=(14, 12))
results_k = {}

for idx_k, k_val in enumerate([3, 4, 5, 6]):
    print(f"\n  --- k = {k_val} ---")
    X_k = build_kmer_matrix(sequences, k_val)
    print(f"  Dimenzije matrice: {X_k.shape}")

    # Skaliranje i PCA
    scaler_k = StandardScaler()
    X_k_scaled = scaler_k.fit_transform(X_k)

    pca_k = PCA(n_components=2)
    X_k_pca = pca_k.fit_transform(X_k_scaled)

    var_pc1 = pca_k.explained_variance_ratio_[0]
    var_pc2 = pca_k.explained_variance_ratio_[1]
    print(f"  PCA varijansa: PC1={var_pc1:.2%}, PC2={var_pc2:.2%}")

    # PCA grafikon
    ax = axes_pca[idx_k // 2][idx_k % 2]
    for color_idx, lab in enumerate(unique_labels):
        indices = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(X_k_pca[indices, 0], X_k_pca[indices, 1],
                   label=SHORT_LABELS.get(lab, lab),
                   color=colors[color_idx], alpha=0.7, edgecolors="black", linewidths=0.3, s=40)
    ax.set_xlabel(f"PC1 ({var_pc1:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_pc2:.1%})", fontsize=10)
    ax.set_title(f"k = {k_val}  (ukupno {X_k.shape[1]} znacajki)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Klasifikacija
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_k_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model_k = LogisticRegression(max_iter=1000, random_state=42)
    model_k.fit(X_tr, y_tr)
    y_pred_k = model_k.predict(X_te)
    accuracy = np.mean(y_pred_k == y_te)
    results_k[k_val] = accuracy
    print(f"  Tacnost klasifikatora: {accuracy:.2%}")
    print(classification_report(y_te, y_pred_k, target_names=class_names))

fig_pca.suptitle("Poredenje PCA vizualizacija za razlicite vrijednosti k",
                 fontsize=14, fontweight="bold", y=1.01)
fig_pca.tight_layout()
fig_pca.savefig(os.path.join(OUTPUT_DIR, "zadatak1_pca_poredenje.png"), dpi=150, bbox_inches="tight")
plt.close(fig_pca)
print("  Sacuvan grafikon: output/zadatak1_pca_poredenje.png")

# Grafikon accuracies po k
plt.figure(figsize=(8, 5))
k_values = list(results_k.keys())
accuracies = list(results_k.values())
plt.bar(k_values, [t * 100 for t in accuracies], color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"],
        edgecolor="black", linewidth=0.5)
plt.xlabel("Vrijednost k", fontsize=12)
plt.ylabel("Tacnost klasifikatora (%)", fontsize=12)
plt.title("Tacnost logisticke regresije za razlicite vrijednosti k", fontsize=13, fontweight="bold")
plt.xticks(k_values)
plt.ylim(0, 105)
for i, t in enumerate(accuracies):
    plt.text(k_values[i], t * 100 + 1.5, f"{t:.1%}", ha="center", fontsize=11, fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "zadatak1_tacnost_po_k.png"), dpi=150)
plt.close()
print("  Sacuvan grafikon: output/zadatak1_tacnost_po_k.png")

# Prikaz najboljeg k
best_k = max(results_k, key=results_k.get)
print(f"\n  Najbolji k = {best_k} sa tacnoscu {results_k[best_k]:.2%}")


# ============================================================================
# 7. ZADATAK 2: Uklanjanje jedne vrste — trening na dvije
# ============================================================================
print("\n" + "=" * 65)
print("  ZADATAK 2: Trening na samo dvije vrste (bez R. temporaria)")
print("=" * 65)

# Filtriranje: zadrzavamo samo Bombina vrste
indices_2species = [i for i, l in enumerate(labels)
                  if l in ("bombina_bombina", "bombina_variegata")]

sequences_2 = [sequences[i] for i in indices_2species]
labels_2 = [labels[i] for i in indices_2species]

unique_2 = sorted(set(labels_2))
map_2 = {lab: i for i, lab in enumerate(unique_2)}
y_2 = np.array([map_2[l] for l in labels_2])
names_2 = [SHORT_LABELS.get(lab, lab) for lab in unique_2]

# Poredenje k=4 i k=6 za dvije Bombina vrste
fig_z2, axes_z2 = plt.subplots(1, 2, figsize=(14, 6))
colors_2 = ["#e74c3c", "#f39c12"]
accuracy_2_by_k = {}

for idx_z2, k_z2 in enumerate([4, 6]):
    X_2 = build_kmer_matrix(sequences_2, k=k_z2)
    scaler_2 = StandardScaler()
    X_2_scaled = scaler_2.fit_transform(X_2)

    pca_2 = PCA(n_components=2)
    X_2_pca = pca_2.fit_transform(X_2_scaled)

    # PCA grafikon
    ax = axes_z2[idx_z2]
    for color_idx, lab in enumerate(unique_2):
        indices = [i for i, l in enumerate(labels_2) if l == lab]
        ax.scatter(X_2_pca[indices, 0], X_2_pca[indices, 1],
                   label=SHORT_LABELS.get(lab, lab),
                   color=colors_2[color_idx], alpha=0.7, edgecolors="black", linewidths=0.3, s=50)
    ax.set_xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Klasifikacija
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_2_scaled, y_2, test_size=0.2, random_state=42, stratify=y_2
    )
    model_2 = LogisticRegression(max_iter=1000, random_state=42)
    model_2.fit(X_tr2, y_tr2)
    y_pred_2 = model_2.predict(X_te2)
    accuracy_2 = np.mean(y_pred_2 == y_te2)
    accuracy_2_by_k[k_z2] = accuracy_2

    ax.set_title(f"k={k_z2} — accuracy: {accuracy_2:.0%}", fontsize=11, fontweight="bold")

    print(f"\n  --- Samo Bombina, k={k_z2} ---")
    print(f"  Tacnost: {accuracy_2:.2%}")
    print(classification_report(y_te2, y_pred_2, target_names=names_2))

fig_z2.suptitle("PCA — samo sestrinske vrste Bombina (bez R. temporaria)",
                fontsize=13, fontweight="bold")
fig_z2.tight_layout()
fig_z2.savefig(os.path.join(OUTPUT_DIR, "zadatak2_pca_2vrste.png"), dpi=150)
plt.close(fig_z2)
print("  Sacuvan grafikon: output/zadatak2_pca_2vrste.png")

# Matrica konfuzije za 2 vrste (k=6)
X_2_k6 = build_kmer_matrix(sequences_2, k=6)
X_2_k6_scaled = StandardScaler().fit_transform(X_2_k6)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_2_k6_scaled, y_2, test_size=0.2, random_state=42, stratify=y_2
)
model_2k6 = LogisticRegression(max_iter=1000, random_state=42)
model_2k6.fit(X_tr2, y_tr2)
y_pred_2k6 = model_2k6.predict(X_te2)

cm_2 = confusion_matrix(y_te2, y_pred_2k6)
fig, ax = plt.subplots(figsize=(6, 5))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=names_2)
disp2.plot(ax=ax, cmap="Oranges", colorbar=True)
ax.set_title("Matrica konfuzije — 2 Bombina vrste (k=6)", fontsize=12, fontweight="bold")
ax.set_xlabel("Predvidjena klasa", fontsize=11)
ax.set_ylabel("Stvarna klasa", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "zadatak2_matrica_konfuzije.png"), dpi=150)
plt.close()
print("  Sacuvan grafikon: output/zadatak2_matrica_konfuzije.png")


# ============================================================================
# 8. ZADATAK 3: SVM i Random Forest umjesto Logisticke regresije
# ============================================================================
print("\n" + "=" * 65)
print("  ZADATAK 3: Poredenje klasifikatora (LR vs SVM vs Random Forest)")
print("=" * 65)

# Koristimo najbolji k iz Zadatka 1 (k=6) za fer poredenje
k_z3 = best_k
X_z3 = build_kmer_matrix(sequences, k_z3)
scaler_z3 = StandardScaler()
X_z3_scaled = scaler_z3.fit_transform(X_z3)

X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
    X_z3_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logisticka regresija": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (linearni)":       SVC(kernel="linear", random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
}

model_results = {}
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))

for idx_m, (model_name, model) in enumerate(models.items()):
    print(f"\n  --- {model_name} (k={k_z3}) ---")
    model.fit(X_tr3, y_tr3)
    y_pred_m = model.predict(X_te3)
    accuracy_m = np.mean(y_pred_m == y_te3)
    model_results[model_name] = accuracy_m

    print(f"  Tacnost: {accuracy_m:.2%}")
    print(classification_report(y_te3, y_pred_m, target_names=class_names))

    # Matrica konfuzije
    cm_m = confusion_matrix(y_te3, y_pred_m)
    disp_m = ConfusionMatrixDisplay(confusion_matrix=cm_m, display_labels=class_names)
    disp_m.plot(ax=axes_cm[idx_m], cmap="Blues", colorbar=False)
    axes_cm[idx_m].set_title(f"{model_name}\n(accuracy: {accuracy_m:.1%})",
                             fontsize=11, fontweight="bold")
    axes_cm[idx_m].set_xlabel("Predvidjena klasa", fontsize=10)
    axes_cm[idx_m].set_ylabel("Stvarna klasa", fontsize=10)

fig_cm.suptitle(f"Poredenje matrica konfuzije — tri klasifikatora (k={k_z3})",
                fontsize=13, fontweight="bold")
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(OUTPUT_DIR, "zadatak3_poredenje_klasifikatora.png"), dpi=150)
plt.close(fig_cm)
print("  Sacuvan grafikon: output/zadatak3_poredenje_klasifikatora.png")

# Grafikon poredenja accuracies
plt.figure(figsize=(9, 5))
names_m = list(model_results.keys())
accuracies_m = [model_results[n] * 100 for n in names_m]
colors_m = ["#3498db", "#e74c3c", "#2ecc71"]
plt.barh(names_m, accuracies_m, color=colors_m, edgecolor="black", linewidth=0.5, height=0.5)
plt.xlabel("Tacnost (%)", fontsize=12)
plt.title(f"Poredenje accuracies klasifikatora (k={k_z3}, sve 3 vrste)",
          fontsize=13, fontweight="bold")
plt.xlim(0, 105)
for i, t in enumerate(accuracies_m):
    plt.text(t + 1, i, f"{t:.1f}%", va="center", fontsize=11, fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "zadatak3_tacnosti_barplot.png"), dpi=150)
plt.close()
print("  Sacuvan grafikon: output/zadatak3_tacnosti_barplot.png")


# ============================================================================
# 9. ZAVRSNI REZIME
# ============================================================================
print("\n" + "=" * 65)
print("  ZAVRSNI REZIME")
print("=" * 65)
print(f"""
  Analizirane vrste:
    1. Bombina bombina (Crveno-trbusna mukacica)
    2. Bombina variegata (Zuto-trbusna mukacica)
    3. Rana temporaria (Europska smedja zaba)

  Podaci: Kompletni mitohondrijalni genomi (NCBI)
  Fragmentacija: prozor=1000bp, korak=500bp
  Ukupno fragmenata: {len(sequences)}

  ZADATAK 1 — Najbolji k:
    k=3: {results_k[3]:.2%}  |  k=4: {results_k[4]:.2%}  |  k=5: {results_k[5]:.2%}  |  k=6: {results_k[6]:.2%}
    Najbolji: k={best_k} ({results_k[best_k]:.2%})

  ZADATAK 2 — Dvije sestrinske vrste (bez R. temporaria):
    Tacnost (k=4): {accuracy_2_by_k.get(4, 0):.2%}
    Tacnost (k=6): {accuracy_2_by_k.get(6, 0):.2%}

  ZADATAK 3 — Poredenje klasifikatora (k={best_k}):""")
for name, acc in model_results.items():
    print(f"    {name}: {acc:.2%}")

print(f"""
  Svi grafikoni sacuvani u: {OUTPUT_DIR}/
""")
print("=" * 65)
print("  PIPELINE ZAVRSEN USPJESNO")
print("=" * 65)
