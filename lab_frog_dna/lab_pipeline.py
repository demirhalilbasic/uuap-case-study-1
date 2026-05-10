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

# Mapiranje naziva fajlova na citljive labele (bosanski)
LABELE_VRSTA = {
    "bombina_bombina":   "B. bombina\n(Crveno-trbusna)",
    "bombina_variegata": "B. variegata\n(Zuto-trbusna)",
    "rana_temporaria":   "R. temporaria\n(Europska smedja)",
}

# Kratke labele za classification report (bez newline)
KRATKE_LABELE = {
    "bombina_bombina":   "B. bombina",
    "bombina_variegata": "B. variegata",
    "rana_temporaria":   "R. temporaria",
}


# ============================================================================
# 2. UCITAVANJE FASTA PODATAKA
# ============================================================================
def ucitaj_fasta(putanja_fajla):
    """Ucitava sekvence iz FASTA fajla. Vraca listu sekvenci (stringova)."""
    sekvence = []
    with open(putanja_fajla, "r") as f:
        seq = ""
        for linija in f:
            if linija.startswith(">"):
                if seq:
                    sekvence.append(seq)
                seq = ""
            else:
                seq += linija.strip().upper()
        if seq:
            sekvence.append(seq)
    return sekvence


def fragmentiraj_sekvencu(sekvenca, velicina_prozora=1000, korak=500):
    """
    Dijeli dugacku sekvencu na preklapajuce fragmente.
    Ovo je potrebno jer svaki FASTA fajl sadrzi samo jednu sekvencu
    (kompletni mitohondrijalni genom ~17kb), a za analizu i klasifikaciju
    trebamo vise uzoraka po vrsti.
    """
    fragmenti = []
    for i in range(0, len(sekvenca) - velicina_prozora + 1, korak):
        fragment = sekvenca[i:i + velicina_prozora]
        # Preskoci fragmente koji sadrze nevalidne karaktere
        if all(c in "ACGT" for c in fragment):
            fragmenti.append(fragment)
    return fragmenti


# ============================================================================
# 3. UCITAVANJE DATASETA I OZNAKA (LABELA)
# ============================================================================
print("=" * 65)
print("  UCITAVANJE PODATAKA")
print("=" * 65)

sekvence = []
labele = []

for naziv_fajla in sorted(os.listdir(DATA_DIR)):
    if naziv_fajla.endswith(".fasta"):
        naziv_vrste = naziv_fajla.replace(".fasta", "")
        putanja = os.path.join(DATA_DIR, naziv_fajla)

        ucitane_sekvence = ucitaj_fasta(putanja)
        print(f"  Fajl: {naziv_fajla}")
        print(f"    Originalne sekvence: {len(ucitane_sekvence)}")
        print(f"    Duzina genoma: {len(ucitane_sekvence[0])} bp")

        # Fragmentacija genoma na manje dijelove
        svi_fragmenti = []
        for seq in ucitane_sekvence:
            fragmenti = fragmentiraj_sekvencu(seq, velicina_prozora=1000, korak=500)
            svi_fragmenti.extend(fragmenti)

        print(f"    Generisani fragmenti: {len(svi_fragmenti)}")
        sekvence.extend(svi_fragmenti)
        labele.extend([naziv_vrste] * len(svi_fragmenti))

print(f"\nUkupno ucitano {len(sekvence)} fragmenata iz {len(set(labele))} vrste")
print()


# ============================================================================
# 4. EKSTRAKCIJA K-MER ZNACAJKI (FEATURE ENGINEERING)
# ============================================================================
def kmer_frekvencija(sekvenca, k):
    """Broji sve k-mere u sekvenci i vraca Counter objekat."""
    kmeri = [sekvenca[i:i+k] for i in range(len(sekvenca) - k + 1)]
    brojac = Counter(kmeri)
    return brojac


def izgradi_kmer_matricu(sekvence, k):
    """
    Gradi matricu znacajki: svaki red je jedan fragment,
    svaki stupac je jedan k-mer, vrijednost je frekvencija.
    """
    svi_kmeri = set()
    kmer_brojaci = []

    for seq in sekvence:
        brojac = kmer_frekvencija(seq, k)
        kmer_brojaci.append(brojac)
        svi_kmeri.update(brojac.keys())

    svi_kmeri = sorted(svi_kmeri)
    X = np.zeros((len(sekvence), len(svi_kmeri)))

    for i, brojac in enumerate(kmer_brojaci):
        for j, kmer in enumerate(svi_kmeri):
            X[i, j] = brojac.get(kmer, 0)

    return X


# ============================================================================
# 5. OSNOVNI PIPELINE (k=4)
# ============================================================================
print("=" * 65)
print("  OSNOVNI PIPELINE (k=4)")
print("=" * 65)

k = 4
X = izgradi_kmer_matricu(sekvence, k)
print(f"  Matrica znacajki: {X.shape[0]} uzoraka x {X.shape[1]} k-mera (k={k})")

# Standardizacija (StandardScaler)
skaler = StandardScaler()
X_skalirano = skaler.fit_transform(X)

# PCA redukcija na 2 komponente
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_skalirano)

print(f"  PCA objasnjena varijansa: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}")

# --- PCA vizualizacija ---
plt.figure(figsize=(9, 7))
jedinstvene_labele = sorted(set(labele))
boje = ["#e74c3c", "#f39c12", "#2ecc71"]  # crvena, zuta, zelena

for idx_boje, lab in enumerate(jedinstvene_labele):
    indeksi = [i for i, l in enumerate(labele) if l == lab]
    plt.scatter(X_pca[indeksi, 0], X_pca[indeksi, 1],
                label=LABELE_VRSTA.get(lab, lab),
                color=boje[idx_boje], alpha=0.7, edgecolors="black", linewidths=0.3, s=50)

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
mapa_labela = {lab: i for i, lab in enumerate(jedinstvene_labele)}
y = np.array([mapa_labela[l] for l in labele])

X_trening, X_test, y_trening, y_test = train_test_split(
    X_skalirano, y, test_size=0.2, random_state=42, stratify=y
)

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_trening, y_trening)
y_pred = model_lr.predict(X_test)

nazivi_klasa = [KRATKE_LABELE.get(lab, lab) for lab in jedinstvene_labele]
print("\n  Rezultati klasifikacije (Logisticka regresija, k=4):")
print(classification_report(y_test, y_pred, target_names=nazivi_klasa))

# Matrica konfuzije
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nazivi_klasa)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title("Matrica konfuzije — Logisticka regresija (k=4)", fontsize=12, fontweight="bold")
ax.set_xlabel("Predvidjena klasa", fontsize=11)
ax.set_ylabel("Stvarna klasa", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "matrica_konfuzije_lr_k4.png"), dpi=150)
plt.close()
print("  Sacuvan grafikon: output/matrica_konfuzije_lr_k4.png")


# ============================================================================
# 6. ZADATAK 1: Promjena k od 3 do 6 — poredenje PCA separacije i tacnosti
# ============================================================================
print("\n" + "=" * 65)
print("  ZADATAK 1: Poredenje razlicitih vrijednosti k (3, 4, 5, 6)")
print("=" * 65)

fig_pca, axes_pca = plt.subplots(2, 2, figsize=(14, 12))
rezultati_k = {}

for idx_k, k_val in enumerate([3, 4, 5, 6]):
    print(f"\n  --- k = {k_val} ---")
    X_k = izgradi_kmer_matricu(sekvence, k_val)
    print(f"  Dimenzije matrice: {X_k.shape}")

    # Skaliranje i PCA
    skaler_k = StandardScaler()
    X_k_skal = skaler_k.fit_transform(X_k)

    pca_k = PCA(n_components=2)
    X_k_pca = pca_k.fit_transform(X_k_skal)

    var_pc1 = pca_k.explained_variance_ratio_[0]
    var_pc2 = pca_k.explained_variance_ratio_[1]
    print(f"  PCA varijansa: PC1={var_pc1:.2%}, PC2={var_pc2:.2%}")

    # PCA grafikon
    ax = axes_pca[idx_k // 2][idx_k % 2]
    for idx_boje, lab in enumerate(jedinstvene_labele):
        indeksi = [i for i, l in enumerate(labele) if l == lab]
        ax.scatter(X_k_pca[indeksi, 0], X_k_pca[indeksi, 1],
                   label=KRATKE_LABELE.get(lab, lab),
                   color=boje[idx_boje], alpha=0.7, edgecolors="black", linewidths=0.3, s=40)
    ax.set_xlabel(f"PC1 ({var_pc1:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_pc2:.1%})", fontsize=10)
    ax.set_title(f"k = {k_val}  (ukupno {X_k.shape[1]} znacajki)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Klasifikacija
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_k_skal, y, test_size=0.2, random_state=42, stratify=y
    )
    model_k = LogisticRegression(max_iter=1000, random_state=42)
    model_k.fit(X_tr, y_tr)
    y_pred_k = model_k.predict(X_te)
    tacnost = np.mean(y_pred_k == y_te)
    rezultati_k[k_val] = tacnost
    print(f"  Tacnost klasifikatora: {tacnost:.2%}")
    print(classification_report(y_te, y_pred_k, target_names=nazivi_klasa))

fig_pca.suptitle("Poredenje PCA vizualizacija za razlicite vrijednosti k",
                 fontsize=14, fontweight="bold", y=1.01)
fig_pca.tight_layout()
fig_pca.savefig(os.path.join(OUTPUT_DIR, "zadatak1_pca_poredenje.png"), dpi=150, bbox_inches="tight")
plt.close(fig_pca)
print("  Sacuvan grafikon: output/zadatak1_pca_poredenje.png")

# Grafikon tacnosti po k
plt.figure(figsize=(8, 5))
k_vrijednosti = list(rezultati_k.keys())
tacnosti = list(rezultati_k.values())
plt.bar(k_vrijednosti, [t * 100 for t in tacnosti], color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"],
        edgecolor="black", linewidth=0.5)
plt.xlabel("Vrijednost k", fontsize=12)
plt.ylabel("Tacnost klasifikatora (%)", fontsize=12)
plt.title("Tacnost logisticke regresije za razlicite vrijednosti k", fontsize=13, fontweight="bold")
plt.xticks(k_vrijednosti)
plt.ylim(0, 105)
for i, t in enumerate(tacnosti):
    plt.text(k_vrijednosti[i], t * 100 + 1.5, f"{t:.1%}", ha="center", fontsize=11, fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "zadatak1_tacnost_po_k.png"), dpi=150)
plt.close()
print("  Sacuvan grafikon: output/zadatak1_tacnost_po_k.png")

# Prikaz najboljeg k
najbolji_k = max(rezultati_k, key=rezultati_k.get)
print(f"\n  Najbolji k = {najbolji_k} sa tacnoscu {rezultati_k[najbolji_k]:.2%}")


# ============================================================================
# 7. ZADATAK 2: Uklanjanje jedne vrste — trening na dvije
# ============================================================================
print("\n" + "=" * 65)
print("  ZADATAK 2: Trening na samo dvije vrste (bez R. temporaria)")
print("=" * 65)

# Filtriranje: zadrzavamo samo Bombina vrste
indeksi_2vrste = [i for i, l in enumerate(labele)
                  if l in ("bombina_bombina", "bombina_variegata")]

sekvence_2 = [sekvence[i] for i in indeksi_2vrste]
labele_2 = [labele[i] for i in indeksi_2vrste]

jedinstvene_2 = sorted(set(labele_2))
mapa_2 = {lab: i for i, lab in enumerate(jedinstvene_2)}
y_2 = np.array([mapa_2[l] for l in labele_2])
nazivi_2 = [KRATKE_LABELE.get(lab, lab) for lab in jedinstvene_2]

# Poredenje k=4 i k=6 za dvije Bombina vrste
fig_z2, axes_z2 = plt.subplots(1, 2, figsize=(14, 6))
boje_2 = ["#e74c3c", "#f39c12"]
tacnost_2_po_k = {}

for idx_z2, k_z2 in enumerate([4, 6]):
    X_2 = izgradi_kmer_matricu(sekvence_2, k=k_z2)
    skaler_2 = StandardScaler()
    X_2_skal = skaler_2.fit_transform(X_2)

    pca_2 = PCA(n_components=2)
    X_2_pca = pca_2.fit_transform(X_2_skal)

    # PCA grafikon
    ax = axes_z2[idx_z2]
    for idx_boje, lab in enumerate(jedinstvene_2):
        indeksi = [i for i, l in enumerate(labele_2) if l == lab]
        ax.scatter(X_2_pca[indeksi, 0], X_2_pca[indeksi, 1],
                   label=KRATKE_LABELE.get(lab, lab),
                   color=boje_2[idx_boje], alpha=0.7, edgecolors="black", linewidths=0.3, s=50)
    ax.set_xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Klasifikacija
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_2_skal, y_2, test_size=0.2, random_state=42, stratify=y_2
    )
    model_2 = LogisticRegression(max_iter=1000, random_state=42)
    model_2.fit(X_tr2, y_tr2)
    y_pred_2 = model_2.predict(X_te2)
    tacnost_2 = np.mean(y_pred_2 == y_te2)
    tacnost_2_po_k[k_z2] = tacnost_2

    ax.set_title(f"k={k_z2} — tacnost: {tacnost_2:.0%}", fontsize=11, fontweight="bold")

    print(f"\n  --- Samo Bombina, k={k_z2} ---")
    print(f"  Tacnost: {tacnost_2:.2%}")
    print(classification_report(y_te2, y_pred_2, target_names=nazivi_2))

fig_z2.suptitle("PCA — samo sestrinske vrste Bombina (bez R. temporaria)",
                fontsize=13, fontweight="bold")
fig_z2.tight_layout()
fig_z2.savefig(os.path.join(OUTPUT_DIR, "zadatak2_pca_2vrste.png"), dpi=150)
plt.close(fig_z2)
print("  Sacuvan grafikon: output/zadatak2_pca_2vrste.png")

# Matrica konfuzije za 2 vrste (k=6)
X_2_k6 = izgradi_kmer_matricu(sekvence_2, k=6)
X_2_k6_skal = StandardScaler().fit_transform(X_2_k6)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_2_k6_skal, y_2, test_size=0.2, random_state=42, stratify=y_2
)
model_2k6 = LogisticRegression(max_iter=1000, random_state=42)
model_2k6.fit(X_tr2, y_tr2)
y_pred_2k6 = model_2k6.predict(X_te2)

cm_2 = confusion_matrix(y_te2, y_pred_2k6)
fig, ax = plt.subplots(figsize=(6, 5))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=nazivi_2)
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
k_z3 = najbolji_k
X_z3 = izgradi_kmer_matricu(sekvence, k_z3)
skaler_z3 = StandardScaler()
X_z3_skal = skaler_z3.fit_transform(X_z3)

X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
    X_z3_skal, y, test_size=0.2, random_state=42, stratify=y
)

modeli = {
    "Logisticka regresija": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (linearni)":       SVC(kernel="linear", random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
}

rezultati_modela = {}
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))

for idx_m, (naziv_modela, model) in enumerate(modeli.items()):
    print(f"\n  --- {naziv_modela} (k={k_z3}) ---")
    model.fit(X_tr3, y_tr3)
    y_pred_m = model.predict(X_te3)
    tacnost_m = np.mean(y_pred_m == y_te3)
    rezultati_modela[naziv_modela] = tacnost_m

    print(f"  Tacnost: {tacnost_m:.2%}")
    print(classification_report(y_te3, y_pred_m, target_names=nazivi_klasa))

    # Matrica konfuzije
    cm_m = confusion_matrix(y_te3, y_pred_m)
    disp_m = ConfusionMatrixDisplay(confusion_matrix=cm_m, display_labels=nazivi_klasa)
    disp_m.plot(ax=axes_cm[idx_m], cmap="Blues", colorbar=False)
    axes_cm[idx_m].set_title(f"{naziv_modela}\n(tacnost: {tacnost_m:.1%})",
                             fontsize=11, fontweight="bold")
    axes_cm[idx_m].set_xlabel("Predvidjena klasa", fontsize=10)
    axes_cm[idx_m].set_ylabel("Stvarna klasa", fontsize=10)

fig_cm.suptitle(f"Poredenje matrica konfuzije — tri klasifikatora (k={k_z3})",
                fontsize=13, fontweight="bold")
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(OUTPUT_DIR, "zadatak3_poredenje_klasifikatora.png"), dpi=150)
plt.close(fig_cm)
print("  Sacuvan grafikon: output/zadatak3_poredenje_klasifikatora.png")

# Grafikon poredenja tacnosti
plt.figure(figsize=(9, 5))
nazivi_m = list(rezultati_modela.keys())
tacnosti_m = [rezultati_modela[n] * 100 for n in nazivi_m]
boje_m = ["#3498db", "#e74c3c", "#2ecc71"]
plt.barh(nazivi_m, tacnosti_m, color=boje_m, edgecolor="black", linewidth=0.5, height=0.5)
plt.xlabel("Tacnost (%)", fontsize=12)
plt.title(f"Poredenje tacnosti klasifikatora (k={k_z3}, sve 3 vrste)",
          fontsize=13, fontweight="bold")
plt.xlim(0, 105)
for i, t in enumerate(tacnosti_m):
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
  Ukupno fragmenata: {len(sekvence)}

  ZADATAK 1 — Najbolji k:
    k=3: {rezultati_k[3]:.2%}  |  k=4: {rezultati_k[4]:.2%}  |  k=5: {rezultati_k[5]:.2%}  |  k=6: {rezultati_k[6]:.2%}
    Najbolji: k={najbolji_k} ({rezultati_k[najbolji_k]:.2%})

  ZADATAK 2 — Dvije sestrinske vrste (bez R. temporaria):
    Tacnost (k=4): {tacnost_2_po_k.get(4, 0):.2%}
    Tacnost (k=6): {tacnost_2_po_k.get(6, 0):.2%}

  ZADATAK 3 — Poredenje klasifikatora (k={najbolji_k}):""")
for naziv, tac in rezultati_modela.items():
    print(f"    {naziv}: {tac:.2%}")

print(f"""
  Svi grafikoni sacuvani u: {OUTPUT_DIR}/
""")
print("=" * 65)
print("  PIPELINE ZAVRSEN USPJESNO")
print("=" * 65)
