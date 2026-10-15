# Semaine 1 — Schémas visuels complets

Cette page rassemble les schémas pédagogiques de la semaine 1 sous une forme directement rendue par GitHub. Ils correspondent aux figures présentes dans le support d'origine.

## 1. Réseau de neurones simple : entrée → couche cachée → sortie

```mermaid
flowchart LR
    X1((x1)) --> H1((h1))
    X1 --> H2((h2))
    X1 --> H3((h3))
    X1 --> H4((h4))
    X2((x2)) --> H1
    X2 --> H2
    X2 --> H3
    X2 --> H4
    X3((x3)) --> H1
    X3 --> H2
    X3 --> H3
    X3 --> H4
    H1 --> Y((ŷ))
    H2 --> Y
    H3 --> Y
    H4 --> Y
```

**Lecture :** l'entrée $x$ est transformée en représentation cachée $z=\sigma(W_1x+b_1)$, puis en sortie $\hat y=W_2z+b_2$.

---

## 2. Fonction paramétrée et apprentissage par la perte

```mermaid
flowchart LR
    X[x — entrée] --> N[Réseau fθ\nfonction paramétrée] --> Y[ŷ — sortie]
    Y --> L[Perte L(ŷ,y)]
    L -->|mise à jour de θ| N
```

---

## 3. CNN : de l'image aux features puis à la classification

```mermaid
flowchart LR
    A[Image d'entrée] --> B[Convolution\nFeature maps]
    B --> C[Pooling\nCondensation]
    C --> D[Fully Connected]
    D --> E1[Chat]
    D --> E2[Chien]
    D --> E3[Oiseau]
```

Le CNN transforme progressivement l'espace brut des pixels en représentations plus utiles.

---

## 4. Convolution locale : image + filtre → feature map

```mermaid
flowchart LR
    X[Image X\n6 × 6] -->|fenêtre locale| W[Filtre W\n3 × 3]
    W -->|glissement| F[Feature map\n4 × 4]
    F --> M[Chaque case indique\nla présence du motif]
```

La même matrice de poids $W$ est réutilisée à différentes positions de l'image.

---

## 5. Hiérarchie de représentations dans un CNN

```mermaid
flowchart LR
    P[Pixels X] --> L1[Couche 1\nMotifs locaux]
    L1 --> L2[Couches profondes\nFeatures]
    L2 --> GAP[Agrégation / pooling\nReprésentation globale]
    GAP --> Z[z]
```

**Intuition :** pixels → bords → textures → parties/objets → représentation globale $z$.

---

## 6. RNN : lecture séquentielle de « Le chat dort »

```mermaid
flowchart LR
    W1[Le] --> C1[RNN]
    W2[chat] --> C2[RNN]
    W3[dort] --> C3[RNN]
    C1 --> C2 --> C3
    C1 --> H1[h1]
    C2 --> H2[h2]
    C3 --> HT[hT]
```

L'état caché est transmis d'un pas de temps au suivant et résume progressivement le passé.

---

## 7. RNN générique : séquence → représentation

```mermaid
flowchart LR
    X1[x1] --> H1((h1))
    X2[x2] --> H2((h2))
    X3[x3] --> H3((h3))
    XT[xT] --> HT((hT))
    H1 --> H2 --> H3 --> DOTS[...] --> HT
```

$h_t$ représente la séquence jusqu'au temps $t$ ; $h_T$ peut servir de représentation globale de la séquence.

---

## 8. Transformer encodeur–décodeur

```mermaid
flowchart LR
    X[Séquence d'entrée] --> SA[Self-Attention]
    SA --> FFN1[FFN]
    FFN1 --> H[Représentations\ncontextuelles H]
    Y[Sortie décalée] --> MSA[Self-Attention masquée]
    MSA --> CA[Attention croisée]
    H --> CA
    CA --> FFN2[FFN]
    FFN2 --> OUT[ŷ]
```

L'encodeur construit des représentations contextuelles ; le décodeur utilise self-attention masquée et attention croisée.

---

## 9. Multi-Head Attention

```mermaid
flowchart LR
    X[X] --> H1[Tête 1\nQ1 K1 V1]
    X --> H2[Tête 2\nQ2 K2 V2]
    X --> HH[Tête h\nQh Kh Vh]
    H1 --> A1[A1]
    H2 --> A2[A2]
    HH --> AH[Ah]
    A1 --> C[Concat]
    A2 --> C
    AH --> C
    C --> WO[Wᴼ]
    WO --> MHA[MHA(X)]
```

Chaque tête apprend des projections différentes et peut capturer des relations complémentaires.

---

## 10. Encodeur CNN : représentation utile mais pas encore mécanisme génératif

```mermaid
flowchart LR
    X[Image x] --> E[Encodeur CNN]
    E --> Z[z]
    Z --> C[Classifieur]
    C --> Y[y]
```

> Le code latent $z$ est obtenu à partir d'une observation $x$. Ce schéma ne définit pas encore comment produire un nouveau $z$ sans observation d'entrée.

---

## 11. Exemple de génération à partir d'un latent échantillonné

```mermaid
flowchart LR
    Z[z ~ p(z)] --> D[Décodeur] --> X[x]
```

Ici, la génération commence par l'échantillonnage d'un latent puis son décodage.

---

## 12. Autoencodeur classique

```mermaid
flowchart LR
    X[x] --> E[Encodeur]
    E --> Z[z]
    Z --> D[Décodeur]
    D --> XH[ x̂ ]
    XH -. perte : ||x - x̂||² .-> X
```

L'objectif principal est de reconstruire l'entrée : $x \approx \hat x$.

---

## 13. Espace latent régularisé vs codes d'un AE classique

```mermaid
flowchart LR
    A["Espace latent régularisé\n• • • •\n • • •\n• • • •\nfacilement échantillonnable"]
    B["AE classique\n• •\n   • • •\n      • •\ncodes concentrés / trous"]
    A --> C[Un point choisi dans une région couverte\na plus de chances d'être plausible]
    B --> C
```

Un autoencodeur classique peut laisser des zones du latent mal couvertes ; une régularisation probabiliste vise à rendre l'échantillonnage plus contrôlé.

---

## 14. Reconstruction et génération : deux chemins différents

```mermaid
flowchart TD
    X[x] --> Z[z = encode(x)]
    Z --> R[Reconstruction\nx̂ = decode(z)\nobjectif : x ≈ x̂]
    Z -. représentation apprise .-> G
    P[z′ ~ p(z)] --> G[Génération\nx′ ~ pθ(x|z′)\nobjectif : x′ plausible / nouveau]
```

**Reconstruction :** on dispose d'une observation $x$ et on cherche à la retrouver.  
**Génération :** on cherche à produire une nouvelle observation plausible, sans observation cible à reconstruire.

---

## Correspondance avec le cours

| Figure | Notion principale |
|---|---|
| 1–2 | NN, couche cachée, représentation et apprentissage |
| 3–5 | CNN, convolution, feature maps et hiérarchie |
| 6–7 | RNN et état caché |
| 8–9 | Transformer, self-attention et multi-head attention |
| 10–13 | espace latent, encodeur/décodeur et échantillonnage |
| 14 | reconstruction vs génération |

Ces schémas complètent les huit chapitres Markdown de la semaine 1 afin que le contenu textuel, les équations, les exemples, les exercices et les représentations visuelles soient tous consultables dans GitHub.
