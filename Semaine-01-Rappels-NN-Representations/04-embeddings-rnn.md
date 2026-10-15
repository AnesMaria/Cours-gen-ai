# A3 — Texte : one-hot → embedding + RNN

## Embeddings & RNN — Notions clés à retenir

- **Problème du symbole :** un ID / one-hot n’a pas de géométrie, donc pas de notion de proximité.
- **Embedding :** une **géométrie apprise** $e(w)\in\mathbb{R}^d$ ; selon l’objectif d’entraînement, certaines proximités peuvent refléter des similarités utiles.
- **RNN :** encode une **séquence** en une **mémoire compressée** : l’état caché $h_t$.
- **Message clé :** **RNN = encoder une séquence dans un état caché qui résume le passé**.
- **Où vit la représentation ?** dans les embeddings des tokens et dans les états cachés $h_t$, souvent $h_T$ ou un pooling.

## A3.1 — One-hot : pourquoi ça échoue ?

**Problème concret :** pour le modèle, `chat` et `chien` devraient être plus proches que `chat` et `voiture`.  
**Échec :** avec one-hot, les vecteurs sont orthogonaux et il n’existe donc aucune notion naturelle de proximité sémantique.  
**Solution :** apprendre une table d’embedding : index $\mapsto$ vecteur dense $e(w)$, où la distance reflète une similarité apprise.

### Illustration one-hot

`chat = (0,0,…,1,…,0)` et `chien = (0,1,0,…,0)`.

$$
\langle \texttt{chat},\texttt{chien}\rangle = 0
$$

Les deux vecteurs sont orthogonaux : le modèle ne voit aucun lien dans cette représentation brute.

### Exemple de table d’embedding

| Token | dim 1 | dim 2 | dim 3 | dim 4 |
|---|---:|---:|---:|---:|
| chat | 0.2 | -0.5 | 0.8 | 0.1 |
| chien | 0.3 | -0.4 | 0.7 | 0.2 |
| voiture | 0.9 | 0.8 | -0.1 | -0.3 |

*Exemple fictif à but pédagogique : ces valeurs ne proviennent pas d’un entraînement réel.*

### Idée essentielle

**Embedding = géométrie apprise.** Selon les données et l’objectif d’entraînement, des mots partageant des propriétés utiles peuvent obtenir des vecteurs proches, par exemple avec un cosinus élevé. Cette proximité n’est toutefois pas une garantie universelle de similarité sémantique.

## A3.2 — RNN : l’ordre compte

**Problème concret :** le sens d’une phrase dépend de l’ordre des mots et du contexte temporel.  
**Échec :** un sac de mots ou une simple moyenne d’embeddings perd l’ordre.  
**Solution :** un RNN lit la séquence pas à pas et maintient un **état caché** $h_t$ qui joue le rôle de mémoire compressée.

La même cellule est appliquée à chaque pas de temps ; l’état caché $h_t$ résume ce qui a été lu jusqu’ici, jusqu’au résumé final $h_T$.

À l’instant $t$ :

$$
h_t=f(h_{t-1},e(x_t))
$$

**Interprétation :** $h_t$ contient un résumé de $x_1,\dots,x_t$.

### Aperçu code

```python
h = h0
for token in sequence:
    x_t = embedding(token)
    h = rnn_cell(x_t, h)    # nouvel etat cache
```

À chaque étape, le nouvel état `h` dépend du token courant et de l’état précédent ; il accumule progressivement le contexte passé.

## Même motif que CNN

Que ce soit en image avec un CNN ou en séquence avec un RNN, le réseau fabrique des **vecteurs qui résument les données**. On se rapproche de l’idée centrale : **l’espace latent**.

## Limites du RNN

1. **Contexte d’un RNN unidirectionnel standard :** lorsqu’il lit de gauche à droite, $h_t$ n’utilise directement que le passé $x_{\leq t}$. Des architectures bidirectionnelles peuvent exploiter les deux directions.
2. **Mémoire limitée :** dans les longues séquences, l’information du début peut se diluer. Le *vanishing gradient* rend notamment l’apprentissage des dépendances longues plus difficile.
3. **Séquentiel obligatoire :** la lecture pas à pas limite la parallélisation et ralentit l’entraînement.

### Mini-interaction

**Question :** pourquoi one-hot est-il mauvais pour représenter la similarité ?  
**Réponse attendue :** parce que les vecteurs sont orthogonaux, donc sans notion de proximité géométrique.

[← Retour à la Semaine 1](README.md)