# Transformers & Attention : représentations contextuelles

## Self-Attention : du token fixe au token contextualisé

### Idée clé

- Un même token n’a pas toujours le même sens : sa représentation doit dépendre du **contexte**.
- La self-attention construit, pour chaque position $t$, un vecteur contextuel $h_t$ comme **combinaison pondérée** des valeurs associées aux positions autorisées de la séquence.

## De $X$ vers $Q,K,V$

Soit $X\in\mathbb{R}^{T\times d}$ la matrice des embeddings. On calcule :

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

où $W_Q,W_K,W_V$ sont des paramètres appris.

Intuition :

- **Q** = ce que le token recherche ;
- **K** = ce que chaque token peut faire correspondre ;
- **V** = l’information effectivement transmise.

## Scaled Dot-Product Attention

$$
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

Le softmax transforme les scores d’attention en poids positifs dont la somme vaut 1.

- $QK^\top$ : matrice de similarité ;
- $1/\sqrt{d_k}$ : stabilise l’amplitude des scores ;
- $M$ : masque, notamment causal, avec 0 pour une position autorisée et une très grande valeur négative pour une position interdite ;
- multiplication par $V$ : agrégation de l’information selon les poids d’attention.

### Aperçu code

```python
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores + mask, dim=-1)
H = weights @ V
```

On calcule les scores de similarité, on les normalise, puis on combine les valeurs $V$ pour obtenir une représentation contextuelle $H$.

## Masque causal

Avec trois tokens $(w_1,w_2,w_3)$, la position 2 peut utiliser $w_1$ et $w_2$, mais ne doit pas regarder $w_3$ s’il appartient au futur. Le masque rend le score futur quasi nul après softmax.

**À retenir :** le masque causal ne change pas le mécanisme d’attention ; il indique quelles positions peuvent être consultées.

### Mini-interaction

Si le score d’attention entre « chat » et « miaule » est élevé, la sortie contextuelle pour « chat » accorde un poids important à la valeur associée à « miaule ».

## Trois mécanismes d’attention du Transformer encodeur–décodeur

- **Self-attention de l’encodeur :** chaque position exploite le contexte de la séquence d’entrée.
- **Self-attention masquée du décodeur :** génération autorégressive sans accès aux tokens futurs.
- **Attention croisée :** le décodeur interroge les représentations produites par l’encodeur.

## Architecture Transformer

Un bloc Transformer contient principalement :

1. **Multi-Head Attention** : contextualise les tokens ;
2. **FFN** : transforme non linéairement chaque position ;
3. **Add & Norm** : connexions résiduelles + normalisation.

L’encodeur empile plusieurs blocs pour construire des représentations contextuelles $H$. Le décodeur combine self-attention masquée, attention croisée sur $H$ et FFN pour produire la sortie.

## Multi-Head Attention

Chaque tête utilise ses propres projections $Q$, $K$ et $V$. Les différentes sorties sont concaténées puis reprojetées.

$$
\mathrm{head}_i=\mathrm{Attention}(XW_i^Q,XW_i^K,XW_i^V)
$$

$$
\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O
$$

Plusieurs relations contextuelles peuvent ainsi être considérées en parallèle.

## Résidus, LayerNorm et encodage positionnel

Les connexions résiduelles facilitent l’optimisation des réseaux profonds. La Layer Normalization aide à stabiliser les activations.

La self-attention seule ne représente pas explicitement l’ordre. On ajoute donc une information de position :

$$
x_t=e(w_t)+p_t
$$

Le terme $p_t$ peut être appris ou construit avec des fonctions sinusoïdales.

## Entraînement et génération d’un décodeur causal

Dans un décodeur autorégressif, le modèle apprend à prédire le prochain token à partir des précédents.

Pour une séquence cible $(y_1^\star,\dots,y_T^\star)$ :

$$
\mathcal{L}=-\sum_{t=1}^{T}\log p(y_t^\star\mid y_{<t}^\star,x)
$$

À chaque position, le modèle doit attribuer une probabilité élevée au vrai token suivant.

- **Pendant l’entraînement :** on utilise généralement les vrais tokens précédents comme contexte — c’est le **teacher forcing**.
- **Pendant la génération :** le vrai token suivant n’est pas disponible ; le modèle produit un token puis réutilise ses propres sorties pour continuer.

Cette mécanique constitue un premier aperçu de la génération **auto-régressive**, qui sera approfondie plus tard dans le cours.

[← Retour à la Semaine 1](README.md)