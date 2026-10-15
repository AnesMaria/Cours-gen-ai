# A2 — CNN : du pixel aux features

## Notions clés à retenir

- **Localité :** un filtre $3\times3$ voit seulement un voisinage et détecte des motifs simples comme des bords ou des coins.
- **Partage de poids :** le même filtre est appliqué partout. Un même motif peut donc être détecté à différentes positions avec les mêmes paramètres. La convolution est **équivariante à la translation** ; le pooling et d’autres agrégations peuvent apporter une certaine robustesse, voire une invariance approximative, aux déplacements.
- **Feature map :** la sortie d’un filtre est une carte de caractéristiques ; chaque position indique si un motif est présent localement.
- **Hiérarchie :** on observe typiquement une transformation progressive :

$$
\text{pixels}\rightarrow\text{bords}\rightarrow\text{textures}\rightarrow\text{parties}\rightarrow\text{concepts}
$$

Cette hiérarchie est une tendance observée, pas une garantie stricte couche par couche.
- **Où vit la représentation ?** dans les feature maps, puis éventuellement dans un vecteur compact $z$ après pooling ou global pooling.

## Problème → échec → solution

**Problème :** une image est une grille de pixels et les motifs utiles se répètent partout.  
**Échec :** un réseau dense regarde tout d’un coup, avec beaucoup de paramètres et sans notion naturelle de voisinage.  
**Solution :** filtres locaux + partage de poids + hiérarchie de features : pixels $\rightarrow$ feature maps $\rightarrow$ latent $z$.

## 1) Localité

Un filtre $3\times3$ calcule une combinaison locale :

$$
(\text{feature map})[i,j]=\sum_{u=-1}^{1}\sum_{v=-1}^{1}W[u,v]X[i+u,j+v]
$$

Le filtre parcourt l’image localement. Si un voisinage ressemble au motif appris, la valeur correspondante de la feature map devient élevée.

### Aperçu code

```python
features = conv(x)          # filtres locaux partages
features = torch.relu(features)
z = pool(features)          # representation plus compacte
```

Le code suit l’idée du cours : filtres locaux, cartes de caractéristiques, puis agrégation vers une représentation plus compacte.

## 2) Partage de poids

Le même filtre $W$ est utilisé à toutes les positions $(i,j)$.

- un bord détecté en haut à gauche ou en bas à droite active le même détecteur ;
- cela réduit le nombre de paramètres ;
- cela améliore la généralisation.

## 3) Image + filtre → feature map

Pour une image $6\times6$ et un filtre $3\times3$, la convolution produit une feature map. Chaque cellule de cette carte indique la présence plus ou moins forte du motif appris à une position donnée.

## 4) Hiérarchie et localisation de la représentation

**CNN = transformer progressivement l’espace des pixels en un espace de représentations plus abstraites.**

$$
\text{pixels}\rightarrow\text{bords}\rightarrow\text{textures}\rightarrow\text{parties/objets}\rightarrow z
$$

Au fil des couches, la résolution spatiale peut diminuer tandis que les représentations deviennent généralement plus adaptées à la tâche.

### Où vit la représentation dans un CNN ?

Il n’existe pas un seul endroit obligatoire :

- les **feature maps intermédiaires** conservent une information spatiale locale ;
- après une agrégation, par exemple global pooling, on peut obtenir un vecteur $z$ plus compact qui résume l’image globalement.

**À retenir :** le CNN ne se contente pas de classifier. Il construit progressivement des représentations qui transforment les pixels bruts en caractéristiques plus utiles.

## Limite du CNN pour la génération

Le CNN utilisé comme encodeur construit $z$ à partir de $x$. Si l’on veut créer une image sans disposer d’un $x$ à encoder, il faut définir un mécanisme permettant de produire l’entrée du générateur ou le latent $z$.

Cette limite mène naturellement vers la question centrale des modèles génératifs : **d’où vient $z$ lorsqu’on veut générer ?**

[← Retour à la Semaine 1](README.md)