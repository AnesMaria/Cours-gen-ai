<img width="1448" height="1086" alt="Schéma RNN _ états cachés et phrase française" src="https://github.com/user-attachments/assets/8bf61aa5-129a-4a02-b77d-221a5ecdf53f" />
# Partie A : Les réseaux construisent des représentations internes

## NN : une fonction paramétrée (base commune à CNN/RNN/Attention)

### Notions clés à retenir

- **Un réseau = fonction paramétrée :** $\hat{y}=f_\theta(x)$.
- **Apprendre = choisir $\theta$ :** minimiser une perte $L(\hat{y},y)$ (descente de gradient).
- **Idée centrale du cours :** le réseau ne produit pas seulement $\hat{y}$, il fabrique aussi une **représentation interne** dans ses couches cachées.
- **Représentation / latent :** on note souvent $z$ l’activation d’une couche intermédiaire :

$$
x \xrightarrow{\ f_{\theta_1}\ } z \xrightarrow{\ f_{\theta_2}\ } \hat{y}
$$
![Schéma RNN — séquence et état caché](rnn-sequence-etat-cache.png)
- **Pont vers CNN/RNN :** ce qui change d’une architecture à l’autre, c’est **comment** on transforme l’espace d’entrée (localité, séquence, contexte).

### Ce que fabrique réellement le réseau

Le réseau ne fait pas seulement une prédiction. Il construit une **représentation interne** :

$$
x \rightarrow z \rightarrow \hat{y}
$$

- $x$ = données brutes
- $z$ = espace latent (représentation utile)
- $\hat{y}$ = décision finale

CNN, RNN et Transformers diffèrent avant tout dans la manière de construire $z$ — même si ce choix a aussi des conséquences importantes sur la parallélisation de l’entraînement et sur la façon dont chaque architecture capture le contexte.

### Aperçu code — retrouver $z$ dans un réseau

```python
z = torch.relu(layer1(x))   # representation interne
y_hat = layer2(z)           # sortie finale
```

**À observer :** la variable `z` correspond à la représentation intermédiaire construite à partir de $x$ ; elle n’est pas la prédiction finale.

### NN = constructeur de représentations

**Problème concret :** on veut qu’un modèle comprenne des données complexes (images, texte, signaux) et généralise, pas seulement mémoriser.  
**Échec :** travailler directement dans l’espace brut (pixels / one-hot) est fragile : les distances et variations ne reflètent pas le sens.  
**Solution :** apprendre une transformation $f_\theta$ qui construit un espace interne où la tâche devient plus simple.

### Du réseau à la représentation
![Uploading Schéma RNN _ états cachés et phrase française.png…]()

Un réseau est une grande fonction composée. On choisit une architecture, puis on ajuste les paramètres $\theta$ pour réduire une erreur. Cela vaut pour un MLP, un CNN, un RNN ou un Transformer : ce qui change, c’est la manière dont on transforme l’espace d’entrée. Ce qui nous intéresse ici n’est pas uniquement la sortie $\hat{y}$, mais ce qu’il se passe au milieu : l’espace latent.

## 1) Visualisation minimale : $x \rightarrow f_\theta \rightarrow \hat{y}$ + perte

Le schéma général est : entrée $x$, réseau $f_\theta$, sortie $\hat{y}$, puis calcul d’une perte $L(\hat{y},y)$ utilisée pour mettre à jour les paramètres $\theta$.

## 2) Où vit la représentation ? Dans les couches cachées

Dans un réseau profond, on peut isoler une couche cachée comme représentation :

$$
x \xrightarrow{\ f_{\theta_1}\ } z \xrightarrow{\ f_{\theta_2}\ } \hat{y}
$$

**Interprétation :** $z$ est un **vecteur** (ou un tenseur) qui résume l’entrée de façon utile.

Exemples :

- dans un **MLP** : $z$ = activations d’une couche fully-connected ;
- dans un **CNN** : $z$ = feature maps (local) ou vecteur après global pooling (global) ;
- dans un **RNN** : $z$ = état caché $h_t$ (ou $h_T$) ;
- dans l’**attention** : $z$ = représentation contextuelle (combinaison pondérée).

## 3) Ce qu’il faut retenir

Que l’on parte de $x \to f_\theta \to \hat{y}$ ou de $x \to f_{\theta_1} \to z \to f_{\theta_2} \to \hat{y}$, c’est **exactement le même réseau** : on a simplement choisi de nommer et d’isoler la couche intermédiaire $z$. C’est précisément cet espace latent / représentation qui va nous intéresser pour le reste du cours.

## 4) Mini-activité : préparer CNN/RNN

**Question :** dans ce schéma, si je change la tâche (classification $\rightarrow$ génération), est-ce que $z$ doit rester identique ?  
**Attendu :** pas nécessairement. Changer de tâche peut modifier la représentation utile. Dans les **modèles génératifs à variables latentes**, on cherchera notamment un espace $z$ compatible avec un mécanisme d’échantillonnage ; d’autres familles génératives procéderont autrement.

### Phrase de lien

- **CNN :** $z$ vient de motifs **locaux** (feature maps).
- **RNN :** $z$ vient d’une **mémoire séquentielle** ($h_t$).
- **Attention :** $z$ devient **contextuel**.

### Complément NN — ce qu’il faut savoir pour la suite générative

- un neurone calcule typiquement une combinaison affine suivie d’une activation : $h=\sigma(Wx+b)$ ;
- la **fonction de perte** mesure l’écart entre ce que le modèle produit et l’objectif d’apprentissage ;
- la **rétropropagation** calcule les gradients nécessaires pour mettre à jour les paramètres $\theta$ ;
- les dimensions d’entrée et de sortie dépendent de la tâche : un réseau peut produire une étiquette, une représentation, les paramètres d’une distribution, ou directement une donnée générée.

Dans un générateur, l’entrée peut aussi être un bruit ou un vecteur latent $z$. Le rôle précis de ce latent dépendra de la famille de modèle étudiée.

[← Retour à la Semaine 1](README.md)
