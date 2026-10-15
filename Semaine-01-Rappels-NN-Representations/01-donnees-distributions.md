# Rappels : données, variables et distributions

Pourquoi ce rappel avant de parler de réseaux ? Dans la suite du cours, des expressions comme $p(x)$ ou $x\sim p(x)$ apparaîtront régulièrement. Avant de demander comment un réseau construit une représentation, il faut donc préciser ce que représentent *la donnée*, *la variable aléatoire*, *l’échantillon* et *la distribution*.

## Une observation n’est pas une distribution

### Notation minimale à connaître

- $X$ : une **variable aléatoire**, c’est-à-dire l’objet abstrait qui peut prendre différentes valeurs ;
- $x$ : une **réalisation** ou observation particulière de $X$ ;
- $x_i$ : une composante de l’observation $x$ ;
- $p_{\mathrm{data}}(x)$ : la distribution réelle des données, inconnue en pratique ;
- $p_\theta(x)$ : la distribution décrite ou approximée par le modèle appris, avec paramètres $\theta$.

### Exemples de données

Une observation $x$ peut être une image, une phrase, un signal audio, une molécule ou une série temporelle. Pour une image $28\times 28$, $x_i$ peut désigner l’intensité du $i$-ème pixel ; pour une phrase, $x_i$ peut désigner le token à la position $i$.

On suppose que les observations proviennent d’un mécanisme réel que l’on résume par :

$$
x\sim p_{\mathrm{data}}(x).
$$

Cette écriture signifie : « l’observation $x$ est un échantillon provenant de la distribution des données ». Elle ne signifie pas que $x$ est lui-même une distribution.

## Variable discrète, variable continue, probabilité et densité

Une variable **discrète** prend des valeurs dans un ensemble fini ou dénombrable, par exemple une étiquette $Y\in\{0,\dots,9\}$. Une variable **continue** prend ses valeurs dans un espace continu, par exemple un vecteur $X\in\mathbb{R}^d$.

### Probabilité versus densité

Pour une variable discrète, $p(X=x)$ peut être interprété comme une probabilité ponctuelle. Pour une variable continue, $p(x)$ désigne plutôt une **densité** évaluée au point $x$ ; la probabilité porte alors sur une région et s’obtient par intégration. Dans les deux cas, la distribution entière est normalisée.

### Première définition d’un modèle génératif

Un modèle génératif cherche à **définir ou induire une distribution sur les données**. Schématiquement :

$$
p_\theta(x)\approx p_{\mathrm{data}}(x).
$$

Une fois le modèle appris, on souhaite notamment pouvoir produire une nouvelle observation :

$$
x_{\mathrm{nouveau}}\sim p_\theta(x).
$$

Exemple : le modèle observe de nombreuses images de chats, apprend les régularités de cette population puis génère une image plausible qui n’était pas dans le jeu d’entraînement.

**Limite volontaire de S1 :** on ne compare pas encore les différentes façons de modéliser une distribution. Cette comparaison sera le sujet de la S2.

### Trois idées à distinguer dès S1

**Évaluer la plausibilité :** pour une donnée $x$, le modèle peut parfois lui associer une quantité probabiliste telle que $p_\theta(x)$.  
**Échantillonner :** produire une nouvelle observation selon le mécanisme appris.  
**Apprendre une représentation :** construire un vecteur ou un tenseur $z$ qui résume utilement $x$.

Pour S1, il suffit de comprendre que ces opérations sont différentes. Leur traitement selon les familles de modèles sera étudié plus tard.

[← Retour à la Semaine 1](README.md)