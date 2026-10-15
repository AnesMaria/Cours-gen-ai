# Partie C : Génération vs Reconstruction

## C1 — Paradigme classique : reconnaissance

**Problème concret :** prédire une étiquette $y$ à partir d’une entrée $x$.  
**Outil :** $x\rightarrow z\rightarrow y$ (encodeur + classifieur).  
**Limite :** $z$ est un sous-produit ; on apprend surtout $p(y|x)$, pas comment **produire** des $x$.

## C2 — Génération : produire une nouvelle observation

**Problème concret :** produire une nouvelle observation plausible sans disposer de cette observation en entrée.  
**Échec :** dans le paradigme précédent, la représentation interne est construite à partir de $x$ ; cela ne suffit donc pas, à lui seul, à définir un mécanisme de génération.  
**Solution :** apprendre un **mécanisme génératif**. Dans une famille importante de modèles à variables latentes, on définit un prior $p(z)$ puis on génère selon :

$$
z\sim p(z),\qquad x\sim p_\theta(x\mid z)
$$

D’autres familles génératives procéderont autrement ; elles seront comparées plus tard.

## C3 — Expérience mentale : l’autoencodeur

**Problème concret :** compresser puis reconstruire :

$$
x\rightarrow z\rightarrow \hat{x}
$$

**Solution :** minimiser une erreur de reconstruction, par exemple :

$$
\|x-\hat{x}\|^2
$$

**Limite :** l’autoencodeur classique n’impose pas de **prior explicite** sur les codes $z$. Rien ne garantit donc que les régions entre les codes observés soient régulièrement occupées ou facilement échantillonnables.

### Pourquoi l’AE classique ne suffit pas pour générer ?

Dans un autoencodeur classique, l’encodeur apprend des codes $z$ utiles pour **reconstruire** les observations, mais il n’impose pas de prior explicite qui organise ces codes pour l’échantillonnage. Certaines régions de l’espace latent peuvent donc être peu couvertes.

Si l’on choisit arbitrairement un point dans une telle région et qu’on le décode, la sortie peut être peu plausible. Les modèles génératifs à variables latentes traiteront précisément cette difficulté ; elle sera approfondie dans les semaines consacrées aux autoencodeurs et aux VAE.

## C4 — Reconstruction et génération : deux objectifs différents

| Reconstruction | Génération |
|---|---|
| Une observation d’entrée est disponible | On cherche à produire une nouvelle observation |
| La sortie est contrainte par cette entrée | La sortie est obtenue par un mécanisme de génération |
| Objectif principal : retrouver/préserver l’information pertinente | Objectif principal : produire un échantillon plausible |
| Exemple AE : $x\rightarrow z\rightarrow\hat{x}$ | Exemple latent : $z\sim p(z)$, puis $x\sim p_\theta(x\mid z)$ |

### Saut conceptuel

**Générer** signifie disposer d’un mécanisme capable de produire de **nouveaux échantillons plausibles**.

Dans les modèles génératifs à variables latentes, cela peut prendre la forme :

$$
z\sim p(z),\qquad x\sim p_\theta(x\mid z)
$$

Dans un autoencodeur classique, la distribution des codes $\mathrm{encode}(x)$ n’est pas explicitement régularisée pour permettre un tel échantillonnage.

### Mini-interaction

**Question :** un modèle qui prend une photo floue et la rend nette — est-ce nécessairement de la reconstruction ou de la génération ?  
**Réponse attendue :** dans le cadre simplifié de cette séance, on peut la voir comme une **reconstruction/restauration conditionnée par une entrée**, puisque la sortie est guidée par la photo floue. Plus généralement, des modèles génératifs conditionnels peuvent aussi réaliser ce type de tâche ; la frontière dépend du modèle et de l’objectif probabiliste.

[← Retour à la Semaine 1](README.md)