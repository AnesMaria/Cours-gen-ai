# A4 — Synthèse : représentations internes et espace latent

## Représentations internes comme langage commun

**Problème :** images et séquences ont des formats différents.  
**Échec :** l’espace brut ne porte pas directement le sens.  
**Solution :** les architectures apprennent des **représentations internes** dans lesquelles certaines relations pertinentes pour la tâche deviennent plus faciles à exprimer.

| Architecture | Où est la représentation ? | Nature |
|---|---|---|
| CNN | feature maps profondes + global pooling $\rightarrow z$ | hiérarchie spatiale |
| RNN | états cachés $h_t$, souvent $h_T$ | séquentielle |
| Embedding | vecteur token $e(w)$ | discret $\rightarrow$ continu |

## Définition de l’espace latent

**Espace latent / espace de représentation :** espace interne appris par le modèle, parfois de dimension réduite, dans lequel sont encodées certaines caractéristiques des données.

Deux points proches **peuvent** correspondre à des exemples similaires selon la représentation apprise et l’objectif d’entraînement ; cette propriété n’est pas garantie universellement.

À ce stade du cours, la lettre $z$ désigne simplement une **représentation interne**. Plus tard, notamment avec les modèles à variables latentes, le rôle de $z$ deviendra plus précis : $z$ sera alors une **variable aléatoire** à laquelle on associera une distribution $p(z)$.

## Trois espaces à ne pas confondre

- **Espace de caractéristiques :** activations internes apprises pour rendre une tâche plus simple.
- **Embedding supervisé :** représentation dont la géométrie est influencée par une tâche et des étiquettes.
- **Espace latent probabiliste :** espace associé à une variable aléatoire $Z$ et à une distribution, par exemple $z\sim p(z)$.

Ces notions peuvent se recouvrir selon l’architecture, mais elles ne sont pas synonymes.

## Exemple MNIST

Supposons qu’un réseau transforme chaque image de chiffre en un point $z\in\mathbb{R}^2$. Après projection, les « 1 » peuvent former un groupe et les « 7 » un autre. Cela montre que la représentation a organisé certaines similarités utiles. Mais deux points proches ne sont pas nécessairement sémantiquement proches dans tout modèle : la géométrie dépend de l’objectif d’apprentissage.

## Interpolation et limite de l’autoencodeur classique

Entre deux codes $z_1$ et $z_2$, on peut explorer un point intermédiaire :

$$
z_\alpha=(1-\alpha)z_1+\alpha z_2,\qquad 0\leq\alpha\leq1.
$$

Décoder ces points peut produire une transition intéressante si l’espace latent est bien structuré. Cependant, un autoencodeur classique ne garantit pas qu’entre deux codes observés, toutes les régions soient plausibles ou bien couvertes. Cette limite préparera directement l’étude des VAE.

## Transition vers la Partie B

On sait maintenant **où** se trouve la représentation. La question suivante est : est-elle **fixe** — un mot correspond toujours au même vecteur — ou **contextuelle** — le vecteur dépend de la phrase ?

[← Retour à la Semaine 1](README.md)