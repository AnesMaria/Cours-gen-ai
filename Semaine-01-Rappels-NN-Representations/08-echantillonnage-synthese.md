# Complément : échantillonnage et plausibilité

## Deux opérations à ne pas confondre

**Évaluer une donnée :** on part d’une observation déjà disponible $x$ et l’on cherche à savoir si elle est compatible avec ce que le modèle a appris. Lorsque le modèle fournit une densité, cette idée peut être reliée à $p_\theta(x)$.  
**Échantillonner :** on utilise le mécanisme appris pour produire une nouvelle réalisation qui n’était pas fournie en entrée.

## Mini-exercice

Indiquez l’opération dominante :

1. compresser une photo puis reconstruire cette même photo ;
2. examiner si une observation est plausible selon le modèle ;
3. tirer un latent aléatoire puis produire une nouvelle image.

**Réponses attendues :** reconstruction ; évaluation de plausibilité ; génération par échantillonnage.

# Synthèse de la semaine 1

Cette semaine, nous avons appris à **construire des représentations** :

- CNN : hiérarchie de représentations spatiales ;
- RNN : représentation séquentielle ;
- Attention : représentation contextuelle ;
- Embeddings : passage de symboles discrets vers une géométrie apprise.

Mais générer exige maintenant de comprendre comment passer d’une représentation utile à un mécanisme capable de produire de nouvelles observations plausibles. Dans certains modèles à variables latentes, cela passera par un espace latent échantillonnable ; d’autres familles procéderont autrement.

Au prochain cours, nous changerons de point de vue : nous étudierons formellement **quelle distribution un modèle cherche à apprendre**.

## Récapitulatif

Les architectures étudiées apprennent des **représentations internes** adaptées à la structure des données : CNN pour l’espace visuel, RNN pour les séquences, attention pour les représentations contextuelles, et embeddings pour associer aux symboles une géométrie apprise.

La grande question qui nous accompagne maintenant est : **comment passer de représentations utiles à un mécanisme capable de générer de nouvelles données plausibles**, et pas seulement de reconnaître ou de reconstruire une entrée ?

**À venir en S2 :** les notions de donnée, distribution et échantillon introduites ici serviront à comparer formellement les différents objectifs probabilistes d’apprentissage. La taxonomie des grandes familles génératives sera alors introduite.

> **Note :** la reconstruction part d’une observation $x$ et cherche à la retrouver ; la génération cherche à produire une nouvelle observation. Le chemin $z\sim p(z)$ puis $x\sim p_\theta(x\mid z)$ illustre ici le cas des modèles génératifs à variables latentes.

# Auto-vérification — S1

Avant de passer à la semaine 2, vérifiez que vous pouvez répondre sans regarder les pages précédentes :

1. Quelle différence existe entre $X$, $x$ et $x_i$ ?
2. Que représentent $p_{\mathrm{data}}(x)$ et $p_\theta(x)$ dans l’intuition introduite en S1 ?
3. Quelle différence existe entre une probabilité discrète et une densité continue ?
4. Que signifie $z=f_\theta(x)$ dans un réseau ?
5. Où se trouve la représentation dans un CNN, un RNN et un mécanisme d’attention ?
6. Pourquoi deux points proches dans un espace latent ne sont-ils pas nécessairement sémantiquement proches ?
7. Quelle différence existe entre reconstruire une observation et générer une nouvelle observation ?
8. Que signifie échantillonner depuis un modèle ?
9. Pourquoi une bonne reconstruction ne garantit-elle pas automatiquement une bonne génération ?

[← Retour à la Semaine 1](README.md)