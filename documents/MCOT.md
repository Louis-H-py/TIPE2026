# MCOT

###### Boucles et cycles

## Titre :

Étude théorique de la tomodensitométrie axiale.

## Ancrage dans le thème de l’année :

Ce TIPE s'intéresse aux méthodes de tomodensitométrie axiale, consistant à reconstruire en deux ou trois dimensions
un organe ou objet ayant été scanné. Les données sont obtenues en effectuant des **boucles** avec un
scanner autour du sujet. Ce TIPE s'inscrit donc dans le thème "boucles et cycles".

## Motivations :

Ce TIPE mêle des questions de traitement du signal, d'analyse de données et de théorie de l'information. Les outils
modernes de résolution de ce problème font appel au *machine learning*, et plus particulièrement à des
réseaux de neurones, sujet qui m'intéresse énormément.

## Positionnement thématique :

Informatique (pratique) — Informatique (théorique) — Physique (ondulatoire)

## Mots clés :

| Français                        | Anglais                      |
|---------------------------------|------------------------------|
| Tomodensitométrie               | *CT scan*                    |
| Reconstruction                  | *Reconstruction*             |
| Sinogramme                      | *Sinogram*                   |
| Réseaux de neurones artificiels | *Artificial neural networks* |
| Imagerie médicale               | *Medical imaging*            |

## Bibliographie commentée :

L'invention de la tomodensitométrie (CT-scan) par Godfrey Hounsfield en 1972 [1] a révolutionné l'imagerie médicale en
permettant de visualiser l'intérieur du corps humain en trois dimensions sans chirurgie. Le principe repose sur la
réalisation de boucles circulaires d'une source de rayons X autour du patient. À chaque angle, on mesure l'atténuation
du faisceau, ce qui permet d'obtenir une projection unidimensionnelle. L'ensemble de ces projections, organisées en
fonction de l'angle de rotation, forme un sinogramme [2].

D'un point de vue mathématique, le passage de l'objet réel au sinogramme est modélisé par la transformée de Radon [2].
Pour reconstruire l'image originale à partir de ces mesures, l'algorithme historique et le plus utilisé est la
rétroprojection filtrée (FBP) [3]. Bien que rapide, la FBP nécessite un grand nombre de projections (échantillonnage
dense) pour produire une image de haute qualité. Si l'on réduit le nombre de prises de vue pour limiter l'exposition du
patient aux radiations — selon le principe ALARA (*As Low As Reasonably Achievable*) — la FBP génère des artefacts de
« vissage » et un bruit important qui dégradent le diagnostic [4].

Pour pallier ces limitations, des méthodes itératives de reconstruction ont été développées. Contrairement à la FBP,
elles considèrent la reconstruction comme un problème d'optimisation statistique [5]. Elles permettent de reconstruire
des images à partir de données éparses (*sparse-view CT*), mais au prix d'un coût computationnel très élevé, rendant
leur
utilisation complexe en routine clinique d'urgence.

C’est ici que l'apport de l’intelligence artificielle devient déterminant. Depuis quelques années, les réseaux de
neurones convolutifs (CNN) [6] transforment le domaine. Plusieurs approches coexistent : certains réseaux sont entraînés
pour « nettoyer » le sinogramme avant la reconstruction, tandis que d'autres interviennent après une FBP de basse
qualité pour supprimer les artefacts (post-traitement). Les architectures les plus performantes, comme les réseaux
U-Net ou les réseaux antagonistes génératifs (GAN) [7], parviennent à restituer des détails anatomiques fins à partir de
scans réalisés avec une dose de rayons X divisée par dix.

L'enjeu actuel de la recherche est la « reconstruction directe » (*end-to-end*), où un réseau de neurones apprend
directement l'opérateur de transformation inverse, passant du sinogramme à l'image finale sans recourir aux
algorithmes analytiques classiques [8]. Cette approche promet une rapidité d'exécution sans précédent et une robustesse
accrue face au bruit de mesure.

L'étude de ces méthodes permet non seulement d'optimiser les outils de diagnostic médical, mais trouve aussi des échos
dans le contrôle non destructif industriel et l'astrophysique, où la reconstruction d'objets à partir de projections
limitées est un défi permanent.

## Problématique retenue :

Comment concevoir un programme informatique capable de réaliser une tomodensitométrie axiale, et comment optimiser la
reconstruction tomographique dans le but de diminuer l'exposition aux rayons X d'un sujet ?

## Objectif du TIPE :

Le TIPE se décompose en plusieurs objectifs :

* Concevoir un générateur de scans aléatoires afin de comparer différentes méthodes sur un grand échantillon
* Coder des algorithmes de reconstruction tomographique existants
* Comparer les performances de ces méthodes en fonction du nombre de prises de vue
* Concevoir un réseau de neurones capable de réaliser une reconstruction tomographique
* Comparer l'ensemble des méthodes envisagées

## Références bibliographiques :

[1] G. N. Hounsfield, Computerized transverse axial scanning (tomography): Part 1. Description of system,  
British Journal of Radiology, vol. 46, no. 552, pp. 1016–1022, 1973.

[2] Wikipédia Théorème de Radon https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Radon (10 octobre 2025)

[3] Feldkamp, L. A., Davis, L. C., & Kress, J. W., Practical cone-beam algorithm. Journal of the Optical Society
of America A, 1(6), 612-619, 1984.

[4] Zhang H, Ma J, Wang J, Liu Y, Han H, Lu H, Moore W, Liang Z. Statistical image reconstruction for low-dose CT using
nonlocal means-based regularization. Part II: An adaptive approach. Comput Med Imaging Graph. 2015

[5] Beister M, Kolditz D, Kalender WA. Iterative reconstruction methods in X-ray CT. Phys Med. 2012

[6] LeCun, Y., Bengio, Y. & Hinton, G. “Deep learning.” Nature 521, 436–444 (2015).

[7] J. M. Wolterink, T. Leiner, M. A. Viergever and I. Išgum, "Generative Adversarial Networks for Noise Reduction in
Low-Dose CT," in IEEE Transactions on Medical Imaging, vol. 36, no. 12, 2017

[8] Zhu, B., Liu, J., Cauley, S. et al. Image reconstruction by domain-transform manifold learning. Nature 555,
487–492 (2018).