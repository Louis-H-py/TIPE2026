# <img src="https://github.com/hampusborgos/country-flags/raw/main/svg/fr.svg" width="40"> TIPE 2026 - Cycle et boucles

Travail d'initiative personnelle encadré (TIPE) réalisé en classe préparatoire, filière MP, en 2026, sur le thème **« Cycle et boucles »**.

Ce projet se situe à la croisée de l'informatique, des mathématiques appliquées et de l'imagerie médicale.

**Auteur :** Louis H.

**Stack :** 🐍 Python | 🧮 NumPy | 📊 Plotly | 🧠 Keras

## Étude théorique de la tomodensitométrie axiale

> Comment concevoir un programme informatique capable de réaliser une tomodensitométrie axiale, et comment optimiser la reconstruction tomographique dans le but de diminuer l'exposition aux rayons X d'un sujet ?

## Description du problème

La tomodensitométrie axiale (CT Scan) permet de reconstruire une représentation interne d'un objet à partir d'une série de projections acquises sous différents angles.

L'étude repose sur un modèle bidimensionnel d'un sujet constitué d'une superposition d'ellipsoïdes de densités aléatoires, représentant un organe.

L'objectif est de développer un programme informatique capable de simuler l'acquisition de projections, puis de reconstruire l'image initiale à partir de ces mesures, tout en limitant le nombre de prises de vues nécessaires.

## Résolution

La reconstruction tomographique est abordée à travers plusieurs méthodes :

- **Rétroprojection** (Back Projection) ;
- **Reconstruction dans le domaine de Fourier**, à partir du théorème de la tranche centrale ;
- **Reconstruction itérative**, formulée comme un problème d'optimisation numérique ;
- **Réseaux de neurones**, avec notamment une architecture inspirée de PointNet et une architecture de type FBPConvNet.

Une étude comparative permet d'analyser l'influence de différents paramètres sur la qualité de reconstruction :

- Nombre de prises de vues ;
- Nombre d'itérations ;
- Complexité du sujet (nombre d'ellipsoïdes).

## Résultats

Les différentes méthodes sont comparées à travers leurs performances de reconstruction et leurs erreurs respectives.

L'étude met notamment en évidence les possibilités d'amélioration de la reconstruction tomographique par des méthodes itératives et des architectures de Deep Learning, dans un contexte où le nombre de projections disponibles est limité.

### Présentation

> [**Présentation.pdf**](https://github.com/Louis-H-py/TIPE2026/blob/main/documents/Pr%C3%A9sentation.pdf)

---

# <img src="https://github.com/hampusborgos/country-flags/raw/main/svg/gb.svg" width="40"> TIPE 2026 - Cycles and Loops

Year-long Research Project completed in 2026 as part of the French preparatory program (MP track), on the theme **"Cycles and Loops"**.

This project lies at the intersection of **Computer Science**, **Applied Mathematics**, and **Medical Imaging**.

**Author:** Louis H.

**Stack:** 🐍 Python | 🧮 NumPy | 📊 Plotly | 🧠 Keras

## Theoretical Study of Computed Tomography

> How can we design a computer program capable of performing computed tomography, and how can tomographic reconstruction be optimized to reduce a subject's exposure to X-rays?

## Problem Description

Computed Tomography (CT) makes it possible to reconstruct an internal representation of an object from a series of projections acquired at different angles.

The study relies on a two-dimensional model of a subject composed of overlapping ellipsoids with randomly assigned densities, representing an organ.

The objective is to develop a computer program capable of simulating projection acquisition and reconstructing the original image from these measurements, while minimizing the number of projections required.

## Approach

Tomographic reconstruction is investigated through several methods:

- **Back Projection**;
- **Fourier-domain Reconstruction**, based on the Fourier Slice Theorem;
- **Iterative Reconstruction**, formulated as a numerical optimization problem;
- **Neural Networks**, including a PointNet-inspired architecture and an FBPConvNet-based architecture.

A comparative study is conducted to investigate the influence of several parameters on reconstruction quality:

- Number of projections;
- Number of iterations;
- Subject complexity (number of ellipsoids).

## Results

The different methods are compared in terms of reconstruction performance and their respective errors.

The study particularly highlights the potential of iterative reconstruction methods and Deep Learning architectures to improve tomographic reconstruction when only a limited number of projections are available.

### Presentation

> [**Presentation.pdf**](https://github.com/Louis-H-py/TIPE2026/blob/main/documents/Pr%C3%A9sentation.pdf)
