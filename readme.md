# Projet de Classification de Sons avec CNN

## Description du Projet
Ce projet, réalisé dans le cadre d'un concours Kaggle, utilise des réseaux de neurones convolutionnels (CNN) pour classifier des sons. La première version de ce projet sert de base, tandis que la deuxième version inclut des améliorations significatives grâce à une extraction de données plus poussée.

## Caractéristiques Audio Utilisées
Pour améliorer la performance du modèle, nous exploitons les caractéristiques audio suivantes:
- **MFCCs (Mel Frequency Cepstral Coefficients)**: Ces coefficients sont cruciaux pour capturer la forme du spectre sonore.
- **Spectral Contrast**: Différencie les pics sonores dans différentes bandes de fréquence.
- **Chroma**: Relatif à l'intensité des différentes notes de musique dans le signal audio.
- **Zero Crossing Rate**: Indique le nombre de fois où le signal change de signe.
- **Harmonic and Percussive**: Sépare le son en composantes harmoniques et percussives, ce qui est utile pour analyser des textures sonores complexes.

## Versions du Projet
### Version 1
- Implémentation de base du modèle CNN.
- Utilisation des caractéristiques audio standard pour la classification.

### Version 2
- Extraction accrue des données audio.
- Intégration des caractéristiques avancées mentionnées ci-dessus pour améliorer la précision de classification.

## Article Associé
L'article lié au projet explore l'efficacité des CNN dans la classification des sons d'oiseaux, analysant les performances de différents modèles CNN et l'impact de l'intégration de diverses caractéristiques audio. Les résultats démontrent comment l'ajout de ces caractéristiques améliore la précision des classifications.

## Utilisation
Pour utiliser ce projet, suivez les étapes ci-dessous:
1. Clonez le répertoire GitHub.
2. Installez les dépendances nécessaires en exécutant `pip install -r requirements.txt`.
3. Exécutez le script principal pour entraîner le modèle : `python train.py`.

## Dépendances
- Python 3.x
- TensorFlow 2.x
- Librosa
- NumPy
- Matplotlib (pour la visualisation)

## Contribution
Les contributions à ce projet sont les bienvenues. Veuillez soumettre vos pull requests sur GitHub ou signaler tout problème que vous rencontrez.