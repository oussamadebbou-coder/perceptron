"""
Fichier : test_perceptron.py
Description : Tests unitaires pour la classe Perceptron (version learning_rate / max_iter)
Auteur : Projet Perceptron
"""

import numpy as np
import pytest
import logging
from perceptron import Perceptron
from sklearn.datasets import make_classification

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================================================================
# 🔹 TEST 1 : Initialisation
# ==========================================================================================
def test_initialisation():
    logger.info("Test: Initialisation du perceptron")
    p = Perceptron(learning_rate=0.1, max_iter=100)
    assert isinstance(p.learning_rate, float)
    assert isinstance(p.max_iter, int)
    assert p.weights is None
    assert p.bias is None

# ==========================================================================================
# 🔹 TEST 2 : Apprentissage sur données linéairement séparables (fonction AND)
# ==========================================================================================
def test_predict_linearly_separable():
    logger.info("Test: Fonction AND logique (linéairement séparable)")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # sortie 0/1 pour la logique AND
    p = Perceptron(learning_rate=0.1, max_iter=20)
    p.fit(X, y)
    preds = p.predict(X)
    assert (preds == y).all(), f"Prédictions incorrectes : {preds} != {y}"

# ==========================================================================================
# 🔹 TEST 3 : Effet du learning rate
# ==========================================================================================
def test_learning_rate_effect():
    logger.info("Test: Effet du learning rate")
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    p1 = Perceptron(learning_rate=0.01, max_iter=10)
    p2 = Perceptron(learning_rate=1.0, max_iter=10)

    p1.fit(X, y)
    p2.fit(X, y)

    assert not np.allclose(p1.weights, p2.weights), "Les poids devraient être différents selon le learning rate."

# ==========================================================================================
# 🔹 TEST 4 : Performance sur un dataset généré avec sklearn
# ==========================================================================================
def test_classification_accuracy():
    logger.info("Test: Dataset sklearn (évaluation de la précision)")
    X, y = make_classification(
        n_samples=100, n_features=2, n_classes=2,
        n_informative=2, n_redundant=0, random_state=42
    )
    p = Perceptron(learning_rate=0.1, max_iter=100)
    p.fit(X, y)
    preds = p.predict(X)
    accuracy = np.mean(preds == y)
    logger.info(f"Précision du modèle : {accuracy:.2f}")
    assert accuracy > 0.7, f"Précision trop faible : {accuracy:.2f}"

# ==========================================================================================
# 🔹 TEST 5 : Gestion d’entrées invalides
# ==========================================================================================
def test_invalid_inputs():
    logger.info("Test: Données invalides")
    p = Perceptron(learning_rate=0.1, max_iter=10)
    with pytest.raises(ValueError):
        p.fit(np.array([]), np.array([]))

# ==========================================================================================
# 🔹 TEST 6 : Cas non séparable (XOR)
# ==========================================================================================
def test_non_separable_data():
    logger.info("Test: Cas XOR non séparable")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])  # XOR
    p = Perceptron(learning_rate=0.1, max_iter=100)
    p.fit(X, y)
    preds = p.predict(X)
    acc = np.mean(preds == y)
    logger.info(f"Précision sur XOR : {acc:.2f}")
    assert acc < 1.0, "Le perceptron ne devrait pas résoudre parfaitement XOR."
