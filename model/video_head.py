"""Classification temporelle des caractéristiques produites par le CNN partagé."""

from torch import Tensor, nn


class VideoHead(nn.Module):
    """Transformer les caractéristiques ordonnées [B, T, 128] en logits bruts [B].

    La première version utilise T=16 images échantillonnées à 8 images/s.
    Chaque séquence doit contenir le même visage, sans remplissage ni
    images manquantes. Les paramètres sont initialisés aléatoirement par
    PyTorch ; aucun poids préentraîné n'est chargé. Utiliser
    BCEWithLogitsLoss pendant l'entraînement et appliquer la sigmoïde
    uniquement pour convertir les logits en scores de suspicion à la prédiction.
    """

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Renvoyer un logit par séquence en conservant la dimension du lot."""
        if features.ndim != 3 or features.shape[-1] != 128:
            raise ValueError("Expected CNN features with shape [B, T, 128].")
        if features.shape[1] == 0:
            raise ValueError("A video sequence must contain at least one frame.")

        _, hidden = self.gru(features)
        # L'état final résume les images dans leur ordre temporel d'origine.
        return self.classifier(hidden[-1]).squeeze(-1)
