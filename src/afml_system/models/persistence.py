"""
Model persistence with metadata.
Handles saving and loading of trained models with associated metadata.
"""
import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import joblib


class ModelPersistence:
    """Handles model saving and loading with metadata."""

    def __init__(self, models_dir: str = "models"):
        """
        Initialize model persistence.

        Args:
            models_dir: Directory to save models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict] = None,
        use_joblib: bool = True
    ) -> str:
        """
        Save model with metadata.

        Args:
            model: Model object to save
            name: Model name
            metadata: Optional metadata dictionary
            use_joblib: Use joblib (True) or pickle (False)

        Returns:
            Path to saved model
        """
        # Create model directory
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        if use_joblib:
            joblib.dump(model, model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Prepare metadata
        full_metadata = {
            'name': name,
            'saved_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'model_path': str(model_path)
        }

        if metadata:
            full_metadata.update(metadata)

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)

        return str(model_path)

    def load_model(self, name: str, use_joblib: bool = True) -> Any:
        """
        Load model by name.

        Args:
            name: Model name
            use_joblib: Use joblib (True) or pickle (False)

        Returns:
            Loaded model
        """
        model_path = self.models_dir / name / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {name}")

        if use_joblib:
            return joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                return pickle.load(f)

    def load_metadata(self, name: str) -> Dict:
        """
        Load model metadata.

        Args:
            name: Model name

        Returns:
            Metadata dictionary
        """
        metadata_path = self.models_dir / name / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for model: {name}")

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def save_model_with_config(
        self,
        model: Any,
        name: str,
        config: Dict,
        metrics: Optional[Dict] = None
    ) -> str:
        """
        Save model with configuration and metrics.

        Args:
            model: Model to save
            name: Model name
            config: Training configuration
            metrics: Training metrics

        Returns:
            Path to saved model
        """
        metadata = {
            'config': config,
            'metrics': metrics or {}
        }

        return self.save_model(model, name, metadata)

    def list_models(self) -> list[str]:
        """List all saved models."""
        if not self.models_dir.exists():
            return []

        models = []
        for item in self.models_dir.iterdir():
            if item.is_dir() and (item / "model.pkl").exists():
                models.append(item.name)

        return sorted(models)

    def delete_model(self, name: str):
        """
        Delete a saved model.

        Args:
            name: Model name to delete
        """
        model_dir = self.models_dir / name

        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)

    def model_exists(self, name: str) -> bool:
        """Check if model exists."""
        model_path = self.models_dir / name / "model.pkl"
        return model_path.exists()

    def get_model_info(self, name: str) -> Dict:
        """
        Get model information.

        Args:
            name: Model name

        Returns:
            Dictionary with model info
        """
        try:
            metadata = self.load_metadata(name)
            model_path = self.models_dir / name / "model.pkl"

            return {
                'name': name,
                'exists': model_path.exists(),
                'size_mb': model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0,
                'metadata': metadata
            }
        except Exception as e:
            return {
                'name': name,
                'exists': False,
                'error': str(e)
            }

    def save_ensemble_models(
        self,
        models: Dict[str, Any],
        ensemble_name: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save multiple models as an ensemble.

        Args:
            models: Dictionary of model_name -> model
            ensemble_name: Name for the ensemble
            metadata: Optional ensemble metadata
        """
        ensemble_dir = self.models_dir / ensemble_name
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save each model
        for model_name, model in models.items():
            model_path = ensemble_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)

        # Save ensemble metadata
        ensemble_metadata = {
            'ensemble_name': ensemble_name,
            'saved_at': datetime.now().isoformat(),
            'models': list(models.keys()),
            'num_models': len(models)
        }

        if metadata:
            ensemble_metadata.update(metadata)

        metadata_path = ensemble_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)

    def load_ensemble_models(self, ensemble_name: str) -> Dict[str, Any]:
        """
        Load ensemble models.

        Args:
            ensemble_name: Ensemble name

        Returns:
            Dictionary of model_name -> model
        """
        ensemble_dir = self.models_dir / ensemble_name

        if not ensemble_dir.exists():
            raise FileNotFoundError(f"Ensemble not found: {ensemble_name}")

        # Load metadata to get model names
        metadata = self.load_metadata(ensemble_name)
        model_names = metadata.get('models', [])

        # Load each model
        models = {}
        for model_name in model_names:
            model_path = ensemble_dir / f"{model_name}.pkl"
            if model_path.exists():
                models[model_name] = joblib.load(model_path)

        return models


# Helper functions for pipeline integration

def get_prado_home() -> Path:
    """Get PRADO home directory (~/.prado)."""
    home = Path.home() / ".prado"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _sanitize_metrics(metrics: Dict) -> Dict:
    """Convert metrics to JSON-serializable format."""
    import pandas as pd
    import numpy as np

    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            sanitized[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
        elif isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
            sanitized[key] = float(value) if 'float' in str(type(value)) else int(value)
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_metrics(value)
        else:
            sanitized[key] = value
    return sanitized


def save_ensemble(
    symbol: str,
    primary_model: Any,
    meta_model: Any,
    strategy_models: Dict[str, Any],
    primary_metrics: Dict,
    meta_metrics: Dict,
    strategy_metrics: Dict[str, Dict]
) -> str:
    """
    Save complete ensemble for a symbol.

    Args:
        symbol: Trading symbol
        primary_model: Primary prediction model
        meta_model: Meta-labeling model
        strategy_models: Dictionary of strategy_name -> model
        primary_metrics: Primary model metrics
        meta_metrics: Meta model metrics
        strategy_metrics: Dictionary of strategy metrics

    Returns:
        Path to saved ensemble
    """
    models_dir = get_prado_home() / "models"
    persistence = ModelPersistence(models_dir=str(models_dir))

    # Create ensemble models dictionary
    models = {
        'primary': primary_model,
        'meta': meta_model,
    }

    # Add strategy models
    for name, model in strategy_models.items():
        models[f'strategy_{name}'] = model

    # Sanitize metrics for JSON serialization
    clean_primary_metrics = _sanitize_metrics(primary_metrics)
    clean_meta_metrics = _sanitize_metrics(meta_metrics)
    clean_strategy_metrics = {name: _sanitize_metrics(m) for name, m in strategy_metrics.items()}

    # Create metadata
    metadata = {
        'symbol': symbol,
        'primary_metrics': clean_primary_metrics,
        'meta_metrics': clean_meta_metrics,
        'strategy_metrics': clean_strategy_metrics,
        'strategies': list(strategy_models.keys())
    }

    # Save ensemble
    persistence.save_ensemble_models(models, symbol, metadata)

    ensemble_path = models_dir / symbol
    return str(ensemble_path)


def load_ensemble(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Load ensemble for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Dictionary with loaded models and metadata, or None if not found
    """
    models_dir = get_prado_home() / "models"
    persistence = ModelPersistence(models_dir=str(models_dir))

    try:
        # Load ensemble models
        models = persistence.load_ensemble_models(symbol)

        # Load metadata
        metadata = persistence.load_metadata(symbol)

        # Extract models
        primary_model = models.get('primary')
        meta_model = models.get('meta')

        # Extract strategy models
        strategy_models = {}
        for key, model in models.items():
            if key.startswith('strategy_'):
                strategy_name = key.replace('strategy_', '')
                strategy_models[strategy_name] = model

        return {
            'primary_model': primary_model,
            'meta_model': meta_model,
            'strategy_models': strategy_models,
            'metadata': metadata
        }

    except FileNotFoundError:
        return None


def list_trained_symbols() -> list[str]:
    """List all symbols with trained models."""
    models_dir = get_prado_home() / "models"
    persistence = ModelPersistence(models_dir=str(models_dir))
    return persistence.list_models()


def delete_ensemble(symbol: str):
    """Delete ensemble for a symbol."""
    models_dir = get_prado_home() / "models"
    persistence = ModelPersistence(models_dir=str(models_dir))
    persistence.delete_model(symbol)


def save_regime_ensemble(
    symbol: str,
    regime_models: Dict[Tuple[str, str], Dict],
    regime_metrics: Dict[Tuple[str, str], Dict]
) -> str:
    """
    Save per-regime ensemble models.

    Args:
        symbol: Trading symbol
        regime_models: Dict of (strategy, regime) -> {primary, meta, strategy_obj}
        regime_metrics: Dict of (strategy, regime) -> {primary, meta, n_samples}

    Returns:
        Path to saved ensemble
    """
    models_dir = get_prado_home() / "models"
    persistence = ModelPersistence(models_dir=str(models_dir))

    # Create models dictionary
    models = {}
    for (strategy, regime), model_dict in regime_models.items():
        key = f"{strategy}_{regime}"
        models[f"{key}_primary"] = model_dict['primary']
        models[f"{key}_meta"] = model_dict['meta']

    # Sanitize metrics
    clean_metrics = {}
    for (strategy, regime), metrics in regime_metrics.items():
        key = f"{strategy}_{regime}"
        clean_metrics[key] = _sanitize_metrics(metrics)

    # Create metadata
    metadata = {
        'symbol': symbol,
        'regime_metrics': clean_metrics,
        'model_keys': [(s, r) for s, r in regime_models.keys()],
        'total_models': len(regime_models)
    }

    # Save ensemble
    persistence.save_ensemble_models(models, symbol, metadata)

    ensemble_path = models_dir / symbol
    return str(ensemble_path)


def load_regime_ensemble(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Load per-regime ensemble models.

    Args:
        symbol: Trading symbol

    Returns:
        Dictionary with loaded models and metadata, or None if not found
    """
    models_dir = get_prado_home() / "models"
    persistence = ModelPersistence(models_dir=str(models_dir))

    try:
        # Load ensemble models
        models = persistence.load_ensemble_models(symbol)

        # Load metadata
        metadata = persistence.load_metadata(symbol)

        # Reconstruct regime models
        regime_models = {}
        model_keys = metadata.get('model_keys', [])

        for strategy, regime in model_keys:
            key = f"{strategy}_{regime}"
            regime_models[(strategy, regime)] = {
                'primary': models.get(f"{key}_primary"),
                'meta': models.get(f"{key}_meta")
            }

        return {
            'regime_models': regime_models,
            'metadata': metadata
        }

    except FileNotFoundError:
        return None
