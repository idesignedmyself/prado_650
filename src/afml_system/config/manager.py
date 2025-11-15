"""
Configuration management for PRADO9 system.
Handles YAML-based configuration loading and validation.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data fetching and processing configuration."""
    symbols: list[str] = field(default_factory=lambda: ["SPY"])
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    interval: str = "1d"
    bar_type: str = "dollar"  # dollar, volume, volatility
    bar_threshold: float = 1e6


@dataclass
class LabelingConfig:
    """Labeling configuration."""
    pt_sl: list[float] = field(default_factory=lambda: [1.0, 1.0])
    min_ret: float = 0.01
    num_threads: int = 4
    vertical_barrier_days: int = 5


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lookback_periods: list[int] = field(default_factory=lambda: [5, 10, 20])
    volatility_window: int = 20
    use_microstructure: bool = True
    use_technical: bool = True


@dataclass
class RegimeConfig:
    """Regime detection configuration."""
    adx_period: int = 14
    adx_threshold: float = 25.0
    ema_period: int = 50
    volume_window: int = 20


@dataclass
class ModelConfig:
    """Model training configuration."""
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    cv_folds: int = 5
    test_size: float = 0.2


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    momentum_lookback: int = 20
    mean_reversion_zscore: float = 2.0
    volatility_threshold: float = 0.02
    pairs_window: int = 60
    scalping_threshold: float = 0.005


@dataclass
class AllocationConfig:
    """Allocation configuration."""
    max_position_size: float = 0.1
    kelly_fraction: float = 0.25
    volatility_target: float = 0.15
    leverage_limit: float = 2.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown: float = 0.2
    max_position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.1


@dataclass
class PRADO9Config:
    """Main PRADO9 configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    state_file: str = "bandit_state.json"
    models_dir: str = "models"
    output_dir: str = "output"


class ConfigManager:
    """Manages configuration loading and access."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> PRADO9Config:
        """Load configuration from file or create default."""
        if self.config_path and os.path.exists(self.config_path):
            return self._load_from_yaml(self.config_path)
        return PRADO9Config()

    def _load_from_yaml(self, path: str) -> PRADO9Config:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return PRADO9Config(
            data=self._dict_to_dataclass(DataConfig, data.get('data', {})),
            labeling=self._dict_to_dataclass(LabelingConfig, data.get('labeling', {})),
            features=self._dict_to_dataclass(FeatureConfig, data.get('features', {})),
            regime=self._dict_to_dataclass(RegimeConfig, data.get('regime', {})),
            model=self._dict_to_dataclass(ModelConfig, data.get('model', {})),
            strategy=self._dict_to_dataclass(StrategyConfig, data.get('strategy', {})),
            allocation=self._dict_to_dataclass(AllocationConfig, data.get('allocation', {})),
            risk=self._dict_to_dataclass(RiskConfig, data.get('risk', {})),
            state_file=data.get('state_file', 'bandit_state.json'),
            models_dir=data.get('models_dir', 'models'),
            output_dir=data.get('output_dir', 'output')
        )

    def _dict_to_dataclass(self, cls, data: Dict[str, Any]):
        """Convert dictionary to dataclass instance."""
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if hasattr(cls, '__dataclass_fields__') and k in cls.__dataclass_fields__})

    def save_to_yaml(self, path: str):
        """Save current configuration to YAML file."""
        config_dict = {
            'data': self._dataclass_to_dict(self.config.data),
            'labeling': self._dataclass_to_dict(self.config.labeling),
            'features': self._dataclass_to_dict(self.config.features),
            'regime': self._dataclass_to_dict(self.config.regime),
            'model': self._dataclass_to_dict(self.config.model),
            'strategy': self._dataclass_to_dict(self.config.strategy),
            'allocation': self._dataclass_to_dict(self.config.allocation),
            'risk': self._dataclass_to_dict(self.config.risk),
            'state_file': self.config.state_file,
            'models_dir': self.config.models_dir,
            'output_dir': self.config.output_dir
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        if not hasattr(obj, '__dataclass_fields__'):
            return obj
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        parts = key.split('.')
        value = self.config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        return value

    def update(self, key: str, value: Any):
        """Update configuration value."""
        parts = key.split('.')
        obj = self.config
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
        setattr(obj, parts[-1], value)
