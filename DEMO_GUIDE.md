# PRADO9 Demo Scripts Guide

## Overview

This guide describes the 8 demo scripts created for the PRADO9 Advanced Financial Machine Learning system. Each demo is self-contained and demonstrates a specific phase of the system.

## Files Created

### 1. __init__.py Files (Updated)
All module `__init__.py` files have been updated with proper imports for public functions and classes:

- `/src/afml_system/config/__init__.py` - Configuration management
- `/src/afml_system/state/__init__.py` - Bandit state management
- `/src/afml_system/data/__init__.py` - Data fetching and processing
- `/src/afml_system/labeling/__init__.py` - Triple barrier labeling
- `/src/afml_system/features/__init__.py` - Feature engineering
- `/src/afml_system/regime/__init__.py` - Regime detection
- `/src/afml_system/models/__init__.py` - Model training
- `/src/afml_system/strategies/__init__.py` - Trading strategies
- `/src/afml_system/execution/__init__.py` - Trade execution
- `/src/afml_system/evaluation/__init__.py` - Backtesting and metrics
- `/src/afml_system/allocation/__init__.py` - Position sizing

### 2. Demo Scripts (Created)

#### Phase 0: Configuration and State Management
**File:** `/examples/phase0_demo.py`
**Size:** 7.1 KB

Demonstrates:
- Configuration manager initialization and defaults
- Loading and updating configuration
- Bandit state manager for multi-armed bandit strategy selection
- Thompson Sampling for arm selection
- State persistence and serialization

Key Functions Shown:
- ConfigManager initialization
- Configuration access and updates
- BanditStateManager registration and updates
- Arm probability calculation
- Best arm selection

#### Phase 1: CUSUM Filter and Information-Driven Bars
**File:** `/examples/phase1_demo.py`
**Size:** 7.4 KB

Demonstrates:
- CUSUM filter for event detection
- Symmetric and adaptive CUSUM filters
- Dollar bars generation
- Volume bars generation
- Volatility bars generation
- Imbalance bars generation
- Bar generation comparison and analysis

Key Functions Shown:
- cusum_filter() - Detects significant price movements
- cusum_filter_symmetric() - Tracks up/down moves
- adaptive_cusum_filter() - Volatility-adjusted threshold
- dollar_bars() - Price x volume sampled bars
- volume_bars() - Share count sampled bars
- volatility_bars() - Absolute return sampled bars

#### Phase 2: Triple Barrier Labeling
**File:** `/examples/phase2_demo.py`
**Size:** 7.2 KB

Demonstrates:
- Triple barrier method for generating labels
- Daily volatility calculation
- Label distribution analysis
- Sample weight calculation for class imbalance
- Label filtering and cleaning
- Bin conversion for multi-class problems

Key Functions Shown:
- triple_barrier_labels() - Generate trading labels
- get_daily_volatility() - Calculate volatility
- get_bins() - Convert labels to bins
- drop_labels() - Filter rare labels
- get_sample_weights() - Weight classes

#### Phase 3: Feature Building
**File:** `/examples/phase3_demo.py`
**Size:** 8.5 KB

Demonstrates:
- Technical indicator calculation (SMA, EMA, RSI, MACD)
- Price-based features
- Volume-based features
- Automatic feature matrix building
- Extended feature building
- Feature selection and normalization
- Z-score vs min-max normalization

Key Functions Shown:
- sma(), ema(), rsi(), macd() - Technical indicators
- build_feature_matrix() - Automatic feature generation
- build_extended_features() - Additional features
- select_top_features() - Feature importance selection
- normalize_features() - Z-score/min-max normalization

#### Phase 4: Regime Detection
**File:** `/examples/phase4_demo.py`
**Size:** 8.5 KB

Demonstrates:
- Trend regime detection (ADX-based)
- Volatility regime detection
- Volume regime detection
- Composite regime detection
- Regime-aware strategy selection
- Strategy recommendations by regime

Key Functions Shown:
- TrendRegimeDetector - ADX-based trend/range detection
- VolatilityRegimeDetector - High/low volatility detection
- VolumeRegimeDetector - High/normal/low volume detection
- CompositeRegimeDetector - Combined regime detection
- detect_all_regimes() - Full regime analysis

#### Phase 5: Strategy Predictions
**File:** `/examples/phase5_demo.py`
**Size:** 9.6 KB

Demonstrates:
- Individual strategy signal generation
- Signal agreement between strategies
- Simple ensemble averaging
- Regime-filtered signals
- Signal quality metrics
- Win/loss ratio analysis
- Hit rate calculation

Key Functions Shown:
- MomentumStrategy.generate_signals()
- MeanReversionStrategy.generate_signals()
- VolatilityStrategy.generate_signals()
- Signal correlation analysis
- Ensemble prediction

#### Phase 6: Model Training
**File:** `/examples/phase6_demo.py`
**Size:** 9.7 KB

Demonstrates:
- Training data preparation
- Primary model training (Random Forest)
- Cross-validation setup
- Model evaluation
- Feature importance analysis
- Model persistence (save/load)
- Meta-model training for strategy selection

Key Functions Shown:
- train_primary_model() - Train classifier
- train_meta_model() - Train strategy selector
- PurgedKFold - Non-overlapping cross-validation
- ModelPersistence - Save/load models
- Feature importance analysis

#### Full Pipeline Demo: End-to-End Integration
**File:** `/examples/full_pipeline_demo.py`
**Size:** 13 KB

Demonstrates:
- Integration of all 6 phases
- Data flow from raw market data to predictions
- Configuration initialization
- Event detection and labeling
- Feature engineering
- Regime detection
- Strategy signal generation
- Model training with cross-validation

## Running the Demos

### Prerequisites
```bash
cd /Users/darraykennedy/Desktop/python_pro/prado9_nov_15
source env/bin/activate  # Activate virtual environment
```

### Phase 0: Configuration and State Management
```bash
python examples/phase0_demo.py
```
Output: Configuration settings, bandit arm statistics, strategy probabilities

### Phase 1: CUSUM and Bars
```bash
python examples/phase1_demo.py
```
Output: Event detection results, bar generation statistics, comparison metrics

### Phase 2: Triple Barrier Labeling
```bash
python examples/phase2_demo.py
```
Output: Label distribution, return statistics, sample weights

### Phase 3: Feature Building
```bash
python examples/phase3_demo.py
```
Output: Technical indicators, feature statistics, normalization results

### Phase 4: Regime Detection
```bash
python examples/phase4_demo.py
```
Output: Regime breakdown, strategy recommendations, probability analysis

### Phase 5: Strategy Predictions
```bash
python examples/phase5_demo.py
```
Output: Strategy signals, signal agreement, ensemble results, win/loss ratios

### Phase 6: Model Training
```bash
python examples/phase6_demo.py
```
Output: Model performance, cross-validation scores, feature importance

### Full Pipeline
```bash
python examples/full_pipeline_demo.py
```
Output: Complete system demonstration with all phases integrated

## Key Concepts Demonstrated

### Configuration Management
- YAML/Python-based configuration
- Hierarchical config structure
- Dynamic configuration updates
- Config persistence

### Multi-Armed Bandits
- Thompson Sampling
- UCB (Upper Confidence Bound)
- Epsilon-Greedy strategies
- Strategy selection and learning

### Event Detection
- CUSUM filter implementation
- Adaptive thresholds
- Tick direction detection
- Event-based vs time-based sampling

### Information-Driven Bars
- Dollar bars (dollar volume)
- Volume bars (share count)
- Volatility bars (price movement)
- Imbalance bars (order flow)

### Labeling Methods
- Triple barrier framework
- Horizontal barriers (profit/loss)
- Vertical barriers (time limit)
- Meta-labels for secondary models
- Sample weighting for imbalance

### Feature Engineering
- Technical indicators (SMA, EMA, RSI, MACD)
- Price-based features (returns, ranges)
- Volume-based features
- Microstructure features
- Stationarity testing
- Feature normalization

### Regime Detection
- ADX for trend/range detection
- Volatility regimes
- Volume regimes
- Composite regime detection
- Regime-aware strategy selection

### Strategy Ensemble
- Individual strategy signals
- Signal aggregation
- Consensus signals
- Regime filtering
- Strategy ranking

### Model Training
- Feature matrix preparation
- Purged K-Fold cross-validation
- Random Forest classification
- Feature importance analysis
- Model persistence

## Output Directories

Demo scripts may create temporary files in:
- `/tmp/prado9_*.json` - State files
- `/tmp/prado9_*.pkl` - Model files

## Customization

Each demo can be customized by:
1. Modifying date ranges
2. Changing symbols (SPY, QQQ, etc.)
3. Adjusting parameters (thresholds, windows, etc.)
4. Changing model hyperparameters
5. Modifying ensemble weights

## Next Steps

After exploring the demos:
1. Combine multiple demos into custom pipelines
2. Add additional symbols and time periods
3. Implement real-time prediction
4. Add backtesting and live trading
5. Monitor model performance
6. Retrain models periodically

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Check that src directory is in Python path
- Verify all dependencies are installed

### Data Errors
- Demos use SPY data by default
- May require internet connection for data fetching
- Check data availability for date ranges

### Model Errors
- Cross-validation requires minimum samples
- Some strategies need sufficient data history
- Feature normalization handles NaN values

## Integration with Production

To integrate these demos into production:
1. Extract key functions into modules
2. Add real-time data feeds
3. Implement position sizing and risk management
4. Add order execution
5. Monitor and alert on performance
6. Implement model retraining schedule

## File Locations

All files are located at:
```
/Users/darraykennedy/Desktop/python_pro/prado9_nov_15/
├── examples/
│   ├── phase0_demo.py
│   ├── phase1_demo.py
│   ├── phase2_demo.py
│   ├── phase3_demo.py
│   ├── phase4_demo.py
│   ├── phase5_demo.py
│   ├── phase6_demo.py
│   └── full_pipeline_demo.py
└── src/afml_system/
    ├── config/__init__.py
    ├── state/__init__.py
    ├── data/__init__.py
    ├── labeling/__init__.py
    ├── features/__init__.py
    ├── regime/__init__.py
    ├── models/__init__.py
    ├── strategies/__init__.py
    ├── execution/__init__.py
    ├── evaluation/__init__.py
    └── allocation/__init__.py
```
