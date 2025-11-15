# PRADO9

Institutional-grade quantitative trading engine based on Advances in Financial Machine Learning (AFML) methodology.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Train ensemble for a symbol
prado train QQQ

# Get prediction
prado predict QQQ

# Run comprehensive backtest
prado backtest QQQ --comprehensive
```

## System Architecture

- **Phase 0**: Infrastructure (config, persistence, state)
- **Phase 1**: Data & Events (CUSUM, bars, microstructure)
- **Phase 2**: Labeling & Targets (triple-barrier, meta-labels, weights)
- **Phase 3**: Feature Engineering (19 AFML features)
- **Phase 4**: Regime Detection (5 regime types)
- **Phase 5**: Strategy Catalog (7 strategies)
- **Phase 6**: Training Protocol (purged CV)
- **Phase 7**: Meta-Selector (3-gate filtering)
- **Phase 8**: Hybrid Allocator (conflict-aware blending)
- **Phase 9**: Execution & Risk (slippage, commission, limits)
- **Phase 10**: Evaluation Suite (4 validation tests)

## Configuration

Config stored at `~/.prado/config.yaml`

Models stored at `~/.prado/models/{symbol}/`

State stored at `~/.prado/state/`
