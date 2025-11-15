"""
PRADO9 CLI using Typer.
Provides train, predict, and backtest commands.
"""
import typer
from typing import List, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from afml_system.pipeline import train_ensemble, predict_ensemble, backtest_comprehensive
from afml_system.data.fetch import fetch_ohlcv
from afml_system.models.persistence import ModelPersistence
from afml_system.config.manager import ConfigManager

app = typer.Typer(
    name="prado9",
    help="PRADO9 - Advanced Financial Machine Learning System"
)


@app.command()
def train(
    symbols: List[str] = typer.Option(
        ["SPY"],
        "--symbol",
        "-s",
        help="Symbols to train on"
    ),
    start_date: str = typer.Option(
        "2020-01-01",
        "--start",
        help="Start date (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(
        "2023-12-31",
        "--end",
        help="End date (YYYY-MM-DD)"
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file"
    ),
    models_dir: str = typer.Option(
        "models",
        "--models-dir",
        "-m",
        help="Directory to save models"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose",
        "-v",
        help="Verbose output"
    )
):
    """
    Train PRADO9 ensemble models.

    Example:
        prado9 train -s SPY -s QQQ --start 2020-01-01 --end 2023-12-31
    """
    typer.echo("🚀 Starting PRADO9 training...")

    # Load config
    if config_file:
        config_manager = ConfigManager(config_file)
        config = config_manager.config
    else:
        config = None

    try:
        # Train ensemble
        results = train_ensemble(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            config=config,
            save_models=True,
            models_dir=models_dir
        )

        typer.echo("\n✅ Training completed successfully!")

        # Display metrics
        typer.echo("\n📊 Primary Model Metrics:")
        metrics = results['primary_metrics']
        typer.echo(f"  CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        typer.echo(f"  Samples: {metrics['n_samples']}")
        typer.echo(f"  Features: {metrics['n_features']}")

        if 'meta_metrics' in results:
            typer.echo("\n📊 Meta Model Metrics:")
            meta = results['meta_metrics']
            typer.echo(f"  Accuracy: {meta['accuracy']:.4f}")
            typer.echo(f"  Precision: {meta['precision']:.4f}")
            typer.echo(f"  F1 Score: {meta['f1']:.4f}")

        typer.echo(f"\n💾 Models saved to: {models_dir}")

    except Exception as e:
        typer.echo(f"\n❌ Error during training: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def predict(
    symbol: str = typer.Option(
        "SPY",
        "--symbol",
        "-s",
        help="Symbol to predict"
    ),
    models_dir: str = typer.Option(
        "models",
        "--models-dir",
        "-m",
        help="Directory with trained models"
    ),
    lookback_days: int = typer.Option(
        100,
        "--lookback",
        "-l",
        help="Days of historical data for prediction"
    )
):
    """
    Generate predictions using trained models.

    Example:
        prado9 predict -s SPY --models-dir models
    """
    typer.echo("🔮 Generating PRADO9 predictions...")

    try:
        # Load models
        persistence = ModelPersistence(models_dir)

        if not persistence.model_exists('primary_model'):
            typer.echo("❌ No trained models found. Run 'train' first.", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"📂 Loading models from {models_dir}...")

        models = {
            'primary_model': persistence.load_model('primary_model'),
            'meta_model': persistence.load_model('meta_model') if persistence.model_exists('meta_model') else None,
            'strategy_models': {}
        }

        # Load strategy models
        for model_name in persistence.list_models():
            if model_name.startswith('strategy_'):
                strategy = model_name.replace('strategy_', '')
                models['strategy_models'][strategy] = persistence.load_model(model_name)

        typer.echo(f"✅ Loaded {len(models['strategy_models'])} strategy models")

        # Fetch recent data
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        typer.echo(f"\n📊 Fetching data for {symbol}...")
        data = fetch_ohlcv(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if isinstance(data.columns, pd.MultiIndex):
            data = data[symbol]

        # Generate predictions
        typer.echo("🤖 Generating predictions...")
        predictions = predict_ensemble(data, models)

        # Display results
        typer.echo("\n" + "=" * 60)
        typer.echo("PREDICTION RESULTS")
        typer.echo("=" * 60)

        for _, row in predictions.iterrows():
            signal = row['signal']
            confidence = row['confidence']

            signal_emoji = "📈" if signal > 0 else "📉" if signal < 0 else "➡️"
            signal_text = "BUY" if signal > 0 else "SELL" if signal < 0 else "HOLD"

            typer.echo(f"\n{signal_emoji} Signal: {signal_text}")
            typer.echo(f"   Strength: {abs(signal):.2f}")
            typer.echo(f"   Confidence: {confidence:.2%}")
            typer.echo(f"   Strategies: {row['num_strategies']}")

        typer.echo("\n" + "=" * 60)

    except Exception as e:
        typer.echo(f"\n❌ Error during prediction: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def backtest(
    symbols: List[str] = typer.Option(
        ["SPY"],
        "--symbol",
        "-s",
        help="Symbols to backtest"
    ),
    start_date: str = typer.Option(
        "2020-01-01",
        "--start",
        help="Start date (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(
        "2023-12-31",
        "--end",
        help="End date (YYYY-MM-DD)"
    ),
    method: str = typer.Option(
        "simple",
        "--method",
        help="Backtest method (simple, walk_forward)"
    ),
    initial_capital: float = typer.Option(
        100000,
        "--capital",
        help="Initial capital"
    ),
    models_dir: Optional[str] = typer.Option(
        None,
        "--models-dir",
        "-m",
        help="Directory with trained models (optional)"
    )
):
    """
    Run comprehensive backtest.

    Example:
        prado9 backtest -s SPY --method walk_forward --capital 100000
    """
    typer.echo("📈 Starting PRADO9 backtest...")

    try:
        # Load models if directory provided
        models = None
        if models_dir:
            persistence = ModelPersistence(models_dir)
            if persistence.model_exists('primary_model'):
                typer.echo(f"📂 Loading models from {models_dir}...")
                models = {
                    'primary_model': persistence.load_model('primary_model')
                }

        # Run backtest
        results = backtest_comprehensive(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            models=models,
            method=method,
            initial_capital=initial_capital
        )

        typer.echo("\n✅ Backtest completed successfully!")

    except Exception as e:
        typer.echo(f"\n❌ Error during backtest: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def info():
    """Display PRADO9 system information."""
    typer.echo("=" * 60)
    typer.echo("PRADO9 - Advanced Financial Machine Learning System")
    typer.echo("=" * 60)
    typer.echo("\nComponents:")
    typer.echo("  ✓ Data: yfinance, CUSUM filter, information-driven bars")
    typer.echo("  ✓ Labels: Triple barrier method, meta-labeling")
    typer.echo("  ✓ Features: 19-feature matrix (stationarity, volatility, microstructure)")
    typer.echo("  ✓ Regimes: 5 regime detectors (trend, volatility, volume, micro, composite)")
    typer.echo("  ✓ Strategies: 7 strategies (momentum, mean reversion, volatility, pairs, seasonality, scalping, sentiment)")
    typer.echo("  ✓ Models: Primary + Meta + Strategy ensemble")
    typer.echo("  ✓ Allocation: Hybrid allocator (Kelly, vol targeting, risk parity)")
    typer.echo("  ✓ Execution: Trade simulation with slippage and costs")
    typer.echo("  ✓ Risk: Position limits, drawdown control, concentration limits")
    typer.echo("\n" + "=" * 60)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
