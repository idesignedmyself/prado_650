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
    symbol: str = typer.Argument(..., help="Symbol to train on (e.g., SPY, QQQ)"),
    start: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    timeframe: str = typer.Option("1d", "--timeframe", help="Data timeframe (1d, 1h, etc.)"),
):
    """
    Train PRADO9 ensemble for a symbol.

    Example:
        prado train QQQ 2020-01-01 2024-12-31
        prado train SPY 2020-01-01 2023-12-31 --timeframe 1h
    """
    typer.echo(f"🚀 Starting PRADO9 training for {symbol.upper()}...")

    try:
        # Train ensemble
        results = train_ensemble(
            symbol=symbol,
            start_date=start,
            end_date=end,
            timeframe=timeframe
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

        typer.echo(f"\n💾 Models saved to: ~/.prado/models/{symbol}/")

    except Exception as e:
        typer.echo(f"\n❌ Error during training: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Symbol to predict (e.g., SPY, QQQ)"),
    show_all: bool = typer.Option(False, "--show-all", help="Show all strategy details")
):
    """
    Get ensemble prediction for a symbol.

    Example:
        prado predict QQQ
        prado predict SPY --show-all
    """
    typer.echo(f"🔮 Generating predictions for {symbol.upper()}...")

    try:
        # Generate predictions
        result = predict_ensemble(symbol=symbol)

        typer.echo("\n✅ Prediction completed!")
        typer.echo(f"   Symbol: {symbol.upper()}")
        typer.echo(f"   Final Position: {result['final_position']:.2f}")
        typer.echo(f"   Confidence: {result['confidence']:.2%}")
        typer.echo(f"   Active Strategies: {len(result['active_strategies'])}")

        if show_all and 'active_strategies' in result:
            typer.echo("\n📊 Strategy Breakdown:")
            for strategy in result['active_strategies']:
                typer.echo(f"   - {strategy}")

    except Exception as e:
        typer.echo(f"\n❌ Error during prediction: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Symbol to backtest (e.g., SPY, QQQ)"),
    comprehensive: bool = typer.Option(True, "--comprehensive", help="Run all 4 backtest methods"),
    standard: bool = typer.Option(False, "--standard", help="Run standard backtest only"),
    walk_forward: bool = typer.Option(False, "--walk-forward", help="Run walk-forward optimization"),
    crisis: bool = typer.Option(False, "--crisis", help="Run crisis stress test"),
    monte_carlo: bool = typer.Option(False, "--monte-carlo", help="Run Monte Carlo analysis"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
):
    """
    Run comprehensive backtest validation suite.

    Example:
        prado backtest QQQ --comprehensive
        prado backtest SPY --standard
        prado backtest QQQ --start 2020-01-01
    """
    typer.echo(f"📈 Starting backtest for {symbol.upper()}...")

    try:
        # Run comprehensive backtest
        report = backtest_comprehensive(
            symbol=symbol,
            start_date=start
        )

        typer.echo("\n✅ Backtest completed!")
        typer.echo(f"\n{report}")

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
