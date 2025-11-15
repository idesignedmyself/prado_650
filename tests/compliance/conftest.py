"""
Pytest fixtures for AFML compliance tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afml_system.pipeline import train_ensemble, predict_ensemble
from afml_system.data.fetch import prepare_training_data


@pytest.fixture(scope="session")
def test_symbol():
    """Test symbol for all compliance tests."""
    return "QQQ"


@pytest.fixture(scope="session")
def test_dates():
    """Test date range with sufficient data."""
    return {
        'start': '2022-01-01',
        'end': '2024-01-01'
    }


@pytest.fixture(scope="session")
def trained_system(test_symbol, test_dates):
    """
    Train a complete system once for all tests.

    Returns dictionary with training results and metadata.
    """
    print(f"\n🧪 Training system for compliance tests...")

    results = train_ensemble(
        symbol=test_symbol,
        start_date=test_dates['start'],
        end_date=test_dates['end']
    )

    return {
        'symbol': test_symbol,
        'results': results,
        'dates': test_dates
    }


@pytest.fixture(scope="session")
def prediction_result(test_symbol, trained_system):
    """
    Generate a prediction using the trained system.

    Returns prediction dictionary.
    """
    print(f"\n🔮 Generating prediction for compliance tests...")

    result = predict_ensemble(symbol=test_symbol)

    return result


@pytest.fixture
def sample_data(test_symbol):
    """
    Fetch sample market data for individual tests.
    """
    data, events = prepare_training_data(
        test_symbol,
        start_date='2023-01-01',
        end_date='2023-06-30',
        use_cusum=True
    )

    return {
        'data': data,
        'events': events,
        'symbol': test_symbol
    }
