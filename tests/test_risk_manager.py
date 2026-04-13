from risk_manager import RiskManager


def test_calculate_position_size_returns_positive_fractional_qty():
    risk = RiskManager(equity=1000.0, open_positions=0)
    qty = risk.calculate_position_size(
        entry_price=50.0,
        stop_price=45.0,
        buying_power=1000.0,
    )

    assert qty > 0


def test_calculate_short_position_size_requires_stop_above_entry():
    risk = RiskManager(equity=1000.0, open_positions=0)

    invalid_qty = risk.calculate_short_position_size(
        entry_price=50.0,
        stop_price=49.0,
        buying_power=1000.0,
    )
    valid_qty = risk.calculate_short_position_size(
        entry_price=50.0,
        stop_price=55.0,
        buying_power=1000.0,
    )

    assert invalid_qty == 0
    assert valid_qty > 0