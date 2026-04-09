"""Tests for technical indicators."""

from koreaalpha_core.technical.indicators import ema, sma, rsi, macd, bollinger_bands


class TestSMA:
    def test_basic(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = sma(values, 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[9] == 9.0  # (8+9+10)/3

    def test_short_data(self):
        result = sma([1, 2], 5)
        assert all(v is None for v in result)


class TestEMA:
    def test_basic(self):
        values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        result = ema(values, 5)
        assert result[3] is None
        assert result[4] is not None  # first EMA at index 4
        assert result[10] is not None

    def test_short_data(self):
        result = ema([1, 2], 5)
        assert all(v is None for v in result)


class TestRSI:
    def test_uptrend(self):
        closes = list(range(100, 130))  # steady uptrend
        result = rsi(closes, 14)
        valid = [v for v in result if v is not None]
        assert len(valid) > 0
        assert valid[-1] > 70  # overbought in steady uptrend

    def test_downtrend(self):
        closes = list(range(130, 100, -1))  # steady downtrend
        result = rsi(closes, 14)
        valid = [v for v in result if v is not None]
        assert len(valid) > 0
        assert valid[-1] < 30  # oversold

    def test_short_data(self):
        result = rsi([100, 101, 102], 14)
        assert all(v is None for v in result)


class TestMACD:
    def test_basic(self):
        closes = [float(100 + i * 0.5) for i in range(60)]
        result = macd(closes)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        assert len(result["macd"]) == 60

    def test_has_values(self):
        closes = [float(100 + i * 0.5) for i in range(60)]
        result = macd(closes)
        valid_macd = [v for v in result["macd"] if v is not None]
        assert len(valid_macd) > 0


class TestBollingerBands:
    def test_basic(self):
        closes = [float(100 + i * 0.3) for i in range(40)]
        result = bollinger_bands(closes, period=20)
        assert "middle" in result
        assert "upper" in result
        assert "lower" in result

    def test_bands_order(self):
        closes = [float(100 + i * 0.3) for i in range(40)]
        result = bollinger_bands(closes, period=20)
        for i in range(19, 40):
            if result["upper"][i] is not None:
                assert result["upper"][i] > result["middle"][i] > result["lower"][i]
