# Bitcoin Trading Strategies Simulator

## Project Purpose
This application aims to regularly fetch the Bitcoin price using the Binance API and simulate five popular and safe trading strategies simultaneously in a virtual environment for comparison. Each strategy trades with its own virtual balance of 10,000 TL, and the results are presented with detailed graphs and profit/loss tables.

## Features
- Fetch the Bitcoin price every five minutes using your Binance API key
- Apply the five most commonly used and trusted trading strategies
- Simulate long/short trades with an independent 10,000 TL balance for each strategy
- Real-time buy/sell simulation adjusts position size according to signal strength
- Optional "full balance" mode executes each trade with the entire balance for easier comparison
- Commission and slippage costs can be simulated for each trade
- Adaptive position scaling reacts to winning streaks and market trend
- Dynamic take-profit and trailing-stop levels adjust with volatility and signal strength
- Adaptive opportunity trigger lowers the threshold after missed moves
- Missed opportunities are marked on charts and summarized with potential profit
- Profit table compares realized profit with expected profit if missed trades were taken
- EMA cross and volume breakout detection trigger initial trades with a small
  position which can pyramid as the trend confirms
- Sells may override the signal strength and liquidate the entire position when the expected profit exceeds 2%
- Separate graph for each strategy:
  - Price curve
  - Buy points (red), sell points (green)
- Collective profit/loss table:
  - Total gains/losses of each strategy
  - Remaining BTC and its value per strategy
  - Final balance for each strategy
- Detailed trade log listing each trade with updated balance
- Easy switching between graphs and table
- Modern and user-friendly interface

## Installation
- Python 3.10+ must be installed
- Install the required libraries:
```bash
pip install -r requirements.txt
```
- Add your Binance API Key and Secret to `settings.py` or the `.env` file.

## Usage
Start the application:
```bash
python main.py
```
Examine the performance and charts of the strategies in the application interface.

Click the profit/loss table icon to see the financial summary of all strategies.

## Project Architecture
- `services/data_service.py`: Fetches price data from Binance
- `strategies/`: Separate Python module for each trading strategy
- `services/simulation.py`: Executes buy/sell actions and balance simulation
- `services/logger.py`: Records executed trades and strategy performance
- `main.py`: Main application flow and interface management

## Strategies
Initially, the following example strategies will be used:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- MA Cross (Moving Average Cross)
- Custom (User defined or added later)
- Gradient (Normalized gradient and volatility adjusted threshold)
- Dynamic Hybrid (ATR & volume filters, adaptive risk management, session-aware
 thresholds, market regime detection, multi-timeframe trend filters, commission/slippage simulation and parameter optimization)
  with missed-opportunity detection, automatic trading of missed signals, adaptive thresholding, and an expected-profit table
Each strategy triggers trades according to its own rules and visualizes the results.

## Development Plan
- New modules for strategy algorithms can be added
- Additional filters, indicators, and performance analysis panels will be developed for the interface
- Integration with real trading environments and an alert system is planned

## Contribution and Contact
Please open a pull request or contact us if you would like to contribute or make suggestions.

## License
This project is licensed under the MIT License.

For the Turkish version of this document, see [readme.md](readme.md).

### Example Parameter Tuning
```python
from strategies.dynamic_hybrid import DynamicHybridStrategy
from services.data_service import DataService

prices = DataService().get_historical_prices(limit=500)
strategy = DynamicHybridStrategy()
grid = {"base_threshold": [0.1, 0.15], "lookback": [40, 60]}
best = strategy.optimize_by_regime(prices, grid)
print(best)
```
