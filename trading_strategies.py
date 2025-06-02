"""
Trading Strategy Implementation Module

This module provides comprehensive trading strategy functionality including:
- Strategy backtesting against historical data
- Custom strategy builder with entry/exit conditions
- Risk management and position sizing
- Portfolio integration and analysis

Built on top of the existing StockPredictor infrastructure.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from predictor_fixed import StockPredictor
except ImportError:
    from predictor import StockPredictor

class TradingStrategy:
    """
    Core trading strategy class that handles strategy definition and execution
    """
    
    def __init__(self, name: str = "Custom Strategy"):
        """
        Initialize a trading strategy
        
        Parameters:
        -----------
        name : str
            Name of the trading strategy
        """
        self.name = name
        self.entry_conditions = []
        self.exit_conditions = []
        self.stop_loss = None
        self.take_profit = None
        self.position_sizing = 'fixed'
        self.risk_tolerance = 0.02  # 2% default risk per trade
        
    def add_entry_condition(self, indicator: str, operator: str, value: Union[float, int]):
        """
        Add an entry condition to the strategy
        
        Parameters:
        -----------
        indicator : str
            Technical indicator name (RSI, SMA_Crossover, BBPosition, etc.)
        operator : str
            Comparison operator ('<', '>', '<=', '>=', '==', '!=')
        value : float or int
            Threshold value for comparison
        """
        self.entry_conditions.append({
            'indicator': indicator,
            'operator': operator,
            'value': value
        })
        
    def add_exit_condition(self, indicator: str, operator: str, value: Union[float, int]):
        """
        Add an exit condition to the strategy
        
        Parameters:
        -----------
        indicator : str
            Technical indicator name
        operator : str
            Comparison operator
        value : float or int
            Threshold value for comparison
        """
        self.exit_conditions.append({
            'indicator': indicator,
            'operator': operator,
            'value': value
        })
        
    def set_risk_management(self, stop_loss: float = None, take_profit: float = None, 
                           position_sizing: str = 'fixed', risk_tolerance: float = 0.02):
        """
        Set risk management parameters
        
        Parameters:
        -----------
        stop_loss : float
            Stop loss percentage (e.g., 0.03 for 3%)
        take_profit : float
            Take profit percentage (e.g., 0.10 for 10%)
        position_sizing : str
            Position sizing method ('fixed', 'percent_risk', 'kelly')
        risk_tolerance : float
            Risk tolerance as percentage of portfolio (e.g., 0.02 for 2%)
        """
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_sizing = position_sizing
        self.risk_tolerance = risk_tolerance

class StrategyBacktester:
    """
    Backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the backtester
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.predictor = StockPredictor()
        
    def _evaluate_condition(self, df_row: pd.Series, condition: dict) -> bool:
        """
        Evaluate a single trading condition
        
        Parameters:
        -----------
        df_row : pd.Series
            Row of data with technical indicators
        condition : dict
            Condition dictionary with indicator, operator, and value
            
        Returns:
        --------
        bool
            True if condition is met, False otherwise
        """
        try:
            indicator_value = float(df_row[condition['indicator']])
            threshold = float(condition['value'])
            operator = condition['operator']
            
            if operator == '<':
                return indicator_value < threshold
            elif operator == '>':
                return indicator_value > threshold
            elif operator == '<=':
                return indicator_value <= threshold
            elif operator == '>=':
                return indicator_value >= threshold
            elif operator == '==':
                return abs(indicator_value - threshold) < 1e-6  # Float comparison
            elif operator == '!=':
                return abs(indicator_value - threshold) >= 1e-6
            else:
                return False
        except (KeyError, ValueError, TypeError):
            return False
    
    def _check_entry_signal(self, df_row: pd.Series, strategy: TradingStrategy) -> bool:
        """
        Check if all entry conditions are met
        
        Parameters:
        -----------
        df_row : pd.Series
            Row of data with technical indicators
        strategy : TradingStrategy
            Trading strategy object
            
        Returns:
        --------
        bool
            True if entry signal is triggered
        """
        if not strategy.entry_conditions:
            return False
            
        return all(self._evaluate_condition(df_row, condition) 
                  for condition in strategy.entry_conditions)
    
    def _check_exit_signal(self, df_row: pd.Series, strategy: TradingStrategy) -> bool:
        """
        Check if any exit conditions are met
        
        Parameters:
        -----------
        df_row : pd.Series
            Row of data with technical indicators
        strategy : TradingStrategy
            Trading strategy object
            
        Returns:
        --------
        bool
            True if exit signal is triggered
        """
        if not strategy.exit_conditions:
            return False
            
        return any(self._evaluate_condition(df_row, condition) 
                  for condition in strategy.exit_conditions)
    
    def _calculate_position_size(self, current_capital: float, current_price: float, 
                               strategy: TradingStrategy, volatility: float = 0.02) -> int:
        """
        Calculate position size based on risk management rules
        
        Parameters:
        -----------
        current_capital : float
            Current available capital
        current_price : float
            Current stock price
        strategy : TradingStrategy
            Trading strategy object
        volatility : float
            Stock volatility estimate
            
        Returns:
        --------
        int
            Number of shares to trade
        """
        if strategy.position_sizing == 'fixed':
            # Fixed dollar amount (10% of capital)
            position_value = current_capital * 0.1
            return int(position_value / current_price)
        
        elif strategy.position_sizing == 'percent_risk':
            # Risk-based position sizing
            risk_amount = current_capital * strategy.risk_tolerance
            
            # Calculate stop loss distance
            if strategy.stop_loss:
                stop_distance = strategy.stop_loss
            else:
                stop_distance = volatility * 2  # Use 2x volatility as default
            
            # Position size = Risk Amount / (Price * Stop Distance)
            position_value = risk_amount / stop_distance
            return int(position_value / current_price)
        
        elif strategy.position_sizing == 'kelly':
            # Simplified Kelly Criterion (would need win rate and avg win/loss in practice)
            # Using conservative 25% of capital for now
            position_value = current_capital * 0.25
            return int(position_value / current_price)
        
        else:
            # Default to 10% of capital
            position_value = current_capital * 0.1
            return int(position_value / current_price)
    
    def backtest_strategy(self, df: pd.DataFrame, strategy: TradingStrategy) -> Dict:
        """
        Backtest a trading strategy against historical data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock price dataframe with OHLCV data
        strategy : TradingStrategy
            Trading strategy to backtest
            
        Returns:
        --------
        dict
            Comprehensive backtest results
        """
        # Engineer features using existing predictor
        feature_df = self.predictor.engineer_features(df)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares held
        entry_price = 0
        entry_date = None
        trades = []
        equity_curve = []
        
        # Track positions and signals
        in_position = False
        
        for idx, row in feature_df.iterrows():
            current_price = float(row['Close'])
            current_date = row['Date']
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * current_price)
            equity_curve.append({
                'Date': current_date,
                'Portfolio_Value': portfolio_value,
                'Price': current_price
            })
            
            if not in_position:
                # Check for entry signals
                if self._check_entry_signal(row, strategy):
                    # Calculate position size
                    volatility = float(row.get('PriceRange', 0.02))
                    shares_to_buy = self._calculate_position_size(
                        capital, current_price, strategy, volatility
                    )
                    
                    # Ensure we have enough capital
                    cost = shares_to_buy * current_price
                    if cost <= capital and shares_to_buy > 0:
                        # Enter position
                        position = shares_to_buy
                        entry_price = current_price
                        entry_date = current_date
                        capital -= cost
                        in_position = True
                        
                        # Record trade entry
                        trades.append({
                            'Type': 'Entry',
                            'Date': current_date,
                            'Price': current_price,
                            'Shares': shares_to_buy,
                            'Value': cost
                        })
            
            else:  # In position
                # Check for exit signals or risk management
                exit_triggered = False
                exit_reason = ""
                
                # Check stop loss
                if strategy.stop_loss and current_price <= entry_price * (1 - strategy.stop_loss):
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                
                # Check take profit
                elif strategy.take_profit and current_price >= entry_price * (1 + strategy.take_profit):
                    exit_triggered = True
                    exit_reason = "Take Profit"
                
                # Check exit conditions
                elif self._check_exit_signal(row, strategy):
                    exit_triggered = True
                    exit_reason = "Exit Signal"
                
                # Execute exit if triggered
                if exit_triggered:
                    # Calculate trade result
                    exit_value = position * current_price
                    profit_loss = exit_value - (position * entry_price)
                    profit_loss_pct = (current_price - entry_price) / entry_price
                    
                    # Update capital
                    capital += exit_value
                    
                    # Record trade exit
                    trades.append({
                        'Type': 'Exit',
                        'Date': current_date,
                        'Price': current_price,
                        'Shares': position,
                        'Value': exit_value,
                        'Entry_Price': entry_price,
                        'Entry_Date': entry_date,
                        'Profit_Loss': profit_loss,
                        'Profit_Loss_Pct': profit_loss_pct,
                        'Exit_Reason': exit_reason,
                        'Days_Held': (current_date - entry_date).days
                    })
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_date = None
                    in_position = False
        
        # Handle open position at end
        if in_position and position > 0:
            final_price = float(feature_df.iloc[-1]['Close'])
            final_date = feature_df.iloc[-1]['Date']
            exit_value = position * final_price
            profit_loss = exit_value - (position * entry_price)
            profit_loss_pct = (final_price - entry_price) / entry_price
            
            capital += exit_value
            
            trades.append({
                'Type': 'Exit',
                'Date': final_date,
                'Price': final_price,
                'Shares': position,
                'Value': exit_value,
                'Entry_Price': entry_price,
                'Entry_Date': entry_date,
                'Profit_Loss': profit_loss,
                'Profit_Loss_Pct': profit_loss_pct,
                'Exit_Reason': 'End of Data',
                'Days_Held': (final_date - entry_date).days
            })
        
        # Calculate performance metrics
        final_value = capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Analyze trades
        exit_trades = [t for t in trades if t['Type'] == 'Exit']
        
        if exit_trades:
            profits = [t['Profit_Loss'] for t in exit_trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p <= 0]
            
            win_rate = len(winning_trades) / len(exit_trades) if exit_trades else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
            
            # Calculate maximum drawdown
            equity_values = [e['Portfolio_Value'] for e in equity_curve]
            peak = equity_values[0]
            max_drawdown = 0
            
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            max_drawdown = 0
        
        # Calculate Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                daily_return = (equity_curve[i]['Portfolio_Value'] - equity_curve[i-1]['Portfolio_Value']) / equity_curve[i-1]['Portfolio_Value']
                returns.append(daily_return)
            
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'strategy_name': strategy.name,
            'initial_capital': self.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': len(exit_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'equity_curve': equity_curve
        }

class PortfolioManager:
    """
    Portfolio management and analysis class
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize portfolio manager
        
        Parameters:
        -----------
        initial_capital : float
            Initial portfolio capital
        """
        self.initial_capital = initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float, 'current_price': float}}
        self.cash = initial_capital
        
    def add_position(self, symbol: str, shares: int, price: float):
        """
        Add or update a position in the portfolio
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        shares : int
            Number of shares
        price : float
            Price per share
        """
        if symbol in self.positions:
            # Update existing position
            total_shares = self.positions[symbol]['shares'] + shares
            total_cost = (self.positions[symbol]['shares'] * self.positions[symbol]['avg_price']) + (shares * price)
            avg_price = total_cost / total_shares if total_shares > 0 else 0
            
            self.positions[symbol] = {
                'shares': total_shares,
                'avg_price': avg_price,
                'current_price': price
            }
        else:
            # New position
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price,
                'current_price': price
            }
        
        # Update cash
        self.cash -= shares * price
    
    def update_prices(self, price_data: Dict[str, float]):
        """
        Update current prices for all positions
        
        Parameters:
        -----------
        price_data : dict
            Dictionary of {symbol: current_price}
        """
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol]['current_price'] = price
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value
        
        Returns:
        --------
        float
            Total portfolio value
        """
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            position_value = position['shares'] * position['current_price']
            total_value += position_value
        
        return total_value
    
    def get_position_weights(self) -> Dict[str, float]:
        """
        Calculate position weights in portfolio
        
        Returns:
        --------
        dict
            Dictionary of {symbol: weight}
        """
        total_value = self.get_portfolio_value()
        weights = {}
        
        for symbol, position in self.positions.items():
            position_value = position['shares'] * position['current_price']
            weight = position_value / total_value if total_value > 0 else 0
            weights[symbol] = weight
        
        return weights
    
    def analyze_new_trade_impact(self, symbol: str, shares: int, price: float) -> Dict:
        """
        Analyze how a new trade would impact the portfolio
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        shares : int
            Number of shares to trade
        price : float
            Trade price
            
        Returns:
        --------
        dict
            Analysis of trade impact
        """
        # Calculate new portfolio state
        trade_value = shares * price
        new_cash = self.cash - trade_value
        
        # Check if we have enough cash
        if new_cash < 0:
            return {
                'feasible': False,
                'reason': 'Insufficient cash',
                'required_cash': trade_value,
                'available_cash': self.cash
            }
        
        # Calculate new position
        if symbol in self.positions:
            new_shares = self.positions[symbol]['shares'] + shares
            total_cost = (self.positions[symbol]['shares'] * self.positions[symbol]['avg_price']) + trade_value
            new_avg_price = total_cost / new_shares if new_shares > 0 else 0
        else:
            new_shares = shares
            new_avg_price = price
        
        # Calculate new portfolio metrics
        new_total_value = self.get_portfolio_value() - trade_value + (new_shares * price)
        new_position_value = new_shares * price
        new_weight = new_position_value / new_total_value if new_total_value > 0 else 0
        
        # Check concentration risk
        concentration_risk = "Low"
        if new_weight > 0.3:
            concentration_risk = "High"
        elif new_weight > 0.2:
            concentration_risk = "Medium"
        
        return {
            'feasible': True,
            'new_shares': new_shares,
            'new_avg_price': new_avg_price,
            'new_position_value': new_position_value,
            'new_weight': new_weight,
            'new_weight_pct': new_weight * 100,
            'concentration_risk': concentration_risk,
            'remaining_cash': new_cash,
            'trade_value': trade_value
        }

class RiskManager:
    """
    Risk management utilities
    """
    
    @staticmethod
    def calculate_position_size(portfolio_value: float, risk_per_trade: float, 
                              entry_price: float, stop_loss_price: float) -> int:
        """
        Calculate position size using fixed risk per trade
        
        Parameters:
        -----------
        portfolio_value : float
            Total portfolio value
        risk_per_trade : float
            Risk percentage per trade (e.g., 0.02 for 2%)
        entry_price : float
            Entry price for the trade
        stop_loss_price : float
            Stop loss price
            
        Returns:
        --------
        int
            Number of shares to trade
        """
        risk_amount = portfolio_value * risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk > 0:
            position_size = int(risk_amount / price_risk)
            return max(1, position_size)  # At least 1 share
        else:
            return 1
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for position sizing
        
        Parameters:
        -----------
        win_rate : float
            Historical win rate (0.0 to 1.0)
        avg_win : float
            Average winning trade amount
        avg_loss : float
            Average losing trade amount (positive number)
            
        Returns:
        --------
        float
            Kelly percentage (0.0 to 1.0)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Cap at 25% for safety
        return max(0.0, min(0.25, kelly_pct))

# Predefined strategy templates
class StrategyTemplates:
    """
    Collection of predefined trading strategy templates
    """
    
    @staticmethod
    def rsi_mean_reversion() -> TradingStrategy:
        """
        RSI-based mean reversion strategy
        
        Returns:
        --------
        TradingStrategy
            Configured RSI mean reversion strategy
        """
        strategy = TradingStrategy("RSI Mean Reversion")
        
        # Entry: RSI oversold
        strategy.add_entry_condition('RSI', '<', 30)
        
        # Exit: RSI overbought
        strategy.add_exit_condition('RSI', '>', 70)
        
        # Risk management
        strategy.set_risk_management(
            stop_loss=0.05,  # 5% stop loss
            take_profit=0.10,  # 10% take profit
            position_sizing='percent_risk',
            risk_tolerance=0.02
        )
        
        return strategy
    
    @staticmethod
    def sma_crossover() -> TradingStrategy:
        """
        SMA crossover momentum strategy
        
        Returns:
        --------
        TradingStrategy
            Configured SMA crossover strategy
        """
        strategy = TradingStrategy("SMA Crossover")
        
        # Entry: SMA20 above SMA50 (bullish crossover)
        strategy.add_entry_condition('SMA_Crossover', '==', 1)
        
        # Exit: SMA20 below SMA50 (bearish crossover)
        strategy.add_exit_condition('SMA_Crossover', '==', 0)
        
        # Risk management
        strategy.set_risk_management(
            stop_loss=0.08,  # 8% stop loss
            position_sizing='fixed',
            risk_tolerance=0.02
        )
        
        return strategy
    
    @staticmethod
    def bollinger_band_breakout() -> TradingStrategy:
        """
        Bollinger Band breakout strategy
        
        Returns:
        --------
        TradingStrategy
            Configured Bollinger Band strategy
        """
        strategy = TradingStrategy("Bollinger Band Breakout")
        
        # Entry: Price near lower band (oversold)
        strategy.add_entry_condition('BBPosition', '<', 0.2)
        
        # Exit: Price near upper band (overbought)
        strategy.add_exit_condition('BBPosition', '>', 0.8)
        
        # Risk management
        strategy.set_risk_management(
            stop_loss=0.06,  # 6% stop loss
            take_profit=0.12,  # 12% take profit
            position_sizing='percent_risk',
            risk_tolerance=0.015
        )
        
        return strategy
    
    @staticmethod
    def momentum_combo() -> TradingStrategy:
        """
        Combined momentum strategy using multiple indicators
        
        Returns:
        --------
        TradingStrategy
            Configured momentum strategy
        """
        strategy = TradingStrategy("Momentum Combo")
        
        # Entry: Multiple bullish conditions
        strategy.add_entry_condition('RSI', '>', 50)  # RSI above neutral
        strategy.add_entry_condition('SMA_Crossover', '==', 1)  # SMA bullish
        strategy.add_entry_condition('MACD', '>', 0)  # MACD positive
        
        # Exit: Any bearish condition
        strategy.add_exit_condition('RSI', '<', 40)  # RSI weakness
        strategy.add_exit_condition('SMA_Crossover', '==', 0)  # SMA bearish
        
        # Risk management
        strategy.set_risk_management(
            stop_loss=0.07,  # 7% stop loss
            take_profit=0.15,  # 15% take profit
            position_sizing='kelly',
            risk_tolerance=0.025
        )
        
        return strategy

# Convenience functions for easy integration
def quick_backtest(df: pd.DataFrame, strategy_name: str = "rsi_mean_reversion", 
                  initial_capital: float = 100000) -> Dict:
    """
    Quick backtest function for predefined strategies
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock price dataframe
    strategy_name : str
        Name of predefined strategy ('rsi_mean_reversion', 'sma_crossover', 
        'bollinger_band_breakout', 'momentum_combo')
    initial_capital : float
        Starting capital
        
    Returns:
    --------
    dict
        Backtest results
    """
    # Get predefined strategy
    if strategy_name == "rsi_mean_reversion":
        strategy = StrategyTemplates.rsi_mean_reversion()
    elif strategy_name == "sma_crossover":
        strategy = StrategyTemplates.sma_crossover()
    elif strategy_name == "bollinger_band_breakout":
        strategy = StrategyTemplates.bollinger_band_breakout()
    elif strategy_name == "momentum_combo":
        strategy = StrategyTemplates.momentum_combo()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Run backtest
    backtester = StrategyBacktester(initial_capital)
    return backtester.backtest_strategy(df, strategy)

def calculate_optimal_position_size(portfolio_value: float, current_price: float, 
                                  stop_loss_pct: float = 0.05, risk_pct: float = 0.02) -> Dict:
    """
    Calculate optimal position size based on risk management
    
    Parameters:
    -----------
    portfolio_value : float
        Total portfolio value
    current_price : float
        Current stock price
    stop_loss_pct : float
        Stop loss percentage (default 5%)
    risk_pct : float
        Risk percentage per trade (default 2%)
        
    Returns:
    --------
    dict
        Position sizing information
    """
    stop_loss_price = current_price * (1 - stop_loss_pct)
    
    shares = RiskManager.calculate_position_size(
        portfolio_value, risk_pct, current_price, stop_loss_price
    )
    
    position_value = shares * current_price
    max_loss = shares * (current_price - stop_loss_price)
    
    return {
        'recommended_shares': shares,
        'position_value': position_value,
        'position_weight': position_value / portfolio_value,
        'position_weight_pct': (position_value / portfolio_value) * 100,
        'max_potential_loss': max_loss,
        'stop_loss_price': stop_loss_price,
        'risk_amount': portfolio_value * risk_pct
    }
