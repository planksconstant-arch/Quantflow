"""
Market Data Fetcher for QuantFlow
Fetches NVDA stock data and option chain using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import joblib
import os

from utils.config import config
from utils.helpers import log_returns, annualized_volatility


class MarketDataFetcher:
    """Fetches and caches market data for NVDA options analysis"""
    
    def __init__(self, ticker: str = None, use_cache: bool = True):
        self.ticker = ticker or config.TICKER
        self.use_cache = use_cache
        self.stock = yf.Ticker(self.ticker)
        
    def fetch_historical_data(self, days: int = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Parameters:
        -----------
        days : int, optional
            Number of days to fetch (default from config)
        
        Returns:
        --------
        pd.DataFrame : Historical price data
        """
        days = days or config.HISTORICAL_DAYS
        
        # Check cache
        cache_file = os.path.join(config.CACHE_DIR, f"{self.ticker}_hist_{days}d.pkl")
        if self.use_cache and os.path.exists(cache_file):
            cache_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
            if cache_age_hours < config.CACHE_TTL_HOURS:
                print(f"✓ Loading cached historical data for {self.ticker}")
                return joblib.load(cache_file)
        
        # Fetch from yfinance
        print(f"Fetching {days} days of historical data for {self.ticker}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = self.stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data received for {self.ticker}")
        
        # Add calculated columns
        df['Returns'] = log_returns(df['Close'])
        df['HV_20'] = annualized_volatility(df['Returns'], window=20)
        df['HV_60'] = annualized_volatility(df['Returns'], window=60)
        
        # Cache result
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        joblib.dump(df, cache_file)
        print(f"✓ Fetched {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
        
        return df
    
    def fetch_option_chain(self, expiry: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch option chain for specified expiry
        
        Parameters:
        -----------
        expiry : str, optional
            Expiration date YYYY-MM-DD (default from config)
        
        Returns:
        --------
        tuple : (calls_df, puts_df)
        """
        expiry = expiry or config.EXPIRY
        
        # Check cache
        cache_file = os.path.join(config.CACHE_DIR, f"{self.ticker}_options_{expiry}.pkl")
        if self.use_cache and os.path.exists(cache_file):
            cache_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
            if cache_age_hours < 1:  # Options data cached for only 1 hour (more dynamic)
                print(f"✓ Loading cached option chain for {expiry}")
                return joblib.load(cache_file)
        
        # Fetch from yfinance
        print(f"Fetching option chain for {self.ticker} expiring {expiry}...")
        
        try:
            opt_chain = self.stock.option_chain(expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            print(f"✓ Found {len(calls)} call options and {len(puts)} put options")
            
            # Cache result
            joblib.dump((calls, puts), cache_file)
            
            return calls, puts
        
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            print(f"Available expiration dates: {self.stock.options}")
            raise
    
    def get_target_option(self, strike: float = None, option_type: str = None) -> pd.Series:
        """
        Get specific option contract
        
        Parameters:
        -----------
        strike : float, optional
            Strike price (default from config)
        option_type : str, optional
            'call' or 'put' (default from config)
        
        Returns:
        --------
        pd.Series : Option data
        """
        strike = strike or config.STRIKE
        option_type = option_type or config.OPTION_TYPE
        
        calls, puts = self.fetch_option_chain()
        
        df = calls if option_type.lower() == 'call' else puts
        
        # Find closest strike
        option = df[df['strike'] == strike]
        
        if option.empty:
            # Find nearest strike
            available_strikes = df['strike'].values
            nearest_strike = available_strikes[np.argmin(np.abs(available_strikes - strike))]
            print(f"⚠ Strike {strike} not found, using nearest: {nearest_strike}")
            option = df[df['strike'] == nearest_strike]
        
        return option.iloc[0]
    
    def get_current_spot_price(self) -> float:
        """Get current stock price"""
        hist = self.fetch_historical_data(days=5)
        return hist['Close'].iloc[-1]
    
    def get_risk_free_rate(self) -> float:
        """
        Get risk-free rate from 3-month T-bill
        
        Returns:
        --------
        float : Annualized risk-free rate
        """
        # Check cache
        cache_file = os.path.join(config.CACHE_DIR, "risk_free_rate.pkl")
        if self.use_cache and os.path.exists(cache_file):
            cache_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
            if cache_age_hours < 24:
                return joblib.load(cache_file)
        
        try:
            # Fetch 3-month T-bill rate (^IRX)
            print("Fetching risk-free rate from Yahoo Finance (^IRX)...")
            tbill = yf.Ticker("^IRX")
            hist = tbill.history(period="5d")
            
            if not hist.empty:
                # ^IRX is already in percentage, convert to decimal
                rate = hist['Close'].iloc[-1] / 100.0
                print(f"✓ Risk-free rate: {rate * 100:.3f}%")
                
                # Cache
                joblib.dump(rate, cache_file)
                return rate
        except Exception as e:
            print(f"Warning: Could not fetch risk-free rate: {e}")
        
        # Fallback to default rate
        default_rate = 0.045  # 4.5% default
        print(f"Using default risk-free rate: {default_rate * 100:.2f}%")
        return default_rate
    
    def get_dividend_yield(self) -> float:
        """Get annual dividend yield"""
        try:
            info = self.stock.info
            div_yield = info.get('dividendYield', 0.0)
            if div_yield is None:
                div_yield = 0.0
            print(f"✓ Dividend yield: {div_yield * 100:.2f}%")
            return div_yield
        except:
            print("ℹ No dividend data available, assuming 0%")
            return 0.0
    
    def get_vix_data(self, days: int = None) -> pd.DataFrame:
        """
        Fetch VIX (volatility index) data
        
        Parameters:
        -----------
        days : int, optional
            Number of days (default from config)
        
        Returns:
        --------
        pd.DataFrame : VIX data
        """
        days = days or config.HISTORICAL_DAYS
        
        cache_file = os.path.join(config.CACHE_DIR, f"VIX_{days}d.pkl")
        if self.use_cache and os.path.exists(cache_file):
            cache_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
            if cache_age_hours < config.CACHE_TTL_HOURS:
                return joblib.load(cache_file)
        
        print(f"Fetching VIX data for {days} days...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        vix = yf.Ticker("^VIX")
        df = vix.history(start=start_date, end=end_date)
        
        if not df.empty:
            joblib.dump(df, cache_file)
            print(f"✓ Fetched VIX data: {len(df)} days")
        else:
            print("⚠ VIX data unavailable")
        
        return df
    
    def get_all_market_data(self) -> Dict:
        """
        Fetch all required market data
        
        Returns:
        --------
        dict : Complete market data package
        """
        print(f"\n{'='*60}")
        print(f"📊 Fetching Market Data for {config.get_option_identifier()}")
        print(f"{'='*60}\n")
        
        data = {
            'historical': self.fetch_historical_data(),
            'spot_price': self.get_current_spot_price(),
            'option': self.get_target_option(),
            'risk_free_rate': self.get_risk_free_rate(),
            'dividend_yield': self.get_dividend_yield(),
            'vix': self.get_vix_data(),
        }
        
        # Add summary
        print(f"\n{'='*60}")
        print(f"✓ Data Summary:")
        print(f"  Current Spot: ${data['spot_price']:.2f}")
        print(f"  Option Strike: ${config.STRIKE}")
        print(f"  Market Price: ${data['option']['lastPrice']:.2f}")
        print(f"  Implied Vol: {data['option']['impliedVolatility']*100:.2f}%")
        print(f"  Risk-Free Rate: {data['risk_free_rate']*100:.2f}%")
        print(f"  Days to Expiry: {(config.expiry_date - datetime.now()).days}")
        print(f"{'='*60}\n")
        
        return data


if __name__ == "__main__":
    # Test data fetcher
    fetcher = MarketDataFetcher()
    data = fetcher.get_all_market_data()
