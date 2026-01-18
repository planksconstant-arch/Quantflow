"""
Volatility Forecasting using GARCH and ML
Combines traditional time series models with machine learning
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from arch import arch_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils.helpers import log_returns, annualized_volatility


class VolatilityForecaster:
    """
    Multi-model volatility forecasting
    Combines GARCH with ML-enhanced predictions
    """
    
    def __init__(self, price_data: pd.DataFrame, vix_data: pd.DataFrame = None):
        """
        Initialize volatility forecaster
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Historical price data (must have 'Close' column)
        vix_data : pd.DataFrame, optional
            VIX data for regime context
        """
        self.price_data = price_data
        self.vix_data = vix_data
        
        # Calculate returns
        self.returns = log_returns(price_data['Close']).dropna()
        self.returns_pct = self.returns * 100  # GARCH works better with percentage returns
        
    def calculate_historical_volatility(self, windows: list = None) -> pd.DataFrame:
        """
        Calculate historical volatility across multiple windows
        
        Parameters:
        -----------
        windows : list, optional
            Window sizes (default: [10, 20, 60])
        
        Returns:
        --------
        pd.DataFrame : Historical volatilities
        """
        if windows is None:
            windows = [10, 20, 60]
        
        hv_data = pd.DataFrame(index=self.price_data.index)
        
        for window in windows:
            hv_data[f'HV_{window}'] = annualized_volatility(self.returns, window)
        
        return hv_data
    
    def fit_garch(self, p: int = 1, q: int = 1) -> Dict:
        """
        Fit GARCH(p,q) model
        
        Parameters:
        -----------
        p : int
            GARCH lag order
        q : int
            ARCH lag order
        
        Returns:
        --------
        dict : GARCH model results
        """
        print(f"\nFitting GARCH({p},{q}) model...")
        
        # Fit GARCH model
        model = arch_model(
            self.returns_pct.dropna(),
            vol='Garch',
            p=p,
            q=q,
            dist='normal'
        )
        
        res = model.fit(disp='off', show_warning=False)
        
        # Extract key parameters
        omega = res.params['omega']
        alpha = res.params['alpha[1]'] if 'alpha[1]' in res.params else 0
        beta = res.params['beta[1]'] if 'beta[1]' in res.params else 0
        
        print(f"GARCH parameters: omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}")
        print(f"  Persistence (alpha+beta): {alpha + beta:.4f}")
        
        # Forecast
        forecast = res.forecast(horizon=5)
        forecast_variance = forecast.variance.values[-1, :]
        forecast_volatility = np.sqrt(forecast_variance) * np.sqrt(252) / 100  # Annualize and convert from %
        
        return {
            'model': res,
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': alpha + beta,
            'forecast_5day': forecast_volatility,
            'current_volatility': np.sqrt(res.conditional_volatility.iloc[-1]**2) * np.sqrt(252) / 100,
            'aic': res.aic,
            'bic': res.bic
        }
    
    def create_ml_features(self, lookback: int = 60) -> pd.DataFrame:
        """
        Create features for ML volatility prediction
        
        Parameters:
        -----------
        lookback : int
            Lookback period for features
        
        Returns:
        --------
        pd.DataFrame : Feature matrix
        """
        features = pd.DataFrame(index=self.price_data.index)
        
        # Historical volatility features
        for window in [10, 20, 60]:
            features[f'hv_{window}'] = annualized_volatility(self.returns, window)
        
        # Returns features
        features['return_1d'] = self.returns
        features['return_5d'] = self.returns.rolling(5).sum()
        features['return_20d'] = self.returns.rolling(20).sum()
        
        # Volume features (if available)
        if 'Volume' in self.price_data.columns:
            features['volume_ratio'] = (
                self.price_data['Volume'] / 
                self.price_data['Volume'].rolling(20).mean()
            )
        
        # Price range features
        if 'High' in self.price_data.columns and 'Low' in self.price_data.columns:
            features['parkinson_vol'] = (
                np.sqrt(1 / (4 * np.log(2)) * 
                       (np.log(self.price_data['High'] / self.price_data['Low'])**2))
            ) * np.sqrt(252)
        
        # Returns autocorrelation
        features['return_autocorr_5'] = self.returns.rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else np.nan
        )
        
        # VIX features (if available)
        if self.vix_data is not None and len(self.vix_data) > 0:
            # Align VIX data
            vix_aligned = self.vix_data['Close'].reindex(features.index, method='ffill')
            features['vix_level'] = vix_aligned
            features['vix_change'] = vix_aligned.pct_change(5)
        
        # Target: Forward 5-day realized volatility
        forward_returns = self.returns.shift(-5).rolling(5).std() * np.sqrt(252)
        
        return features, forward_returns
    
    def train_ml_forecaster(self, train_size: float = 0.8) -> Dict:
        """
        Train ML model for volatility forecasting
        
        Parameters:
        -----------
        train_size : float
            Fraction of data for training
        
        Returns:
        --------
        dict : ML model and evaluation results
        """
        print(f"\nTraining ML volatility forecaster...")
        
        # Create features
        X, y = self.create_ml_features()
        
        # Drop NaN rows
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) < 100:
            print("! Insufficient data for ML training")
            return None
        
        # Train-test split (walk-forward)
        split_idx = int(len(X_clean) * train_size)
        X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Gradient Boosting model
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Evaluation metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f" Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance,
            'test_predictions': pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred_test
            })
        }
    
    def forecast_volatility(self, horizon: int = 5) -> Dict[str, float]:
        """
        Generate ensemble volatility forecast
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon in days
        
        Returns:
        --------
        dict : Volatility forecasts from all models
        """
        print(f"\n{'='*70}")
        print(f"VOLATILITY FORECAST ({horizon}-day horizon)")
        print(f"{'='*70}\n")
        
        forecasts = {}
        
        # 1. Historical volatility (simple baseline)
        hv_20 = annualized_volatility(self.returns, 20).iloc[-1]
        hv_60 = annualized_volatility(self.returns, 60).iloc[-1]
        forecasts['historical_20d'] = hv_20
        forecasts['historical_60d'] = hv_60
        print(f"Historical Vol (20-day): {hv_20*100:.2f}%")
        print(f"Historical Vol (60-day): {hv_60*100:.2f}%")
        
        # 2. GARCH forecast
        garch_result = self.fit_garch(p=1, q=1)
        garch_forecast = garch_result['forecast_5day'][horizon-1]
        forecasts['garch'] = garch_forecast
        print(f"GARCH Forecast: {garch_forecast*100:.2f}%")
        
        # 3. ML forecast (if possible)
        ml_result = self.train_ml_forecaster()
        if ml_result is not None:
            # Get latest features
            X, _ = self.create_ml_features()
            X_latest = X.iloc[[-1]]
            
            if not X_latest.isna().any().any():
                X_scaled = ml_result['scaler'].transform(X_latest)
                ml_forecast = ml_result['model'].predict(X_scaled)[0]
                forecasts['ml'] = ml_forecast
                print(f"ML Forecast: {ml_forecast*100:.2f}%")
        
        # 4. Ensemble (average of available forecasts)
        ensemble_forecast = np.mean([v for v in forecasts.values()])
        forecasts['ensemble'] = ensemble_forecast
        
        print(f"\nENSEMBLE FORECAST: {ensemble_forecast*100:.2f}%")
        
        return forecasts


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    nvda = yf.Ticker("NVDA")
    hist = nvda.history(period="1y")
    vix = yf.Ticker("^VIX").history(period="1y")
    
    forecaster = VolatilityForecaster(hist, vix)
    forecasts = forecaster.forecast_volatility(horizon=5)
