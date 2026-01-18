"""
Mispricing Detection using XGBoost with SHAP Explainability
Identifies options trading cheaper/richer than fair value
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')


class MispricingDetector:
    """
    ML-powered option mispricing detection
    Uses XGBoost classifier with SHAP for explainability
    """
    
    def __init__(self):
        """Initialize mispricing detector"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.shap_explainer = None
        
    def create_features(self, 
                       market_price: float,
                       model_price: float,
                       implied_vol: float,
                       forecast_vol: float,
                       greeks: Dict,
                       spot: float,
                       strike: float,
                       moneyness: float,
                       bid_ask_spread: float,
                       volume: float,
                       open_interest: float,
                       vix_level: float = None) -> pd.DataFrame:
        """
        Create feature vector for mispricing detection
        
        Parameters:
        -----------
        market_price : float
            Current market price
        model_price : float
            Theoretical model price
        implied_vol : float
            Implied volatility
        forecast_vol : float
            Forecasted volatility
        greeks : dict
            Greeks (delta, gamma, vega, theta)
        spot : float
            Current spot price
        strike : float
            Strike price
        moneyness : float
            Spot / Strike
        bid_ask_spread : float
            Bid-ask spread in dollars
        volume : float
            Trading volume
        open_interest : float
            Open interest
        vix_level : float, optional
            Current VIX level
        
        Returns:
        --------
        pd.DataFrame : Feature vector
        """
        features = {}
        
        # === PRICING DEVIATION FEATURES ===
        pricing_error = (model_price - market_price) / market_price
        features['pricing_error_pct'] = pricing_error * 100
        features['pricing_error_abs'] = abs(pricing_error)
        features['model_overpriced'] = 1 if pricing_error > 0 else 0
        
        # Volatility spread
        vol_spread = implied_vol - forecast_vol
        features['iv_forecast_spread'] = vol_spread
        features['iv_forecast_spread_pct'] = (vol_spread / forecast_vol) * 100 if forecast_vol > 0 else 0
        
        # === MARKET MICROSTRUCTURE FEATURES ===
        features['bid_ask_spread_pct'] = (bid_ask_spread / market_price) * 100
        features['volume_oi_ratio'] = volume / open_interest if open_interest > 0 else 0
        features['liquidity_score'] = np.log1p(volume) * np.log1p(open_interest)
        
        # === GREEKS SIGNALS ===
        features['delta'] = greeks.get('delta', 0)
        features['gamma'] = greeks.get('gamma', 0)
        features['vega_pct'] = greeks.get('vega_percent', 0)
        features['theta_per_day'] = greeks.get('theta_per_day', 0)
        
        # Delta vs expected (based on moneyness)
        # ATM call should have delta ~0.5, ITM higher, OTM lower
        if moneyness > 0:
            expected_delta = 0.5 + 0.5 * np.tanh((moneyness - 1) * 5)
        else:
            expected_delta = 0
        features['delta_abnormality'] = abs(greeks.get('delta', 0) - expected_delta)
        
        # Gamma concentration (high gamma = sensitive to small moves)
        features['gamma_weighted_iv'] = greeks.get('gamma', 0) * implied_vol
        
        # === MONEYNESS & POSITION FEATURES ===
        features['moneyness'] = moneyness
        features['moneyness_log'] = np.log(moneyness) if moneyness > 0 else -10
        features['itm_flag'] = 1 if moneyness > 1 else 0
        features['atm_distance'] = abs(moneyness - 1)
        
        # === VOLATILITY FEATURES ===
        features['implied_vol'] = implied_vol
        features['forecast_vol'] = forecast_vol
        features['vol_ratio'] = implied_vol / forecast_vol if forecast_vol > 0 else 1
        
        # === VIX FEATURES (if available) ===
        if vix_level is not None:
            features['vix_level'] = vix_level
            # VIX percentile (rough approximation: 10-20 is low, 20-30 is normal, >30 is high)
            features['vix_regime'] = 0 if vix_level < 20 else (1 if vix_level < 30 else 2)
        
        return pd.DataFrame([features])
    
    def create_training_data(self, 
                            historical_options: pd.DataFrame,
                            lookforward_days: int = 5,
                            profit_threshold: float = 0.05) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training dataset from historical option data
        
        This is a SIMPLIFIED version. In production, you'd need:
        - Historical option chain data
        - Price evolution over time
        - Realized P&L from buying at T and selling at T+5
        
        Parameters:
        -----------
        historical_options : pd.DataFrame
            Historical option data with features
        lookforward_days : int
            Days ahead to check performance
        profit_threshold : float
            Profit threshold to label as "mispriced" (>5% gain)
        
        Returns:
        --------
        tuple : (X, y) features and labels
        """
        # This is a placeholder - would need real historical option chain data
        # For demo purposes, create synthetic labels based on pricing error
        
        if 'pricing_error_pct' in historical_options.columns:
            # Synthetic labeling: large pricing errors tend to correct
            # Label=1 if pricing error is significant (option underpriced)
            y = ((historical_options['pricing_error_pct'] > 3) | 
                 (historical_options['pricing_error_pct'] < -3)).astype(int)
        else:
            y = pd.Series(np.zeros(len(historical_options)), index=historical_options.index)
        
        return historical_options, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """
        Train XGBoost mispricing classifier
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Labels (1 = mispriced, 0 = fairly priced)
        test_size : float
            Test set fraction
        
        Returns:
        --------
        dict : Training results
        """
        print(f"\nTraining Mispricing Detector...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        y_prob_test = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob_test)
        else:
            auc = np.nan
        
        print(f" Train Accuracy: {train_acc:.3f} | Test Accuracy: {test_acc:.3f}")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | AUC: {auc:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_prob_test': y_prob_test
        }
    
    def predict_mispricing(self, features: pd.DataFrame) -> Dict:
        """
        Predict mispricing score for a single option
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature vector
        
        Returns:
        --------
        dict : Mispricing score and explanation
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probability
        mispricing_prob = self.model.predict_proba(features_scaled)[0, 1]
        mispricing_score = mispricing_prob * 100  # Convert to 0-100 scale
        
        # SHAP explanation
        shap_values = self.shap_explainer.shap_values(features_scaled)
        
        # Get top contributing features
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        shap_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'feature_value': features.iloc[0].values
        }).sort_values('shap_value', key=abs, ascending=False)
        
        # Interpretation
        if mispricing_score > 70:
            assessment = "STRONG MISPRICING SIGNAL - High conviction trade"
        elif mispricing_score > 50:
            assessment = "MODERATE MISPRICING - Worth investigating"
        else:
            assessment = "FAIRLY PRICED - No strong signal"
        
        return {
            'mispricing_score': mispricing_score,
            'mispricing_probability': mispricing_prob,
            'assessment': assessment,
            'top_drivers': shap_contributions.head(5),
            'all_shap_values': shap_contributions
        }
    
    def explain_prediction(self, features: pd.DataFrame, verbose: bool = True) -> str:
        """
        Generate human-readable explanation of mispricing prediction
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature vector
        verbose : bool
            Print explanation
        
        Returns:
        --------
        str : Explanation text
        """
        result = self.predict_mispricing(features)
        
        explanation = f"\n{'='*70}\n"
        explanation += f"MISPRICING ANALYSIS\n"
        explanation += f"{'='*70}\n\n"
        explanation += f"Mispricing Score: {result['mispricing_score']:.1f}/100\n"
        explanation += f"Assessment: {result['assessment']}\n\n"
        explanation += f"Top Drivers (SHAP Analysis):\n"
        
        for idx, row in result['top_drivers'].iterrows():
            direction = "UP" if row['shap_value'] > 0 else "DOWN"
            contribution_pct = abs(row['shap_value']) / result['top_drivers']['shap_value'].abs().sum() * 100
            explanation += f"   {direction} {row['feature']}: {row['feature_value']:.4f} "
            explanation += f"(contributes {contribution_pct:.1f}% to score)\n"
        
        if verbose:
            print(explanation)
        
        return explanation


if __name__ == "__main__":
    # Test mispricing detector
    detector = MispricingDetector()
    
    # Create sample features
    features = detector.create_features(
        market_price=10.50,
        model_price=11.20,
        implied_vol=0.35,
        forecast_vol=0.30,
        greeks={'delta': 0.55, 'gamma': 0.03, 'vega_percent': 0.15, 'theta_per_day': -0.05},
        spot=145,
        strike=140,
        moneyness=145/140,
        bid_ask_spread=0.20,
        volume=1500,
        open_interest=5000,
        vix_level=18.5
    )
    
    print("Sample features created:")
    print(features.T)
