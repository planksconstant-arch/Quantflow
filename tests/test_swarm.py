"""
Unit Tests for Biomimetic Mormyrid Swarm Intelligence Module
"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.swarm import (
    FishAgent,
    ScoutFishAgent,
    PredatorFishAgent,
    SchoolingFishAgent,
    SentinelFishAgent,
    AgentRole,
    JammingAvoidanceEngine,
    MormyridSwarmConsensusEngine,
    SwarmConsensusSignal,
    ArtificialFishSwarmOptimizer,
)


class TestFishAgents:
    """Test individual fish agents and active electrolocation"""
    
    def test_agent_initialization(self):
        scout = ScoutFishAgent(agent_id=1, feature_dim=5)
        assert scout.role == AgentRole.SCOUT
        assert len(scout.position) == 5
        assert scout.frequency > 0
        assert not scout.is_jammed

    def test_eod_pulse_discharge(self):
        predator = PredatorFishAgent(agent_id=2, feature_dim=5)
        market_vector = np.array([0.5, 0.2, 1.2, 0.1, 0.05])
        fitness = predator.emit_eod_pulse(market_vector)
        assert isinstance(fitness, float)
        assert fitness > 0
        assert 0.0 <= predator.confidence <= 1.0

    def test_agent_kinematics(self):
        schooler = SchoolingFishAgent(agent_id=3, feature_dim=5)
        center = np.zeros(5)
        best = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        repulsion = np.zeros(5)
        initial_pos = schooler.position.copy()
        schooler.update_kinematics(center, best, repulsion)
        assert not np.allclose(initial_pos, schooler.position)


class TestJammingAvoidance:
    """Test Jamming Avoidance Response (JAR)"""
    
    def test_jar_frequency_shift(self):
        jar = JammingAvoidanceEngine(jamming_frequency_threshold=5.0, spatial_crowding_radius=1.0)
        agent1 = ScoutFishAgent(agent_id=1, base_frequency=80.0)
        agent2 = ScoutFishAgent(agent_id=2, base_frequency=81.0)
        
        # Place them very close
        agent1.position = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        agent2.position = np.array([0.11, 0.11, 0.11, 0.11, 0.11])
        
        freq1_before = agent1.frequency
        freq2_before = agent2.frequency
        
        repulsions, crowding, jammed_ratio = jar.process_swarm_jar([agent1, agent2])
        
        assert jammed_ratio > 0
        assert agent1.is_jammed
        assert agent2.is_jammed
        # Higher frequency should shift up, lower down
        if freq2_before > freq1_before:
            assert agent2.frequency > freq2_before
            assert agent1.frequency < freq1_before
        else:
            assert agent1.frequency > freq1_before
            assert agent2.frequency < freq2_before


class TestSwarmConsensus:
    """Test full Mormyrid Swarm Consensus Engine"""
    
    def test_consensus_prediction_generation(self):
        engine = MormyridSwarmConsensusEngine(
            n_scouts=3,
            n_predators=4,
            n_schoolers=4,
            n_sentinels=2,
            feature_dim=5
        )
        signal = engine.step_market_state(
            ofi=0.45,
            vpin=0.25,
            hawkes_intensity=1.8,
            micro_price_dev=1.5,
            relative_spread=0.8,
            hawkes_branching_ratio=0.72,
        )
        
        assert isinstance(signal, SwarmConsensusSignal)
        assert -50.0 <= signal.predicted_drift_bps <= 50.0
        assert 0.0 <= signal.jump_probability <= 1.0
        assert -1.0 <= signal.optimal_quote_skew <= 1.0
        assert 0.0 <= signal.market_crowding_index <= 1.0
        assert 0.0 <= signal.adverse_selection_risk <= 1.0
        assert signal.dominant_regime in ["HAWKES_MOMENTUM", "MEAN_REVERSION", "TOXIC_DRAIN", "STABLE_LOB"]


class TestAFSAOptimizer:
    """Test Artificial Fish Swarm Algorithm optimizer"""
    
    def test_sphere_function_optimization(self):
        # Minimize Sphere function f(x) = sum(x^2), optimal at x=0
        def sphere(x):
            return -float(np.sum(x ** 2)) # Negative for maximization
            
        afsa = ArtificialFishSwarmOptimizer(
            objective_func=sphere,
            dim=3,
            bounds=[(-2.0, 2.0)] * 3,
            pop_size=15,
            visual_radius=0.8,
            step_size=0.2,
            maximize=True,
        )
        best_x, best_fit, history = afsa.optimize(max_iter=10)
        assert len(best_x) == 3
        assert best_fit > -5.0
