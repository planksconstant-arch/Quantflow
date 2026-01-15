QuantFlow Documentation
=======================

Welcome to QuantFlow's API documentation. This documentation covers the complete API for options pricing, Greeks calculation, and quantitative analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   RISK_ANALYSIS
   MODEL_VALIDATION

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Start
===========

Installation::

    pip install -r requirements.txt

Basic Usage::

    from main import QuantFlow
    
    qf = QuantFlow(
        ticker="NVDA",
        option_type="call",
        strike=140,
        expiry="2026-04-17"
    )
    
    qf.fetch_data()
    pricing = qf.get_ensemble_pricing()
    greeks = qf.get_greeks()

API Reference
=============

.. automodule:: models.pricing.black_scholes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: models.greeks.calculator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: analysis.scenario_analysis
   :members:
   :undoc-members:
   :show-inheritance:
