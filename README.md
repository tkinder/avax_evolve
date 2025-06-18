# AVAX Evolve

## Project Overview
…short description…

## Structure
- **data/**: raw & processed data files  
- **scripts/**: standalone runners (fetch, backtest, analysis)  
- **src/**: reusable modules  
  - **core/**: data loading, logging, performance, params  
  - **evolution/**: GA phases, fitness utils  
- **tests/**: pytest suites  
- **notebooks/**: exploratory analysis  
- **docs/**: design docs, roadmap  
- **logs/**: log output files  

## Getting Started
```bash
pip install -r requirements.txt
python scripts/fetch_and_store_ohlcv.py
