# Triangular Arb App

## Setup

1. Create `.env` in `app/` (see placeholder file).
2. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
3. Run:
   ```bash
   python app/main.py
   ```

## Notes

- Uses Binance bookTicker stream.
- Hourly heartbeat + opportunity push (throttled).
- Configurable via `.env`.
