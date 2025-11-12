# TCP Reset Research — Lab-only PoC

**Short description**  
This repository contains research / analysis artifacts to study TCP connection termination behavior (RST packets) and defensive detection techniques. **THIS REPOSITORY IS FOR EDUCATIONAL AND DEFENSIVE USE ONLY** — see `ethical_disclaimer.md`.

---

## ⚠️ Important — Legal & Ethical Notice
Do **not** run experiments outside an isolated lab you control. Never target external networks or services without explicit, written permission. By using anything in this repo you agree to follow applicable laws and institutional policies.

---

## What’s in this repo
- `app.py` — analysis utilities / application entrypoint (avoid running network-active code on public networks).  
- `requirements.txt` — Python packages used for analysis.  
- `data/`, `notebooks/`, `models/` — sanitized datasets and notebooks for offline analysis.  
- `docs/` — methodology, detection heuristics, mitigation notes (additions encouraged).

---

## Quick setup (safe / local)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
