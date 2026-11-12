# Termius update and expiry research

## Research sweep (does not change live signals)

The repository's lab includes a broad set of technical rule families. Run the added
`lab_expiry_sweep.py` to evaluate each existing rule on fixed expirations from 3 through
60 minutes. It uses the same minute-history source and validation procedure as `lab.py`.

```bash
cd /path/to/Racio_1bot
python3 -m pip install -r requirements.txt
python3 lab_expiry_sweep.py --download
```

The downloader adds recent Yahoo minute bars to `lab_data/`; repeat it over time to build
more history, then run `python3 lab_expiry_sweep.py` without `--download` to analyze the
accumulated dataset. The sweep is substantially larger than the default lab run and can
take a long time. Keep the generated `lab_summary.txt`, `lab_report.csv`, and
`lab_validated.json` for review.

A PASS in historical research is not a guarantee of future performance. In particular,
the sweep does not establish that every PASS configuration can be safely deployed by the
current live engine: live signals presently use the three presets in `reversion_engine.py`
on closed 5-minute bars. Do not copy arbitrary lab configurations into live trading or
claim an expected win rate without independent review and demo observation.

## Updating through Termius

After changes have been reviewed and merged into the branch used by the server:

```bash
cd /path/to/Racio_1bot
cp -a . "../Racio_1bot-backup-$(date +%Y%m%d-%H%M%S)"
git pull --ff-only
python3 -m py_compile *.py
python3 -m unittest discover -v
```

Then restart the bot using the same process manager used on the server (for example,
`systemctl restart <your-service>` if it is managed by systemd). Do not paste secrets or
`.env` contents into chat. Confirm the bot starts and responds in Telegram before removing
the backup.
