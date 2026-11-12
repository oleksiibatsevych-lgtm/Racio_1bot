"""Run the existing strategy lab across every fixed expiry from 3 to 60 minutes.

This is research-only: results are not automatically enabled in the live bot.
Use: python3 lab_expiry_sweep.py [the same arguments as lab.py]
"""

import lab

# Include each whole-minute fixed expiry requested, while keeping the lab's existing
# timeframes, strategy definitions, data split, clustered statistics, and controls.
lab.EXPIRATIONS = list(range(3, 61))


if __name__ == "__main__":
    lab.main()
