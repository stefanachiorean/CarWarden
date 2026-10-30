import sys

import numpy as np
import pandas as pd

out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
s = pd.read_csv(f"{out_dir}/scored_windows.csv")
pd.set_option("display.width", 220)

print("1) Are false alarms close to attack windows?")
for name, p in s.groupby("capture"):
    p = p.sort_values("t_rel").reset_index(drop=True)
    attack_pos = np.where(p["is_attack"])[0]
    dist = np.array([np.abs(attack_pos - i).min() for i in range(len(p))])
    normal = ~p["is_attack"].to_numpy()
    false_alarm = p["flagged"].to_numpy() & normal
    near = dist <= 5
    print(f"  {name}: {int((false_alarm & near).sum())} of {int(false_alarm.sum())} false alarms "
          f"are within 5 windows (0.5 s) of an attack; "
          f"{int((normal & near).sum())} of {int(normal.sum())} normal windows are that close")

print()
print("2) Attack windows: caught (flagged=True) vs missed (flagged=False), mean features")
columns = ["n_msgs", "n_unique_ids", "id_entropy", "frac_unknown_id", "iat_min",
           "iat_mean", "payload_change_rate", "attack_frac"]
attack = s[s["is_attack"]]
print(attack.groupby(["capture", "flagged"])[columns].mean().round(4).to_string())