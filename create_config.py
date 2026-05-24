#!/usr/bin/env python3
#!/usr/bin/env python3

import argparse
import configparser
import os
from pathlib import Path

# ----------------------------
# Argument parsing
# ----------------------------

parser = argparse.ArgumentParser(
    description="Create modified config file for WxFactory"
)

parser.add_argument("case_number",type=int)
parser.add_argument("dt", type=float)
parser.add_argument("t_end", type=float)
parser.add_argument("num_solpts", type=int)
parser.add_argument("num_elements_horizontal", type=int)
parser.add_argument("num_elements_vertical", type=int)
parser.add_argument("output_freq", type=int)

args = parser.parse_args()

# ----------------------------
# Paths
# ----------------------------

base_config = Path("config/entropy_wave.ini")
output_dir = Path("config/param_change")

output_dir.mkdir(parents=True, exist_ok=True)

dt_str = f"{args.dt:.0e}".replace("e-0", "e-").replace("e+0", "e+")
t_end_str = int(args.t_end) if args.t_end.is_integer() else args.t_end

output_file = output_dir / (
    f"case_{args.case_number}_"
    f"dt_{dt_str}_"
    f"tend_{t_end_str}_"
    f"solpts_{args.num_solpts}_"
    f"nx_{args.num_elements_horizontal}_"
    f"ny_{args.num_elements_vertical}.ini"
)

# ----------------------------
# Read config
# ----------------------------

config = configparser.ConfigParser()
config.read(base_config)

# ----------------------------
# Modify parameters
# ----------------------------

# Change these section names/keys
# to match your actual config structure

config["Test_case"]["case_number"] = str(args.case_number)

config["Time_integration"]["dt"] = str(args.dt)
config["Time_integration"]["t_end"] = str(args.t_end)

config["Spatial_discretization"]["num_solpts"] = str(args.num_solpts)
config["Spatial_discretization"]["num_elements_horizontal"] = str(
    args.num_elements_horizontal
)
config["Spatial_discretization"]["num_elements_vertical"] = str(
    args.num_elements_vertical
)

config["Output_options"]["output_freq"] = str(
    args.output_freq
)

# ----------------------------
# Write new config
# ----------------------------

with open(output_file, "w") as f:
    config.write(f)

print(f"Created config file: {output_file}")