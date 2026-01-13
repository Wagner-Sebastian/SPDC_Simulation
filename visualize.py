import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog="SPDC_Visualizer",
    description="Takes in data produced by SPDC-simulation and creates a Plot",
    epilog=".",
)
parser.add_argument("data_path")
parser.add_argument("-x", "--x_count", type=int, default=800)
parser.add_argument("-y", "--y_count", type=int, default=800)
parser.add_argument("-q_inc_x", "--q_incremental_x", type=float, default=9e-06)
parser.add_argument("-q_inc_y", "--q_incremental_y", type=float, default=9e-06)
parser.add_argument("save_name")

args = parser.parse_args()
x_min = -args.x_count * 1000 / 2 * args.q_incremental_x
x_max = args.x_count * 1000 / 2 * args.q_incremental_x
y_min = -args.y_count * 1000 / 2 * args.q_incremental_y
y_max = args.y_count * 1000 / 2 * args.q_incremental_y
print(x_min, y_min)

read_data = np.fromfile(args.data_path, dtype=np.float32).reshape(
    args.x_count, args.y_count
)
plt.figure(figsize=(8, 6))
plt.imshow(read_data, origin="lower", extent=(x_min, x_max, y_min, y_max), cmap="gray")

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("SPDC BBO Entanglement")
plt.savefig(f"./{args.save_name}_spdc_sim.png", dpi=300)
plt.close()
