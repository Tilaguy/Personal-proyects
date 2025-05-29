import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Functions.robotic_arm import robotic_arm

def generate_arm_report():
  arm = robotic_arm()

  servos_op, arm_operation = arm.operation_point()
  servos_rated, arm_rated = arm.rated_point()
  servos_safe, arm_safe = arm.safe_point()
  servos_max, arm_max = arm.max_point()

  print("="*90)
  print("🔧 Robotic Arm - Operating at Edge Conditions:")
  print(servos_op)
  print("-"*90)
  print("✅ Robotic Arm - Rated Operation:")
  print(servos_rated)
  print("-"*90)
  print("🛡️ Robotic Arm - Rated Operation + Safety Factor:")
  print(servos_safe)
  print("-"*90)
  print("⚡ Robotic Arm - Maximum Capacity:")
  print(servos_max)

  arm_report_df = pd.concat([arm_operation, arm_rated, arm_safe, arm_max], axis=1)
  print("-"*90)
  print("📊 Summary: Robotic Arm Operating Points")
  print(arm_report_df)

  safe_I = arm_report_df.loc["I(A)", "Rated with safety factor"]
  safe_P = arm_report_df.loc["P(W)", "Rated with safety factor"]
  op_I = arm_report_df.loc["I(A)", "Operation"]
  op_P = arm_report_df.loc["P(W)", "Operation"]
  rated_I = arm_report_df.loc["I(A)", "Rated"]
  rated_P = arm_report_df.loc["P(W)", "Rated"]

  print("-"*90)
  print(f"🔌 Power Source Requirements (with safety factor):")
  print(f"- Output voltage: 24 VDC (stable)")
  print(f"- Output current ≥ {safe_I:.2f} A")
  print(f"- Output power ≥ {max(safe_P, safe_I * 24):.2f} W")

  if op_I > safe_I or op_P > safe_P:
      print(f"💥 WARNING: Operating point exceeds safe design limits!")
      print(f"  - Current: {op_I:.2f} A > Safe: {safe_I:.2f} A")
      print(f"  - Power  : {op_P:.2f} W > Safe: {safe_P:.2f} W")

  elif op_I > rated_I or op_P > rated_P:
      print(f"⚠️ WARNING: Operating point exceeds rated values.")
      print(f"  - Current: {op_I:.2f} A > Rated: {rated_I:.2f} A")
      print(f"  - Power  : {op_P:.2f} W > Rated: {rated_P:.2f} W")

  print("="*90)

  category_names = ['Operation', 'Rated', 'Rated with safety factor', 'Maximum']
  report_subset = arm_report_df.loc[["I(A)", "P(W)"]]

  raw_data = report_subset.values
  data_max = np.max(raw_data, axis=1).reshape(-1, 1)
  normalized = raw_data / data_max

  index = np.arange(normalized.shape[0])
  bar_height = 0.2
  category_colors = plt.colormaps['Paired'](np.linspace(0.15, 0.41, len(category_names)))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.invert_yaxis()
  ax.xaxis.set_visible(False)
  ax.set_xlim(0, 1)

  for j in range(len(category_names)):
      bars = ax.barh(index + j * bar_height, normalized[:, j], height=bar_height,
                    color=category_colors[j], label=category_names[j])
      labels = [f"{v:.2f}" for v in raw_data[:, j]]
      ax.bar_label(bars, labels=labels, label_type='center', fontsize=14, color='black')

  ax.set_yticks(index + (bar_height * (len(category_names) - 1)) / 2)
  ax.set_yticklabels(report_subset.index)
  ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')

  plt.title('⚙️ Power and Current Consumption per Operation Case', fontsize=16)
  plt.tight_layout()
  window = plt.get_current_fig_manager().window
  window.state('zoomed')
  plt.show()

if __name__ == '__main__':
  generate_arm_report()
  