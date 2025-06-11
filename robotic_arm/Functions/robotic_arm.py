import numpy as np
import pandas as pd
from Functions.servomotors import v4_x_series 

class robotic_arm:
  """
  A class to simulate and analyze the behavior of a robotic arm composed of servomotors.

  Attributes:
      arm_structure (pd.DataFrame): Table containing servo model references and torques for each motor.
      current (float): Total current drawn by the arm in Amperes.
      voltage (float): Average voltage across the motors in Volts.
      efficiency (float): Electrical efficiency (%) of the entire arm.
      power_in (float): Total input electrical power (Watts).
      power_out (float): Total mechanical output power (Watts).
      safety_factor (float): Safety margin applied to rated current in `safe_point()`.

  Methods:
      operation_point():
          Calculates the system-wide electrical and mechanical parameters for user-defined torque values.
      rated_point():
          Calculates system parameters using rated values from datasheets.
      max_point():
          Calculates system parameters using maximum values of each servo.
      safe_point():
          Calculates parameters using rated values with a safety factor applied to current.
  """
  
  def __init__(self, arm_file="Robotic_arm_motors.xlsx"):
    """
    Initializes the robotic_arm with motor configuration data.

    Args:
        arm_file (str): Path to the Excel file containing motor references and desired torques.
    """
    self.arm_structure = pd.read_excel(arm_file)
    
    self.current = 0
    self.voltage = 24
    self.efficiency = 0
    self.power_in = 0
    self.power_out = 0
    self.safety_factor = .2
    
  def __compute_arm_point(self, point_getter, modify_current=False):
    """
    Internal method to calculate the system's aggregate performance based on a specified
    point evaluation function for each servo.

    Args:
        point_getter (function): A callback that returns the operation point data for a servo.
        modify_current (bool): If True, applies a safety margin to each servo's current.

    Returns:
        pd.DataFrame: Operation parameters per servo at the evaluated point.
    """
    self.voltage = self.current = self.power_in = self.power_out = 0
    servos_op = pd.DataFrame()
    voltage_acum = 0

    for servo_idx in self.arm_structure.index:
        motor_name = f"M{servo_idx+1}"
        servo_data = self.arm_structure.loc[servo_idx]
        servo = v4_x_series(servo_data["Servomotor reference"])
        op_servo = point_getter(servo, servo_data)

        if servos_op.empty:
            servos_op = pd.DataFrame(columns=op_servo.index)
        servos_op.loc[motor_name] = op_servo

        # Optional safety modification
        if modify_current:
            op_servo["I(A)"] *= (1 + self.safety_factor)
            servos_op.loc[motor_name, "I(A)"] = op_servo["I(A)"]
            servos_op.loc[motor_name, "Pin(W)"] = op_servo["U(V)"] * op_servo["I(A)"]

        voltage_acum += op_servo["U(V)"]
        self.current += op_servo["I(A)"]
        self.power_in += op_servo["U(V)"] * op_servo["I(A)"]
        self.power_out += op_servo["N(r/min)"] * op_servo["T(N.m)"] * 2 * np.pi / 60

    self.voltage = voltage_acum / len(self.arm_structure.index)
    self.efficiency = 100 * self.power_out / self.power_in
    return servos_op
    
  def operation_point(self):
    """
    Calculates performance using custom torque values defined in the Excel file.

    Returns:
        tuple:
            - pd.DataFrame: Servo-specific performance parameters.
            - pd.Series: Aggregate arm performance summary.
    """
    def getter(s, d): return s.get_operation_point(d["Operation torque [N.m]"])
    servos_op = self.__compute_arm_point(getter)
    return servos_op, self.__arm_summary("Operation")

  def rated_point(self):
    """
    Calculates performance based on each motor's rated (nominal) values.

    Returns:
        tuple:
            - pd.DataFrame: Servo-specific rated performance data.
            - pd.Series: Aggregate arm performance summary.
    """
    servos_op = self.__compute_arm_point(lambda s, _: s.get_rated_point())
    return servos_op, self.__arm_summary("Rated")
  
  def max_point(self):
    """
    Calculates performance assuming each servo is operating at its maximum output levels.

    Returns:
        tuple:
            - pd.DataFrame: Servo-specific max performance data.
            - pd.Series: Aggregate arm performance summary.
    """
    servos_op = self.__compute_arm_point(lambda s, _: s.get_max_point())
    return servos_op, self.__arm_summary("Maximum")
  
  def safe_point(self):
    """
    Calculates performance using rated values while applying a safety factor to the current
    to simulate overload handling or conservative operation.

    Returns:
        tuple:
            - pd.DataFrame: Servo-specific rated data with modified current.
            - pd.Series: Aggregate arm performance summary.
    """
    servos_op = self.__compute_arm_point(lambda s, _: s.get_rated_point(), modify_current=True)
    return servos_op, self.__arm_summary("Rated with safety factor")
  
  def __arm_summary(self, name):
    """
    Summarizes the arm's total voltage, current, efficiency, and input power under the
    last evaluated condition.

    Args:
        name (str): Label to assign to the resulting Series.

    Returns:
        pd.Series: Aggregate summary with ["U(V)", "I(A)", "Eff(%)", "P(W)"].
    """
    return pd.Series(
        data=[self.voltage, self.current, self.efficiency, self.power_in],
        index=["U(V)", "I(A)", "Eff(%)", "P(W)"],
        name=name
    )
