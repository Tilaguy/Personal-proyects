import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt

def estimate_funct(torque, variable, n_knots=5, degree=3):
  """
  Estimates a regression model that approximates a variable as a function of torque
  using a spline transformation and ridge regression.

  Parameters:
      torque (np.ndarray): Array of torque values.
      variable (np.ndarray): Array of the target variable to approximate.
      n_knots (int): Number of knots for the spline transformer.
      degree (int): Degree of the spline function.

  Returns:
      model (Pipeline): Trained pipeline model to estimate the variable.
  """
  X_train = torque[:, np.newaxis]
  model = make_pipeline(SplineTransformer(n_knots=n_knots, degree=degree), Ridge(alpha=1e-3))
  model.fit(X_train, variable)
  return model

class v4_x_series:
  """
  Represents a V4-X series servomotor, capable of estimating operating variables
  at given torque levels using fitted regression models.

  Attributes:
      speed (float): Estimated speed [r/min].
      power_output (float): Estimated output power [W].
      voltage (float): Estimated voltage [V].
      current (float): Estimated current [A].
      power_input (float): Estimated input power [W].
      efficiency (float): Estimated efficiency [%].
  """
  def __init__(self, ref: str, servo_name: str = ""):
    """
    Initializes the servomotor model using data from CSV databases and fits
    regression models for all operational variables.

    Parameters:
        ref (str): Reference name of the motor model (e.g., "4-p36").
        servo_name (str): Optional custom name for the motor.
    """
    self.speed = 0
    self.power_output = 0
    self.voltage = 0
    self.voltage = 0
    self.current = 0
    self.power_input = 0
    self.efficiency = 0
    self.__ref = ref
    
    __file_name_dict = {
      "4-p36": "Databases/4_p36.csv",
      "2-p28": "Databases/2_p28.csv",
    }
    
    if ref not in __file_name_dict.keys():
      raise ValueError("Motor reference is not into database")
    
    self.__db = pd.read_csv(
      filepath_or_buffer=__file_name_dict[ref],
      sep=";",
      header=0,
      )
    
    # Fit estimation models
    self.__speed_func = estimate_funct(self.__db["T(N.m)"].values, self.__db["N(r/min)"].values)
    self.__power_out_func = estimate_funct(self.__db["T(N.m)"].values, self.__db["Pout(W)"].values, n_knots=6)
    self.__voltage_func = estimate_funct(self.__db["T(N.m)"].values, self.__db["U(V)"].values)
    self.__current_func = estimate_funct(self.__db["T(N.m)"].values, self.__db["I(A)"].values, n_knots=7)
    self.__power_input_func = estimate_funct(self.__db["T(N.m)"].values, self.__db["Pin(W)"].values)
    self.__efficiency_func = estimate_funct(self.__db["T(N.m)"].values, self.__db["Eff(%)"].values, degree=3, n_knots=10)
    
  def get_operation_point(self, torque: float = 0.0):
    """
    Predicts the motor's operating parameters at a given torque.

    Parameters:
        torque (float): Torque value to evaluate.

    Returns:
        pd.Series: A row containing all estimated variables at the given torque.
    """
    self.speed = self.__speed_func.predict([[torque]])[0]
    self.power_output = self.__power_out_func.predict([[torque]])[0]
    self.voltage = self.__voltage_func.predict([[torque]])[0]
    self.current = self.__current_func.predict([[torque]])[0]
    self.power_input = self.__power_input_func.predict([[torque]])[0]
    self.efficiency = self.__efficiency_func.predict([[torque]])[0]
    
    operation_point = self.__db.copy()
    operation_point.loc[len(operation_point)] = [torque, self.speed, self.power_output, self.voltage, self.current, self.power_input, self.efficiency]
    
    return operation_point.iloc[-1, :]
  
  def get_rated_point(self):
    """
    Returns the rated point (nominal torque) for the motor.

    Returns:
        pd.Series: A row containing all estimated variables at nominal torque.
    """
    nom_torque_dict = {
      "4-p36": 10.5,
      "2-p28": 2.5,
    }
    torque = nom_torque_dict[self.__ref]
    return self.get_operation_point(torque)
  
  def get_max_point(self):
    """
    Returns the motor's estimated variables at the maximum torque in the database.

    Returns:
        pd.Series: A row containing all estimated variables at the max torque.
    """
    torque = self.__db.loc[self.__db.index[-1], 'T(N.m)']
    return self.get_operation_point(torque)
  
  def validate_model(self, model_name: str):
    """
    Visualizes the fitted model and actual data to assess the model's accuracy.

    Parameters:
        model_name (str): One of:
          - "Speed [r/min]"
          - "Pout [W]"
          - "Voltage [V]"
          - "Current [A]"
          - "Pin [W]"
          - "Efficiency (%)"
    """
    model_map  = {
      "Speed [r/min]": self.__speed_func,
      "Pout [W]": self.__power_out_func,
      "Voltage [V]": self.__voltage_func,
      "Current [A]": self.__current_func,
      "Pin [W]": self.__power_input_func,
      "Efficiency (%)": self.__efficiency_func,
    }
    variable_map  = {
      "Speed [r/min]":  "N(r/min)",
      "Pout [W]": "Pout(W)",
      "Voltage [V]": "U(V)",
      "Current [A]": "I(A)",
      "Pin [W]": "Pin(W)",
      "Efficiency (%)": "Eff(%)",
    }
    
    if model_name not in model_map:
      raise ValueError(f"Invalid model name: '{model_name}'")
    
    model = model_map[model_name]
    variable = variable_map[model_name]
    
    x = self.__db["T(N.m)"].values[:, np.newaxis]
    y_true = self.__db[variable].values
    y_pred = model.predict(x)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(x, y_true, 'o-', label=f"Ground truth ({model_name})")
    ax.plot(x, y_pred, '--', label=f"{model_name} model")
    ax.set_title(f"Model Validation - {model_name}")
    ax.set_xlabel("Torque [N·m]")
    ax.set_ylabel(model_name)
    ax.legend()
    ax.grid(True)
    window = plt.get_current_fig_manager().window
    window.state('zoomed')
    plt.show()
