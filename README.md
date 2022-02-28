# FourDVar-Lorenz96

This is scripts for 4D-Var using a neural network surrogate model obtained by machine learning for Lorenz 96 model.


# Contents

* Misc
  * README.md: this file
  * LICENSE: license file

* For Experiments
  * lorenz96.py: Lorenz 96 model
  * net.py: neural network model
  * train_1step.py: for one-step learing
  * train_10step.py: for ten-step learing
  * 4dvar.py: for 4D-Var experiment without observation error
  * 4dvar_err.py: for 4D-Var experiment with observation error
  * 4dvar_phy.py: for 4D-Var experiment with manually constructed adjoint model

* For Analysis and Visualization
  * run_err.py
  * learning_curve_1step.py
  * graph_loss.py
  * graph_4dvar.py
  * accuracy_4dvar.py
  * lorenz96.rb
  * learning.rb
  * cost.rb
  * 4dvar.rb
  * 4dvar_rmse.rb

