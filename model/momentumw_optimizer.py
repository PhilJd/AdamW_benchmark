# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Momentum with weight decay for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import momentum
from tensorflow.python.training import training_ops


class MomentumWOptimizer(WeightDecayHelper, momentum.MomentumOptimizer):
  """Optimizer that implements the Momentum algorithm with weight_decay.

  Computes (if `use_nesterov = False`):

  ```
  accumulation = momentum * accumulation + gradient
  variable -= learning_rate * accumulation
  ```

  Note that in the dense version of this algorithm, `accumulation` is updated
  and applied regardless of a gradient's value, whereas the sparse version (when
  the gradient is an `IndexedSlices`, typically because of `tf.gather` or an
  embedding) only updates variable slices and corresponding `accumulation` terms
  when that part of the variable was used in the forward pass.
  """

  def __init__(self, learning_rate, momentum, schedule_multiplier, weight_decay,
               use_locking=False, name="MomentumW", use_nesterov=False):
    """Construct a new Momentum optimizer.

tensor    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
      use_nesterov: If `True` use Nesterov Momentum.
        See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
        This implementation always computes gradients at the value of the
        variable(s) passed to the optimizer. Using Nesterov Momentum makes the
        variable(s) track the values called `theta_t + mu*v_t` in the paper.

    @compatibility(eager)
    When eager execution is enabled, learning_rate and momentum can each be a
    callable that takes no arguments and returns the actual value to use. This
    can be useful for changing these values across different invocations of
    optimizer functions.
    @end_compatibility
    """
    super(MomentumWOptimizer, self).__init__(
        learning_rate, momentum, use_locking=use_locking, name=name,
        use_nesterov=use_nesterov)
    self._weight_decay = weight_decay
    self._schedule_multiplier = schedule_multiplier

  def _prepare(self):
    super(MomentumWOptimizer, self)._prepare()
    weight_decay = self._weight_decay
    if callable(weight_decay):
      weight_decay = weight_decay()
    self._weight_decay_tensor = ops.convert_to_tensor(
        weight_decay, name="weight_decay")

    schedule_multiplier = self._schedule_multiplier
    if callable(schedule_multiplier):
      schedule_multiplier = schedule_multiplier()
    self._schedule_multiplier_tensor = ops.convert_to_tensor(
        schedule_multiplier, name="schedule_multiplier")

  def _apply_dense(self, grad, var):
    with ops.control_dependencies([self._decay_weights(var)]):
      return super(MomentumWOptimizer, self)._apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    with ops.control_dependencies([self._decay_weights(var, True)]):
      return super(MomentumWOptimizer, self)._resource_apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    with ops.control_dependencies([self._decay_weights(var)]):
      return super(MomentumWOptimizer, self)._apply_sparse(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    with ops.control_dependencies([self._decay_weights(var, True)]):
      return super(MomentumWOptimizer, self)._resource_apply_sparse(grad, var, indices)

  def _decay_weights(self, var, is_resource_var=False):
    # assign_op = (state_ops.assign_sub if is_resource_var else
    #              resource_variable_ops.assign_sub)
    # assign_op(var, self._schedule_multiplier_tensor * self._weight_decay * var)
    return var.assign_sub(self._schedule_multiplier_tensor * self._weight_decay * var)
