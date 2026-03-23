import casadi as ca


class ComplementarityReLU_MLP:
	"""ReLU network expressed with complementarity constraints.

	Hidden layers enforce: y >= 0, y >= z, and y * (y - z) <= 0.
	The last layer is linear.
	"""

	def __init__(self, layer_sizes, sym_type="SX", name="nn", use_bias=True):
		if len(layer_sizes) < 2:
			raise ValueError("layer_sizes must have at least input and output sizes")
		self.layer_sizes = list(layer_sizes)
		self.sym_type = sym_type
		self.name = name
		self.use_bias = bool(use_bias)
		self.n_params = self._count_parameters()

	def _count_parameters(self):
		"""Count total number of parameters for the network."""
		count = 0
		for layer_idx in range(len(self.layer_sizes) - 1):
			n_in = self.layer_sizes[layer_idx]
			n_out = self.layer_sizes[layer_idx + 1]
			count += n_in * n_out
			if self.use_bias:
				count += n_out
		return count

	def _sym(self, label, nrow, ncol=1):
		if self.sym_type.upper() == "MX":
			return ca.MX.sym(label, nrow, ncol)
		return ca.SX.sym(label, nrow, ncol)

	def create_parameters(self):
		"""Create symbolic parameters (weights and biases).

		Returns
		-------
		params : list of tuple
			List of (W, b) pairs per layer. If use_bias is False, b is None.
		params_flat : casadi.SX or casadi.MX
			Flattened parameter vector.
		"""
		params = []
		flat_parts = []
		for layer_idx in range(len(self.layer_sizes) - 1):
			n_in = self.layer_sizes[layer_idx]
			n_out = self.layer_sizes[layer_idx + 1]
			W = self._sym(f"{self.name}_W{layer_idx}", n_out, n_in)
			b = None
			if self.use_bias:
				b = self._sym(f"{self.name}_b{layer_idx}", n_out, 1)
			params.append((W, b))
			flat_parts.append(ca.reshape(W, -1, 1))
			if b is not None:
				flat_parts.append(ca.reshape(b, -1, 1))
		params_flat = ca.vertcat(*flat_parts) if flat_parts else self._sym(f"{self.name}_params", 0, 1)
		return params, params_flat

	def build(self, x, params=None, tau=1.0):
		"""Build network output and complementarity constraints.

		Parameters
		----------
		x : casadi.SX or casadi.MX
			Input vector.
		params : list of tuple, optional
			(W, b) pairs per layer. If None, they are created symbolically. When
			use_bias is False, b should be None.
		tau : float, optional
			Relaxation parameter for complementarity constraints. Default is 1.0 (no relaxation).

		Returns
		-------
		result : dict
			Keys: output, params, params_flat, vars, lbw, ubw, g, lbg, ubg.
		"""
		if params is None:
			params, params_flat = self.create_parameters()
		else:
			params_flat = None

		a = x
		vars_list = []
		lbw = []
		ubw = []
		g = []
		lbg = []
		ubg = []

		for layer_idx, (W, b) in enumerate(params):
			z = W @ a
			if b is not None:
				z = z + b
			is_last = layer_idx == len(params) - 1
			if is_last:
				a = z
				continue

			n_out = self.layer_sizes[layer_idx + 1]
			y_layer = self._sym(f"{self.name}_y{layer_idx}", n_out, 1)

			# Here we store the hidden layer outputs as optimization variables to enforce the complementarity constraints.
			vars_list.extend([y_layer])
			lbw.extend([0.0] * n_out)
			ubw.extend([ca.inf] * n_out)

			# y - z >= 0 (elementwise)
			g.append(y_layer - z)
			lbg.extend([0.0] * n_out)
			ubg.extend([ca.inf] * n_out)

			# y * (y - z) <= tau (elementwise)
			g.append(y_layer * (y_layer - z))
			lbg.extend([-ca.inf] * n_out)
			ubg.extend([tau] * n_out)

			a = y_layer

		return {
			"output": a,
			"params": params,
			"params_flat": params_flat,
			"vars": vars_list,
			"lbw": lbw,
			"ubw": ubw,
			"g": g,
			"lbg": lbg,
			"ubg": ubg,
		}

	def build_function(self):
		"""Create a CasADi function for evaluation.

		The returned function accepts both numeric and symbolic inputs.
		The function also returns the constraint
		values and requires the hidden layer outputs as inputs.
		"""
		x = self._sym(f"{self.name}_x", self.layer_sizes[0], 1)
		params, params_flat = self.create_parameters()
		result = self.build(x, params=params)

		inputs = [x, params_flat, *result["vars"]]
		outputs = [result["output"]]
		outputs.append(ca.vertcat(*result["g"]) if result["g"] else self._sym(f"{self.name}_g", 0, 1))

		return ca.Function(f"{self.name}_eval", inputs, outputs)

def main():
	# Example usage and testing of the ComplementarityReLUNetwork
	net = ComplementarityReLU_MLP([2, 6, 6, 2])
	x = ca.SX.sym("x", 2)
	result = net.build(x)
	# print("Output expression:", result["output"])
	print("vars:", result["vars"])
	print("Symbolic vars:", ca.symvar(result["output"]))
	print("Symbolic constr vars:", [ca.symvar(g) for g in result["g"]])
	print("Number of variables:", sum(var.numel() for var in result["vars"]))
	print("Dimension of each variable:", [var.shape for var in result["vars"]])
	print("Number of parameters:", net.n_params)
	print("Number of constraints:", sum(g.numel() for g in result["g"]))

	# Output-only function: inputs are x and params_flat
	# f_out = net.build_function(include_constraints=False)
	x0 = ca.DM([1.0, 2.0])
	# out = f_out(x0, ca.DM.zeros(sum(W.numel() + (b.numel() if b is not None else 0) for W, b in result["params"])))
	# print("Output for x0:", out)

	# Output + constraints: inputs are x, params_flat, and each hidden-layer y
	f_all = net.build_function()
	numerical_constraint_inputs = [ca.DM.zeros(var.shape) for var in result["vars"]]
	symbolic_constraint_inputs = [ca.SX.sym(f"{net.name}_y{layer_idx}", var.shape) for layer_idx, var in enumerate(result["vars"])]
	print("Symbolic constraint inputs (hidden layer outputs):", symbolic_constraint_inputs)
	out = f_all(x0, ca.DM.zeros(sum(W.numel() + (b.numel() if b is not None else 0) for W, b in result["params"])), *symbolic_constraint_inputs)
	print("Output and constraints for x0:", out)
  
if __name__ == "__main__":
    main()