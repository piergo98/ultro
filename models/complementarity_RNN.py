import casadi as ca

class ComplementarityRNN:
	"""RNN with complementarity-encoded activation constraints.

	Supports standard ReLU complementarity and a lifted complementarity form
	where z = a - s with a, s >= 0 and a * s <= tau.
	"""

	def __init__(
		self,
		input_size,
		hidden_size,
		output_size,
		sym_type="SX",
		name="rnn",
		use_bias=True,
		output_bias=True,
		activation="relu",
		complementarity=False,
	):
		"""Initialize the ComplementarityRNN.

		Parameters
		----------
		input_size : int
			Number of features in the input.
		hidden_size : int, list
			Number of features in the hidden state.
			Can be an int for single layer or list of ints for multiple layers.
		output_size : int
			Number of features in the output.
		sym_type : str, optional
			CasADi symbolic type, "SX" or "MX", by default "SX".
		name : str, optional
			Base name for symbolic variables, by default "rnn".
		use_bias : bool, optional
			Whether to include bias terms, by default True.
		output_bias : bool, optional
			Whether to include bias in the output layer, by default True.
		activation : str, optional
			Activation function, currently only "relu" is supported, by default "relu".
		complementarity : bool, optional
			Whether to use complementarity constraints for ReLU, by default False.
    
    
    """
		self.input_size = int(input_size)
		# Check if hidden_size is a list for multiple layers
		if isinstance(hidden_size, list):
			self.hidden_size = [int(size) for size in hidden_size]
		else:
			self.hidden_size = [int(hidden_size)]
		if output_size > 0:
			self.output_size = int(output_size)
			self.output_layer = True
		else:
			self.output_layer = False
			self.output_size = 0
		self.sym_type = sym_type
		self.name = name
		self.use_bias = bool(use_bias)
		self.output_bias = bool(output_bias)
		self.activation = activation
		self.complementarity = bool(complementarity)
		self.n_params = self._count_parameters()

	def _count_parameters(self):
		"""Count total number of parameters for the RNN."""
		count = 0
		for i, size in enumerate(self.hidden_size):
			if i == 0:
				input_dim = self.input_size
			else:
				input_dim = self.hidden_size[i - 1]
			count += size * input_dim  # W_x
			count += size * size  # W_h
			if self.use_bias:
				count += size  # b_h
		if self.output_layer:
			count += self.output_size * self.hidden_size[-1]  # W_y
			if self.output_bias:
				count += self.output_size  # b_y
		return count

	def _sym(self, label, nrow, ncol=1):
		if self.sym_type.upper() == "MX":
			return ca.MX.sym(label, nrow, ncol)
		return ca.SX.sym(label, nrow, ncol)

	def create_parameters(self):
		"""Create symbolic parameters (weights and biases)."""
		params = []
		flat_parts = []
		for i, size in enumerate(self.hidden_size):
			if i == 0:
				input_dim = self.input_size
			else:
				input_dim = self.hidden_size[i - 1]
			W_x = self._sym(f"{self.name}_W_x{i}", size, input_dim)
			W_h = self._sym(f"{self.name}_W_h{i}", size, size)
			params.extend([W_x, W_h])
			flat_parts.extend([ca.reshape(W_x, -1, 1), ca.reshape(W_h, -1, 1)])
			b_h = None
			if self.use_bias:
				b_h = self._sym(f"{self.name}_b_h{i}", size, 1)
				params.append(b_h)
				flat_parts.append(ca.reshape(b_h, -1, 1))
    
		if self.output_layer:
			size = self.hidden_size[-1]
			W_y = self._sym(f"{self.name}_W_y", self.output_size, size)
			params.append(W_y)
			flat_parts.append(ca.reshape(W_y, -1, 1))
   
			b_y = None
			if self.output_bias:
				b_y = self._sym(f"{self.name}_b_y", self.output_size, 1)
				params.append(b_y)
				flat_parts.append(ca.reshape(b_y, -1, 1))

		# Create the params tuple in the expected order
		params = tuple(params)
		
		# The flat parameter vector is created by concatenating the flattened weight and bias matrices.
		params_flat = ca.vertcat(*flat_parts) if flat_parts else self._sym(f"{self.name}_params", 0, 1)
   
		return params, params_flat

	def build(self, x_seq, h0=None, tau=1.0):
		"""Build RNN output and complementarity constraints for a sequence.

		Parameters
		----------
		x_seq : casadi.SX or casadi.MX
			Input sequence matrix with shape (input_size, horizon).
		h0 : casadi.SX or casadi.MX, optional
			Initial hidden state (hidden_size, 1). If None, a symbolic input is created.
		tau : float, optional
			Relaxation parameter for complementarity constraints.

		Returns
		-------
		result : dict
			Keys: output, hidden, params, params_flat, vars, lbw, ubw, g, lbg, ubg.
		"""
		if self.activation != "relu":
			raise ValueError("Only 'relu' activation is supported in complementarity form.")
		
		# Extract parameters
		params, params_flat = self.create_parameters()
		
		W_x = params[0]
		W_h = params[1]
		b_h = params[2] if self.use_bias else None

		if h0 is None:
			h_prev = self._sym(f"{self.name}_h0", self.hidden_size[0], 1)
		else:
			h_prev = h0
		# TODO: add support for multiple RNN layers
		self.horizon = int(x_seq.shape[1])
		vars_list = []
		lbw = []
		ubw = []
		g = []
		lbg = []
		ubg = []
		hidden_list = []
		output_list = []

		for t in range(self.horizon):
			x_t = x_seq[:, t]
			for i in range(len(self.hidden_size)):
				if self.use_bias:
					W_x = params[3 * i]
					W_h = params[3 * i + 1]
					b_h = params[3 * i + 2]
				else:
					W_x = params[2 * i]
					W_h = params[2 * i + 1]
					b_h = None
				z_t = W_x @ x_t + W_h @ h_prev
				if b_h is not None:
					z_t = z_t + b_h

				if self.complementarity:
					h_t = self._sym(f"{self.name}_h{t}_{i}", self.hidden_size[i], 1)
					vars_list.append(h_t)
					lbw.extend([0.0] * self.hidden_size[i])
					ubw.extend([ca.inf] * self.hidden_size[i])

					# h - z >= 0
					g.append(h_t - z_t)
					lbg.extend([0.0] * self.hidden_size[i])
					ubg.extend([ca.inf] * self.hidden_size[i])

					# h * (h - z) <= tau
					# g.append(h_t * (h_t - z_t))
					# lbg.extend([-ca.inf] * self.hidden_size[i])
					# ubg.extend([tau] * self.hidden_size[i])

				else:
					h_t = ca.fmax(z_t, 0)

				# If there are multiple layers, the output of the current layer becomes the input to the next layer
				x_t = h_t
			if self.output_layer:
				W_y = params[-2] if self.output_bias else params[-1]
				b_y = params[-1] if self.output_bias else None
				y_t = W_y @ h_t
				if b_y is not None:
					y_t = y_t + b_y
			output_list.append(y_t)
			hidden_list.append(h_t)
			h_prev = h_t

		output_seq = ca.hcat(output_list) if output_list else self._sym(f"{self.name}_y", 0, 1)
		hidden_seq = ca.hcat(hidden_list) if hidden_list else self._sym(f"{self.name}_h", 0, 1)

		return {
			"output": output_seq,
			"hidden": hidden_seq,
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
		"""Create a CasADi function for evaluation over a fixed horizon."""
		x_seq = self._sym(f"{self.name}_x", self.input_size, self.horizon)
		h0 = self._sym(f"{self.name}_h0", self.hidden_size, 1)
		params, params_flat = self.create_parameters()
		result = self.build(x_seq, h0=h0, params=params)

		if self.complementarity:
			inputs = [x_seq, h0, params_flat, *result["vars"]]
		else:
			inputs = [x_seq, h0, params_flat]
		outputs = [result["output"], result["hidden"]]
  
		if self.complementarity:
			outputs.append(ca.vertcat(*result["g"]) if result["g"] else self._sym(f"{self.name}_g", 0, 1))

		return ca.Function(f"{self.name}_eval", inputs, outputs)

def main():
	# Example usage and testing of the ComplementarityRNN
	net = ComplementarityRNN(2, 4, 1, complementarity=False)
	# Build a RNN cell for a sequence of length 3 of 2D inputs
	x_seq = ca.SX.sym("x", 2, 3)
	result = net.build(x_seq)
	print("vars:", result["vars"])
	print("Symbolic vars:", ca.symvar(result["output"]))
	print("Symbolic constr vars:", [ca.symvar(g) for g in result["g"]])
	print("Number of variables:", sum(var.numel() for var in result["vars"]))
	print("Dimension of each variable:", [var.shape for var in result["vars"]])
	print("Number of parameters:", net.n_params)
	print("Number of constraints:", sum(g.numel() for g in result["g"]))

	# Output + constraints: inputs are x_seq, h0, params_flat, and each hidden variable
	f_all = net.build_function()
	x0 = ca.DM([[1.0, 0.5, -0.2], [0.0, 1.0, 0.3]])
	h0 = ca.DM.zeros(4, 1)
	params_zero = ca.DM.zeros(net.n_params)
	constraint_inputs = [ca.DM.zeros(var.shape) for var in result["vars"]]
	out = f_all(x0, h0, params_zero, *constraint_inputs)
	print("Output and constraints for x_seq:", out)
  
if __name__ == "__main__":
    main()