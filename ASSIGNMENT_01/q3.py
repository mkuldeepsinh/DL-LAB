import numpy as np

def perceptron(inputs, weights, bias):
    activation = np.dot(inputs, weights) + bias
    return 1 if activation >= 0 else 0

def solve_gate(gate_type):
    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
    if gate_type == "AND":
        w, b = np.array([1, 1]), -1.5
    elif gate_type == "OR":
        w, b = np.array([1, 1]), -0.5
    
    return [perceptron(i, w, b) for i in inputs]

and_results = solve_gate("AND")
or_results = solve_gate("OR")