import numpy as np

class Node:
    def __init__(self, name, condition):
        self.name = name
        self.condition = condition

    def evaluate(self, x):
        return eval(self.condition)

def evaluate_statements(nodes, x):
    results = []
    
    for node in nodes:
        try:
            result = node.evaluate(x)
            results.append((node.name, node.condition, result))
        except Exception as e:
            results.append((node.name, node.condition, str(e)))
    
    return results

def main():
    # Define the nodes and their conditions
    nodes = [
        Node("A", "np.linalg.norm(x, np.inf) < 0.25"),  # Example condition for node A
        Node("B", "np.linalg.norm(x - np.array([-1, 1]), 1) > 2"),  # Example condition for node B
        Node("C", "np.linalg.norm(x - np.array([0, 1]), 2) < 1")  # Example condition for node C
    ]
    
    #Update the nodes list with the actual conditions of your classification tree nodes.

    # Example coordinates to test
    test_points = [
        np.array([0.5, 0.5]),
        np.array([0.1, 0.1]),
        np.array([-0.5, 0.5])
    ]
    
    # Update the test_points list with the coordinates you want to test against the node conditions.
    
    for x in test_points:
        print(f"Evaluating for point: {x}")
        results = evaluate_statements(nodes, x)
        
        for node_name, condition, result in results:
            print(f"Node {node_name} with condition '{condition}' - Result: {result}")

if __name__ == "__main__":
    main()
