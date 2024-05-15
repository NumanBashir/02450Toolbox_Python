import numpy as np

# Definer fejlrate data
features = ['None', 'x1', 'x2', 'x3', 'x4', 'x1,x2', 'x1,x3', 'x1,x4', 'x2,x3', 'x2,x4', 'x3,x4',
            'x1,x2,x3', 'x1,x2,x4', 'x1,x3,x4', 'x2,x3,x4', 'x1,x2,x3,x4']
E_test = [0.9, 0.622, 0.721, 0.446, 0.645, 0.574, 0.311, 0.503, 0.428, 0.515, 0.458, 0.324, 0.534, 0.314, 0.391, 0.641]

# Definer en funktion til at finde det bedste træk ved at tilføje en funktion ad gangen
def forward_selection(E_test, features):
    selected_features = []
    best_error = E_test[0]  # Start med ingen funktioner
    available_features = features[1:]  # Alle funktioner bortset fra 'None'
    while True:
        candidates = []
        for i, error in enumerate(E_test[1:], start=1):
            if error < best_error and features[i] in available_features:
                candidates.append((error, features[i]))
        if not candidates:
            break
        best_error, best_feature = min(candidates)
        selected_features.append(best_feature)
        available_features.remove(best_feature)
    return selected_features

# Definer en funktion til at finde det bedste træk ved at fjerne en funktion ad gangen
def backward_selection(E_test, features):
    selected_features = features[-1].split(',')  # Start med alle funktioner
    best_error = E_test[-1]
    current_features = selected_features.copy()
    while True:
        candidates = []
        for i, error in enumerate(E_test[:-1]):
            if error < best_error and all(f in selected_features for f in features[i].split(',')):
                candidates.append((error, features[i]))
        if not candidates:
            break
        best_error, best_feature_set = min(candidates)
        current_features = best_feature_set.split(',')
        selected_features = current_features.copy()
    return selected_features

# Find de bedste træk ved forward og backward selection
forward_selected = forward_selection(E_test, features)
backward_selected = backward_selection(E_test, features)

print(f"Forward selection features: {forward_selected}")
print(f"Backward selection features: {backward_selected}")
