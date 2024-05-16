class Attribute:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.type = self.determine_type(description)

    def determine_type(self, description):
        # Simplified rules to determine the type of an attribute based on its description
        if "species" in description.lower() or "category" in description.lower():
            return "nominal"
        elif "score" in description.lower() or "rank" in description.lower():
            return "ordinal"
        elif "year" in description.lower() or "temperature" in description.lower():
            return "interval"
        elif "height" in description.lower() or "weight" in description.lower() or "length" in description.lower():
            return "ratio"
        else:
            return "unknown"

def evaluate_statements(attributes, statements):
    type_dict = {
        "nominal": "Nominal",
        "ordinal": "Ordinal",
        "interval": "Interval",
        "ratio": "Ratio"
    }
    
    results = []
    
    for statement in statements:
        try:
            eval_statement = eval(statement)
            results.append((statement, eval_statement))
        except Exception as e:
            results.append((statement, str(e)))
    
    return results

def main():
    # Define the attribute descriptions
    attribute_descriptions = [
        ("x1", "Species (Oak, pine, ...)"),
        ("x2", "Year planted (e.g. 1946)"),
        ("x3", "Tree height (in feet)"),
        ("x4", "Tree quality score (1, 2, ..., 5)")
    ]
    #her skal man ændre det man får fra tabellen.
    
    # Create Attribute objects
    attributes = [Attribute(name, description) for name, description in attribute_descriptions]
    
    # Print attributes and their types
    for attr in attributes:
        print(f"{attr.name}: {attr.description} -> {attr.type}")

    # Define the statements to evaluate
    statements = [
        'attributes[1].type == "interval" and attributes[2].type == "ratio"',
        'attributes[1].type == "ratio" and attributes[0].type == "nominal"',
        'attributes[0].type == "ordinal" and attributes[3].type == "ordinal"',
        'all(attr.type in ["nominal", "ordinal"] for attr in attributes)',
        'any(attr.type == "interval" for attr in attributes)',
        'attributes[0].type == "nominal"',
    ]

    results = evaluate_statements(attributes, statements)
    
    for statement, result in results:
        print(f"Statement: {statement} - Result: {result}")

if __name__ == "__main__":
    main()
