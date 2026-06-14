from expression_graph import parse_program_to_tree, tree_to_edges, tree_to_string


EXAMPLES = [
    "sin(), exp()",
    "sin(), poly([0,3]), exp()\nsinc()",
    "poly( [0, 0.2] )",
]


def main():
    for program in EXAMPLES:
        tree = parse_program_to_tree(program)
        print("Program:")
        print(program)
        print()
        print("Expression:")
        print(tree_to_string(tree))
        print()
        print("Edges:")
        for parent, child in tree_to_edges(tree):
            print(f"{parent} -> {child}")
        print()


if __name__ == "__main__":
    main()
