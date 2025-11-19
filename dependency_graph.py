import os
import json
import re
from loguru import logger

class DependencyGraphBuilder:
    """
    Build a dependency graph from function_map.json.

    Input (function_map.json):
    {
        "src/file1.c": {
            "funcA": { "start": 10, "end": 30, "code": "...", "summary": "..." },
            "funcB": {...}
        }
    }

    Output (dependencies.json):
    {
        "src/file1.c": {
            "funcA": {
                "calls": ["funcB"],
                "called_by": [],
                "file": "src/file1.c"
            }
        }
    }
    """

    # Matches possible function calls:   foo(   ,   foo (  
    CALL_REGEX = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

    # Known language constructs / false matches
    IGNORED_CALLS = {
        "if", "for", "while", "switch", "return", "sizeof",
        "else", "do",
        # common stdlib functions
        "printf", "scanf", "puts", "putchar", "malloc", "calloc",
        "free", "realloc", "exit", "fprintf", "fscanf", "strcpy",
        "strcmp", "strlen", "tolower", "toupper"
    }

    # skip tests? (optional)
    SKIP_TEST_FUNCTIONS = False

    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.function_map_path = os.path.join(repo_path, "function_map.json")
        self.dependencies_path = os.path.join(repo_path, "dependencies.json")

    # ------------------------------------------------------------
    def load_function_map(self):
        if not os.path.exists(self.function_map_path):
            raise FileNotFoundError(f"{self.function_map_path} not found.")

        with open(self.function_map_path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------
    def extract_calls_from_code(self, code, current_function):
        """
        Extract function calls using regex.

        Removes:
        - self calls
        - keywords
        - false positives
        """
        found = set()

        for m in self.CALL_REGEX.finditer(code):
            name = m.group(1)

            # skip language constructs and known junk
            if name in self.IGNORED_CALLS:
                continue

            # skip self call (avoid false recursive detection)
            if name == current_function:
                continue

            # Optional: skip test function calls unless same file
            if self.SKIP_TEST_FUNCTIONS and name.startswith("test_"):
                continue

            found.add(name)

        return sorted(found)

    # ------------------------------------------------------------
    def build_graph(self):
        logger.info("Loading function_map.json...")
        function_map = self.load_function_map()

        logger.info("Building dependency graph...")

        graph = {}

        # ---------------------------------------
        #     Build CALLS list for each function
        # ---------------------------------------
        for file, funcs in function_map.items():
            graph[file] = {}

            for fn, entry in funcs.items():
                code = entry["code"]
                calls = self.extract_calls_from_code(code, current_function=fn)

                graph[file][fn] = {
                    "calls": calls,
                    "called_by": [],
                    "file": file
                }

        # ---------------------------------------
        #  Reverse edges: populate CALLED_BY
        # ---------------------------------------
        all_functions = {}

        # global lookup name → list of (file, fn)
        for file, funcs in graph.items():
            for fn in funcs:
                all_functions.setdefault(fn, []).append((file, fn))

        # fill called_by
        for file, funcs in graph.items():
            for fn, data in funcs.items():
                for callee in data["calls"]:
                    if callee in all_functions:
                        # many files can define a function with same name
                        for (callee_file, _) in all_functions[callee]:

                            # ignore self-calls
                            if fn == callee and file == callee_file:
                                continue

                            # register reverse edge
                            graph[callee_file][callee]["called_by"].append(fn)

        return graph

    # ------------------------------------------------------------
    def save_graph(self, graph):
        with open(self.dependencies_path, "w") as f:
            json.dump(graph, f, indent=2)

        logger.info(f"Saved dependencies.json → {self.dependencies_path}")

    # ------------------------------------------------------------
    def run(self):
        graph = self.build_graph()
        self.save_graph(graph)
        return graph


# =====================================================================
#                        COMMAND LINE ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python dependency_graph.py <repo_path>")
        exit(1)

    repo_path = sys.argv[1]
