import os
import re
import json
import requests
import time
from loguru import logger

# ===============================================================
#            JSON REPAIR + FLEXIBLE PARSING HELPERS
# ===============================================================

def strip_markdown_fences(text: str) -> str:
    """Remove ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return text.strip()


def fix_json_common_errors(text: str) -> str:
    """Fix common JSON mistakes by LLM."""
    text = text.replace("\r", "").replace("\n", " ")
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r"(\w+):", r'"\1":', text)     # missing quotes on keys
    return text


def safe_parse_json(text: str, function_names=None):
    """
    Try strict JSON, then loose JSON, then fallback to minimal summaries.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try json5 (tolerant)
    try:
        import json5
        return json5.loads(text)
    except Exception:
        pass

    # Final fallback → return minimal blank summaries
    logger.error("FINAL FALLBACK: Returning blank summaries for each function.")
    return {fn: "Summary failed" for fn in function_names or []}


# ======================================================================
#                      GROQ LLM CLIENT
# ======================================================================

class GroqLLMClient:

    """
    Wrapper around Groq API to generate 2-line function summaries.
    Now with robust JSON cleaning and error healing.
    """

    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model

    def summarize_functions(self, file_path, full_text, function_names, retries=5):
        func_list_str = "\n".join(f"- {fn}" for fn in function_names)

        prompt = f"""
You are analyzing a C source file. For EACH function listed, give EXACTLY 2-3 lines summarizing its purpose.

Return ONLY VALID JSON:
{{
    "func_name": "two-line summary",
    ...
}}

FILE: {file_path}
FUNCTIONS:
{func_list_str}

FULL CODE:
{full_text}
"""

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }

        # ==========================
        #   RETRY LOOP WITH BACKOFF
        # ==========================
        for attempt in range(retries):
            response = requests.post(url, headers=headers, data=json.dumps(body))

            if response.status_code == 200:
                break  # SUCCESS → exit loop

            err = response.json().get("error", {})

            # ---- RATE LIMIT → WAIT + RETRY ----
            if err.get("code") == "rate_limit_exceeded":
                msg = err.get("message", "")
                wait_time = self._extract_wait_time(msg)

                logger.warning(f"Rate limited. Waiting {wait_time}s before retry (attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
                continue

            # ---- OTHER ERRORS → FAIL FAST ----
            logger.error(f"Groq API error: {response.text}")
            return {fn: "Summary failed" for fn in function_names}

        else:
            # ran out of attempts
            logger.error("Exceeded retry attempts for rate limits.")
            return {fn: "Summary failed" for fn in function_names}

        # -------- SUCCESSFUL RESPONSE --------
        raw = response.json()["choices"][0]["message"]["content"]
        cleaned = strip_markdown_fences(raw)
        cleaned = fix_json_common_errors(cleaned)

        parsed = safe_parse_json(cleaned, function_names)
        return parsed


    def _extract_wait_time(self, msg):
        """Extracts X seconds from Groq's rate-limit error message."""
        m = re.search(r"try again in ([0-9.]+)s", msg)
        if m:
            return float(m.group(1))
        return 15  # default fallback


# ======================================================================
#                         FUNCTION PARSER
# ======================================================================

class FunctionParser:

    FUNC_REGEX = re.compile(
        r"""
        (?P<signature>
            (?P<rtype>
                (?:unsigned|signed|static|extern|inline|const|struct|enum|union)  # keywords
                [A-Za-z0-9_\*\s]*?                                               # optional modifiers
            )
            \s+
            (?P<name>[A-Za-z_][A-Za-z0-9_]*)                                     # function name
            \s*\([^;{}]*\)                                                       # argument list
            \s*\{                                                                # opening brace
        )
        """,
        re.MULTILINE | re.VERBOSE
    )


    def __init__(self, groq_api_key):
        self.llm = GroqLLMClient(groq_api_key)

    # -------------------------------------------------------------

    def parse_repo(self, repo_path, save_as="function_map.json"):
        logger.info(f"Parsing repository: {repo_path}")

        function_map = {}

        for root, _, files in os.walk(repo_path):
            for filename in files:
                if filename.endswith(".c"):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, repo_path)

                    try:
                        file_result = self.parse_file(full_path, rel_path)
                        if file_result:
                            function_map[rel_path] = file_result
                    except Exception as e:
                        logger.error(f"Error parsing file {full_path}: {e}")

        # save results
        out_path = os.path.join(repo_path, save_as)
        with open(out_path, "w") as f:
            json.dump(function_map, f, indent=2)

        logger.info(f"Saved function map to: {out_path}")
        return function_map

    # -------------------------------------------------------------

    def parse_file(self, full_path, rel_path):
        logger.info(f"Extracting functions from: {rel_path}")

        text = open(full_path).read()

        matches = list(self.FUNC_REGEX.finditer(text))
        if not matches:
            return {}

        extracted = {}
        names = []

        # ---- extract function code blocks ----
        for match in matches:
            func_name = match.group("name")

            if func_name in {"if", "else", "while", "for", "switch", "do"}:
                continue

            names.append(func_name)

            sig_start = match.start()
            start_line = text[:sig_start].count("\n") + 1

            end_index = self._find_matching_brace(text, match.end() - 1)
            if end_index is None:
                continue

            end_line = text[:end_index].count("\n") + 1
            code_block = text[sig_start:end_index + 1]

            extracted[func_name] = {
                "start": start_line,
                "end": end_line,
                "code": code_block,
                "summary": "<pending>"
            }

        # ---- call LLM ONCE ----
        summaries = self.llm.summarize_functions(rel_path, text, names)

        # ---- assign summaries ----
        for fn in extracted:
            extracted[fn]["summary"] = summaries.get(fn, "Summary not generated")

        return extracted

    # -------------------------------------------------------------

    def _find_matching_brace(self, text, open_brace_index):
        depth = 0
        for i in range(open_brace_index, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None


# ======================================================================
#                               ENTRYPOINT
# ======================================================================

def run_parser(repo_path, groq_api_key):
    parser = FunctionParser(groq_api_key)
    return parser.parse_repo(repo_path)


if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python parser.py <repo_path>")
    #     exit(1)
    print("hi")

    # repo = sys.argv[1]
