import os
import re
import json
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingManager:
    """
    Embedding manager maintaining FIVE FAISS indexes:

    1. context_index     → PR + Issue descriptions
    2. commit_index      → Commit message + diff chunks
    3. function_index    → Per-function summary + signature + deps
    4. dependency_index  → Caller/callee relationships of functions
    5. file_index        → File level summary + list of functions

    Uses cosine similarity (FAISS IndexFlatIP over normalized vectors).
    """

    def __init__(self, repo_path: str, model_name="all-MiniLM-L6-v2"):
        self.repo_path = repo_path
        self.embed_dir = os.path.join(repo_path, "embeddings")
        os.makedirs(self.embed_dir, exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Cosine similarity (inner product over normalized vectors)
        self.indexes = {
            "context": faiss.IndexFlatIP(self.dimension),
            "commit": faiss.IndexFlatIP(self.dimension),
            "function": faiss.IndexFlatIP(self.dimension),
            "dependency": faiss.IndexFlatIP(self.dimension),
            "file": faiss.IndexFlatIP(self.dimension),
        }

        # Metadata for each index
        self.metadatas = {k: [] for k in self.indexes}

        # commit_id → summary
        self.commit_summaries = {}

        self._load_all_indexes()

    # -----------------------------------------------------------
    # Helpers for file paths
    # -----------------------------------------------------------

    def _get_index_path(self, key):
        return os.path.join(self.embed_dir, f"{key}.index")

    def _get_meta_path(self, key):
        return os.path.join(self.embed_dir, f"{key}.json")

    def _get_summary_map_path(self):
        return os.path.join(self.embed_dir, "commit_summaries.json")

    # -----------------------------------------------------------
    # Loading existing indexes
    # -----------------------------------------------------------

    def _load_all_indexes(self):
        """Load FAISS indexes + metadata if already present."""
        for key in self.indexes:
            index_path = self._get_index_path(key)
            meta_path = self._get_meta_path(key)

            if os.path.exists(index_path):
                try:
                    self.indexes[key] = faiss.read_index(index_path)
                    logger.info(f"Loaded {key} index.")
                except Exception as e:
                    logger.error(f"Failed loading {key} index: {e}")

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        self.metadatas[key] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed loading {key} metadata: {e}")

        # Load commit summary map
        summary_path = self._get_summary_map_path()
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    self.commit_summaries = json.load(f)
            except Exception as e:
                logger.error(f"Failed loading commit summaries: {e}")

    # -----------------------------------------------------------
    # Saving indexes
    # -----------------------------------------------------------

    def save_index(self):
        """Persist FAISS indexes + metadata."""
        for key, index in self.indexes.items():
            try:
                faiss.write_index(index, self._get_index_path(key))
            except Exception as e:
                logger.error(f"Failed writing {key} index: {e}")
            try:
                with open(self._get_meta_path(key), "w") as f:
                    json.dump(self.metadatas[key], f, indent=2)
            except Exception as e:
                logger.error(f"Failed writing {key} metadata: {e}")

        try:
            with open(self._get_summary_map_path(), "w") as f:
                json.dump(self.commit_summaries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed writing commit summaries: {e}")

        logger.info("Saved all FAISS indexes + metadata.")

    # -----------------------------------------------------------
    # Vector helpers
    # -----------------------------------------------------------

    def _normalize(self, vec: np.ndarray):
        """Normalize embeddings for cosine (inner product) similarity.
        Accepts 1D or 2D arrays, returns 2D float32 array.
        """
        if vec is None:
            return None
        v = np.array(vec, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        v = v / norms
        return v.astype(np.float32)

    def _extract_identifiers(self, text: str):
        """
        Advanced C++ identifier extractor.
        Handles:
        - namespaces (std::vector)
        - class::method tokens
        - template types (Foo<Bar>)
        - member access (a.b, a->b)
        - function names
        - types, macros
        - avoids keywords, literals, comments

        Returns a list of unique identifiers.
        """
        if not text:
            return []

        # Remove comments first (C/C++)
        text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        # Remove strings and char literals
        text = re.sub(r'"(?:\\.|[^"\\])*"', "", text)
        text = re.sub(r"'(?:\\.|[^'\\])+'", "", text)

        # Extract C++ identifiers:
        # Includes namespace::symbol, template<T>, and single words.
        pattern = r"""
            [A-Za-z_][A-Za-z0-9_]*              # identifier
            (?:
                ::[A-Za-z_][A-Za-z0-9_]*        # ::namespace or ::method
            )*
            |
            [A-Za-z_][A-Za-z0-9_]*\s*<[^>]+>     # template types Foo<Bar>
        """

        tokens = re.findall(pattern, text, flags=re.VERBOSE)

        # Normalize:
        # Strip whitespace inside template params
        cleaned = []
        for t in tokens:
            t = re.sub(r"\s+", "", t)
            cleaned.append(t)

        # Remove C/C++ keywords to avoid noise
        cpp_keywords = {
            "if","else","for","while","switch","case","default","break","continue",
            "class","struct","enum","namespace","using","typename","template","return",
            "new","delete","auto","void","int","long","float","double","char","bool",
            "public","private","protected","virtual","override","const","static","mutable",
            "inline","constexpr","volatile","throw","catch","try","this","true","false",
        }

        result = [t for t in cleaned if t not in cpp_keywords]

        return result


    # -----------------------------------------------------------
    # Helper: Chunk long text
    # -----------------------------------------------------------

    def _chunk_text(self, text, chunk_size=200, overlap=50):
        """Split long text into overlapping chunks (by words)."""
        if not text:
            return []
        words = text.split()
        if not words:
            return []
        step = max(1, chunk_size - overlap)
        chunks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), step)
        ]
        return chunks

    # -----------------------------------------------------------
    # Add PR / Issue Embeddings
    # -----------------------------------------------------------

    def add_pr_issue_embedding(self, source_type, source_id, state, text):
        """Embed PR/Issue description into context index."""
        if not text or not text.strip():
            return

        chunks = self._chunk_text(text)
        if not chunks:
            return

        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        embeddings = self._normalize(embeddings)
        if embeddings is None or len(embeddings) == 0:
            return

        try:
            self.indexes["context"].add(embeddings)
        except Exception as e:
            logger.error(f"Failed adding to context index: {e}")
            return

        for i, chunk in enumerate(chunks):
            self.metadatas["context"].append({
                "source_type": source_type,
                "source_id": source_id,
                "state": state,
                "chunk_id": i,
                "text": chunk
            })

        logger.info(f"[context] Added {len(chunks)} chunks for {source_type} #{source_id}")

    def add_commit_embedding(self, commit_id, summary, diff_text, pr_number=None):
        """
        Add commit embedding.
        Stores per-diff-chunk embeddings (one metadata row per chunk).
        """

        if not summary and not diff_text:
            return

        # Save summary for retrieval
        self.commit_summaries[commit_id] = summary or ""

        # Split diff into file blocks
        file_blocks = re.split(r"^diff --git ", diff_text or "", flags=re.MULTILINE)
        file_blocks = file_blocks[1:]  # drop preamble

        for block in file_blocks:
            m = re.match(r"a/(.+?) b/(.+?)\n", block)
            if not m:
                continue

            changed_file = m.group(2)
            file_diff = block.split("\n", 1)[1] if "\n" in block else ""

            diff_chunks = self._chunk_text(file_diff, chunk_size=250, overlap=60)
            if not diff_chunks:
                diff_chunks = [summary or ""]

            embeddings = self.model.encode(diff_chunks, convert_to_numpy=True)
            embeddings = self._normalize(embeddings)
            if embeddings is None or len(embeddings) == 0:
                continue

            try:
                self.indexes["commit"].add(embeddings)
            except Exception as e:
                logger.error(f"Failed adding commit embeddings for {commit_id}: {e}")
                continue

            for i, chunk in enumerate(diff_chunks):
                md = {
                    "source_type": "commit_diff",
                    "source_id": commit_id,
                    "chunk_id": i,
                    "text": chunk,
                    "changed_file": changed_file,
                }
                if pr_number:
                    md["pr_number"] = pr_number

                self.metadatas["commit"].append(md)

            logger.info(f"[commit] {commit_id}: {len(diff_chunks)} chunks | file={changed_file}")

    # -----------------------------------------------------------
    # Add Function Embeddings
    # -----------------------------------------------------------

    def add_function_embedding(self, file_path, func_name, summary, signature, dependencies):
        """
        Embed function-level summary + signature + dependency info.
        """
        text_blob = f"""
FUNCTION: {func_name}
FILE: {file_path}

SUMMARY:
{summary}

SIGNATURE:
{signature}

DEPENDENCIES:
Calls: {', '.join(dependencies.get('calls', []))}
Called By: {', '.join(dependencies.get('called_by', []))}
"""
        embedding = self.model.encode([text_blob], convert_to_numpy=True)
        embedding = self._normalize(embedding)
        if embedding is None or len(embedding) == 0:
            return

        try:
            self.indexes["function"].add(embedding)
        except Exception as e:
            logger.error(f"Failed adding function embedding: {e}")
            return

        self.metadatas["function"].append({
            "file": file_path,
            "function": func_name,
            "summary": summary,
            "signature": signature,
            "dependencies": dependencies,
        })

    # -----------------------------------------------------------
    # Add Dependency Embeddings
    # -----------------------------------------------------------

    def add_dependency_embedding(self, func_name, dependencies):
        text_blob = f"""
FUNCTION DEPENDENCY CONTEXT:
Function: {func_name}
Calls: {', '.join(dependencies.get('calls', []))}
Called By: {', '.join(dependencies.get('called_by', []))}
"""
        embedding = self.model.encode([text_blob], convert_to_numpy=True)
        embedding = self._normalize(embedding)
        if embedding is None or len(embedding) == 0:
            return

        try:
            self.indexes["dependency"].add(embedding)
        except Exception as e:
            logger.error(f"Failed adding dependency embedding: {e}")
            return

        self.metadatas["dependency"].append({
            "function": func_name,
            "dependencies": dependencies,
        })

    # -----------------------------------------------------------
    # Add File Embeddings
    # -----------------------------------------------------------

    def add_file_embedding(self, file_path, description, functions):
        text_blob = f"""
FILE: {file_path}

DESCRIPTION:
{description}

FUNCTIONS:
{', '.join(functions)}
"""
        embedding = self.model.encode([text_blob], convert_to_numpy=True)
        embedding = self._normalize(embedding)
        if embedding is None or len(embedding) == 0:
            return

        try:
            self.indexes["file"].add(embedding)
        except Exception as e:
            logger.error(f"Failed adding file embedding: {e}")
            return

        self.metadatas["file"].append({
            "file": file_path,
            "description": description,
            "functions": functions,
        })

    # -----------------------------------------------------------
    # Build embeddings (keeps your original pipeline)
    # -----------------------------------------------------------

    def build_embeddings(self, commit_to_pr):
        """
        Build ALL embeddings for the repo (context, commit, function, dependency, file).
        """
        # 1) PR + ISSUE CONTEXT EMBEDDINGS
        for file_name, source_type in [
            ("issues.json", "issue"),
            ("pull_requests.json", "pr")
        ]:
            path = os.path.join(self.repo_path, file_name)
            if not os.path.exists(path):
                continue

            with open(path, "r") as f:
                data = json.load(f)

            for entry in data:
                text = ""
                title = entry.get("title", "")
                body = entry.get("body", "")
                labels = entry.get("labels", [])
                state = entry.get("state", "")

                if title:
                    text += title + "\n"
                if body:
                    text += body + "\n"
                if labels:
                    text += "Labels: " + ", ".join(labels) + "\n"
                if labels:
                    text += "State: " + state + "\n"

                if not text.strip():
                    continue

                self.add_pr_issue_embedding(
                    source_type,
                    str(entry.get("number", "")),
                    state,
                    text
                )

        # 2) COMMITS
        commits_path = os.path.join(self.repo_path, "commits.json")

        if os.path.exists(commits_path):
            with open(commits_path, "r") as f:
                commits_data = json.load(f)

            for commit in commits_data[:100]:
                commit_id = commit.get("sha")
                summary = commit.get("message", "") or ""
                diff_text = ""

                diff_dir = os.path.join(self.repo_path, "commit_diffs")
                diff_fp = os.path.join(diff_dir, f"{commit_id}.diff")

                if os.path.exists(diff_fp):
                    try:
                        with open(diff_fp, "r") as df:
                            diff_text = df.read()
                    except Exception as e:
                        logger.warning(f"Failed reading diff for {commit_id}: {e}")

                self.add_commit_embedding(
                    commit_id,
                    summary,
                    diff_text,
                    pr_number=commit_to_pr.get(commit_id)
                )

        # 3) FUNCTION EMBEDDINGS
        function_map_path = os.path.join(self.repo_path, "function_map.json")
        if os.path.exists(function_map_path):
            with open(function_map_path, "r") as f:
                function_map = json.load(f)
        else:
            function_map = {}

        dependencies_path = os.path.join(self.repo_path, "dependencies.json")
        if os.path.exists(dependencies_path):
            with open(dependencies_path, "r") as f:
                dep_graph = json.load(f)
        else:
            dep_graph = {}

        for file_path, funcs in function_map.items():
            for func_name, info in funcs.items():
                summary = info.get("summary", "")
                code = info.get("code", "")
                signature = code.split("{")[0].strip().replace("\n", " ")
                dependencies = dep_graph.get(file_path, {}).get(func_name, {
                    "calls": [],
                    "called_by": []
                })

                self.add_function_embedding(
                    file_path=file_path,
                    func_name=func_name,
                    summary=summary,
                    signature=signature,
                    dependencies=dependencies
                )

        # 4) DEPENDENCY EMBEDDINGS
        for file_path, funcs in dep_graph.items():
            for func_name, info in funcs.items():
                self.add_dependency_embedding(
                    func_name,
                    dependencies=info
                )

        # 5) FILE-LEVEL EMBEDDINGS
        for file_path, funcs in function_map.items():
            all_function_names = list(funcs.keys())
            file_description = "\n".join(
                f"{fn}: {funcs[fn].get('summary', '')}"
                for fn in all_function_names
            )

            self.add_file_embedding(
                file_path=file_path,
                description=file_description,
                functions=all_function_names
            )

        # SAVE
        self.save_index()
        logger.info(f"All embeddings built and saved for repo: {self.repo_path}")

    # -----------------------------------------------------------
    # Optimized Query (cosine + lexical + identifier + dependency + file proximity)
    # -----------------------------------------------------------

    def query_similar(self, index_type: str, query_text: str, top_k=5):
        """Optimized retrieval that returns metadata entries with composite scores."""
        if index_type not in self.indexes:
            raise ValueError(f"Invalid index_type: {index_type}")

        if not self.metadatas.get(index_type):
            logger.warning(f"No entries in {index_type} index.")
            return []

        if not query_text or not query_text.strip():
            return []

        # Encode + normalize query
        try:
            query_emb = self.model.encode([query_text], convert_to_numpy=True)
            query_emb = self._normalize(query_emb)
        except Exception as e:
            logger.error(f"Failed encoding query: {e}")
            return []

        # run FAISS search (inner product = cosine for normalized vectors)
        try:
            D, I = self.indexes[index_type].search(query_emb, 2*top_k)
        except Exception as e:
            logger.error(f"FAISS search failed on {index_type}: {e}")
            return []

        # feature helpers
        def lexical_overlap(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        def identifier_overlap(a: str, b: str) -> float:
            ia = set(self._extract_identifiers(a))
            ib = set(self._extract_identifiers(b))
            if not ia or not ib:
                return 0.0
            return len(ia & ib) / len(ia | ib)

        def dependency_score(md):
            deps = md.get("dependencies", {})
            if not deps:
                return 0.0
            calls = len(deps.get("calls", []))
            called_by = len(deps.get("called_by", []))
            return 0.3 * calls + 0.3 * called_by

        def file_proximity(md):
            fp = md.get("file")
            if not fp:
                return 0.0
            return max(0.0, 1.0 - 0.1 * fp.count("/"))

        def same_file_bonus(md):
            if md.get("changed_file") and md.get("file"):
                if os.path.normpath(md["changed_file"]) == os.path.normpath(md["file"]):
                    return 0.25
            return 0.0

        # index weights (keeps relative importance if needed)
        index_weights = {
            "context": 1.0,
            "commit": 1.0,
            "function": 1.0,
            "dependency": 1.0,
            "file": 1.0,
        }

        results = []
        for sim, idx in zip(D[0], I[0]):
            # idx may be -1 for missing; ignore
            if idx < 0 or idx >= len(self.metadatas[index_type]):
                continue

            # Convert the raw FAISS score (sim) to a consistent similarity score in [0..1]
            # Otherwise (L2 distance), convert distance -> similarity using 1 - 1/2(L2^2).
            cos_sim = 1.0 - (0.5 * (float(sim)**2))

            md = self.metadatas[index_type][idx].copy()
            meta_text = json.dumps(md)

            lex = lexical_overlap(query_text, meta_text)
            ident = identifier_overlap(query_text, meta_text)
            dep = dependency_score(md)
            fprox = file_proximity(md)
            sfb = same_file_bonus(md)

            # Composite score: primary = cos_sim (normalized similarity), others are additive boosts
            final_score = (
                0.60 * cos_sim
                + 0.15 * lex
                + 0.10 * ident
                + 0.05 * dep
                + 0.05 * fprox
                + 0.05 * sfb
            )

            md["raw_faiss_score"] = float(sim)
            md["cosine_similarity"] = float(cos_sim)
            md["final_score"] = float(final_score)
            md["lexical_overlap"] = lex
            md["identifier_overlap"] = ident
            md["dependency_score"] = dep
            md["file_proximity"] = fprox
            md["same_file_bonus"] = sfb

            if index_type == "commit":
                cid = md.get("source_id")
                md["commit_summary"] = self.commit_summaries.get(cid, "")

            results.append(md)

        # sort by composite final score (descending)
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    # -------------------------- Retrieval Helpers --------------------------

    def load_pull_requests(self):
        """Return list of PR dicts (from pull_requests.json)."""
        pr_path = os.path.join(self.repo_path, "pull_requests.json")
        if not os.path.exists(pr_path):
            return []
        try:
            with open(pr_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pull_requests.json: {e}")
            return []

    def get_pr_by_number(self, pr_number):
        """Find PR entry by number (int or str)."""
        prs = self.load_pull_requests()
        for p in prs:
            if str(p.get("number")) == str(pr_number):
                return p
        return None

    def retrieve_pr_context(self, pr_number, params=None):
        """
        Build an aggregated context object for a PR using existing JSON files and indexes.
        Uses the optimized query_similar internally.
        """

        params = params or {}
        top_k = params.get("top_k", {})
        top_context = top_k.get("context", 5)
        top_commit = top_k.get("commit", 10)
        top_function = top_k.get("function", 5)
        # top_dependency = top_k.get("dependency", 5)
        # top_file = top_k.get("file", 3)

        pr = self.get_pr_by_number(pr_number)
        if not pr:
            logger.warning(f"PR #{pr_number} not found in pull_requests.json")
            return None

        pr_query_text = (str(pr.get("title", "")) + "\n" + str(pr.get("body", ""))).strip()

        # 1) context matches
        context_matches = self.query_similar("context", pr_query_text, top_k=top_context)

        # 2) function matches
        function_matches = self.query_similar("function", pr_query_text, top_k=top_function)

        # 3) dependency matches
        # dependency_matches = self.query_similar("dependency", pr_query_text, top_k=top_dependency)

        # 4) file matches
        # file_matches = self.query_similar("file", pr_query_text, top_k=top_file)

        # 5) Detect linked issues inside PR body/title
        linked_issues = []
        issue_ids = set()
        patterns = [
            r"(?:fixes|closes|resolves)\s*#(\d+)",
            r"#(\d+)"
        ]
        pr_text_lower = pr_query_text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, pr_text_lower, flags=re.IGNORECASE)
            for mid in matches:
                issue_ids.add(str(mid))

        for ctx in self.metadatas.get("context", []):
            if ctx.get("source_type") == "issue":
                if ctx.get("source_id") in issue_ids:
                    linked_issues.append(ctx)

        # 6) Per Changed File: collect context
        files_data = {}
        pr_files = pr.get("files", [])
        unified_files = []

        for f in pr_files:
            if isinstance(f, dict) and f.get("filename"):
                unified_files.append({
                    "filename": f.get("filename"),
                    "patch": f.get("patch", "") or ""
                })

        # fallback: PR diff file
        if not unified_files:
            pr_diff_dir = os.path.join(self.repo_path, "pr_diffs")
            diff_fp = os.path.join(pr_diff_dir, f"pr_{pr.get('number')}.diff")
            if os.path.exists(diff_fp):
                try:
                    with open(diff_fp, "r") as df:
                        unified_files.append({
                            "filename": None,
                            "patch": df.read()
                        })
                except Exception:
                    pass

        for f in unified_files:
            fname = f.get("filename") or "<unknown>"
            patch = f.get("patch", "") or ""
            file_entry = {
                "patch": patch,
                "commit_matches": [],
                "file_summary": None,
                "functions": [],
            }

            # commit-level search
            if patch.strip():
                commit_hits = self.query_similar("commit", patch, top_k=top_commit)
            else:
                commit_hits = self.query_similar("commit", f"file: {fname}", top_k=top_commit)

            file_entry["commit_matches"] = commit_hits

            # file metadata
            file_meta = [
                m for m in self.metadatas.get("file", [])
                if m.get("file") and os.path.normpath(m.get("file")) == os.path.normpath(fname)
            ]
            if file_meta:
                file_entry["file_summary"] = file_meta[0]

            # function metadata
            funcs_here = []
            for fm in self.metadatas.get("function", []):
                if fm.get("file") and os.path.normpath(fm.get("file")) == os.path.normpath(fname):
                    funcs_here.append({
                        "function": fm.get("function"),
                        "summary": fm.get("summary"),
                        "signature": fm.get("signature")
                    })

            file_entry["functions"] = funcs_here
            files_data[fname] = file_entry

        # 7) Final PR Review Result
        return {
            "pr": pr,
            "context_matches": context_matches,
            "function_matches": function_matches,
            # "dependency_matches": dependency_matches,
            # "file_matches": file_matches,
            "linked_issues": linked_issues,
            "files": files_data
        }
