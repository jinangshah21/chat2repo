import pandas as pd
import streamlit as st
from loguru import logger
from repo_service import RepoManager
from token_count import num_messages, num_tokens_from_string
import json
import pne


class StreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def process_token(self, token: str):
        if not isinstance(token, str):
            token = str(token)
        self.text += token
        self.container.markdown(self.text)


def refresh_repos():
    logger.info("Refreshing repositories")
    if "repoManager" not in st.session_state:
        st.session_state["repoManager"] = RepoManager()
    st.session_state["repoManager"].load_repos()
    st.success("Refreshed repositories")


def create_app():
    st.set_page_config(page_title="ChatWithRepo", page_icon="🤖")

    if "repoManager" not in st.session_state:
        st.session_state["repoManager"] = RepoManager()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "pr_review_context" not in st.session_state:
        st.session_state["pr_review_context"] = None

    repoManager: RepoManager = st.session_state["repoManager"]

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        # ----------------------- LLM CONFIG -------------------------
        st.title("Settings for LLM")

        model_options = [
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "deepseek/deepseek-chat",
            "zhipu/glm-4",
            "ollama/llama2",
            "groq/llama-3.3-70b-versatile",
            "claude-3-5-sonnet-20240620",
        ]
        model_options.insert(0, "Custom Model")

        selected_model = st.selectbox("Language Model Name", options=model_options)

        model_name = selected_model
        if selected_model == "Custom Model":
            model_name = st.text_input(
                "Enter Custom Model Name",
                placeholder="e.g. groq/llama3-70b-8192",
            )

        api_key = st.text_input("API Key", type="password")
        if api_key:
            st.session_state["api_key"] = api_key
            repoManager.api_key = api_key
        api_base = st.text_input("OpenAI Proxy URL (Optional)")
        temperature = st.slider(
            "Temperature", 0.0, 1.0, value=0.7, step=0.1
        )

        system_prompt = st.text_area(
            "System Prompt",
            value="You are an AI assistant. Use repository context to answer user questions.",
        )

         # ----------- GitHub Token Input -----------
        st.title("GitHub Authentication")

        github_token = st.text_input(
            "GitHub Access Token",
            type="password",
            placeholder="Enter your GitHub token"
        )

        # Save token in session state
        if github_token:
            st.session_state["github_token"] = github_token
            repoManager.github_token = github_token

        st.markdown(
            """
            <small>Token is used for authenticated GitHub API calls  
            (private repos, higher rate limits).</small>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")   # separator

        st.title("Settings for Repo")
        custom_repo_url = st.text_input("Custom Repository URL")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Custom Repository"):
                if repoManager.add_repo(custom_repo_url):
                    st.success(f"Added custom repository: {custom_repo_url}")
                else:
                    st.error(f"Repository add failed: {custom_repo_url}")

        with col2:
            if st.button("Refresh Repositories"):
                refresh_repos()

        repo_url = st.selectbox("Repository URL", options=repoManager.get_repo_urls())

        # ============ Repository Selection UI ===============
        if repoManager.check_if_repo_exists(repo_url):
            repo = repoManager.get_repo_service(repo_url)

            selected_folder = st.multiselect(
                "Select Folder", options=repo.get_folders_options()
            )
            selected_files = st.multiselect(
                "Select Files", options=repo.get_files_options(), default="README.md"
            )
            selected_languages = st.multiselect(
                "Filtered by Language", options=repo.get_languages_options()
            )

            limit = st.number_input("Limit", value=100000, step=10000)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Count Tokens"):
                    file_string = repo.get_filtered_files(
                        selected_folders=selected_folder,
                        selected_files=selected_files,
                        selected_languages=selected_languages,
                        limit=limit,
                    )
                    st.write(f"Total Tokens: {num_tokens_from_string(file_string)}")

            with col2:
                if st.button("Update Repo"):
                    if repo.update_repo():
                        st.success(f"Updated repository: {repo_url}")
                    else:
                        st.error(f"Failed updating repo: {repo_url}")
                    st.rerun()

            with col3:
                if st.button("Delete Repo"):
                    if repo.delete_repo():
                        st.success(f"Deleted repository: {repo_url}")
                    refresh_repos()
                    st.rerun()
        else:
            st.error("Please select a valid repository.")

        # # ---------------- PR REVIEW SECTION ----------------
        # st.subheader("PR Review")

        # pr_number = st.text_input("Enter PR Number to Review")

        # if st.button("Review PR"):
        #     if repoManager.check_if_repo_exists(repo_url):
        #         repo = repoManager.get_repo_service(repo_url)
        #         context = repo.review_pr(pr_number)

        #         if context is None:
        #             st.error(f"PR #{pr_number} not found or no context available.")
        #         else:
        #             st.session_state["pr_review_context"] = context
        #             st.success(f"Loaded review context for PR #{pr_number}")
            
        #     pr_review_prompt = (
        #         f"You are performing a Pull Request review for PR #{pr_number}.\n"
        #         f"Identify correctness issues, memory bugs, logic problems,\n"
        #         f"security concerns, missing edge cases, dependency implications,\n"
        #         f"and give improvement suggestions.\n\n"
        #         f"Here is the structured PR context:\n"
        #         f"{json.dumps(context, indent=2)}\n"
        #     )

        #     llm_messages = [
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": pr_review_prompt},
        #     ]

        #     # Display in chat
        #     with st.chat_message("assistant"):
        #         stream_handler = StreamHandler(st.empty())

        #         response = pne.chat(
        #             model=model_name,
        #             stream=True,
        #             messages=llm_messages,
        #             model_config={"api_base": api_base, "api_key": api_key},
        #         )

        #         for chunk in response:
        #             stream_handler.process_token(chunk)

        #     # append to chat history
        #     st.session_state["messages"].append(
        #         {"role": "assistant", "content": stream_handler.text}
        #     )


        if st.button("Clear Chat"):
            st.session_state["messages"] = []

    # -----------------------------------------------------------

    if repoManager.isEmpty():
        st.info("Add a repository URL to begin.")
        st.stop()

    if not repoManager.check_if_repo_exists(repo_url):
        st.info("Repository has not been added yet.")
        st.stop()

    repo = repoManager.get_repo_service(repo_url)
    st.title(f"Repo: {repo.repo_name}")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # ---------------- CHAT INPUT ----------------
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # load filtered files
        file_string = repo.get_filtered_files(
            selected_folders=selected_folder,
            selected_files=selected_files,
            selected_languages=selected_languages,
            limit=limit,
        )
        metadata_string = repo.get_metadata_summary(limit=limit)
        file_string += "\n\n=== Repository Metadata ===\n" + metadata_string

        # # ---------- PR Context Injection ----------
        # pr_context_text = ""
        # if st.session_state.get("pr_review_context"):
        #     pr_context_text = (
        #         "=== PR REVIEW CONTEXT ===\n"
        #         + json.dumps(st.session_state["pr_review_context"], indent=2)
        #         + "\n=== END CONTEXT ===\n"
        #     )

        # ---------- Build LLM messages ----------
        messages = [{"role": "system", "content": system_prompt}]

        # if pr_context_text:
        #     messages.append({
        #         "role": "system",
        #         "content": (
        #             "You are reviewing a Pull Request (mention potential new bugs or improvements). Here is critical repository context:\n"
        #             + pr_context_text
        #         ),
        #     })

        # Add repo content (optional)
        messages.append({"role": "user", "content": file_string})

        # Chat history
        messages += st.session_state.messages

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())

            response = pne.chat(
                model=model_name,
                stream=True,
                messages=messages,
                model_config={"api_base": api_base, "api_key": api_key},
            )

            for chunk in response:
                stream_handler.process_token(chunk)

            st.session_state.messages.append(
                {"role": "assistant", "content": stream_handler.text}
            )
    
    # -----------------------------------------
    # PR REVIEW SECTION (Placed below chat input)
    # -----------------------------------------
    st.markdown("---")
    st.subheader("Review a Pull Request")

    col_a, col_b = st.columns([3, 1])

    with col_a:
        pr_number_input = st.text_input("PR Number", key="pr_number_below")

    with col_b:
        run_pr_review = st.button("Review PR", key="run_pr_review_below")

    if run_pr_review:
        if repoManager.check_if_repo_exists(repo_url):
            repo = repoManager.get_repo_service(repo_url)

            context = repo.review_pr(pr_number_input)

            if context is None:
                st.error(f"PR #{pr_number_input} not found or no context available.")
            else:
                st.session_state["pr_review_context"] = context
                st.success(f"Loaded review context for PR #{pr_number_input}")

                # Build review prompt
                pr_review_prompt = f"""
                    Review PR #{pr_number_input}. 
                    Only analyze the code CHANGES in the diff — nothing else.

                    Report ONLY:
                    - correctness issues introduced by this diff
                    - logic or behavioral bugs caused by this diff
                    - memory safety issues from this diff
                    - edge cases this diff breaks
                    - Give fixed change if there is some incorrect code written in patch

                    Do NOT: discuss unrelated code, suggest improvements already in the file,
                    comment on style, or propose new features.

                    Here is the PR context:
                    {json.dumps(context, indent=2)}
                    """


                llm_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pr_review_prompt},
                ]

                # Output in main chat
                with st.chat_message("assistant"):
                    stream_handler = StreamHandler(st.empty())

                    response = pne.chat(
                        model=model_name,
                        stream=True,
                        messages=llm_messages,
                        model_config={"api_base": api_base, "api_key": api_key},
                    )

                    for chunk in response:
                        stream_handler.process_token(chunk)

                # log in history
                st.session_state["messages"].append(
                    {"role": "assistant", "content": stream_handler.text}
                )

                # reset PR context after using it
                st.session_state["pr_review_context"] = None

if __name__ == "__main__":
    create_app()
