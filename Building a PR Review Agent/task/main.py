import os
import asyncio
from github import Github, Auth
from dotenv import load_dotenv
from llama_index.core.agent import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow, ToolCall, ToolCallResult, AgentOutput
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool


# =======================
# Load environment
# =======================
load_dotenv()
repo_url = os.getenv("GIT_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER")
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Autenticación con GitHub
git = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = git.get_repo(f"{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}")


# =======================
# Tools
# =======================
def get_pr_details(pr_number: int):
    """Fetch PR details given the PR number."""
    pull_request = repo.get_pull(pr_number)
    commit_SHAs = [c.sha for c in pull_request.get_commits()]
    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_SHAs": commit_SHAs,
    }

def get_commit_details(head_sha: str):
    """Fetch commit details given the commit SHA."""
    commit = repo.get_commit(head_sha)
    return [
        {
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        }
        for f in commit.files
    ]

def add_context_to_state(state: dict, context: str):
    """Save gathered context into state."""
    state["gathered_contexts"] = context
    return state

def add_comment_to_state(state: dict, draft_comment: str):
    """Save draft comment into state."""
    state["draft_comment"] = draft_comment
    return state

def add_final_review_to_state(state: dict, final_review: str):
    """Save final review into state."""
    state["final_review_comment"] = final_review
    return state

def post_review_to_github(pr_number: int, comment: str):
    """Post the final review comment to the PR on GitHub."""
    pr = repo.get_pull(pr_number)
    review = pr.create_review(body=comment, event="COMMENT")
    return {"status": "success", "review_id": review.id}


# Wrap tools
get_pr_tool = FunctionTool.from_defaults(fn=get_pr_details, name="get_pr_details")
get_commit_tool = FunctionTool.from_defaults(fn=get_commit_details, name="get_commit_details")
add_context_tool = FunctionTool.from_defaults(fn=add_context_to_state, name="add_context_to_state")
add_comment_tool = FunctionTool.from_defaults(fn=add_comment_to_state, name="add_comment_to_state")
add_final_review_tool = FunctionTool.from_defaults(fn=add_final_review_to_state, name="add_final_review_to_state")
post_review_tool = FunctionTool.from_defaults(fn=post_review_to_github, name="post_review_to_github")


# =======================
# LLM
# =======================
llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)


# =======================
# Agents
# =======================
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context about the PR: details, changed files, and repo files.",
    tools=[get_pr_tool, get_commit_tool, add_context_tool],
    system_prompt=(
        "You are the context gathering agent. When gathering context, you MUST gather:\n"
        "- PR details: author, title, body, diff_url, state, head_sha;\n"
        "- Changed files;\n"
        "- Any requested repo files.\n"
        "Once finished, you MUST hand control back to the CommentorAgent."
    ),
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a PR review comment.",
    tools=[add_comment_tool],
    system_prompt=(
        "You are the commentor agent that writes review comments for pull requests as a human reviewer would.\n"
        "Ensure to:\n"
        "- Request PR details, changed files, and repo files from ContextAgent.\n"
        "- Then write a ~200-300 word review in markdown with:\n"
        "  * What's good about the PR?\n"
        "  * Did the author follow contribution rules?\n"
        "  * Are there tests and migrations?\n"
        "  * Are endpoints documented?\n"
        "  * Which lines could be improved? Quote them and suggest fixes.\n"
        "- Address the author directly (e.g., 'Thanks for fixing this...').\n"
        "- You must hand off to the ReviewAndPostingAgent once you are done drafting a review."
    ),
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment, refines it if needed, and posts it to GitHub.",
    tools=[add_final_review_tool, post_review_tool],
    system_prompt=(
        "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.\n"
        "Once a review is generated, you need to run a final check and post it to GitHub.\n"
        "   - The review must:\n"
        "   - Be a ~200-300 word review in markdown format.\n"
        "   - Specify what is good about the PR.\n"
        "   - Did the author follow ALL contribution rules? What is missing?\n"
        "   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?\n"
        "   - Are there notes on whether new endpoints were documented?\n"
        "   - Are there suggestions on which lines could be improved upon? Are these lines quoted?\n"
        "If the review does not meet this criteria, you must ask the CommentorAgent to rewrite it.\n"
        "When you are satisfied, post the review to GitHub."
    ),
    can_handoff_to=["CommentorAgent"],
)


# =======================
# Workflow
# =======================
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review_comment": ""
    },
)


# =======================
# Runner
# =======================
async def main():
    pr_number = os.getenv("PR_NUMBER") or "1"
    query = f"Write a review for PR number {pr_number}"
    handler = workflow_agent.run(query)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"\n🔄 Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n📝 Final response:", event.response.content)
            if event.tool_calls:
                print("Selected tools:", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCall):
            print(f"\n⚙️ Calling tool: {event.tool_name}, with args: {event.tool_kwargs}")
        elif isinstance(event, ToolCallResult):
            print(f"\n🛠 Output from tool:", event.tool_output)


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
