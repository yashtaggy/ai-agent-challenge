import os
import argparse
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Any
import operator
import pdfplumber

# --- Agent Libraries ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Import Tools ---
from tools import run_test, save_code, cleanup

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# --- Define Constants ---
# FIX: Define DATA_DIR based on your project structure
DATA_DIR = "data" 

# --- 1. Define the Agent State ---
class AgentState(TypedDict):
    """Represents the state of the agent's workflow."""
    bank_name: str
    target_pdf_path: str
    target_csv_path: str
    code: str
    feedback: str
    attempt_count: int
    max_attempts: int = 3 # T1 Constraint: self-fix (<=3 attempts)
    messages: Annotated[List[Any], operator.add]

# --- 2. Initialize LLM and Tools ---
# Use the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

# Tools the agent can call (we will call tools externally for control)
# The agent will use the internal flow logic (nodes) to simulate tool use.

# --- 3. Define Nodes/Functions ---

def extract_pdf_context(state: AgentState) -> str:
    """Extracts text from the first page of the PDF to provide context to the LLM."""
    try:
        # Use an f-string or os.path.join for clarity, but the state path should be correct
        with pdfplumber.open(state['target_pdf_path']) as pdf:
            first_page = pdf.pages[0]
            # Get the page text, which is often enough for simple parsers
            text_content = first_page.extract_text()
            return text_content if text_content else "Could not extract text from PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

def generate_initial_code(state: AgentState) -> AgentState:
    """The first call to the LLM to write the initial parser code."""
    
    # Extract context for the LLM
    pdf_context = extract_pdf_context(state)
    
    system_prompt = (
        "You are an expert Python programmer specializing in parsing complex PDF documents, "
        "especially bank statements. Your task is to write a COMPLETE Python module "
        "that adheres to a strict contract."
    )
    
    user_prompt = (
        f"Generate the full Python code for a parser file named '{state['bank_name']}_parser.py'.\n\n"
        f"**The parser MUST contain a single function:**\n"
        f"```python\n"
        f"def parse(pdf_path: str) -> pd.DataFrame:\n"
        f"    # Your implementation here\n"
        f"```\n\n"
        f"**Contract Details (T3):**\n"
        f"1. The output must be a pandas DataFrame (`pd.DataFrame`).\n"
        f"2. The DataFrame must match the schema and content of the expected data in the CSV: '{state['target_csv_path']}'.\n"
        f"3. You MUST include `import pandas as pd` and any other necessary imports (like `pdfplumber`).\n\n"
        f"**PDF Context (First Page):**\n"
        f"Use the following text extracted from the PDF at '{state['target_pdf_path']}' to understand the data layout:\n"
        f"--- PDF TEXT START ---\n{pdf_context}\n--- PDF TEXT END ---\n\n"
        f"**Instructions:**\n"
        f"1. Do not use any external tools.\n"
        f"2. Return ONLY the complete, correct Python code block (using `parse(pdf_path)`) within triple backticks (```python...```)."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    print("Agent: Generating initial parser code...")
    response = llm.invoke(messages)
    
    # Simple extraction of the code block
    code = ""
    try:
        # Look for the triple backticks
        start = response.content.find("```python") + len("```python")
        end = response.content.find("```", start)
        code = response.content[start:end].strip()
    except:
        # Fallback if extraction fails
        code = response.content.strip() 

    return {
        "code": code,
        "messages": [HumanMessage(content=user_prompt), response]
    }

def save_and_test_code(state: AgentState) -> AgentState:
    """Saves the code and runs the test."""
    bank_name = state['bank_name']
    code = state['code']
    attempt_count = state.get('attempt_count', 0) + 1
    
    # T2/T3: Parser Contract (file name)
    target_file = f"{bank_name}_parser.py"

    # 1. Save Code
    save_result = save_code(code, target_file)
    print(f"Agent: Save result: {save_result['status']}")

    if save_result['status'] == 'failure':
          return {
              "feedback": f"Code Save FAILED: {save_result['message']}",
              "attempt_count": attempt_count,
              "messages": []
           }

    # 2. Run Test (T4: Test)
    test_result = run_test(bank_name, state['max_attempts'], attempt_count)
    
    # Update state with results
    new_state = {
        "feedback": test_result['feedback'],
        "attempt_count": attempt_count,
        "messages": [] # We don't need to save the test output in the history for every loop
    }
    
    if test_result['test_result'] == "PASS":
        print("Test Result: PASS")
        new_state['test_result'] = "PASS"
    else:
        print("Test Result: FAIL")
        new_state['test_result'] = "FAIL"
        
    return new_state

def fix_code(state: AgentState) -> AgentState:
    """The LLM refines the code based on test feedback."""
    
    system_prompt = (
        "You are an expert debugging agent. Your previous code failed the test. "
        "Analyze the provided test feedback (error messages, DataFrame head mismatches) "
        "and generate the COMPLETE, CORRECTED Python code block for the parser file. "
        "Do not explain your reasoning; return ONLY the code."
    )
    
    # Combine the previous code and the new feedback
    previous_code = state['code']
    feedback = state['feedback']
    
    # Construct the message for the LLM
    user_prompt = (
        f"The previous code was:\n\n"
        f"```python\n{previous_code}\n```\n\n"
        f"**TEST FEEDBACK (CRITICAL):**\n"
        f"The tests failed with the following feedback. Use this information to fix the code:\n\n"
        f"--- FEEDBACK START ---\n{feedback}\n--- FEEDBACK END ---\n\n"
        f"**Instructions:**\n"
        f"1. Generate the COMPLETE, CORRECTED Python code block, adhering to the `parse(pdf_path)` contract.\n"
        f"2. Return ONLY the complete, corrected Python code block within triple backticks (```python...```)."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    print(f"Agent: Attempt {state['attempt_count']} failed. Generating fix...")
    response = llm.invoke(messages)
    
    # Simple extraction of the code block
    code = ""
    try:
        start = response.content.find("```python") + len("```python")
        end = response.content.find("```", start)
        code = response.content[start:end].strip()
    except:
        code = response.content.strip() 
        
    return {
        "code": code,
        "messages": [HumanMessage(content=user_prompt), response]
    }

# --- 4. Define Edge Logic (Router) ---
def should_continue(state: AgentState) -> str:
    """Determines whether to loop for a fix or terminate."""
    if state.get('test_result') == "PASS":
        return "end" # Success!
    
    if state['attempt_count'] >= state['max_attempts']:
        return "end_failure" # T1 Constraint: max attempts reached
    
    return "fix" # Loop for self-correction

# --- 5. Build the LangGraph ---
def build_agent_graph():
    """Builds and compiles the Agent-as-Coder graph."""
    workflow = StateGraph(AgentState)

    # 5.1 Add Nodes
    workflow.add_node("generate_code", generate_initial_code)
    workflow.add_node("save_and_test", save_and_test_code)
    workflow.add_node("fix_code", fix_code)

    # 5.2 Set Entry Point
    workflow.set_entry_point("generate_code")

    # 5.3 Define Edges
    # Initial run
    workflow.add_edge("generate_code", "save_and_test")
    
    # Self-correction loop (T1: Loop: plan -> generate code -> run tests -> self-fix)
    workflow.add_conditional_edges(
        "save_and_test",
        should_continue,
        {
            "end": END,
            "end_failure": END,
            "fix": "fix_code",
        }
    )
    
    # After a fix, go back to test
    workflow.add_edge("fix_code", "save_and_test")

    return workflow.compile()

# --- 6. Main CLI Function (T2: CLI) ---
def main():
    """T2: CLI interface to run the agent."""
    parser = argparse.ArgumentParser(description="Agent-as-Coder Challenge Runner.")
    # T2: python agent.py --target icici
    parser.add_argument('--target', required=True, help='Bank name (e.g., icici)')
    args = parser.parse_args()
    
    bank_name = args.target.lower()
    
    # Define paths based on target
    # FIX: Corrected path to match file system: 'icici sample.pdf'
    target_pdf_path = os.path.join(DATA_DIR, bank_name, f"{bank_name} sample.pdf")
    # Assuming CSV name is correct (e.g., 'icici_sample.csv')
    target_csv_path = os.path.join(DATA_DIR, bank_name, "result.csv") 
    
    print(f"\n--- Starting Agent for {bank_name.upper()} ---")
    print(f"Goal: Write parser at custom_parsers/{bank_name}_parser.py")
    print(f"PDF: {target_pdf_path}")
    print(f"CSV: {target_csv_path}")

    if not os.path.exists(target_pdf_path) or not os.path.exists(target_csv_path):
        print("\nFATAL ERROR: Required data files not found.")
        print(f"Please ensure {target_pdf_path} and {target_csv_path} exist.")
        return

    # Clear previous runs for a fresh start
    cleanup(bank_name)

    # Initial state
    initial_state = {
        "bank_name": bank_name,
        "target_pdf_path": target_pdf_path,
        "target_csv_path": target_csv_path,
        "code": "",
        "feedback": "Initial run.",
        "attempt_count": 0,
        "max_attempts": 3, 
        "messages": []
    }

    # Build and run the graph
    app = build_agent_graph()
    
    final_state = app.invoke(initial_state)

    # --- Final Output ---
    final_result = final_state.get('test_result', 'FAIL')
    
    if final_result == "PASS":
        print("\n\n#############################################")
        print(f"## SUCCESS: Parser for {bank_name.upper()} is complete! ##")
        print("#############################################")
    else:
        print("\n\n#####################################################")
        print(f"## FAILURE: Agent failed to converge after 3 attempts. ##")
        print(f"## Last Feedback: {final_state['feedback'].splitlines()[0]} ##")
        print("#####################################################")
        print(f"Final code saved to custom_parsers/{bank_name}_parser.py (may be incomplete/buggy).")


if __name__ == "__main__":
    main()