## Agent-as-Coder Challenge

This project implements an "Agent-as-Coder" using LangGraph and the Gemini API to autonomously write and debug Python parsers for bank statement PDFs.

### Agent Architecture (T5 Diagram/Explanation)

The agent operates on a **four-node self-correction loop**. It begins at the `generate_code` node, which uses the LLM and PDF context to create the initial `parse()` function. This code is then passed to the `save_and_test` node, which writes the file and runs the validation test (comparing output DataFrame via `pd.DataFrame.equals`). A conditional edge routes the flow: if the test **PASSES**, the agent terminates successfully. If the test **FAILS**, and the agent is below its $\le 3$ attempt limit, it routes to the `fix_code` node. The `fix_code` node feeds the detailed test failure output (error messages, DataFrame head mismatches) back to the LLM to generate a corrected version of the code, restarting the test cycle.

### 5-Step Run Instructions (T5 Instructions)

1.  **Clone and Setup:** Fork this repository and install dependencies (`pip install -r requirements.txt`, etc.).
2.  **API Key:** Set your `GEMINI_API_KEY` in the `.env` file.
3.  **Data:** Ensure your sample PDF (`icici_sample.pdf`) and the ground truth CSV (`icici_sample.csv`) are placed in the `data/icici` folder.
4.  **Run Agent (T2 CLI):** Execute the agent script via the command line:
    ```bash
    python agent.py --target icici
    ```
5.  **Verify:** The agent will output "SUCCESS" upon completion. The final, passing parser can be found at `custom_parsers/icici_parser.py`.