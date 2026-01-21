import argparse  # Import argparse
import sys

import ollama

# --- Configuration removed, will be passed via args ---


def read_question_from_file(filepath: str) -> str | None:
    """Reads the question from the specified file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Question file not found at '{filepath}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading question file '{filepath}': {e}", file=sys.stderr)
        return None


def ask_ollama(model: str, question: str) -> str:
    """
    Sends a question to a model hosted by a local Ollama instance.

    Args:
        model: The name of the model registered with Ollama.
        question: The question to ask the model.

    Returns:
        The model's response content as a string.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': question},
            ],
        )
        # The response object contains details; we extract the message content
        return response['message']['content']
    except Exception as e:
        # Handle potential errors like connection issues or model not found
        print(f"Error interacting with Ollama model '{model}': {e}", file=sys.stderr)
        # You might want to raise the exception or handle it differently
        return "Error: Could not get a response from the Ollama model."


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Ask a question to an Ollama model.")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",  # Default model if not specified
        help="Name of the Ollama model to use (e.g., llama3.2:latest)",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        required=True,  # Make question file mandatory
        help="Path to the file containing the question to ask.",
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Read question from file
    question = read_question_from_file(args.question_file)
    if question is None:
        sys.exit(1)  # Exit if question file couldn't be read

    print(f"Using Ollama model: '{args.model}'")
    print(f"Reading question from: '{args.question_file}'")
    print(f'Question: "{question}"')
    print("-" * 20)

    answer = ask_ollama(args.model, question)

    print("\nResponse:")
    print(answer)

    print("-" * 20)
    print("Ensure Ollama service is running and model is available.")


if __name__ == "__main__":
    main()  # Call main function
