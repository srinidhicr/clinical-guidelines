"""Interactive-only Gemini connectivity check; never writes credentials or prompts to logs."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Add GOOGLE_API_KEY to .env in the repository root.")

    from google import genai
    from google.genai import types
    from src.generation.schema import GuidelineAnswer, gemini_response_schema

    configured_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model=configured_model, contents="Reply with: connection ok")
        print(f"Basic Gemini connection succeeded with model '{configured_model}': {response.text}")
        structured_response = client.models.generate_content(
            model=configured_model,
            contents=(
                "Return a JSON abstention response. The answer must say the corpus does not support an answer, "
                "with no citations, grounded false, and confidence zero."
            ),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=gemini_response_schema(),
            ),
        )
        GuidelineAnswer.model_validate_json(str(structured_response.text))
        print("Structured JSON/Pydantic request succeeded.")
    except Exception as error:
        # This diagnostic is intentionally terminal-only. Do not copy API credentials
        # into its error output, source code, or issue reports.
        print(f"Gemini connection failed for model '{configured_model}'.")
        print(f"Provider error type: {type(error).__name__}")
        print(f"Provider message: {error}")
        raise SystemExit(1)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
