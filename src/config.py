from dotenv import load_dotenv
import os

# Load .env into environment
load_dotenv()

# Remove problematic SSL/requests cert env vars if present
for _k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_k, None)

def ensure_openai_key(prompt_if_missing: bool = True) -> None:
    """Ensure `OPENAI_API_KEY` is present in the environment.

    If `prompt_if_missing` is True, prompt the user interactively when the
    variable is not present.
    """
    if not os.environ.get("OPENAI_API_KEY") and prompt_if_missing:
        import getpass
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
