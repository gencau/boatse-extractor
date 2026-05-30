# inference_worker.py
import sys, json, warnings, os
from contextlib import redirect_stderr

def main():
    # Hide noisy UserWarnings from libs (optional)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Silence stderr from libs that write banners to stderr (optional)
    devnull = open(os.devnull, "w")
    with redirect_stderr(devnull):
        from openai import OpenAI

        payload = json.load(sys.stdin)
        messages = payload["messages"]
        model_kwargs = payload["model_kwargs"]

        api_key = model_kwargs.pop("api_key", os.environ.get("OPENROUTER_API_KEY", ""))
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        resp = client.chat.completions.create(messages=messages, **model_kwargs)
        content = resp.choices[0].message.content

    json.dump({"ok": True, "content": content}, sys.stdout)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
