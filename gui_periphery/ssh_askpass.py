"""OpenSSH prompt bridge. Answers go only to OpenSSH's private stdout pipe."""

import json
import os
import sys
from urllib.request import Request, urlopen


def main():
    request = Request(os.environ["GPE_AUTH_URL"],
                      data=json.dumps({"prompt": sys.argv[1]}).encode(),
                      headers={"Content-Type": "application/json",
                               "X-GPE-Auth": os.environ["GPE_AUTH_TOKEN"]})
    try:
        with urlopen(request, timeout=180) as response:
            answer = json.load(response)["answer"]
        sys.stdout.write(answer + "\n")
    except Exception:
        # No credential values or HTTP bodies are written to logs.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
