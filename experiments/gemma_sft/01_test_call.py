"""One API call to check the key and the model. Needs GEMINI_API_KEY in the environment."""
import os
from google import genai

MODEL = "gemma-4-26b-a4b-it"     # gemma-4-31b-it returned HTTP 500 in the first test; retry it later

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
prompt = ("Answer the instruction below in 2 to 4 plain sentences. No markdown, no lists.\n\n"
          "Instruction: Why is the sky blue?")
r = client.models.generate_content(model=MODEL, contents=prompt)
print(r.text)
