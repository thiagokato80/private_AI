from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_google_genai import ChatGoogleGenerativeAI

import getpass
import os

#if "GOOGLE_API_KEY" not in os.environ:
    #os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                             )

result = llm.invoke("Tell me about the history of AI")
print(result.content)