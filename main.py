from dotenv import load_dotenv

load_dotenv()
from langchain_groq import ChatGroq

from graph.graph import app

if __name__ == "__main__":
    print("main function")
    print(app.invoke(input={"question":"what is distance bw chennai to anantapur?"}))