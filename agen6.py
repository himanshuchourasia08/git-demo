from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key='*****')

history=InMemoryChatMessageHistory()

history.add_message(HumanMessage("""Build an AI Detective that Collects clues during a conversation and 
                                    Solves a mystery only at the end and You should only short answer in any query like "Noted","Got it","Information Recorded",
                                    and the last one is query is asking to solve the case in clues."""))


print("Type 'exit', to quit. \n")
while True:
    query=input(" Your: ")
    if query.lower()=="exit":
        break

    history.add_message(HumanMessage(query))

    response=llm.invoke(history.messages)

    history.add_message(AIMessage(response.content))

    print("AI",response.content)

for i, msg in enumerate(history.messages, start=1):
    print(f"{i}.{msg.type.upper()}: {msg.content}")
