import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage

llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key='******************')

history=[]

if os.path.exists('memory.json'):
    with open('memory.json', 'r') as file:
        data = json.load(file)
    for msg in data:
        if msg['role'] == "human":
            history.append(HumanMessage(content=msg["content"]))

        else:
            history.append(AIMessage(content=msg["content"]))

print(' \n Type here to Exit or Quit \n ')

while True:
    query=input(" You : ")
    if query.lower().strip()=="exit":
        break
    history.append(HumanMessage(content=query))
    response=llm.invoke(history)

    history.append(AIMessage(content=response.content))

    print(" AI : ",response.content)

print("============================================================================")
print(history)
print("--------------------------------------------------------------------------------")

data=[]

for msg in history:
    data.append(
        {
            "role":msg.type,
            "content":msg.content
        }
    )

with open('memory.json','w') as file:
    json.dump(data,file,indent=4)
    print(' \n Memory saved to memory.json')