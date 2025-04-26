import bs4
import llm_policy
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader

llm = llm_policy.LLMAgent()
embeddings = OllamaEmbeddings(model="nomic-embed-text")

### Construct retriever ###

markdown_path = "https://raw.githubusercontent.com/openatx/uiautomator2/master/README_CN.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
assert len(data) == 1
assert isinstance(data[0], Document)
readme_content = data[0].page_content
api_start_index = readme_content.find("API Documents")
if api_start_index != -1:
    api_content = readme_content[api_start_index:]
else:
    api_content = ""
    print("没有找到API Documents")

api_document = Document(page_content=api_content)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents([api_document])

vector_store = Chroma.from_documents(documents=all_splits, embedding=embeddings)
# docs = [Document(page_content="ui testing")]
# vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vector_store.as_retriever()


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
system_prompt = (
    "You are an expert in App GUI testing to guide the testing tool to enhance the coverage of "
    "functional scenarios in testing the App based on your extensive App testing experience."
    "Your task is to select an action based on the current GUI Infomation to perform next and "
    "help the app prevent entering or escape the UI tarpit."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Example usage:
# var = conversational_rag_chain.invoke(
#     {"input": "what is the API document?"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]
