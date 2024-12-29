from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from ..config import get_settings

settings = get_settings()

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def get_response(self, query: str, relevant_docs: list[str]) -> str:
        context = "\n".join(relevant_docs)
        prompt = f"""Based on the following context, please answer the question.
        If you cannot find the answer in the context, say so.
        
        Context: {context}
        
        Question: {query}
        """
        response = self.llm.predict(prompt)
        return response
