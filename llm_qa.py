from typing import List, Dict, Any
from langchain_community.chat_models import ChatOllama
# We wrap this import in try/except so it doesn't crash locally if you didn't install groq yet
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import config

class QAEngine:
    def __init__(self):
        """
        Initializes the RAG engine.
        Switches between Ollama (Local) and Groq (Cloud) based on config.
        """
        print(f"🤖 Initializing QA Engine in mode: {config.DEPLOYMENT_MODE}...")
        
        # 1. Select Model Provider
        if config.DEPLOYMENT_MODE == "local":
            # Local Mode: Use Ollama
            self.llm = ChatOllama(
                model=config.LLM_MODEL_NAME,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.3,
                keep_alive="1h"
            )
        else:
            # Cloud Mode: Use Groq
            if not config.GROQ_API_KEY:
                raise ValueError("CRITICAL: DEPLOYMENT_MODE is 'cloud' but GROQ_API_KEY is missing.")
            
            if ChatGroq is None:
                raise ImportError("langchain-groq is not installed. Please add it to requirements.txt")

            self.llm = ChatGroq(
                model=config.LLM_MODEL_NAME,
                api_key=config.GROQ_API_KEY,
                temperature=0.3
            )

        # 2. Define the "Kid-Friendly" Prompt Template
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a smart, friendly teacher explaining Qatar's economy to a student. 
            Your goal is to make complex financial facts sound simple and interesting.

            Guidelines:
            1. 👶 **Simplify**: Don't use big words like "fiscal consolidation" without explaining them. Use plain English.
            2. 🗣️ **Tone**: Be conversational and encouraging.
            3. 🎯 **Directness**: Answer the question first, then explain it.
            4. 🚫 **No Jargon**: If you see a table with numbers, just tell the story of the numbers.
            5. 📖 **Sources**: You MUST put the [Page Number] at the end of every sentence you take from the text.

            Context from the report:
            {context}

            Student's Question: 
            {question}

            Your Simple Explanation:"""
        )

        # 3. Create the Chain
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Prepares documents for the LLM context window.
        """
        formatted_text = ""
        for doc in docs:
            # Clean content slightly for the LLM
            content = doc.page_content.replace("### TEXT CONTENT", "").replace("### TABLES", "Table Data:")
            source = doc.metadata.get("source", "Report")
            page = doc.metadata.get("page", "?")
            
            formatted_text += f"--- INFO FROM PAGE {page} ({source}) ---\n{content}\n\n"
        return formatted_text

    def answer_question(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        if not retrieved_docs:
            return {
                "answer": "I looked through the report, but I couldn't find the answer to that specific question.",
                "citations": []
            }

        # 1. Prepare Context
        context_text = self._format_docs(retrieved_docs)

        # 2. Generate Answer
        try:
            response_text = self.chain.invoke({
                "context": context_text,
                "question": query
            })
        except Exception as e:
            print(f"Error during LLM inference: {e}")
            return {
                "answer": "Oops! I had a little trouble thinking about that. Could you ask me again?",
                "citations": []
            }

        # 3. Format Citations (Raw data for the app to clean up)
        citations = []
        for i, doc in enumerate(retrieved_docs):
            citations.append({
                "rank": i + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "snippet": doc.page_content
            })

        return {
            "answer": response_text,
            "citations": citations,
            "context_used": len(retrieved_docs)
        }

if __name__ == "__main__":
    try:
        qa = QAEngine()
        print("✅ QA Engine initialized successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")