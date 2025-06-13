# -*- coding: utf-8 -*-
# File: nda_chatbot.py
# -*- coding: utf-8 -*-
# File: nda_chatbot.py
import os
import warnings
from typing import Dict, Any, List

warnings.filterwarnings("ignore")
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document


class NDADocumentChatbot:
    """End-to-end assistant for loading, analysing and chatting about NDA PDFs."""

    # ------------------------------------------------------------------ #
    # 1. INITIALISATION
    # ------------------------------------------------------------------ #
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o") -> None:
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.2,
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = None
        self.documents = None
        self.pdf_path = None

        # Conversation memory (last 10 exchanges)
        self.memory = ConversationBufferWindowMemory(
            k=10, return_messages=True, memory_key="chat_history"
        )

        # NDA analysis master prompt
        self.nda_prompt = """You are a legal assistant AI with expertise in M&A transactions. The user
has uploaded a Non-Disclosure Agreement (NDA) related to a potential merger or
acquisition. Extract and summarise the most relevant clauses and information for
internal due diligence.

Return a • **bullet-point summary** covering:

1. **Parties Involved** – disclosing and receiving parties
2. **Purpose of Disclosure**
3. **Definition of Confidential Information** (plus exclusions)
4. **Obligations of the Receiving Party**
5. **Permitted Disclosures** (affiliates, advisers…)
6. **Term & Duration**
7. **Return / Destruction of Information**
8. **Remedies / Legal Recourse**
9. **Jurisdiction & Governing Law**
10. **Special Clauses** (non-solicitation, standstill, exclusivity, …)

Important:
* If something is missing, say **“Not specified”**.
* Flag ambiguous language with **[REQUIRES LEGAL REVIEW]**.
* Flag unusual clauses with **[UNUSUAL CLAUSE]**.

### Output
* Use clear bullet points.
* Keep it concise, accurate, and well-structured.
* Start with a one-sentence note summarising the NDA’s overall intent and term.

Document:
{text}"""  # the summarisation chains inject {text}

        # Intent classifier pipeline
        self.intent_classifier = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an intent classifier for an NDA analysis chatbot.
Classify the user's message into **exactly one** of:

1. SUMMARIZE – user wants a summary of the NDA
2. QUESTION  – user asks something about the NDA’s content
3. GENERAL   – greetings, small-talk, meta-questions

Respond with one word: SUMMARIZE, QUESTION or GENERAL.""",
                ),
                ("user", "{user_message}"),
            ]
        )
        self.intent_chain = self.intent_classifier | self.llm | StrOutputParser()

    # ------------------------------------------------------------------ #
    # 2. DOCUMENT HANDLING
    # ------------------------------------------------------------------ #
    def load_nda_document(self, pdf_path: str) -> bool:
        """Load a PDF NDA into memory."""
        try:
            print(f"Loading NDA document: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            self.documents = loader.load()
            self.pdf_path = pdf_path
            print(f"✅ NDA loaded successfully ({len(self.documents)} pages)")
            return True
        except Exception as e:
            print(f"❌ Error loading NDA: {e}")
            return False

    # ------------------------------------------------------------------ #
    # 3. SUMMARISATION CHAINS
    # ------------------------------------------------------------------ #
    def setup_summarization_chain(self, chain_type: str = "stuff"):
        """Create a LangChain summarisation chain compatible with v0.1.15+."""
        base_prompt = ChatPromptTemplate.from_template(self.nda_prompt)

        if chain_type == "stuff":
            # One-shot ("stuff") summarisation
            return load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=base_prompt,
            )

        # ---------- map-reduce prompts ----------
        map_prompt = ChatPromptTemplate.from_template(
            """Analyse this section of the NDA. Focus on key legal provisions,
party obligations, dates, and unusual clauses.

{text}

Section analysis:"""
        )

        reduce_prompt = ChatPromptTemplate.from_template(
            """You are a legal assistant AI. Combine the section analyses below
into a single, comprehensive NDA summary covering: parties, confidentiality
obligations, term, governing law, and special clauses.

{text}

Comprehensive NDA summary:"""
        )

        return load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt,  # <- crucial name
        )

    def determine_chain_strategy(self) -> str:
        """Pick 'stuff' or 'map_reduce' based on rough token count."""
        if not self.documents:
            return "stuff"

        total_text = "\n".join(doc.page_content for doc in self.documents)
        word_count = len(total_text.split())
        est_tokens = int(word_count * 1.3)

        print(f"📊 Document: {word_count:,} words, ~{est_tokens:,} tokens")

        if est_tokens > 4_000:
            print("📋 Large document detected – using map_reduce strategy")
            return "map_reduce"
        print("📋 Standard document size – using stuff strategy")
        return "stuff"

    def analyze_nda_comprehensive(self) -> str:
        """High-level NDA analysis. Falls back to manual chunking if needed."""
        if not self.documents:
            return "❌ No NDA document loaded"

        try:
            chain_type = self.determine_chain_strategy()

            # Very large files → manual chunk routine
            if chain_type == "map_reduce":
                return self.analyze_large_nda_manual()

            # Smaller files → normal summarisation
            print(f"🔍 Setting up {chain_type} strategy")
            summarize_chain = self.setup_summarization_chain(chain_type=chain_type)

            print(f"🔍 Analysing NDA with {chain_type} strategy")
            summary = summarize_chain.run(self.documents)
            print("✅ NDA analysis completed")
            return summary

        except Exception as e:
            print(f"❌ Error in standard analysis: {e}")
            return self.analyze_large_nda_manual()

    # ------------------------------------------------------------------ #
    # 4. MANUAL CHUNK FALLBACK (for huge NDAs)
    # ------------------------------------------------------------------ #
    def analyze_large_nda_manual(self) -> str:
        """Chunk-by-chunk analysis to avoid context-window limits."""
        try:
            print("🔍 Using manual chunking approach for large NDA")

            full_text = "\n\n".join(doc.page_content for doc in self.documents)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3_000,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " "],
            )

            temp_doc = Document(page_content=full_text)
            chunks = text_splitter.split_documents([temp_doc])
            print(f"📄 Split document into {len(chunks)} chunks")

            analyses = []
            for idx, chunk in enumerate(chunks, 1):
                print(f"🔍 Analysing chunk {idx}/{len(chunks)}")
                chunk_prompt = f"""Analyse this NDA section. Identify key
provisions, parties, obligations, terms and important clauses.

Document section:
{chunk.page_content}

Analysis:"""
                analyses.append(f"Chunk {idx}: {self.llm.invoke(chunk_prompt).content}")

            combined = "\n\n".join(analyses)
            final_prompt = f"""You are a legal assistant AI. Based on the
following analyses, produce a comprehensive NDA summary following the bullet
structure in the main prompt.

Section analyses:
{combined}

Comprehensive NDA summary:"""

            final_response = self.llm.invoke(final_prompt)
            print("✅ Manual analysis completed")
            return final_response.content

        except Exception as e:
            return f"❌ Error in manual analysis: {e}"

    # ------------------------------------------------------------------ #
    # 5. RETRIEVAL-AUGMENTED QA (RAG)
    # ------------------------------------------------------------------ #
    def setup_rag_chain(self):
        """Prepare a retriever-QA chain for question answering."""
        if not self.documents:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1_000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(self.documents)

        self.vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory="./nda_chroma_db",
        )

        qa_prompt = PromptTemplate(
            template="""Use these NDA excerpts to answer the question. Focus on
confidentiality obligations, parties, terms, and legal provisions.

If the answer is not in the NDA, say so.

NDA context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"],
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )

    def ask_nda_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about the NDA via RAG."""
        qa_chain = self.setup_rag_chain()
        if qa_chain is None:
            return {"answer": "❌ No NDA loaded for Q&A"}

        try:
            result = qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
            }
        except Exception as e:
            return {"answer": f"❌ Error answering question: {e}"}

    # ------------------------------------------------------------------ #
    # 6. CONVERSATION & MEMORY
    # ------------------------------------------------------------------ #
    def classify_intent(self, user_message: str) -> str:
        """SUMMARIZE / QUESTION / GENERAL."""
        try:
            intent = (
                self.intent_chain.invoke({"user_message": user_message}).strip().upper()
            )
            return intent if intent in {"SUMMARIZE", "QUESTION", "GENERAL"} else "QUESTION"
        except Exception:
            return "QUESTION"

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return formatted conversation history."""
        history = []
        msgs = self.memory.chat_memory.messages
        for i in range(0, len(msgs), 2):
            if i + 1 < len(msgs):
                history.append(
                    {"user": msgs[i].content, "assistant": msgs[i + 1].content}
                )
        return history

    def get_conversation_context(self, max_exchanges: int = 3) -> str:
        """Last few exchanges as plain text for context injection."""
        history = self.get_conversation_history()[-max_exchanges:]
        lines = []
        for h in history:
            lines.append(f"User: {h['user']}")
            lines.append(
                f"Assistant: {h['assistant'][:200]}"
                + ("…" if len(h["assistant"]) > 200 else "")
            )
        return "\n".join(lines)

    def chat_with_nda(self, user_message: str) -> Dict[str, Any]:
        """Main public interface: chat, remembering context."""
        if not self.documents:
            return {
                "response": "❌ Please load an NDA first (load_nda_document)",
                "intent": "ERROR",
                "sources": [],
            }

        print(f"💬 User: {user_message}")

        intent = self.classify_intent(user_message)
        print(f"🎯 Intent: {intent}")

        context = self.get_conversation_context()

        if intent == "SUMMARIZE":
            print("📋 Generating NDA analysis")
            response_text = self.analyze_nda_comprehensive()
            sources = []

        elif intent == "QUESTION":
            print("❓ Searching NDA for answer")
            if context:
                question_with_ctx = (
                    f"Previous context:\n{context}\n\n"
                    f"Current question: {user_message}\n"
                    "Answer considering the conversation history."
                )
                result = self.ask_nda_question(question_with_ctx)
            else:
                result = self.ask_nda_question(user_message)
            response_text = result["answer"]
            sources = result.get("source_documents", [])

        else:  # GENERAL
            print("💬 Handling general conversation")
            general_prompt = f"""You are an NDA analysis assistant having a
conversation with a user who has loaded an NDA.

{f"Conversation context:\n{context}" if context else ""}

User message: {user_message}

Respond naturally, professionally, and helpfully:"""
            response_text = self.llm.invoke(general_prompt).content
            sources = []

        # Add to memory
        self.memory.chat_memory.add_user_message(user_message)
        self.memory.chat_memory.add_ai_message(response_text)

        print(f"🤖 Assistant: {response_text[:200]}{'…' if len(response_text) > 200 else ''}")
        if sources:
            print(f"📚 Returned {len(sources)} source sections")

        return {"response": response_text, "intent": intent, "sources": sources}

    def clear_memory(self) -> None:
        self.memory.clear()
        print("🗑️ Chat history cleared")

    # ------------------------------------------------------------------ #
    # 7. STATS & UTILITIES
    # ------------------------------------------------------------------ #
    def get_memory_stats(self) -> Dict[str, Any]:
        total_msgs = len(self.memory.chat_memory.messages)
        return {
            "total_exchanges": total_msgs // 2,
            "memory_messages": total_msgs,
            "memory_limit": self.memory.k * 2,
            "memory_usage_percent": (total_msgs / (self.memory.k * 2)) * 100
            if self.memory.k
            else 0,
        }

    def get_nda_stats(self) -> Dict[str, Any]:
        if not self.documents:
            return {"error": "No NDA loaded"}

        full_text = "\n".join(doc.page_content for doc in self.documents)
        words = len(full_text.split())
        pages = len(self.documents)
        reading_time = words / 200  # 200 WPM

        return {
            "file_path": self.pdf_path,
            "pages": pages,
            "words": words,
            "reading_time": f"{reading_time:.1f} min",
            "estimated_tokens": int(words * 1.3),
        }
