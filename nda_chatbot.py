# -*- coding: utf-8 -*-
# File: nda_chatbot.py
# -*- coding: utf-8 -*-
# File: nda_chatbot.py
import os
import warnings
warnings.filterwarnings('ignore')

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
from typing import Dict, Any, List

class NDADocumentChatbot:
    def __init__(self, openai_api_key: str, model_name: str = 'gpt-4o'):
        """Initialize the NDA chatbot"""
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.2
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = None
        self.documents = None
        self.pdf_path = None
        self.chat_history = []
        
        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # NDA analysis prompt
        self.nda_prompt = '''You are a legal assistant AI with expertise in M&A transactions. The user has uploaded a Non-Disclosure Agreement (NDA) related to a potential merger or acquisition. Your task is to extract and summarize the most relevant clauses and information for internal due diligence.

Please analyze the document thoroughly and return a **bullet-point summary** covering the following:

1. **Parties Involved:** Identify the disclosing and receiving parties (entities or individuals).
2. **Purpose of Disclosure:** Explain why confidential information is being exchanged (e.g. due diligence for a potential M&A deal).
3. **Definition of Confidential Information:** How is it defined in the document? Are there exclusions?
4. **Obligations of the Receiving Party:** Summarize what the receiving party is allowed or not allowed to do with the information.
5. **Permitted Disclosures:** Who (if anyone) can the receiving party share the information with (e.g. affiliates, advisors)?
6. **Term & Duration:** When does the NDA take effect and how long do confidentiality obligations last?
7. **Return or Destruction of Info:** What happens to confidential materials at the end of the relationship or deal?
8. **Remedies / Legal Recourse:** Are there any penalties or legal consequences for breaching the NDA?
9. **Jurisdiction & Governing Law:** Which jurisdiction and laws govern the agreement?
10. **Special Clauses (if any):** Flag any non-standard clauses such as:
    - Non-solicitation
    - Standstill agreements
    - Exclusivity

### Important Notes:
- If any section is not clearly defined or missing, explicitly state "Not specified" or "Standard provisions apply"
- Flag any ambiguous language that may need legal review with "[REQUIRES LEGAL REVIEW]"
- Highlight any unusual or potentially problematic clauses with "[UNUSUAL CLAUSE]"

### Output Format:
- Use **clear bullet points**
- Keep it concise, accurate, and well-structured
- Include a short note at the top summarizing the overall intent and duration of the NDA

Document: {text}'''

        # Intent classifier
        self.intent_classifier = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for an NDA analysis chatbot.
            Classify the user's message into one of these categories:

            1. SUMMARIZE - User wants a summary of the NDA or specific sections
            2. QUESTION - User is asking specific questions about the NDA content
            3. GENERAL - General conversation or greetings

            Examples for NDA context:
            - "Summarize this NDA" → SUMMARIZE
            - "Give me an analysis of this agreement" → SUMMARIZE
            - "What are the confidentiality obligations?" → QUESTION
            - "Who are the parties involved?" → QUESTION
            - "What is the governing law?" → QUESTION
            - "Hello" → GENERAL

            Respond with only one word: SUMMARIZE, QUESTION, or GENERAL"""),
            ("user", "{user_message}")
        ])
        
        self.intent_chain = self.intent_classifier | self.llm | StrOutputParser()

    def is_initialized(self):
        """Check if the chatbot is initialized"""
        return hasattr(self, 'llm') and hasattr(self, 'chat_history')

    def load_nda_document(self, pdf_path: str) -> bool:
        """Load NDA PDF document"""
        try:
            print(f"Loading NDA document: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            self.documents = loader.load()
            self.pdf_path = pdf_path
            print(f"✅ NDA loaded successfully! ({len(self.documents)} pages)")
            return True
        except Exception as e:
            print(f"❌ Error loading NDA: {str(e)}")
            return False

    def setup_summarization_chain(self, chain_type: str = "stuff"):
        """Setup the summarization chain using load_summarize_chain with fixed map_reduce"""
        
        if chain_type == "stuff":
            # For stuff chain, use single prompt
            prompt = PromptTemplate(
                template=self.nda_prompt,
                input_variables=["text"]
            )
            
            return load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt
            )
        
        else:  # map_reduce - Fixed version
            # Map prompt - simpler format for individual chunks
            map_prompt = PromptTemplate(
                template="""Analyze this section of an NDA document. Focus on identifying:
- Key legal provisions and clauses
- Party obligations and rights
- Important dates, terms, and conditions
- Any unusual or notable clauses

Document section:
{text}

Key findings from this section:""",
                input_variables=["text"]
            )
            
            # Combine prompt - comprehensive analysis using the original detailed prompt
            combine_prompt = PromptTemplate(
                template="""You are a legal assistant AI with expertise in M&A transactions. Based on the following analyses of different sections of an NDA document, provide a comprehensive summary covering:

1. **Parties Involved:** Identify the disclosing and receiving parties
2. **Purpose of Disclosure:** Why confidential information is being exchanged  
3. **Definition of Confidential Information:** How it's defined and any exclusions
4. **Obligations of the Receiving Party:** What they can/cannot do
5. **Permitted Disclosures:** Who can receive the information
6. **Term & Duration:** When it takes effect and how long obligations last
7. **Return or Destruction of Info:** What happens to materials afterward
8. **Remedies / Legal Recourse:** Penalties for breaching the NDA
9. **Jurisdiction & Governing Law:** Which laws govern the agreement
10. **Special Clauses:** Any non-standard provisions like non-solicitation, standstill, exclusivity

Use clear bullet points and flag any ambiguous language with "[REQUIRES LEGAL REVIEW]" or unusual clauses with "[UNUSUAL CLAUSE]".

Section analyses:
{text}

Comprehensive NDA Analysis:""",
                input_variables=["text"]
            )
            
            return load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt
            )

    def determine_chain_strategy(self) -> str:
        """Determine whether to use 'stuff' or 'map_reduce' based on document size"""
        if not self.documents:
            return "stuff"

        # Calculate total tokens (rough estimate)
        total_text = "\n".join([doc.page_content for doc in self.documents])
        word_count = len(total_text.split())
        estimated_tokens = word_count * 1.3
        
        print(f"📊 Document analysis: {word_count:,} words, ~{estimated_tokens:.0f} tokens")
        
        # Use map_reduce for documents over ~8000 tokens to be more conservative
        if estimated_tokens > 8000:
            print(f"📋 Large document detected - using map_reduce strategy")
            return "map_reduce"
        else:
            print(f"📋 Standard document size - using stuff strategy")
            return "stuff"

    def analyze_nda_comprehensive(self) -> str:
        """Comprehensive NDA analysis using load_summarize_chain"""
        if not self.documents:
            return "❌ No NDA document loaded"
        
        try:
            # Determine best chain to use based on document size
            chain_type = self.determine_chain_strategy()
            
            print(f"🔍 Setting up {chain_type} strategy...")
            summarize_chain = self.setup_summarization_chain(chain_type=chain_type)
            
            print(f"🔍 Analyzing NDA using {chain_type} strategy...")
            summary = summarize_chain.run(self.documents)
            print("✅ NDA analysis completed!")
            return summary
            
        except Exception as e:
            print(f"❌ Error in {chain_type} analysis: {str(e)}")
            # Fallback to manual chunking for any error
            return self.analyze_large_nda_manual()

    def analyze_large_nda_manual(self) -> str:
        """Manual analysis for large NDAs as fallback"""
        try:
            print("🔍 Using manual chunking approach as fallback...")
            
            # Combine all documents into single text
            full_text = "\n\n".join([doc.page_content for doc in self.documents])
            
            # Split into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,  # Smaller chunks
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            # Create temporary documents for chunking
            temp_doc = Document(page_content=full_text)
            chunks = text_splitter.split_documents([temp_doc])
            
            print(f"📄 Split document into {len(chunks)} chunks")
            
            # Analyze each chunk
            chunk_analyses = []
            for i, chunk in enumerate(chunks):
                print(f"🔍 Analyzing chunk {i+1}/{len(chunks)}...")
                
                # Simple prompt for chunk analysis
                chunk_prompt = f"""Analyze this section of an NDA document. Identify key legal provisions, parties, obligations, terms, and any important clauses.

Document section:
{chunk.page_content}

Analysis:"""
                
                response = self.llm.invoke(chunk_prompt)
                chunk_analyses.append(f"Chunk {i+1} Analysis: {response.content}")
            
            # Combine all chunk analyses
            combined_analysis = "\n\n".join(chunk_analyses)
            
            # Generate final comprehensive summary
            final_prompt = f"""You are a legal assistant AI. Based on the following analyses of different sections of an NDA, provide a comprehensive summary following this structure:

1. **Parties Involved:** Identify the disclosing and receiving parties
2. **Purpose of Disclosure:** Why confidential information is being exchanged  
3. **Definition of Confidential Information:** How it's defined and any exclusions
4. **Obligations of the Receiving Party:** What they can/cannot do
5. **Permitted Disclosures:** Who can receive the information
6. **Term & Duration:** When it takes effect and how long obligations last
7. **Return or Destruction of Info:** What happens to materials afterward
8. **Remedies / Legal Recourse:** Penalties for breaching the NDA
9. **Jurisdiction & Governing Law:** Which laws govern the agreement
10. **Special Clauses:** Any non-standard provisions like non-solicitation, standstill, exclusivity

Section analyses:
{combined_analysis}

Comprehensive NDA Analysis:"""
            
            final_response = self.llm.invoke(final_prompt)
            print("✅ Manual analysis completed!")
            return final_response.content
            
        except Exception as e:
            return f"❌ Error in manual analysis: {str(e)}"

    def setup_rag_chain(self):
        """Setup RAG chain for NDA Q&A"""
        if not self.documents:
            return None
            
        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(self.documents)

        # Create vectorstore
        self.vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory="./nda_chroma_db"
        )
        
        # NDA-specific Q&A prompt
        qa_prompt_template = """Use the following pieces of the NDA document to answer the question at the end.
Focus on providing accurate information about confidentiality obligations, parties involved, terms, and legal provisions.

If you don't know the answer based on the NDA content, just say that the information is not specified in this NDA.

NDA Context:
{context}

Question: {question}

Answer:"""
        
        qa_prompt = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        
        return qa_chain

    def ask_nda_question(self, question: str) -> Dict[str, Any]:
        """Answer questions about the NDA using RAG"""
        try:
            qa_chain = self.setup_rag_chain()
            if qa_chain is None:
                return {"answer": "❌ No NDA document loaded for Q&A"}
            
            result = qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            return {"answer": f"❌ Error answering question: {str(e)}"}

    def classify_intent(self, user_message: str) -> str:
        """Classify user intent"""
        try:
            intent = self.intent_chain.invoke({"user_message": user_message}).strip().upper()
            return intent if intent in ["SUMMARIZE", "QUESTION", "GENERAL"] else "QUESTION"
        except:
            return "QUESTION"  # Default to question if classification fails

    def get_conversation_context(self, max_exchanges: int = 3) -> str:
        """Get recent conversation context for better responses"""
        history = self.get_conversation_history()
        if not history:
            return ""
        
        # Get last few exchanges for context
        recent_history = history[-max_exchanges:] if len(history) > max_exchanges else history
        
        context_parts = []
        for i, exchange in enumerate(recent_history):
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant'][:200]}{'...' if len(exchange['assistant']) > 200 else ''}")
        
        return "\n".join(context_parts)

    def chat_with_nda(self, user_message: str) -> Dict[str, Any]:
        """Main chat interface for NDA analysis with enhanced memory"""
        if not self.documents:
            return {
                "response": "❌ Please load an NDA document first using load_nda_document(pdf_path)",
                "intent": "ERROR",
                "sources": []
            }
        
        print(f"💬 User: {user_message}")
        
        # Classify user intent
        intent = self.classify_intent(user_message)
        print(f"🎯 Intent: {intent}")
        
        # Get conversation context for better responses
        conversation_context = self.get_conversation_context()
        
        if intent == "SUMMARIZE":
            # Summarize the NDA
            print("📋 Generating NDA analysis...")
            analysis = self.analyze_nda_comprehensive()
            response = f"📄 **NDA ANALYSIS:**\n\n{analysis}"
            sources = []
            
        elif intent == "QUESTION":
            # Answer questions about the NDA with conversation context
            print("❓ Searching NDA for answer...")
            
            # Enhanced question with context for better answers
            if conversation_context:
                contextual_question = f"""Previous conversation context:
{conversation_context}

Current question: {user_message}

Please answer the current question while considering the conversation history for better context."""
                qa_result = self.ask_nda_question(contextual_question)
            else:
                qa_result = self.ask_nda_question(user_message)
                
            response = qa_result["answer"]
            sources = qa_result.get("source_documents", [])
            
        else:  # GENERAL
            # General conversation with memory context
            print("💬 Handling general conversation...")
            general_prompt = f"""You are an NDA analysis assistant. The user has loaded an NDA document and is having a conversation with you.

{f"Previous conversation context: {conversation_context}" if conversation_context else ""}

You should be helpful, professional, and friendly. You can:
- Greet users and explain your capabilities
- Answer questions about NDAs in general
- Provide guidance on what kinds of analysis you can perform
- Have natural conversations while staying focused on your role as an NDA assistant
- Reference previous parts of the conversation when relevant

Current conversation context: The user has an NDA document loaded and ready for analysis.

User message: {user_message}

Respond naturally and helpfully:"""
            
            response = self.llm.invoke(general_prompt).content
            sources = []
        
        # Store conversation in memory
        self.memory.chat_memory.add_user_message(user_message)
        self.memory.chat_memory.add_ai_message(response)
        
        # Response preview logging
        print(f"🤖 Assistant: {response[:200]}{'...' if len(response) > 200 else ''}")
        if sources:
            print(f"📚 Found {len(sources)} relevant document sections")
        
        return {
            "response": response,
            "intent": intent,
            "sources": sources
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        history = []
        messages = self.memory.chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                history.append({
                    "user": human_msg.content,
                    "assistant": ai_msg.content
                })
        
        return history

    def clear_memory(self):
        """Clear memory function"""
        self.memory.clear()
        print("🗑️ Chat history cleared")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the current memory state"""
        history = self.get_conversation_history()
        memory_messages = len(self.memory.chat_memory.messages)
        
        return {
            "total_exchanges": len(history),
            "memory_messages": memory_messages,
            "memory_limit": self.memory.k * 2,  # k exchanges = k*2 messages
            "memory_usage_percent": (memory_messages / (self.memory.k * 2)) * 100 if self.memory.k > 0 else 0
        }

    def get_nda_stats(self) -> Dict[str, Any]:
        """Get NDA document statistics"""
        if not self.documents:
            return {"error": "No NDA loaded"}
        
        total_text = "\n".join([doc.page_content for doc in self.documents])
        word_count = len(total_text.split())
        char_count = len(total_text)
        page_count = len(self.documents)

        # Estimate reading time (average 200 WPM)
        reading_time_minutes = word_count / 200
        reading_time = f"{int(reading_time_minutes)}m" if reading_time_minutes < 60 else f"{int(reading_time_minutes // 60)}h {int(reading_time_minutes % 60)}m"

        return {
            "file_path": self.pdf_path,
            "pages": page_count,
            "words": word_count,
            "characters": char_count,
            "reading_time": reading_time,
            "estimated_tokens": word_count * 1.3  # Rough estimate
        }
