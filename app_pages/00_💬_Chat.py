import streamlit as st
import os
from typing import List, Literal, Dict, Any
from dotenv import load_dotenv
import pprint
import ast
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.utils.llm_service import RetrievedContext, LLMService
from app.utils.multi_query import MultiQueryGenerator
from app.utils.reranker import GeminiReranker
from app.utils.embeddings import EmbeddingService, VectorStore
from app.utils.hybrid_search import HybridSearcher
from app.utils.feedback import FeedbackService
from app.utils.database import verify_team_access
from app.components.agent import Agent
from app.utils.genai_client import configured_client, configured_llm_service

# Load environment variables
load_dotenv()

RetrievalMethod = Literal["vector", "hybrid", "multi_query", "reranked", "agentic"]

# Initialize authentication state if not already set by main app
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'team_id' not in st.session_state:
    st.session_state.team_id = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

def show_login():
    """Display login form and handle authentication."""
    st.title("🔐 Login")
    
    team_id = st.text_input("Team ID")
    access_code = st.text_input("Access Code", type="password")
    
    if st.button("Login"):
        if verify_team_access(team_id, access_code):
            st.session_state.authenticated = True
            st.session_state.team_id = team_id
            st.session_state.is_admin = (team_id == 'admin')
            st.success(f"Logged in as {'Admin' if st.session_state.is_admin else f'Team {team_id}'}")
            st.rerun()
        else:
            st.error("Invalid team ID or access code")
    
    # Add logout button if already authenticated
    if st.session_state.authenticated:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.team_id = None
            st.session_state.is_admin = False
            st.session_state.messages = []  # Clear chat history
            st.rerun()

def get_recent_chat_history(n=2):
    """Return the last n user/assistant turns as a string for context."""
    history = []
    for msg in st.session_state.get("messages", [])[-2*n:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {msg['content']}")
    return "\n".join(history)

def process_query(
    full_query: str,
    prompt: str,
    k: int = 4,
    retrieval_method: RetrievalMethod = "vector"
) -> List[RetrievedContext]:
    """
    Process a query through the RAG pipeline to retrieve relevant context.
    
    Args:
        full_query: The user's query plus recent chat history (for vector search)
        prompt: The latest user input (for FTS)
        k: Number of results to retrieve
        retrieval_method: Method to use for retrieval ("vector", "hybrid", "multi_query", or "reranked")
        
    Returns:
        List of retrieved context chunks
    """
    processing_steps = []
    team_id = st.session_state.team_id
    is_admin = st.session_state.is_admin

    # Enable agentic mode only if retrieval_method is 'agentic'
    use_agentic = retrieval_method == "agentic"

    # Track query processing steps
    if use_agentic:
        processing_steps.append("Query Assessment (Agentic)")
        processing_steps.append(f"Original Query: {full_query}")
        import asyncio
        agent = st.session_state.agent
        async def retrieve_chunks(query):
            embedding = st.session_state.embedding_service.generate_embedding(query)
            results = st.session_state.vector_store.similarity_search(
                query_embedding=embedding,
                team_id=team_id,
                k=k,
                admin_override=is_admin
            )
            return [r["text"] for r in results]
        async def generate_answer(query, chunks):
            return "\n---\n".join(chunks)
        result = asyncio.run(agent.process_query(
            prompt,
            retrieve_chunks,
            generate_answer,
            schema_config={}
        ))
        pprint.pprint(result.get("process_log"))
        st.session_state.last_processing_steps = [str(step) for step in result.get("process_log", [])]
        return [RetrievedContext(doc_id="agentic_result", content=result.get("response", "No answer found."), metadata={}, score=1.0)]
    else:
        processing_steps.append("ℹ️ Agentic query assessment is disabled")
    
    # Generate embedding for the full query (for vector search)
    query_embedding = st.session_state.embedding_service.generate_embedding(full_query)
    
    # Execute retrieval based on selected method
    if retrieval_method == "hybrid":
        hybrid_weight = st.session_state.get("hybrid_weight", 0.5)
        search_results = st.session_state.hybrid_search.search(
            query=prompt,  # Use only the latest user input for FTS
            query_embedding=query_embedding,  # Embedding of full_query for vector search
            team_id=team_id,
            k=k,
            weight=hybrid_weight,
            admin_override=is_admin
        )
        contexts = [
            RetrievedContext(
                doc_id=result.doc_id,
                content=result.text,
                metadata={
                    **result.metadata,
                    'vector_score': f"{result.vector_score:.3f}",
                    'text_score': f"{result.text_score:.3f}",
                    'combined_score': f"{result.combined_score:.3f}",
                    'weight': f"{hybrid_weight:.2f}"
                },
                score=result.combined_score
            ) for result in search_results
        ]
    elif retrieval_method == "multi_query":
        # Use only the latest user prompt for multi-query generation
        queries = st.session_state.multi_query_generator.generate_queries(prompt)
        print("[Multi-Query] Generated queries:", queries)  # Print to CLI for debugging
        all_results = []
        per_query_k = max(3, k // len(queries))
        for q in queries:
            print(f"[Multi-Query] Running similarity search for: '{q}'")
            q_embedding = st.session_state.embedding_service.generate_embedding(q)
            results = st.session_state.vector_store.similarity_search(
                query_embedding=q_embedding,
                team_id=team_id,
                k=per_query_k,
                admin_override=is_admin
            )
            print(f"[Multi-Query] Results for '{q}':")
            for r in results:
                print(f"  chunk_id={r['doc_id']} score={r['similarity']:.6f}")
            all_results.extend([
                RetrievedContext(
                    doc_id=r["doc_id"],
                    content=r["text"],
                    metadata={
                        **r["metadata"], 
                        "generated_query": q,
                        "vector_score": r["similarity"]
                    },
                    score=r["similarity"]
                ) for r in results
            ])
        seen_docs = set()
        contexts = []
        for ctx in sorted(all_results, key=lambda x: x.score, reverse=True):
            if ctx.doc_id not in seen_docs and len(contexts) < k:
                contexts.append(ctx)
                seen_docs.add(ctx.doc_id)
    elif retrieval_method == "reranked":
        initial_k = min(k * 2, 12)
        initial_results = st.session_state.vector_store.similarity_search(
            query_embedding=query_embedding,
            team_id=team_id,
            k=initial_k,
            admin_override=is_admin
        )
        chunks_to_rerank = [
            (r["text"], r["similarity"])
            for r in initial_results
        ]
        reranked_results = st.session_state.reranker.rerank_chunks(
            query=full_query,
            chunks=chunks_to_rerank
        )
        contexts = []
        for i, (text, score) in enumerate(reranked_results[:k]):
            original_result = next(r for r in initial_results if r["text"] == text)
            original_index = next(i for i, r in enumerate(initial_results) if r["text"] == text)
            position_change = original_index - i
            contexts.append(
                RetrievedContext(
                    doc_id=original_result["doc_id"],
                    content=text,
                    metadata={
                        **original_result["metadata"],
                        "initial_score": f"{original_result['similarity']:.3f}",
                        "llm_score": f"{score:.1f}",
                        "reranked": "true",
                        "rank": str(i + 1),
                        "initial_rank": str(original_index + 1),
                        "position_change": str(position_change)
                    },
                    score=score
                )
            )
    else:  # Default to vector search
        results = st.session_state.vector_store.similarity_search(
            query_embedding=query_embedding,
            team_id=team_id,
            k=k,
            admin_override=is_admin
        )
        contexts = [
            RetrievedContext(
                doc_id=r["doc_id"],
                content=r["text"],
                metadata=r["metadata"],
                score=r["similarity"]
            ) for r in results
        ]
    if use_agentic and contexts:
        print("\n[Agentic] Starting agentic retrieval assessment...")
        print(f"[Agentic] Query being assessed: {full_query}")
        print(f"[Agentic] Contexts being passed to LLM (first 2 shown):")
        for i, ctx in enumerate(contexts[:2]):
            print(f"  Context {i+1}: {ctx.content[:200]}...")  # Print first 200 chars for brevity

        processing_steps.append("\nRetrieval Assessment (Agentic)")
        max_retrieval_attempts = 2
        attempt = 1
        import asyncio
        while attempt <= max_retrieval_attempts:
            context_texts = [ctx.content for ctx in contexts]
            print(f"[Agentic] Attempt {attempt}: Passing to agent.assess_retrieval()")
            assessment = asyncio.run(
                st.session_state.agent.assess_retrieval(full_query, context_texts)
            )
            print(f"[Agentic] Assessment result: is_sufficient={assessment.is_sufficient}, needs_clarification={assessment.needs_clarification}")
            print(f"[Agentic] Reasoning: {assessment.reasoning}")
            if assessment.needs_clarification:
                print(f"[Agentic] Clarification suggested: {assessment.suggested_clarification}")
            processing_steps.append(f"\nAttempt {attempt}:")
            processing_steps.append(f"Relevance Check: {'✓ Sufficient' if assessment.is_sufficient else '⚠️ Insufficient'}")
            processing_steps.append(f"Analysis: {assessment.reasoning}")
            if assessment.needs_clarification:
                processing_steps.append("\n❓ Clarification Needed")
                processing_steps.append(f"Question: {assessment.suggested_clarification}")
                st.session_state.needs_clarification = True
                st.session_state.clarification_question = assessment.suggested_clarification
                return []
            if assessment.is_sufficient:
                break
            if attempt < max_retrieval_attempts and not assessment.is_sufficient:
                processing_steps.append("\n🔄 Attempting retrieval with expanded parameters...")
                expanded_k = k * 2
                results = st.session_state.vector_store.similarity_search(
                    query_embedding=query_embedding,
                    team_id=team_id,
                    k=expanded_k,
                    admin_override=is_admin
                )
                contexts = [
                    RetrievedContext(
                        doc_id=r["doc_id"],
                        content=r["text"],
                        metadata=r["metadata"],
                        score=r["similarity"]
                    ) for r in results
                ]
            else:
                processing_steps.append("\n⚠️ Warning: Retrieved context may not fully answer the query")
            attempt += 1
    st.session_state.last_processing_steps = processing_steps
    return contexts

def process_citations(response: str, contexts: List[RetrievedContext]) -> str:
    """
    Process citations in the response to add tooltips and improve readability.
    """
    import re
    
    # Find all source citations in the format [Source #] or [Source #, #] or (Source #, #)
    citation_pattern = r'(\[|\()Source (\d+(?:,\s*\d+)*)[\]|\)]'
    
    def replace_citation(match):
        # Get the source numbers from the citation
        source_nums = [int(num.strip()) for num in match.group(2).split(',')]
        
        # Build tooltip content for each source
        tooltips = []
        for num in source_nums:
            if 1 <= num <= len(contexts):
                ctx = contexts[num - 1]
                # Always use 'filename' as the key, fallback to doc_id
                source_name = ctx.metadata.get('filename') or ctx.doc_id
                
                # Build a simple tooltip with just the essential information
                tooltip_parts = [f"Source: {source_name}"]
                
                # Add the most relevant score based on retrieval method
                if 'llm_score' in ctx.metadata:
                    tooltip_parts.append(f"Score: {ctx.metadata['llm_score']}/10")
                elif 'combined_score' in ctx.metadata:
                    tooltip_parts.append(f"Score: {ctx.metadata['combined_score']}")
                elif 'similarity' in ctx.metadata:
                    tooltip_parts.append(f"Score: {ctx.metadata['similarity']}")
                
                tooltips.append(" | ".join(tooltip_parts))
        
        # Join tooltips with a clear separator
        tooltip_content = " || ".join(tooltips)
        
        # Create a simple span with escaped tooltip
        citation_text = match.group(0)
        return f'<span title="{tooltip_content}">{citation_text}</span>'
    
    # Replace all citations with tooltips
    processed_response = re.sub(citation_pattern, replace_citation, response)
    
    return processed_response

def main():
    # Check authentication first
    if not st.session_state.authenticated:
        show_login()
        st.stop()
    
    # Ensure last_processing_steps is always initialized
    if "last_processing_steps" not in st.session_state:
        st.session_state.last_processing_steps = []
    
    st.title("💬 Chat with Your Knowledge Base")

    # Initialize chat history and message counter in session state if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Always ensure message_counter exists
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 0

    # Use the unified GenAI client abstraction
    if "genai_client" not in st.session_state:
        st.session_state.genai_client = configured_client
    client = st.session_state.genai_client

    if "llm_service" not in st.session_state:
        st.session_state.llm_service = configured_llm_service

    # Initialize base services in session state
    if "embedding_service" not in st.session_state:
        st.session_state.embedding_service = EmbeddingService(client)

    # Build db_params from environment variables and create SQLAlchemy engine
    if "db_engine" not in st.session_state:
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD') # Allowed to be empty string
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')
        db_name = os.getenv('DB_NAME')
        
        # Check required variables: user, host, port, name must be non-empty
        # Password must be present (can be empty string), so check it's not None
        required_vars_present = all([db_user, db_host, db_port, db_name])
        password_is_present = db_password is not None
        
        if not (required_vars_present and password_is_present):
            missing = []
            if not db_user: missing.append('DB_USER')
            if db_password is None: missing.append('DB_PASSWORD') # Check for None explicitly
            if not db_host: missing.append('DB_HOST')
            if not db_port: missing.append('DB_PORT')
            if not db_name: missing.append('DB_NAME')
            st.error(f"Database configuration environment variables missing or empty: {', '.join(missing)}")
            st.stop()
            
        db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Create the engine with the specified search path
        st.session_state.db_engine = create_engine(
            db_url,
            connect_args={'options': '-c search_path=app_schema,extensions,public'}
        )
        # Optional: Create a session factory if services expect sessions
        # st.session_state.SessionFactory = sessionmaker(bind=st.session_state.db_engine)


    # Initialize other core services if not already set
    # TODO: Update these initializations to accept the engine or session factory
    if "vector_store" not in st.session_state:
        # Example: Modify VectorStore to accept engine
        st.session_state.vector_store = VectorStore(engine=st.session_state.db_engine) 
    if "hybrid_search" not in st.session_state:
        # Example: Modify HybridSearcher to accept engine
        st.session_state.hybrid_search = HybridSearcher(engine=st.session_state.db_engine) 
    if "multi_query_generator" not in st.session_state:
        st.session_state.multi_query_generator = MultiQueryGenerator(client)
    if "reranker" not in st.session_state:
        st.session_state.reranker = GeminiReranker(client)
    if "feedback_service" not in st.session_state:
        # Example: Modify FeedbackService to accept engine
        st.session_state.feedback_service = FeedbackService(engine=st.session_state.db_engine)
    if "agent" not in st.session_state:
        st.session_state.agent = Agent()

    # Sidebar controls
    with st.sidebar:
        st.session_state.show_context = st.checkbox(
            "Show sources used",
            value=st.session_state.get("show_context", False),
            help="Toggle to show/hide the source documents used to generate responses"
        )

        # Retrieval method selector
        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=[
                "Vector Search",
                "Hybrid Search",
                "Multi-Query",
                "Reranked Search",
                "Agentic (ReAct)"
            ],
            index=0,  # Default to Vector Search
            help="""
            Select the method for retrieving relevant context:
            - Vector Search: Default semantic search using embeddings
            - Hybrid Search: Combines semantic and keyword search
            - Multi-Query: Generates multiple queries for better recall
            - Reranked Search: Uses LLM to rerank initial vector search results
            - Agentic (ReAct): Uses vector search with reasoning and action to enhance retrieval
            """
        )
        st.session_state["retrieval_method"] = retrieval_method

        # Show info block if Agentic mode is selected (in sidebar, under dropdown)
        if retrieval_method == "Agentic (ReAct)":
            st.info("\n**Agentic (ReAct) mode is experimental.**\n\nThis mode uses an LLM agent to iteratively reason about and improve retrieval results. It may be slower or behave unexpectedly. Feedback is welcome!\n")

        # Retrieval settings (simple, always visible)
        st.markdown("---")
        st.markdown("**Retrieval Settings**")
        st.session_state.num_chunks = st.number_input(
            "Number of Chunks to Retrieve",
            min_value=1,
            max_value=20,
            value=st.session_state.get("num_chunks", 5),
            help="Number of document chunks to retrieve"
        )
        st.session_state.similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("similarity_threshold", 0.7),
            help="Minimum similarity score for retrieval"
        )
        if retrieval_method == "Hybrid Search":
            st.session_state.hybrid_search_weight = st.slider(
                "Hybrid Search Weight",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("hybrid_search_weight", 0.5),
                step=0.1,
                help="Weight between vector and text search (higher favors vector search)"
            )

    # Update the CSS to be simpler and more reliable
    st.markdown("""
        <style>
        span[title] {
            border-bottom: 1px dotted #4CAF50;
            color: #4CAF50;
            cursor: help;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Display the processed content with citations if it exists
            content_to_display = message["content"]
            st.markdown(content_to_display, unsafe_allow_html=True)
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant":
                # Create unique keys using both index and message counter
                msg_id = message.get("id", idx)  # Fallback to index for older messages
                col1, col2, col3, spacer = st.columns([1, 1, 1, 7])
                with col1:
                    if st.button("👍 Helpful", key=f"helpful_{msg_id}_{idx}"):
                        # Store feedback in database
                        feedback_stored = st.session_state.feedback_service.store_feedback(
                            message_id=msg_id,
                            feedback_type="helpful",
                            query=message.get("query", None),
                            response=message.get("raw_content", None),
                            context_used=[ctx.__dict__ for ctx in message.get("context_used", [])] if message.get("context_used") else None
                        )
                        if feedback_stored:
                            st.success("Thank you for your feedback!")
                        else:
                            st.error("Failed to store feedback. Please try again.")
                with col2:
                    if st.button("👎 Not Helpful", key=f"not_helpful_{msg_id}_{idx}"):
                        feedback_stored = st.session_state.feedback_service.store_feedback(
                            message_id=msg_id,
                            feedback_type="not_helpful",
                            query=message.get("query", None),
                            response=message.get("raw_content", None),
                            context_used=[ctx.__dict__ for ctx in message.get("context_used", [])] if message.get("context_used") else None
                        )
                        if feedback_stored:
                            st.success("Thank you for your feedback!")
                        else:
                            st.error("Failed to store feedback. Please try again.")
                with col3:
                    if st.button("⚠️ Report Issue", key=f"report_{msg_id}_{idx}"):
                        issue = st.text_area(
                            "Please describe the issue:",
                            key=f"issue_{msg_id}_{idx}"
                        )
                        if issue:
                            feedback_stored = st.session_state.feedback_service.store_feedback(
                                message_id=msg_id,
                                feedback_type="issue",
                                query=message.get("query", None),
                                response=message.get("raw_content", None),
                                context_used=[ctx.__dict__ for ctx in message.get("context_used", [])] if message.get("context_used") else None,
                                issue_description=issue
                            )
                            if feedback_stored:
                                st.success("Thank you for reporting the issue!")
                            else:
                                st.error("Failed to store feedback. Please try again.")
            
            # Display processing steps if available and agent is enabled
            if "processing_steps" in message and st.session_state.get("use_agent", False):
                with st.expander("🤔 Query Processing Steps", expanded=False):
                    for step in message["processing_steps"]:
                        st.markdown(step)
            
            # Display context if available
            if "context_used" in message and st.session_state.get("show_context", False):
                with st.expander("📚 Sources Used", expanded=False):
                    # If using multi-query, show the generated queries first
                    if (
                        message["context_used"] and 
                        "generated_query" in message["context_used"][0].metadata
                    ):
                        st.markdown("**🔍 Generated Search Queries:**")
                        unique_queries = {
                            ctx.metadata["generated_query"] 
                            for ctx in message["context_used"]
                        }
                        for i, query in enumerate(unique_queries, 1):
                            st.markdown(f"{i}. {query}")
                        st.markdown("---")
                    
                    # Display sources with better organization
                    for i, ctx in enumerate(message["context_used"], 1):
                        with st.container():
                            source_name = ctx.metadata.get('filename') or ctx.doc_id
                            st.markdown(f"**Source {i}:** {source_name}")
                            
                            # Create two columns with better width ratio
                            score_col, content_col = st.columns([1, 3])
                            
                            with score_col:
                                # Display scoring information based on retrieval method
                                if "reranked" in ctx.metadata:
                                    position_change = int(ctx.metadata['position_change'])
                                    change_str = f"+{-position_change}" if position_change < 0 else str(-position_change)
                                    
                                    st.markdown("**📊 Scoring:**")
                                    st.markdown(f"- Rank: #{ctx.metadata['rank']} ({change_str})")
                                    st.markdown(f"- Initial Score: {ctx.metadata['initial_score']}")
                                    st.markdown(f"- LLM Score: {ctx.metadata['llm_score']}/10")
                                    
                                    # Visual score indicators
                                    initial_score = float(ctx.metadata['initial_score'])
                                    llm_score = float(ctx.metadata['llm_score']) / 10
                                    st.progress(initial_score, text="Vector Score")
                                    st.progress(llm_score, text="LLM Score")
                                    
                                elif "vector_score" in ctx.metadata:
                                    # Show Hybrid Scores only if both vector and text scores are present
                                    if 'text_score' in ctx.metadata and 'combined_score' in ctx.metadata:
                                        st.markdown("**📊 Hybrid Scores:**")
                                        st.markdown(f"- Vector: {ctx.metadata['vector_score']}")
                                        st.markdown(f"- Text: {ctx.metadata['text_score']}")
                                        st.markdown(f"- Combined: {ctx.metadata['combined_score']}")
                                    else:
                                        st.markdown("**📊 Vector Score:**")
                                        st.markdown(f"- Vector: {ctx.metadata['vector_score']}")
                                    
                                    # Visual score indicators
                                    vector_score = float(ctx.metadata['vector_score'])
                                    st.progress(vector_score, text="Vector")
                                    if 'text_score' in ctx.metadata:
                                        text_score = float(ctx.metadata['text_score'])
                                        st.progress(text_score, text="Text")
                                    
                                elif "generated_query" in ctx.metadata:
                                    st.markdown("**📊 Score:**")
                                    st.markdown(f"- Vector Score: {ctx.metadata['vector_score']}")
                                    st.markdown(f"- Generated Query: {ctx.metadata['generated_query']}")
                                    
                                    # Visual score indicator
                                    vector_score = float(ctx.metadata['vector_score'])
                                    st.progress(vector_score, text="Vector Score")
                                    
                                else:
                                    st.markdown("**📊 Score:**")
                                    st.markdown(f"- Similarity: {ctx.score:.3f}")
                                    st.progress(ctx.score, text="Similarity")
                            
                            with content_col:
                                # Display content with better formatting
                                st.markdown("**📄 Content:**")
                                content = ctx.content
                                if any(marker in content.lower() for marker in ['```', 'def ', 'class ', 'import ']):
                                    st.code(content, language="python")
                                else:
                                    st.markdown(content)
                        
                            # Add a visual separator between sources
                            st.markdown("---")
            
            # Show agentic reasoning steps only for Agentic retrieval method
            if (
                "processing_steps" in message
                and st.session_state.get("retrieval_method", "") == "Agentic (ReAct)"
                and message["processing_steps"]
            ):
                with st.expander("🤔 Agentic Reasoning Steps", expanded=False):
                    for step in message["processing_steps"]:
                        # If stored as stringified dict, parse it
                        if isinstance(step, str) and step.startswith("{"):
                            try:
                                step_dict = ast.literal_eval(step)
                            except Exception:
                                step_dict = {"step": "Unparsable", "raw": step}
                        else:
                            step_dict = step
                        st.markdown(f"**Step:** {step_dict.get('step', '')}")
                        for k, v in step_dict.items():
                            if k == "step":
                                continue
                            if isinstance(v, list):
                                st.markdown(f"- **{k}:**")
                                for item in v:
                                    st.markdown(f"    - {item}")
                            else:
                                st.markdown(f"- **{k}:** {v}")
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask a question about your team's documents"):
        # Add user message to chat history with unique ID
        st.session_state.message_counter += 1
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "id": st.session_state.message_counter
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Show a more detailed loading state
                with st.status("Processing your question...", expanded=True) as status:
                    status.write("🔍 Retrieving relevant information...")
                    
                    # Map UI selection to retrieval method
                    method_map = {
                        "Vector Search": "vector",
                        "Hybrid Search": "hybrid",
                        "Multi-Query": "multi_query",
                        "Reranked Search": "reranked",
                        "Agentic (ReAct)": "agentic"
                    }
                    
                    # Get recent chat history (last 2 turns)
                    recent_history = get_recent_chat_history(n=2)
                    if recent_history:
                        full_query = f"{recent_history}\nUser: {prompt}"
                    else:
                        full_query = prompt
                    
                    # Get relevant context through RAG pipeline
                    contexts = process_query(
                        full_query,
                        prompt,  # Pass the latest user input as a separate argument
                        k=st.session_state.num_chunks,
                        retrieval_method=method_map[retrieval_method]
                    )
                    
                    status.update(label="Generating response...", state="running")
                    status.write("🤖 Analyzing context and composing answer...")
                
                # Check if we need clarification
                if st.session_state.get("needs_clarification", False):
                    clarification_question = st.session_state.get("clarification_question")
                    message_placeholder.warning(
                        f"🤔 I need some clarification to better answer your question:\n\n"
                        f"{clarification_question}"
                    )
                    # Clear the flags
                    st.session_state.needs_clarification = False
                    st.session_state.clarification_question = None
                    return
                
                if not contexts:
                    message_placeholder.warning(
                        "❌ I couldn't find any relevant information in the knowledge base. "
                        "Please try rephrasing your question or ask something else."
                    )
                    return
                
                # Generate streaming response
                response_stream = st.session_state.llm_service.generate_response_stream(
                    query=full_query,
                    contexts=contexts,
                    temperature=0.7
                )
                
                if response_stream:
                    # Initialize response text
                    full_response = ""
                    
                    # Stream the response with a typing indicator
                    for chunk in response_stream:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    processed_response = process_citations(full_response, contexts)

                    # --- DEBUG: Print raw LLM response to CLI before citation processing ---
                    print("\n[DEBUG] Raw LLM response before citation processing:\n" + full_response + "\n")
                    # --- END DEBUG ---
                    
                    message_placeholder.markdown(processed_response, unsafe_allow_html=True)
                    
                    # Add assistant response to chat history with unique ID
                    st.session_state.message_counter += 1
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": processed_response,  # Store processed response
                        "raw_content": full_response,  # Store original response
                        "context_used": contexts,
                        "processing_steps": st.session_state.last_processing_steps,
                        "id": st.session_state.message_counter,
                        "query": prompt  # Store the original query
                    })
                    
                    # Update status to complete
                    status.update(label="✅ Response complete!", state="complete")
                    
                    # Force a rerun to display the sources immediately
                    st.rerun()
                else:
                    message_placeholder.error(
                        "❌ I encountered an error while generating the response. "
                        "Please try again."
                    )
                    
            except Exception as e:
                message_placeholder.error(
                    f"❌ An error occurred: {str(e)}. Please try again."
                )

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main() 