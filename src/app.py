import streamlit as st
from rag_pipeline import rag_chat
from config import DEFAULT_MODEL
from memory import get_long_term_memory


st.set_page_config(page_title="Customer Support Chatbot with Memory", layout="wide")
st.title("AI Customer Support Chatbot (RAG) with Memory")

# Sidebar for user management and memory controls
with st.sidebar:
    st.header("User & Memory Management")
    
    # User selection
    user_id = st.text_input("User ID:", value="default_user", help="Enter a unique user ID to maintain personalized memory")
    
    # Memory controls
    st.subheader("Memory Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Short-term Memory"):
            st.session_state.clear_short_term = True
            st.rerun()
    
    with col2:
        if st.button("Clear Episodic Memory"):
            st.session_state.clear_episodic = True
            st.rerun()
    
    if st.button("Clear All Memory"):
        st.session_state.clear_all_memory = True
        st.rerun()
    
    # Memory statistics
    st.subheader("Memory Statistics")
    try:
        memory = get_long_term_memory(user_id)
        stats = memory.get_stats()
        
        st.metric("Total Facts", stats['total_facts'])
        st.metric("Extracted Entities", stats['extracted_entities'])
        st.metric("Retrieved Facts", stats['retrieved_facts'])
        
        if stats['fact_types']:
            st.write("**Fact Types:**")
            for fact_type, count in stats['fact_types'].items():
                st.write(f"- {fact_type.capitalize()}: {count}")
    except:
        st.write("Memory statistics unavailable")

# Main chat interface
st.markdown("### Ask a medical or health-related question:")

# Chat input
query = st.text_area("Your question:", placeholder="e.g., What are the side effects of ibuprofen?", height=150)
top_k = st.slider("Number of context chunks:", 1, 10, 3)

# Submit button with enhanced feedback
if st.button("Submit") and query.strip():
    with st.spinner("Retrieving information, checking memory, and generating answer..."):
        result = rag_chat(query, top_k=top_k, model=DEFAULT_MODEL, user_id=user_id)
        
        # Display results
        st.markdown("### Chatbot Answer:")
        st.write(result["formatted_answer"], unsafe_allow_html=True)
        
        # Display cache information
        if result.get("cache_hit"):
            st.success("✅ Answer retrieved from cache - faster response!")
        else:
            st.info("📝 New answer generated and cached for future use.")
        
        # Display memory statistics
        if "memory_stats" in result:
            st.markdown("### Memory Usage:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Short-term Messages", result["memory_stats"]["short_term"]["current_messages"])
                st.metric("Short-term Tokens", result["memory_stats"]["short_term"]["total_tokens"])
            
            with col2:
                st.metric("Episodic Summaries", result["memory_stats"]["episodic"]["current_summaries"])
                st.metric("Extracted Facts", result["memory_stats"]["episodic"]["extracted_facts"])
            
            with col3:
                st.metric("Long-term Facts", result["memory_stats"]["long_term"]["total_facts"])
                st.metric("Database Size", result["memory_stats"]["long_term"]["database_size"])
        
        # Display timing information
        if "timings" in result:
            st.markdown("### Performance Metrics:")
            timings = result["timings"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cache Check", f"{timings.get('cache_check', 0):.3f}s")
            
            with col2:
                st.metric("Context Retrieval", f"{timings.get('retrieve_context', 0):.3f}s")
            
            with col3:
                st.metric("Answer Generation", f"{timings.get('generate_answer', 0):.3f}s")
            
            with col4:
                st.metric("Memory Update", f"{timings.get('update_memories', 0):.3f}s")
        
        # Display retrieved contexts
        with st.expander("View Retrieved Contexts"):
            for i, c in enumerate(result["contexts"]):
                st.markdown(f"**[Context {i+1}]**")
                st.write(c["content"])
else:
    st.info("Type a question above and click **Submit** to start. Your conversation history will be remembered across sessions!")

# Display user medical profile
st.markdown("---")
st.markdown("### User Medical Profile")
try:
    memory = get_long_term_memory(user_id)
    profile = memory.get_user_profile()
    
    if "message" in profile:
        st.info(profile["message"])
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Medications")
            if profile["medications"]:
                for med in profile["medications"][:5]:  # Show last 5
                    st.write(f"- {med['text']} (Source: {med['source']})")
            else:
                st.write("No medication history recorded.")
        
        with col2:
            st.subheader("Conditions")
            if profile["conditions"]:
                for cond in profile["conditions"][:5]:  # Show last 5
                    st.write(f"- {cond['text']} (Source: {cond['source']})")
            else:
                st.write("No condition history recorded.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Symptoms")
            if profile["symptoms"]:
                for symp in profile["symptoms"][:5]:  # Show last 5
                    st.write(f"- {symp['text']} (Source: {symp['source']})")
            else:
                st.write("No symptom history recorded.")
        
        with col4:
            st.subheader("Allergies")
            if profile["allergies"]:
                for allerg in profile["allergies"][:5]:  # Show last 5
                    st.write(f"- {allerg['text']} (Source: {allerg['source']})")
            else:
                st.write("No allergy history recorded.")
        
        st.caption(f"Last updated: {profile.get('last_updated', 'N/A')}")
        
except Exception as e:
    st.error(f"Error loading medical profile: {e}")
