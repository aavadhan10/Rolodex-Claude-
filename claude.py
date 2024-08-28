import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Initialize Claude API using environment variable
claude_api_key = st.secrets["claude"]["CLAUDE_API_KEY"]
claude_api_url = "https://api.claude.ai/v1/chat/completions"

st.write("Starting app...")

# Function to call Claude
def call_claude(messages):
    headers = {
        "Authorization": f"Bearer {claude_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "claude-v1",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.9,
    }
    
    try:
        st.write("Sending request to Claude...")
        response = requests.post(claude_api_url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        response_json = response.json()
        st.write("Received response from Claude.")
        return response_json['choices'][0]['message']['content'].strip()
    except requests.exceptions.Timeout:
        st.error("The request to Claude timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while communicating with Claude: {e}")
        return None

# Vectorize the questions and data
def vectorize_data(matters_data, text_column='Matter Description'):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(matters_data[text_column].fillna(''))
    return vectors, vectorizer

# Query Claude with Data
def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    try:
        st.write("Vectorizing the question...")
        question = ' '.join(question.split()[-3:])
        question_vec = matters_vectorizer.transform([question])
        
        st.write("Performing vector search...")
        D, I = matters_index.search(normalize(question_vec).toarray(), k=10)
        if I.size > 0 and not (I == -1).all():
            relevant_data = matters_data.iloc[I[0]]
        else:
            relevant_data = matters_data.head(1)  # Fallback
        
        st.write("Filtering the data...")
        filtered_data = relevant_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Role Detail']]
        filtered_data = filtered_data.rename(columns={'Role Detail': 'Role'}).drop_duplicates(subset=['Attorney'])
        
        if filtered_data.empty:
            filtered_data = matters_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Role Detail']]
            filtered_data = filtered_data.rename(columns={'Role Detail': 'Role'}).dropna(subset=['Attorney']).drop_duplicates(subset=['Attorney']).head(1)

        context = filtered_data.to_string(index=False)
        messages = [
            {"role": "system", "content": "You are the CEO of a prestigious law firm..."},
            {"role": "user", "content": f"Based on the following information, please make a recommendation:\n\n{context}\n\nRecommendation:"}
        ]
        
        st.write("Calling Claude API with filtered data...")
        claude_response = call_claude(messages)
        if not claude_response:
            return  # Exit if Claude API failed
        
        recommendations = claude_response.split('\n')
        recommendations = [rec for rec in recommendations if rec.strip()]
        recommendations = list(dict.fromkeys(recommendations))
        recommendations_df = pd.DataFrame(recommendations, columns=['Recommendation Reasoning'])
        
        st.write("Displaying results...")
        top_recommended_lawyers = filtered_data.drop_duplicates(subset=['Attorney'])
        st.write("All Potential Lawyers with Recommended Skillset:")
        st.write(top_recommended_lawyers.to_html(index=False), unsafe_allow_html=True)
        st.write("Recommendation Reasoning:")
        st.write(recommendations_df.to_html(index=False), unsafe_allow_html=True)

        for lawyer in top_recommended_lawyers['Attorney'].unique():
            st.write(f"**{lawyer}'s Matters:**")
            lawyer_matters = matters_data[matters_data['Attorney'] == lawyer][['Practice Area', 'Matter Description']]
            st.write(lawyer_matters.to_html(index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error querying Claude: {e}")

# Example usage:
# Assuming matters_data is your DataFrame and matters_index is a FAISS index
# matters_vectors, matters_vectorizer = vectorize_data(matters_data)
# query_claude_with_data("What are the top lawyers for this case?", matters_data, matters_index, matters_vectorizer)
