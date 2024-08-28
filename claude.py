import requests

# Initialize Claude API using environment variable
claude_api_key = st.secrets["CLAUDE_API_KEY"]
claude_api_url = "https://api.claude.ai/v1/chat/completions"  # Example endpoint

# Function to call Claude
def call_claude(messages):
    headers = {
        "Authorization": f"Bearer {claude_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "claude-v1",  # Adjust model name if necessary
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.9,
    }
    
    response = requests.post(claude_api_url, headers=headers, json=data)
    response_json = response.json()
    
    return response_json['choices'][0]['message']['content'].strip()

# Update your GPT querying function
def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    try:
        question = ' '.join(question.split()[-3:])
        question_vec = matters_vectorizer.transform([question])
        
        D, I = matters_index.search(normalize(question_vec).toarray(), k=10)
        relevant_data = matters_data.iloc[I[0]] if I.size > 0 and not (I == -1).all() else matters_data.head(1)
        
        filtered_data = relevant_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Role Detail']].rename(columns={'Role Detail': 'Role'}).drop_duplicates(subset=['Attorney'])
        
        if filtered_data.empty:
            filtered_data = matters_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Role Detail']].rename(columns={'Role Detail': 'Role'}).dropna(subset=['Attorney']).drop_duplicates(subset=['Attorney']).head(1)

        context = filtered_data.to_string(index=False)
        messages = [
            {"role": "system", "content": "You are the CEO of a prestigious law firm..."},
            {"role": "user", "content": f"Based on the following information, please make a recommendation:\n\n{context}\n\nRecommendation:"}
        ]
        
        claude_response = call_claude(messages)
        recommendations = claude_response.split('\n')
        recommendations = [rec for rec in recommendations if rec.strip()]
        recommendations = list(dict.fromkeys(recommendations))
        recommendations_df = pd.DataFrame(recommendations, columns=['Recommendation Reasoning'])
        
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
