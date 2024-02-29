import streamlit as st


@st.cache_data


def show_home_page():
    # Create three columns
    col1, col2, col3, col4, col5 = st.columns([1,1,3,1,1])

    # Use the middle column to display the image, achieving a centered effect
    with col3:
        st.image("logo2.png", width=250)

    st.markdown("<h1 style='text-align: center; color: "";'>Financial News Categorization using NLP</h1>", unsafe_allow_html=True)
    st.write(
        """
    #### In this project we will be using a pre-trained model called FinBERT to categorize financial news articles."""
    )
    st.markdown("""
        <div style="text-align: justify">
             Sentiment analysis is the statistical analysis of simple sentiment cues. 
             Essentially, it involves making statistical analyses on polarized statements 
             (i.e., statements with a positive, negative and neutral sen timent), which 
             are usually collected in the form of social media posts, reviews, and news articles. 
             Financial sentiment analysis is a challenging task due to the specialized language and
             lack of labeled data in that domain.
         </div>
""", unsafe_allow_html=True)
    
    
    st.markdown(""" """)
        
    st.markdown(""" """)

    st.markdown(""" """)
        
            # Example of using an expander to show/hide information
    with st.expander("More information about this project"):
        st.markdown("""
        - **Project:** Financial News Categorization using NLP            
        - **Authors of the App:** 
            - Fausto Bravo Cuvi
            - Daniel Espinoza
        - **Assigment:** DATA DRIVEN BUSINESS CLASS
        - **Teacher:** Ricardo García
    
        """)

    st.markdown(""" """)
    
    st.markdown(""" """)
    
    st.markdown(""" """)
 
   # Continue with the rest of your app below
    st.markdown("""
    <div style='font-size: 22px; color: red; text-align: center;'>
        <span style='display: inline-block; transform: rotate(180deg);'>&#10145;</span> Navigate to the sidebar to see the FinBERT model in action!
    </div>
    """, unsafe_allow_html=True)


    

 
    
    
    
    
    
