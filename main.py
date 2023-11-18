import streamlit as st
import gc
from dotenv import load_dotenv


def load_result(user_query):
    from langchain.embeddings import GooglePalmEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.llms import GooglePalm
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # load garbage collector
    gc.enable()

    # create embeddings
    embeddings = GooglePalmEmbeddings()

    # store embeddings
    vectordb = FAISS.load_local('palm_index', embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    print('vectordb loaded')

    # create template
    prompt_template = PromptTemplate.from_template(
        "As a legal advisor well-versed in the Indian Penal Code, provide expert guidance on the following "
        "query:\n\n'{user_query}'.\n\nExplain the law principles, relevant articles, and any precedents that apply. Answer to the best of your knowledge. If you dont know the answer reply with I don't know"
    )
    user_prompt = prompt_template.format(user_query=user_query)
    bad_chars = [';', ':', '!', "*", "?", '.', ',']
    user_prompt = ''.join(letter for letter in user_prompt if (letter not in bad_chars))

    # question answer chain
    llm = GooglePalm()
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)

    try:
        response = chain(user_prompt)
        if response:
            print(response['result'], end='\n')
            st.write(response['result'])

            with st.expander("References"):
                for doc in response['source_documents']:
                    st.write(doc.page_content)
        del response
    except:
        st.write("Please try again!")

    # delete all unused variables
    del embeddings
    del vectordb
    del retriever
    del prompt_template
    del user_prompt
    del llm
    del chain

    # free garbage collector
    gc.collect()


if __name__ == '__main__':
    # load env files
    load_dotenv()

    prompt1 = "I'm going through a divorce and wondering about child custody. Can you provide guidance on how the Indian Law influences such cases?"
    prompt2 = "If I meet an  car accident and the other person dies because my driving mistake, what will be the punishment for me ?"
    prompt3 = "What legal actions can be taken against online fraud and scams in India?"
    prompt4 = "My landlord is refusing to return my security deposit. What should I do in context to Laws in India ?"

    # sidebar
    with st.sidebar:
        st.header('About')
        st.write(
            "LawGPT harnesses the power of a state-of-the-art language model trained on the intricacies of the Indian Laws and Acts.")
        st.write("Start asking to your Legal advisor with the below questions or ask you own 😊")
        st.info(prompt1)
        st.info(prompt2)
        st.info(prompt3)
        st.info(prompt4)
        st.write("\n\n")
        st.write("\n\n")

        st.write("Made with ❤ by [Hamza Khan](https://hamzakhan07.netlify.app/)")

    st.header('📚⚖️ LawGPT')

    st.write(
        "Got a legal question? Meet your go-to LawGPT! 🌟 Ask anything about Indian law, and get straightforward advice! No legal mumbo-jumbo, just clear answers! Know Your Rights, Duties, and Legal Insights 💬")

    query = st.text_area('Enter your question')
    # query submitted
    submitted = st.button('Submit')
    if query or submitted:
        load_result(query)
