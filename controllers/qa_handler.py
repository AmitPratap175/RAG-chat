from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from services.google_ai import GoogleAI
from config import Config

class QAHandler:
    def __init__(self):
        self.llm = GoogleAI().get_llm()
        with open(Config.PROMPT_FILE, "r") as file:
            custom_prompt = file.read()
        
        self.qa_prompt = PromptTemplate(
            template=custom_prompt,
            input_variables=["context", "question"]
        )

    def get_answer(self, query, retriever):
        """Execute QA chain"""
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )
        return qa_chain.invoke(query)
