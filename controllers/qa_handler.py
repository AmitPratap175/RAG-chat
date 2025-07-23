from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from services.google_ai import GoogleAI
from config import Config

class QAHandler:
    """
    QAHandler manages the question-answering process using a configured language model and prompt.

    Responsibilities:
    -----------------
    - Load and prepare a custom prompt template.
    - Instantiate the language model interface via GoogleAI.
    - Build and run a RetrievalQA chain to answer user queries based on provided retriever.

    Attributes:
    -----------
    llm : ChatGoogleGenerativeAI
        The language model client used to generate answers.
    qa_prompt : PromptTemplate
        The prompt template providing structure for QA chain input.

    Methods:
    --------
    get_answer(query, retriever)
        Executes the QA chain to generate an answer and return source documents.
    """

    def __init__(self):
        """
        Initializes QAHandler by loading the custom prompt and getting the language model client.
        """
        self.llm = GoogleAI().get_llm()
        with open(Config.PROMPT_FILE, "r") as file:
            custom_prompt = file.read()
        
        self.qa_prompt = PromptTemplate(
            template=custom_prompt,
            input_variables=["context", "question"]
        )

    def get_answer(self, query, retriever):
        """
        Execute the RetrievalQA chain with the given query and retriever.

        Parameters:
        -----------
        query : str
            The user question or query text to be answered.
        retriever : BaseRetriever
            Retriever instance used to fetch relevant documents for context.

        Returns:
        --------
        dict
            Result containing answer text and source documents, as produced by the QA chain.
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )
        return qa_chain.invoke(query)
