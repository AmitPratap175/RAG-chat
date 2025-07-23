from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
# from services.sentiment import SentimentAnalyzer  # Assume you have this
from controllers.qa_handler import QAHandler  # Your existing QA handler
from services.google_ai import GoogleAI
from langchain.prompts import ChatPromptTemplate

class BotState(TypedDict):
    user_input: str
    sentiment: Annotated[str, "negative|positive"]
    qa_answer: str
    needs_human: bool
    response: str

class CustomerSupportBot:
    """
    CustomerSupportBot is a workflow-driven customer support assistant that analyzes user inputs for sentiment,
    routes the conversation accordingly, retrieves answers from a knowledge base, and manages human escalation.

    Responsibilities:
    -----------------
    - Analyze incoming customer messages for sentiment.
    - Route customer queries to either automated answering or human escalation based on sentiment.
    - Retrieve answers using the QA handler and knowledge base.
    - Generate final responses for delivery to the end user.

    Attributes:
    -----------
    llm : ChatGoogleGenerativeAI
        Language model for both sentiment analysis and response generation.
    qa_handler : QAHandler
        Component to handle QA retrieval operations.
    workflow : StateGraph
        LangGraph state-driven workflow definition.
    retriever : Optional[BaseRetriever]
        Current retriever instance for sourcing answers.

    Methods:
    --------
    sentiment_analyzer(text: str) -> str
        Analyzes customer input for sentiment ('positive' or 'negative') using the LLM.
    analyze_sentiment(state: BotState) -> dict
        Workflow node to perform sentiment analysis and return result.
    decide_routing(state: BotState) -> str
        Directs workflow path based on sentiment output ('escalate' or 'answer').
    retrieve_answer(state: BotState) -> dict
        Uses QAHandler to get answer from knowledge base and return it in state.
    escalate_to_human(state: BotState) -> dict
        Marks that human escalation is needed.
    generate_response(state: BotState) -> dict
        Finalizes the bot response, returning either human escalation message or retrieved answer.
    handle_query(user_input: str, retriever) -> str
        Orchestrates the end-to-end workflow execution given user input and retriever.
    """

    def __init__(self):
        self.llm = GoogleAI().get_llm()
        self.qa_handler = QAHandler()
        self.workflow = StateGraph(BotState)
        self.retriever = None
        
        # Define workflow nodes
        self.workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        self.workflow.add_node("retrieve_answer", self.retrieve_answer)
        self.workflow.add_node("escalate_to_human", self.escalate_to_human)
        self.workflow.add_node("generate_response", self.generate_response)

        # Define workflow edges
        self.workflow.set_entry_point("analyze_sentiment")
        self.workflow.add_edge("generate_response", END)
        
        # Conditional workflow routing based on sentiment
        self.workflow.add_conditional_edges(
            "analyze_sentiment",
            self.decide_routing,
            {
                "escalate": "escalate_to_human",
                "answer": "retrieve_answer"
            }
        )
        self.workflow.add_edge("retrieve_answer", "generate_response")
        self.workflow.add_edge("escalate_to_human", "generate_response")

        # Compile workflow graph for runtime execution
        self.app = self.workflow.compile()
    
    def sentiment_analyzer(self, text: str) -> str:
        """
        Analyze sentiment of the customer input using the large language model
        and a prompt template for classification.

        Parameters:
        -----------
        text : str
            Customer query input.

        Returns:
        --------
        str
            Either 'positive' or 'negative' as determined by the LLM.
        """
        prompt = ChatPromptTemplate.from_template(
            """
        You are a customer support chatbot. Analyze the sentiment of the following customer query. 
        Ignore any previous chat history. Focus solely on the current query.
        Consider the context of customer support interactions when determining the sentiment.

        Respond with one of the following exactly: 'positive', or 'negative'.

        Here are some guidelines:

        - 'positive': The customer expresses satisfaction, appreciation, or positive feelings. The customer's query is informational, factual, or lacks strong emotional expression, and the chatbot can potentially provide a resolution.
          If the customer uses inappropriate language, but the query can still be answered by the bot, respond with neutral.
        - 'negative': The customer explicitly requests to speak to a human, or the customer's query indicates that the chatbot is unable to provide a satisfactory answer and requires human intervention.

        Query: {query}
        """
        )
        chain = prompt | self.llm
        sentiment = chain.invoke({"query": text}).content
        # print(f"Sentiment analyzed: {sentiment}")
        return sentiment

    def analyze_sentiment(self, state: BotState) -> dict:
        """
        Workflow node for analyzing sentiment of the user's input.

        Parameters:
        -----------
        state : BotState
            Current workflow state with 'user_input' field.

        Returns:
        --------
        dict
            Dictionary with the sentiment result.
        """
        sentiment = self.sentiment_analyzer(state["user_input"])
        return {"sentiment": sentiment}

    def decide_routing(self, state: BotState) -> str:
        """
        Decide workflow routing based on sentiment.

        If sentiment is 'negative', escalate to human, else continue with answer retrieval.

        Parameters:
        -----------
        state : BotState

        Returns:
        --------
        str
            'escalate' or 'answer' indicating routing decision.
        """
        if state["sentiment"] == "negative":
            print("Routing to human escalation due to negative sentiment.")
            return "escalate"
        return "answer"

    def retrieve_answer(self, state: BotState) -> dict:
        """
        Workflow node to retrieve an answer from the knowledge base.

        Parameters:
        -----------
        state : BotState

        Returns:
        --------
        dict
            Dictionary with the answer as 'qa_answer'.
        """
        answer = self.qa_handler.get_answer(state["user_input"], self.retriever)
        return {"qa_answer": answer}

    def escalate_to_human(self, state: BotState) -> dict:
        """
        Workflow node for escalation; marks the need for human intervention.

        Parameters:
        -----------
        state : BotState

        Returns:
        --------
        dict
            Sets 'needs_human' flag True.
        """
        return {"needs_human": True}

    def generate_response(self, state: BotState) -> dict:
        """
        Generate the final response based on whether escalation occurred.

        If escalation is needed, return escalation message. Otherwise, return QA answer.

        Parameters:
        -----------
        state : BotState

        Returns:
        --------
        dict
            Dictionary with the field 'response'.
        """
        if state.get("needs_human"):
            return {"response": "Let me connect you to a human representative..."}
        return {"response": state["qa_answer"]}

    def handle_query(self, user_input: str, retriever) -> str:
        """
        Run the entire customer support workflow given the user's input and a retriever instance.

        Parameters:
        -----------
        user_input : str
            The user's question or issue.
        retriever : BaseRetriever
            Retriever instance for document/question answering.

        Returns:
        --------
        str
            The chatbot's final response to the user.
        """
        self.retriever = retriever
        results = self.app.invoke({"user_input": user_input})
        return results["response"]
