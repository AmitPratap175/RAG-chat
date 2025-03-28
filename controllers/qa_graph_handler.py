from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from services.sentiment import SentimentAnalyzer  # Assume you have this
from controllers.qa_handler import QAHandler  # Your existing QA handler
from services.google_ai import GoogleAI
import operator

class BotState(TypedDict):
    user_input: str
    sentiment: Annotated[str, "negative|neutral|positive"]
    qa_answer: str
    needs_human: bool
    response: str

class CustomerSupportBot:
    def __init__(self):
        self.llm = GoogleAI().get_llm()
        self.qa_handler = QAHandler()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.workflow = StateGraph(BotState)
        self.retriever = None
        
        # Define nodes
        self.workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        self.workflow.add_node("retrieve_answer", self.retrieve_answer)
        self.workflow.add_node("escalate_to_human", self.escalate_to_human)
        self.workflow.add_node("generate_response", self.generate_response)

        # Define edges
        self.workflow.set_entry_point("analyze_sentiment")
        self.workflow.add_edge("generate_response", END)
        
        # Conditional routing
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

        # Compile the graph
        self.app = self.workflow.compile()

    def analyze_sentiment(self, state: BotState) -> dict:
        """Analyze user input sentiment"""
        sentiment = self.sentiment_analyzer.analyze(state["user_input"])
        print(f"Detected sentiment: {sentiment}")
        return {"sentiment": sentiment}

    def decide_routing(self, state: BotState) -> str:
        """Route based on sentiment"""
        if state["sentiment"] == "negative":
            return "escalate"
        return "answer"

    def retrieve_answer(self, state: BotState) -> dict:
        """Get answer from knowledge base"""
        answer = self.qa_handler.get_answer(state["user_input"], self.retriever)
        return {"qa_answer": answer}

    def escalate_to_human(self, state: BotState) -> dict:
        """Handle escalation"""
        return {"needs_human": True}

    def generate_response(self, state: BotState) -> dict:
        """Final response formatting"""
        if state.get("needs_human"):
            return {"response": "Let me connect you to a human representative..."}
        return {"response": state["qa_answer"]}

    def handle_query(self, user_input: str, retriever) -> str:
        """Execute the workflow"""
        self.retriever = retriever
        results = self.app.invoke({"user_input": user_input})
        return results["response"]