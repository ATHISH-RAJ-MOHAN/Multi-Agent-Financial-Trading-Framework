from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_finbert_sentiment
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news, get_finbert_sentiment
        ]

        '''system_message = (
            "You are a social media and company-specific sentiment analyst. "
            "Your job is to analyze social media posts, recent company news, and public sentiment "
            "for the given company over the past week. "
            "You must **call the tool get_news(query, start_date, end_date)** to fetch company-related news, "
            "and then **call the tool get_finbert_sentiment(texts)** on the fetched headlines and summaries "
            "to obtain quantitative sentiment scores (positive, neutral, negative probabilities). "
            "Use these FinBERT scores to compute overall sentiment statistics "
            "(average pos/neg/neu values, sentiment skew, and the dominant label). "
            "In your final output, include a JSON field named 'scores' with these values: "
            "{'pos_mean': float, 'neu_mean': float, 'neg_mean': float, 'n': int}, "
            "and list the top 3 headlines with their individual FinBERT scores in a Markdown table. "
            "Do not just say 'mixed sentiment' — explain what FinBERT indicates quantitatively "
            "and what it implies for traders or investors."
        )'''

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )


        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
