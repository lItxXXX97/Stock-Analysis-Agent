# 导入所需的库
import os
import logging
from dotenv import load_dotenv
import yfinance as yf
from llama_agents.launchers.local import LocalLauncher
from llama_agents.services import AgentService, ToolService
from llama_agents.tools import MetaServiceTool
from llama_agents.control_plane.server import ControlPlaneServer
from llama_agents.message_queues.simple import SimpleMessageQueue
from llama_agents.orchestrators.agent import AgentOrchestrator
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# 设置日志级别为INFO
logging.basicConfig(level=logging.INFO)

# 加载.env文件中的环境变量
load_dotenv()

# 从环境变量中获取baichuan_key密钥
api_key = os.getenv("OPENAI_API_KEY")
api_base = "https://api.aigc369.com/v1"
# 确保API密钥已设置,否则抛出异常
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# 设置baichuan_key密钥为环境变量
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = api_base
# 定义获取股票当前价格的函数
def get_stock_price(symbol: str) -> str:
    """获取给定股票代码的当前价格"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            return f"The current price of {symbol} is ${current_price:.2f}"
        else:
            return f"Unable to fetch the current price for {symbol}. The stock data is empty."
    except Exception as e:
        logging.error(f"Error fetching stock price for {symbol}: {str(e)}")
        return f"Error fetching stock price for {symbol}: {str(e)}"

# 定义获取公司信息的函数
def get_company_info(symbol: str) -> str:
    """获取给定股票代码的公司信息"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return f"{info['longName']} ({symbol}) is in the {info.get('sector', 'Unknown')} sector. {info.get('longBusinessSummary', '')[:200]}..."
    except Exception as e:
        logging.error(f"Error fetching company info for {symbol}: {str(e)}")
        return f"Error fetching company info for {symbol}: {str(e)}"

# 定义获取财务比率的函数
def get_financial_ratios(symbol: str) -> str:
    """获取给定股票的关键财务比率"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield = f"{dividend_yield * 100:.2f}%"
        return f"{symbol} financial ratios: P/E: {pe_ratio}, P/B: {pb_ratio}, Dividend Yield: {dividend_yield}"
    except Exception as e:
        logging.error(f"Error fetching financial ratios for {symbol}: {str(e)}")
        return f"Error fetching financial ratios for {symbol}: {str(e)}"

# 定义获取分析师推荐的函数
def get_analyst_recommendations(symbol: str) -> str:
    """获取分析师对给定股票的推荐"""
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            latest_rec = recommendations.iloc[-1]
            return f"Latest analyst recommendation for {symbol}: {latest_rec['To Grade']} as of {latest_rec.name.date()}"
        else:
            return f"No analyst recommendations available for {symbol}"
    except Exception as e:
        logging.error(f"Error fetching analyst recommendations for {symbol}: {str(e)}")
        return f"Unable to fetch analyst recommendations for {symbol} due to an error: {str(e)}"

# 定义获取最新新闻的函数
def get_recent_news(symbol: str) -> str:
    """获取与给定股票相关的最新新闻"""
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        if news:
            latest_news = news[0]
            return f"Latest news for {symbol}: {latest_news['title']} - {latest_news['link']}"
        else:
            return f"No recent news available for {symbol}"
    except Exception as e:
        logging.error(f"Error fetching recent news for {symbol}: {str(e)}")
        return f"Error fetching recent news for {symbol}: {str(e)}"

# 定义获取行业比较的函数
def get_industry_comparison(symbol: str) -> str:
    """获取股票与行业平均水平的比较"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        pe_ratio = info.get('trailingPE', 'N/A')
        industry_pe = info.get('industryPE', 'N/A')

        comparison = f"{symbol} is in the {sector} sector, specifically in the {industry} industry. "
        if pe_ratio != 'N/A' and industry_pe != 'N/A':
            if pe_ratio < industry_pe:
                comparison += f"Its P/E ratio ({pe_ratio:.2f}) is lower than the industry average ({industry_pe:.2f}), which could indicate it's undervalued compared to its peers."
            elif pe_ratio > industry_pe:
                comparison += f"Its P/E ratio ({pe_ratio:.2f}) is higher than the industry average ({industry_pe:.2f}), which could indicate it's overvalued compared to its peers."
            else:
                comparison += f"Its P/E ratio ({pe_ratio:.2f}) is in line with the industry average ({industry_pe:.2f})."
        else:
            comparison += "Unable to compare P/E ratio with industry average due to lack of data."

        return comparison
    except Exception as e:
        logging.error(f"Error fetching industry comparison for {symbol}: {str(e)}")
        return f"Unable to fetch industry comparison for {symbol} due to an error: {str(e)}"

# 创建工具对象,每个工具对应一个函数
stock_price_tool = FunctionTool.from_defaults(fn=get_stock_price)
company_info_tool = FunctionTool.from_defaults(fn=get_company_info)
financial_ratios_tool = FunctionTool.from_defaults(fn=get_financial_ratios)
analyst_recommendations_tool = FunctionTool.from_defaults(fn=get_analyst_recommendations)
recent_news_tool = FunctionTool.from_defaults(fn=get_recent_news)
industry_comparison_tool = FunctionTool.from_defaults(fn=get_industry_comparison)

# 指定使用的OpenAI模型
llm = OpenAI(model="gpt-3.5-turbo",
             temperature=0,
             api_base=api_base
             )


# 创建消息队列
message_queue = SimpleMessageQueue()

# 创建工具服务,包含所有定义的工具
tool_service = ToolService(
    message_queue=message_queue,
    tools=[stock_price_tool, company_info_tool, financial_ratios_tool, analyst_recommendations_tool, recent_news_tool,
           industry_comparison_tool],
    running=True,
    step_interval=0.5,
)

# 创建控制平面服务器
control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=llm),
)

# 创建元工具列表,每个元工具对应一个实际工具
meta_tools = [
    MetaServiceTool(
        tool_metadata=tool.metadata,
        message_queue=message_queue,
        tool_service_name=tool_service.service_name,
    ) for tool in
    [stock_price_tool, company_info_tool, financial_ratios_tool, analyst_recommendations_tool, recent_news_tool,
     industry_comparison_tool]
]

# 创建代理工作器,设置系统提示
worker1 = FunctionCallingAgentWorker.from_tools(
    meta_tools,
    llm=llm,
    system_prompt="""你是一个专业的股票分析师。你的任务是分析给定的股票,并根据所有可用信息提供是否购买的建议。
    请使用所有可用工具来收集相关信息,然后给出全面的分析和明确的建议。
    考虑当前价格、公司信息、财务比率、分析师推荐、最新新闻和行业比较。
    解释你的推荐理由,并提供一个清晰的"买入"、"持有"或"卖出"建议。
    如果某些信息无法获取,请在分析中说明,并基于可用信息给出最佳判断。
    """
)

# 将工作器转换为代理
agent1 = worker1.as_agent()

# 创建代理服务
agent_server_1 = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for analyzing stocks and providing investment recommendations.",
    service_name="stock_analysis_agent",
)

# 创建本地启动器
launcher = LocalLauncher(
    [agent_server_1, tool_service],
    control_plane,
    message_queue,
)

# 执行股票分析
result = launcher.launch_single("""
分析 AAPL 股票是否值得购买。
请考虑以下因素:
1. 当前股价
2. 公司基本信息
3. 关键财务比率（如 P/E、P/B、股息收益率）
4. 分析师推荐
5. 最新相关新闻
6. 与行业平均水平的比较
根据这些信息，给出你的投资建议（买入、持有或卖出）并详细解释理由。
如果某些信息无法获取，请在分析中说明，并基于可用信息给出最佳判断。
""")

# 打印分析结果
print(f"Result: {result}")