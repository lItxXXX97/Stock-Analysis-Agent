import os
import logging
from dotenv import load_dotenv

# 导入必要的 llama_index 模块
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
# 导入 llama_agents 相关模块
from llama_agents import (
    AgentService,
    ToolService,
    LocalLauncher,
    MetaServiceTool,
    ControlPlaneServer,
    SimpleMessageQueue,
    AgentOrchestrator,
)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.llms.openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()
api_base = "https://api.aigc369.com/v1"
# 从环境变量中获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")

# 确保 API 密钥已设置,否则抛出异常
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# 设置 OpenAI API 密钥为环境变量
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = api_base
# 设置 llama_agents 的日志级别为 INFO
logging.getLogger("llama_agents").setLevel(logging.INFO)

# 加载并索引数据
def load_and_index_data():
    try:
        # 尝试从已保存的存储中加载索引
        storage_context = StorageContext.from_defaults(persist_dir="./storage/lyft")
        lyft_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(persist_dir="./storage/uber")
        uber_index = load_index_from_storage(storage_context)
    except:
        # 如果索引不存在,则创建新的索引
        lyft_docs = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"]).load_data()
        uber_docs = SimpleDirectoryReader(input_files=["./data/10k/uber_2021.pdf"]).load_data()

        lyft_index = VectorStoreIndex.from_documents(lyft_docs)
        uber_index = VectorStoreIndex.from_documents(uber_docs)

        # 保存新创建的索引
        lyft_index.storage_context.persist(persist_dir="./storage/lyft")
        uber_index.storage_context.persist(persist_dir="./storage/uber")

    return lyft_index, uber_index

# 设置查询引擎和工具
def setup_query_engines_and_tools(lyft_index, uber_index):
    # 创建 Lyft 和 Uber 的查询引擎
    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)

    # 创建查询引擎工具列表
    query_engine_tools = [
        QueryEngineTool(
            query_engine=lyft_engine,
            metadata=ToolMetadata(
                name="lyft_10k",
                description="Provides information about Lyft financials for year 2021. "
                            "Use a detailed plain text question as input to the tool.",
            ),
        ),
        QueryEngineTool(
            query_engine=uber_engine,
            metadata=ToolMetadata(
                name="uber_10k",
                description="Provides information about Uber financials for year 2021. "
                            "Use a detailed plain text question as input to the tool.",
            ),
        ),
    ]

    return query_engine_tools

# 设置代理和服务
async def setup_agents_and_services(query_engine_tools):
    # 创建消息队列
    message_queue = SimpleMessageQueue()
    # 创建控制平面服务器
    control_plane = ControlPlaneServer(
        message_queue=message_queue,
        orchestrator=AgentOrchestrator(llm=OpenAI(model="gpt-3.5-turbo",
                                                  api_base=api_base)),
    )

    # 创建工具服务
    tool_service = ToolService(
        message_queue=message_queue,
        tools=query_engine_tools,
        running=True,
        step_interval=0.5,
    )

    # 创建元工具列表
    meta_tools = [
        await MetaServiceTool.from_tool_service(
            t.metadata.name,
            message_queue=message_queue,
            tool_service=tool_service,
        )
        for t in query_engine_tools
    ]

    # 创建函数调用代理工作器
    worker1 = FunctionCallingAgentWorker.from_tools(
        meta_tools,
        llm=OpenAI(),
    )
    # 将工作器转换为代理
    agent1 = worker1.as_agent()
    # 创建代理服务
    agent_server_1 = AgentService(
        agent=agent1,
        message_queue=message_queue,
        description="Used to answer questions over Uber and Lyft 10K documents",
        service_name="uber_lyft_10k_analyst_agent",
    )

    # 创建本地启动器
    launcher = LocalLauncher(
        [agent_server_1, tool_service],
        control_plane,
        message_queue,
    )

    return launcher

# 主函数,用于运行整个脚本
async def main():
    # 加载并索引数据
    lyft_index, uber_index = load_and_index_data()
    # 设置查询引擎和工具
    query_engine_tools = setup_query_engines_and_tools(lyft_index, uber_index)
    # 设置代理和服务
    launcher = await setup_agents_and_services(query_engine_tools)

    # 示例查询
    queries = [
        "What are the risk factors for Uber?",
        "What was Lyft's revenue growth in 2021?",
    ]

    # 执行查询并打印结果
    for query in queries:
        print(f"Query: {query}")
        result = await launcher.alaunch_single(query)  # 使用 alaunch_single 而不是 launch_single
        print(f"Result: {result}\n")

# 运行主函数
if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()