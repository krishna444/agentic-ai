import logging
from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from app.config import settings

logger= logging.getLogger(__name__)

class PromptService:
    def __init__(self):
        logger.info(f"Initializing models")
        model=init_chat_model(
            settings.GROQ_MODEL,
            model_provider=settings.GROQ_PROVIDER,
            temperature=0.5,
            timeout=600,
            max_tokens=5500,
            streaming=True,
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL
        )
        
        self.llm=model
        self.prompt=ChatPromptTemplate.from_messages([
            ("system",settings.SYSTEM),
            ("human",settings.HUMAN)            
        ])
        
        self.chain=self.prompt | self.llm| StrOutputParser()
    
    async def execute(self, inputs:Dict[str, Any])->str:
        try:
            logger.info("Executing generation chain")
            response=await self.chain.ainvoke(inputs)
            return response
        except Exception as e:
            logger.critical(f"Failed execution: {str(e)}", exc_info=True)
            raise RuntimeError("Fatal degradation of AI pipeline across multiple provider networks.") from e