import asyncio
from app.prompt_service import PromptService
async def main() ->None:
    service=PromptService()
    payload={"topic": "python panda"}
    result=await service.execute(inputs=payload)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
