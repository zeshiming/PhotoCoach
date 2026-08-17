"""PhotoCoach 主运行时使用的 Pydantic 请求模型。"""

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """一次 Agent 运行的标准输入。

    使用 Pydantic 后，入口数据会在进入 Agents SDK 前完成基本类型校验。
    """

    text: str = ""
    images: list[str] = Field(default_factory=list)
    session_id: str | None = None

    @property
    def has_image(self) -> bool:
        return bool(self.images)
