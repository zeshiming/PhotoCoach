"""用于扩展 PhotoCoach 路由的能力注册表和工具注册表。

注册表故意和具体 LLM Provider 解耦。
未来的召回模块可以索引 ``retrieval_text`` 和工具描述，
而不需要修改任务理解的数据契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import CapabilitySpec, RiskLevel, ToolSpec


@dataclass
class CapabilityRegistry:
    """应用已知能力的内存注册表。

    生产环境中可以替换成配置文件或带索引的存储。
    路由器只依赖当前这个小型注册表接口。
    """

    _items: dict[str, CapabilitySpec] = field(default_factory=dict)

    def register(self, capability: CapabilitySpec) -> None:
        # register 严格禁止重名。
        # 这样启动时就能发现配置错误，而不是静默覆盖能力契约。
        if capability.name in self._items:
            raise ValueError(f"capability already registered: {capability.name}")
        self._items[capability.name] = capability

    def upsert(self, capability: CapabilitySpec) -> None:
        self._items[capability.name] = capability

    def get(self, name: str) -> CapabilitySpec | None:
        return self._items.get(name)

    def require(self, name: str) -> CapabilitySpec:
        capability = self.get(name)
        if capability is None:
            raise KeyError(f"unknown capability: {name}")
        return capability

    def list(self) -> list[CapabilitySpec]:
        return sorted(self._items.values(), key=lambda item: item.name)

    def names(self) -> set[str]:
        return set(self._items)


@dataclass
class ToolRegistry:
    """用于发现能力具体实现的工具注册表。

    当前只做元数据查询。
    未来执行器会使用 ToolSpec 校验参数，再调用真正的工具函数。
    """

    _items: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._items:
            raise ValueError(f"tool already registered: {tool.name}")
        self._items[tool.name] = tool

    def upsert(self, tool: ToolSpec) -> None:
        self._items[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._items.get(name)

    def require(self, name: str) -> ToolSpec:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool

    def list(self, *, enabled_only: bool = False) -> list[ToolSpec]:
        tools = self._items.values()
        if enabled_only:
            tools = (tool for tool in tools if tool.enabled)
        return sorted(tools, key=lambda item: item.name)

    def for_capability(
        self, capability: str, *, enabled_only: bool = True
    ) -> list[ToolSpec]:
        # 一个能力可以有多个实现。
        # 后续路由阶段可以按照可用性、成本、延迟或质量进行排序。
        return [
            tool
            for tool in self.list(enabled_only=enabled_only)
            if capability in tool.provides
        ]


def build_default_capability_registry() -> CapabilityRegistry:
    """Create the initial open-ended capability vocabulary.

    These are capability contracts, not a closed list of user intents.  New
    capabilities can be added without changing the task-understanding model.
    """

    registry = CapabilityRegistry()
    # 这是开放的能力词表，不是固定的 Intent 枚举。
    # 当需求对应不同工具、权限或执行流程时，再新增能力。
    capabilities = [
        CapabilitySpec(
            name="photo_knowledge_qa",
            description="回答摄影知识、器材和拍摄技巧问题",
        ),
        CapabilitySpec(
            name="image_understanding",
            description="理解图片中的主体、场景、构图和光线",
            requires_image=True,
            retrieval_text="图片理解、视觉分析、主体、场景、构图、光线",
        ),
        CapabilitySpec(
            name="exposure_analysis",
            description="分析曝光、明暗、逆光和动态范围问题",
            requires_image=True,
            retrieval_text="曝光分析、脸部太暗、过曝、欠曝、光比、逆光",
        ),
        CapabilitySpec(
            name="composition_analysis",
            description="分析构图、主体位置、平衡和视觉引导",
            requires_image=True,
            retrieval_text="构图分析、主体位置、画面平衡、视觉引导",
        ),
        CapabilitySpec(
            name="shoot_planning",
            description="设计拍摄方案、时间、地点、机位和参数建议",
            retrieval_text="拍摄计划、人像方案、时间地点、机位、参数",
        ),
        CapabilitySpec(
            name="web_search",
            description="查询外部网站或平台上的最新信息",
            risk_level=RiskLevel.MEDIUM,
            retrieval_text="联网搜索、网页查询、最新推荐、网友评价",
        ),
        CapabilitySpec(
            name="location_recommendation",
            description="根据拍摄目标筛选和比较拍摄地点",
            retrieval_text="拍摄地点推荐、日落、人像、夜景、地点比较",
        ),
        CapabilitySpec(
            name="image_generation",
            description="根据文字描述生成新图片",
            risk_level=RiskLevel.MEDIUM,
            retrieval_text="生成图片、文生图、生成一张照片",
        ),
        CapabilitySpec(
            name="image_editing",
            description="根据指令修改已有图片",
            requires_image=True,
            risk_level=RiskLevel.MEDIUM,
            retrieval_text="图片编辑、换背景、调色、删除物体、胶片风格",
        ),
    ]
    for capability in capabilities:
        registry.register(capability)
    return registry
