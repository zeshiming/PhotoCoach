"""PhotoCoach 能力注册表。

Capability 描述用户目标，Tool 是具体执行函数。一个 Capability 可以不需要
Tool（由 Agent 直接回答），也可以映射到一个或多个 Tool。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    description: str
    tools: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    risk_level: str = "low"


@dataclass
class CapabilityRegistry:
    specs: dict[str, CapabilitySpec] = field(default_factory=dict)

    def register(self, spec: CapabilitySpec) -> None:
        self.specs[spec.id] = spec

    def get(self, capability_id: str) -> CapabilitySpec | None:
        return self.specs.get(capability_id)

    def validate(self, capability_ids: list[str]) -> list[str]:
        """过滤未知能力并保持模型返回顺序。"""

        return [capability_id for capability_id in capability_ids if capability_id in self.specs]

    def tool_names_for(self, capability_ids: list[str]) -> set[str]:
        names: set[str] = set()
        for capability_id in self.validate(capability_ids):
            names.update(self.specs[capability_id].tools)
        return names

    def capability_for_tool(self, tool_name: str) -> str | None:
        """返回一个 Tool 的主能力，用于执行前的错配检测。"""

        for spec in self.specs.values():
            if tool_name in spec.tools:
                return spec.id
        return None

    def prompt_text(self) -> str:
        lines = ["可选择的 Capability（只能从以下 ID 中选择）："]
        for spec in self.specs.values():
            examples = "；".join(spec.examples) or "无示例"
            lines.append(
                f"- {spec.id}: {spec.description}；示例：{examples}；"
                f"工具：{', '.join(spec.tools) or '无（直接回答）'}"
            )
        return "\n".join(lines)

    def infer(self, text: str) -> list[str]:
        """语义模型未返回能力时的保守本地兜底，不代替模型判断。"""

        value = text.lower()
        if any(item in value for item in ("exif", "相机参数", "尺寸", "格式", "快门", "光圈", "iso")):
            return ["read_metadata"]
        if any(item in value for item in ("照片", "图片", "构图", "光线", "曝光", "虚化")):
            return ["analyze_image"]
        if any(item in value for item in ("怎么拍", "如何拍", "摄影", "镜头")):
            return ["explain_photography"]
        return []


def build_default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        CapabilitySpec(
            id="analyze_image",
            description="分析图片的构图、光线、曝光、色彩和主体关系",
            tools=("analyze_current_image",),
            examples=("分析这张照片的构图", "为什么人物脸部很暗"),
            required_inputs=("image",),
        )
    )
    registry.register(
        CapabilitySpec(
            id="read_metadata",
            description="读取图片尺寸、格式、EXIF 和相机参数",
            tools=("read_image_metadata",),
            examples=("这张照片的 ISO 是多少", "图片尺寸是多少"),
            required_inputs=("image",),
        )
    )
    registry.register(
        CapabilitySpec(
            id="explain_photography",
            description="解释摄影概念并给出拍摄建议",
            examples=("怎么拍出背景虚化", "35mm 适合拍人像吗"),
        )
    )
    registry.register(
        CapabilitySpec(
            id="shoot_planning",
            description="根据主题、地点和时间制定拍摄计划",
            examples=("帮我安排周末街拍计划",),
        )
    )
    registry.register(
        CapabilitySpec(
            id="photo_recommendation",
            description="提供摄影地点、器材和技巧推荐",
            examples=("推荐适合新手的人像镜头",),
        )
    )
    return registry


DEFAULT_CAPABILITY_REGISTRY = build_default_capability_registry()
