"""Deterministic intent routing baseline for PhotoCoach."""

from .models import Intent, RouteRequest, RouteResult


class IntentRouter:
    """Route a normalized request to one supported photography intent."""

    GENERATION_KEYWORDS = (
        "生成",
        "创作",
        "画一张",
        "参考图",
        "生成图片",
    )
    EDITING_KEYWORDS = (
        "修改",
        "换背景",
        "背景换成",
        "天空换成",
        "替换背景",
        "调色",
        "胶片风格",
        "裁剪",
        "裁成",
        "去掉",
        "移除",
    )
    AMBIGUOUS_EDITING_KEYWORDS = (
        "优化这张",
        "处理一下这张",
        "修一下这张",
    )
    ANALYSIS_KEYWORDS = (
        "分析",
        "构图",
        "曝光",
        "光线",
        "脸很暗",
        "脸比较暗",
        "脸暗",
        "噪点",
        "清晰度",
        "高光",
        "阴影",
        "主体",
        "视觉引导线",
        "适合发到小红书",
        "拍摄参数",
        "读取参数",
        "EXIF",
    )
    PLANNING_KEYWORDS = (
        "拍摄计划",
        "拍摄方案",
        "去哪拍",
        "地点",
        "机位",
        "天气",
        "日落",
        "时间段",
        "小红书推荐",
        "网友推荐",
        "大家推荐",
        "适合拍夜景",
        "设计方案",
        "三套",
        "拍一组",
        "方案",
    )
    SENSITIVE_INFERENCE_KEYWORDS = (
        "真实身份",
        "年龄",
        "国籍",
        "性别",
    )
    MISSING_IMAGE_REFERENCE_KEYWORDS = (
        "这张照片",
        "这张图片",
        "这张图",
    )
    OUT_OF_SCOPE_KEYWORDS = (
        "写简历",
        "求职简历",
        "写代码",
        "股票",
        "法律咨询",
    )

    def route(self, request: RouteRequest) -> RouteResult:
        text = request.text.strip()

        # 1. 图片没有文字：目标不明确，先澄清。
        if request.image and not text:
            return RouteResult(
                intent=Intent.CLARIFICATION,
                confidence=0.99,
                should_clarify=True,
                reason="用户上传了图片，但没有说明希望完成什么任务",
                matched_rules=["image_only_without_instruction"],
            )

        # Safety routing: keep the task in photo analysis so a later policy
        # layer can refuse sensitive identity/attribute inference.
        if request.image and self._contains(text, self.SENSITIVE_INFERENCE_KEYWORDS):
            return RouteResult(
                intent=Intent.PHOTO_ANALYSIS,
                confidence=0.99,
                should_clarify=False,
                reason="图片分析请求包含敏感身份或属性推断，回答层必须拒绝该部分",
                matched_rules=["sensitive_inference_guard"],
            )

        # 2. 明确生成请求优先于图片分析，支持参考图生成。
        if self._contains(text, self.GENERATION_KEYWORDS):
            return RouteResult(
                intent=Intent.IMAGE_GENERATION,
                confidence=0.95,
                should_clarify=False,
                reason="用户明确要求生成新图片",
                matched_rules=["explicit_image_generation"],
            )

        # 3. 模糊的“优化”先澄清，避免误调用编辑工具。
        if self._contains(text, self.AMBIGUOUS_EDITING_KEYWORDS):
            return RouteResult(
                intent=Intent.CLARIFICATION,
                confidence=0.92,
                should_clarify=True,
                reason="用户表达了图片优化意图，但没有明确要分析还是直接编辑",
                matched_rules=["ambiguous_image_editing"],
            )

        # 4. 明确编辑请求必须带有原图。
        if self._contains(text, self.EDITING_KEYWORDS):
            if request.image:
                return RouteResult(
                    intent=Intent.IMAGE_EDITING,
                    confidence=0.95,
                    should_clarify=False,
                    reason="用户提供了原图，并明确要求修改图片",
                    matched_rules=["explicit_image_editing"],
                )
            return RouteResult(
                intent=Intent.CLARIFICATION,
                confidence=0.96,
                should_clarify=True,
                reason="用户要求编辑图片，但没有提供原图",
                matched_rules=["image_editing_missing_image"],
            )

        # 5. 有图片且询问视觉问题：图片分析。
        if request.image and self._contains(text, self.ANALYSIS_KEYWORDS):
            return RouteResult(
                intent=Intent.PHOTO_ANALYSIS,
                confidence=0.90,
                should_clarify=False,
                reason="用户提供了图片，并询问图片中的摄影问题",
                matched_rules=["image_analysis_keyword"],
            )

        # 6. 地点、天气、机位和社区推荐：拍摄计划。
        if self._contains(text, self.PLANNING_KEYWORDS):
            return RouteResult(
                intent=Intent.SHOOT_PLANNING,
                confidence=0.88,
                should_clarify=False,
                reason="用户需要制定拍摄计划或查询拍摄地点信息",
                matched_rules=["shoot_planning_keyword"],
            )

        # 7. 空请求或明显越界请求：澄清并引导回摄影范围。
        if not text or self._contains(text, self.OUT_OF_SCOPE_KEYWORDS):
            return RouteResult(
                intent=Intent.CLARIFICATION,
                confidence=0.90,
                should_clarify=True,
                reason="请求为空或不属于当前摄影助手范围",
                matched_rules=["empty_or_out_of_scope"],
            )

        if not request.image and self._contains(
            text, self.MISSING_IMAGE_REFERENCE_KEYWORDS
        ):
            return RouteResult(
                intent=Intent.CLARIFICATION,
                confidence=0.93,
                should_clarify=True,
                reason="用户引用了一张当前未提供的图片",
                matched_rules=["referenced_image_missing"],
            )

        # 8. 其余摄影相关文字问题暂时归入统一入口。
        return RouteResult(
            intent=Intent.PHOTO_QUESTION,
            confidence=0.50,
            should_clarify=False,
            reason="暂未命中特定规则，使用摄影问题默认路由",
            matched_rules=["default_photo_question"],
        )

    @staticmethod
    def _contains(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)


def route_message(
    text: str = "",
    image: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> RouteResult:
    """Convenience wrapper used by the future agent loop."""

    request = RouteRequest(text=text, image=image, history=history or [])
    return IntentRouter().route(request)
