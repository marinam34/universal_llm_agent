from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FieldDescriptor(BaseModel):
    selector: str
    name: str | None = None
    input_type: str | None = None
    label: str | None = None
    placeholder: str | None = None
    required: bool = False
    options: list[str] = Field(default_factory=list)


class LinkDescriptor(BaseModel):
    text: str
    selector: str
    href: str | None = None
    title: str | None = None


class ButtonDescriptor(BaseModel):
    text: str
    selector: str
    button_type: str | None = None
    target_url: str | None = None
    title: str | None = None


class FormDescriptor(BaseModel):
    selector: str
    action: str | None = None
    method: str | None = None
    fields: list[FieldDescriptor] = Field(default_factory=list)
    buttons: list[ButtonDescriptor] = Field(default_factory=list)


class PageSnapshot(BaseModel):
    url: str
    title: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    links: list[LinkDescriptor] = Field(default_factory=list)
    buttons: list[ButtonDescriptor] = Field(default_factory=list)
    forms: list[FormDescriptor] = Field(default_factory=list)
    inputs: list[FieldDescriptor] = Field(default_factory=list)
    text_content: str | None = None
    text_excerpt: str | None = None
    visible_text_sample: list[str] = Field(default_factory=list)
    preview_image_url: str | None = None
    embedding: list[float] = Field(default_factory=list)


class TransitionEdge(BaseModel):
    from_url: str
    to_url: str
    trigger_text: str | None = None
    selector: str | None = None
    trigger_type: Literal["link", "button", "form", "navigation"] = "link"


class SiteKnowledgeBase(BaseModel):
    root_url: HttpUrl
    domain: str
    created_at: datetime = Field(default_factory=utcnow)
    pages: list[PageSnapshot] = Field(default_factory=list)
    transitions: list[TransitionEdge] = Field(default_factory=list)


ActionType = Literal["goto", "click", "type", "select", "wait_for"]


class ActionStep(BaseModel):
    action: ActionType
    url: str | None = None
    selector: str | None = None
    value: str | None = None
    option: str | None = None
    text: str | None = None
    reason: str | None = None
    expected_url_contains: str | None = None
    expected_selector: str | None = None
    expected_text: str | None = None


class ActionPlan(BaseModel):
    goal: str
    possible: bool
    missing_inputs: list[str] = Field(default_factory=list)
    steps: list[ActionStep] = Field(default_factory=list)
    stop_before: list[ActionStep] = Field(default_factory=list)
    explanation_for_user: str


class CrawlRequest(BaseModel):
    url: HttpUrl
    max_pages: int = Field(default=75, ge=1, le=300)


class CrawlResponse(BaseModel):
    kb_path: str
    pages_discovered: int
    transition_count: int


class CreateAgentResponse(BaseModel):
    agent_name: str
    site_url: str
    kb_path: str
    pages_discovered: int
    transition_count: int
    greeting: str
    preview_image_url: str | None = None
    live_preview_allowed: bool = True


class PlanRequest(BaseModel):
    url: HttpUrl
    user_goal: str
    user_inputs: dict[str, Any] = Field(default_factory=dict)


class ExecuteRequest(BaseModel):
    url: HttpUrl
    plan: ActionPlan
    user_inputs: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    url: HttpUrl
    message: str
    user_inputs: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    mode: Literal["answer", "action", "clarification"]
    reply: str
    collected_inputs: dict[str, Any] = Field(default_factory=dict)
    suggested_url: str | None = None
    preview_image_url: str | None = None
    live_preview_allowed: bool = True
    relevant_pages: list[str] = Field(default_factory=list)
    plan: ActionPlan | None = None
    execution: "ExecutionResult | None" = None
    plan_path: str | None = None
    log_path: str | None = None


class ExecutionStepLog(BaseModel):
    timestamp: datetime = Field(default_factory=utcnow)
    step: ActionStep
    status: Literal["pending", "success", "failed", "blocked"]
    detail: str
    current_url: str | None = None


class ExecutionResult(BaseModel):
    success: bool
    stopped_for_human: bool
    current_url: str | None = None
    preview_image_url: str | None = None
    logs: list[ExecutionStepLog] = Field(default_factory=list)
    log_path: str | None = None
