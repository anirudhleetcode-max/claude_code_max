"""Permission policy: decides ALLOW / ASK / DENY for each tool call."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import StrEnum

from ..config.settings import Mode, PermissionLevel, PermissionsSettings
from ..security.command_risk import Risk
from .base import ActionAssessment


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: str
    level: PermissionLevel


MODE_CEILING: dict[Mode, PermissionLevel] = {
    Mode.SAFE: PermissionLevel.READ_ONLY,
    Mode.ASSISTED: PermissionLevel.PRIVILEGED,
    Mode.DEVELOPER: PermissionLevel.PRIVILEGED,
    Mode.AUTONOMOUS: PermissionLevel.PRIVILEGED,
}


def level_for_risk(risk: Risk, read_only: bool) -> PermissionLevel:
    if risk >= Risk.HIGH:
        return PermissionLevel.PRIVILEGED
    if risk == Risk.LOW and read_only:
        return PermissionLevel.READ_ONLY
    return PermissionLevel.DEVELOPMENT


def _matches(command: str, patterns: list[str]) -> str | None:
    normalized = " ".join(command.split())
    for pattern in patterns:
        if fnmatch.fnmatchcase(normalized, pattern) or fnmatch.fnmatchcase(command, pattern):
            return pattern
    return None


class PermissionPolicy:
    def __init__(self, settings: PermissionsSettings) -> None:
        self.settings = settings

    @property
    def mode(self) -> Mode:
        return self.settings.mode

    @property
    def ceiling(self) -> PermissionLevel:
        return min(MODE_CEILING[self.settings.mode], self.settings.max_level)

    def evaluate(self, a: ActionAssessment) -> PolicyDecision:
        level = a.level
        if a.risk is not None:
            level = max(level, level_for_risk(a.risk, a.read_only))
        if a.command:
            denied = _matches(a.command, self.settings.deny_commands)
            if denied:
                return PolicyDecision(Decision.DENY, f"command matches deny_commands pattern '{denied}'", level)
        if a.risk == Risk.CRITICAL:
            return PolicyDecision(Decision.DENY, "critical-risk action is never permitted", level)
        ceiling = self.ceiling
        if level > ceiling:
            return PolicyDecision(
                Decision.DENY,
                f"requires {level.name} but mode '{self.mode}' allows at most {ceiling.name}",
                level,
            )
        if a.command:
            allowed = _matches(a.command, self.settings.allow_commands)
            if allowed:
                return PolicyDecision(Decision.ALLOW, f"pre-approved by allow_commands pattern '{allowed}'", level)
        if level >= PermissionLevel.PRIVILEGED:
            return PolicyDecision(Decision.ASK, "high-risk action requires approval", level)
        if self.mode == Mode.ASSISTED and level >= PermissionLevel.SAFE_WRITE:
            return PolicyDecision(Decision.ASK, "assisted mode: writes and commands require approval", level)
        return PolicyDecision(Decision.ALLOW, "within policy", level)
