import React, { useState, useMemo } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import type { ConfirmRequest } from "../store/types.js";

interface ConfirmDialogProps {
  request: ConfirmRequest;
  onApprove: (id: number) => void;
  onDeny: (id: number, comments: string) => void;
}

type Mode = "choice" | "commenting";

export function ConfirmDialog({
  request,
  onApprove,
  onDeny,
}: ConfirmDialogProps): React.ReactElement {
  const [mode, setMode] = useState<Mode>("choice");
  const [selected, setSelected] = useState(0);
  const [comment, setComment] = useState("");
  const [scrollOffset, setScrollOffset] = useState(0);
  const { stdout } = useStdout();
  const terminalHeight = stdout?.rows ?? 40;
  const previewHeight = Math.max(5, terminalHeight - 12);

  const previewLines = useMemo(() => {
    return buildPreviewLines(request);
  }, [request]);

  const maxScroll = Math.max(0, previewLines.length - previewHeight);

  const options = ["Approve", "Deny with comments", "Cancel"];

  useInput((input, key) => {
    if (mode === "choice") {
      if (key.leftArrow) {
        setSelected((s) => Math.max(0, s - 1));
      } else if (key.rightArrow) {
        setSelected((s) => Math.min(options.length - 1, s + 1));
      } else if (key.upArrow) {
        setScrollOffset((s) => Math.max(0, s - 1));
      } else if (key.downArrow) {
        setScrollOffset((s) => Math.min(maxScroll, s + 1));
      } else if (input === "j") {
        setScrollOffset((s) => Math.min(maxScroll, s + 3));
      } else if (input === "k") {
        setScrollOffset((s) => Math.max(0, s - 3));
      } else if (key.return) {
        if (selected === 0) {
          onApprove(request.id);
        } else if (selected === 1) {
          setMode("commenting");
        } else {
          onDeny(request.id, "");
        }
      }
    } else if (mode === "commenting") {
      if (key.return) {
        onDeny(request.id, comment);
      } else if (key.escape) {
        setMode("choice");
        setComment("");
      } else if (key.backspace || key.delete) {
        setComment((c) => c.slice(0, -1));
      } else if (input && !key.ctrl && !key.meta) {
        setComment((c) => c + input);
      }
    }
  });

  const title =
    request.type === "program"
      ? "Confirm Program"
      : request.type === "criteria"
        ? "Confirm Success Criteria"
        : "Confirm Skills";

  const visibleLines = previewLines.slice(scrollOffset, scrollOffset + previewHeight);
  const scrollIndicator = previewLines.length > previewHeight
    ? ` [${scrollOffset + 1}-${Math.min(scrollOffset + previewHeight, previewLines.length)}/${previewLines.length}]`
    : "";

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1}>
      <Box>
        <Text bold color="yellow">{title}</Text>
        <Text dimColor>{scrollIndicator}</Text>
      </Box>

      <Box flexDirection="column" flexGrow={1} overflow="hidden" marginTop={1}>
        {visibleLines.map((line, i) => (
          <Text key={i} wrap="truncate">{line}</Text>
        ))}
      </Box>

      {mode === "choice" && (
        <Box marginTop={1}>
          {options.map((opt, i) => (
            <Box key={i} marginRight={2}>
              <Text color={i === selected ? "cyan" : undefined} bold={i === selected}>
                {i === selected ? "[" : " "}
                {opt}
                {i === selected ? "]" : " "}
              </Text>
            </Box>
          ))}
          <Text dimColor>  |  ←→: select  ↑↓/jk: scroll  Enter: confirm</Text>
        </Box>
      )}

      {mode === "commenting" && (
        <Box flexDirection="column" marginTop={1}>
          <Text>Comments (Enter to submit, Esc to go back):</Text>
          <Box>
            <Text color="cyan">{"> "}</Text>
            <Text>{comment}</Text>
            <Text color="cyan">█</Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}

function buildPreviewLines(request: ConfirmRequest): string[] {
  if (request.type === "program") {
    return buildProgramLines(request.payload);
  } else if (request.type === "criteria") {
    return buildCriteriaLines(request.payload);
  } else {
    return buildSkillsLines(request.payload);
  }
}

function buildProgramLines(payload: unknown): string[] {
  const data = payload as Record<string, unknown>;
  const program = data?.program as Record<string, unknown> | undefined;
  const lines: string[] = [];

  if (!program) {
    const raw = JSON.stringify(data, null, 2);
    return raw.split("\n");
  }

  if (program.name) lines.push(`Program: ${program.name}`);
  if (program.description) {
    lines.push(`${program.description}`);
  }
  lines.push("");

  const steps = (program.steps as Array<Record<string, unknown>>) ?? [];
  const code = program.code as string | undefined;

  if (steps.length > 0) {
    lines.push("Steps:");
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const prefix = `  ${(i + 1).toString().padStart(2)}. `;
      let line = prefix + String(step.skill ?? "");
      if (typeof step.description === "string") {
        line += ` - ${step.description}`;
      }
      lines.push(line);
      if (typeof step.inputs === "object" && step.inputs !== null && Object.keys(step.inputs).length > 0) {
        lines.push(`      ${formatInputs(step.inputs as Record<string, unknown>)}`);
      }
    }
  }

  if (code) {
    lines.push("Code:");
    const codeLines = code.split("\n");
    for (const cl of codeLines) {
      lines.push(`  ${cl}`);
    }
  }

  return lines;
}

function buildCriteriaLines(payload: unknown): string[] {
  const data = payload as Record<string, unknown>;
  const criteria = (data?.criteria as Array<Record<string, unknown>>) ?? [];
  const goalSummary = data?.goal_summary as string | undefined;
  const mustPassAll = data?.must_pass_all as boolean | undefined;
  const lines: string[] = [];

  if (!criteria.length && !goalSummary) {
    return JSON.stringify(data, null, 2).split("\n");
  }

  if (goalSummary) {
    lines.push(`Goal: ${goalSummary}`);
    lines.push("");
  }
  lines.push(`Success Criteria${mustPassAll === false ? " (any one must pass)" : " (all must pass)"}:`);
  for (const c of criteria) {
    lines.push(`  [${c.kind}] ${c.description}`);
    if (typeof c.visual_claim === "string") {
      lines.push(`    claim: "${c.visual_claim}"`);
    }
  }

  return lines;
}

function buildSkillsLines(payload: unknown): string[] {
  const data = payload as Record<string, unknown>;
  const skills = (data?.skills as Array<Record<string, unknown>>) ?? [];
  const lines: string[] = [];

  if (skills.length === 0) {
    if (data && Object.keys(data).length > 0) {
      return JSON.stringify(data, null, 2).split("\n");
    }
    lines.push("No skills proposed");
    return lines;
  }

  lines.push("Proposed New Skills:");
  for (const s of skills) {
    lines.push(`  - ${(s.proposed_id ?? s.name ?? "unknown") as string}`);
    if (typeof s.description === "string") {
      lines.push(`    ${s.description}`);
    }
  }

  return lines;
}

function formatInputs(inputs: Record<string, unknown>): string {
  const entries = Object.entries(inputs).slice(0, 4);
  const parts = entries.map(([k, v]) => {
    if (typeof v === "string") return `${k}="${truncateStr(v, 25)}"`;
    if (typeof v === "number" || typeof v === "boolean") return `${k}=${v}`;
    return `${k}=…`;
  });
  if (Object.keys(inputs).length > 4) parts.push("…");
  return parts.join(", ");
}

function truncateStr(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "…";
}
