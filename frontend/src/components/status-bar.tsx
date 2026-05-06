import React from "react";
import { Box, Text } from "ink";
import { useAgentStore } from "../store/agent-store.js";

const PHASE_LABELS: Record<string, string> = {
  idle: "Idle",
  explorer: "Explorer",
  strategy: "Strategy",
  planner: "Planner",
  executor: "Executor",
  evaluator: "Evaluator",
  learner: "Learner",
};

function ContextGauge({ used, max }: { used: number; max: number }): React.ReactElement {
  const pct = Math.min(100, Math.round((used / max) * 100));
  const color = pct >= 85 ? "red" : pct >= 60 ? "yellow" : "green";
  const filled = Math.round(pct / 10);
  const bar = "●".repeat(filled) + "○".repeat(10 - filled);
  return (
    <Text>
      <Text color={color}>{bar}</Text>
      <Text dimColor> ctx </Text>
      <Text color={color} bold>{pct}%</Text>
    </Text>
  );
}

interface StatusBarProps {
  isReplay?: boolean;
}

export function StatusBar({ isReplay }: StatusBarProps): React.ReactElement {
  const currentPhase = useAgentStore((s) => s.currentPhase);
  const phases = useAgentStore((s) => s.phases);
  const attempt = useAgentStore((s) => s.attempt);
  const error = useAgentStore((s) => s.error);
  const contextUsage = useAgentStore((s) => s.contextUsage);

  const phase = phases[currentPhase];
  const progressStr = phase?.progress
    ? ` ${phase.progress.current}/${phase.progress.total}`
    : "";

  const attemptStr = attempt > 0 ? `  Attempt ${attempt + 1}` : "";

  return (
    <Box
      borderStyle="single"
      borderColor="gray"
      paddingX={1}
      justifyContent="space-between"
      width="100%"
    >
      <Box>
        {error ? (
          <Text color="red" bold>
            Error: {error}
          </Text>
        ) : (
          <Text>
            {isReplay && <Text color="magenta" bold>[Replay] </Text>}
            <Text color="yellow" bold>
              [{PHASE_LABELS[currentPhase] ?? currentPhase}]
            </Text>
            <Text dimColor>{progressStr}</Text>
            {phase?.summary && <Text dimColor>{" — "}{phase.summary}</Text>}
            <Text dimColor>{attemptStr}</Text>
          </Text>
        )}
      </Box>
      <Box gap={2}>
        {contextUsage && (currentPhase === "explorer" || currentPhase === "learner") && (
          <ContextGauge used={contextUsage.used} max={contextUsage.max} />
        )}
        <Text dimColor>Ctrl+C: exit</Text>
      </Box>
    </Box>
  );
}
