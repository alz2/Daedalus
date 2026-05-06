import React, { useState, useCallback, useRef } from "react";
import { resolve } from "node:path";
import { readFileSync, existsSync } from "node:fs";
import { Box, Text, useInput } from "ink";
import Spinner from "ink-spinner";
import { parse as parseYaml } from "yaml";
import { useAgentStore } from "./store/agent-store.js";
import { ProcessManager } from "./bridge/process-manager.js";
import { loadTrace, resolveTraceDir } from "./bridge/trace-loader.js";
import { loadState, pushGoal, setLastConfig } from "./store/persistence.js";
import { HeaderBar } from "./components/header-bar.js";
import { StatusBar } from "./components/status-bar.js";
import { GoalDisplay } from "./components/goal-display.js";
import { ChatFeed } from "./components/chat-feed.js";
import { ConfirmDialog } from "./components/confirm-dialog.js";
import { ConfigScreen } from "./components/config-screen.js";
import { TraceBrowser } from "./components/trace-browser.js";
import { useKeybindings } from "./hooks/use-keybindings.js";

const LOGO = `     ___                  __      __
    / _ \\ ___ _ ___  ___ / /___ _/ /__ __ ___
   / // // _ \`// -_)/ _ \\ / _ \`// // // /(_-<
  /____/ \\_,_/ \\__/ \\__/_/\\_,_//_/ \\_,_//___/`;

interface AppProps {
  goal?: string;
  configPath?: string;
  projectRoot: string;
  tracePath?: string;
}

export function App({ goal, configPath, projectRoot, tracePath }: AppProps): React.ReactElement {
  const [manager] = useState(() => new ProcessManager(projectRoot));
  const persistedState = useRef(loadState());
  const goalHistory = useRef(persistedState.current.goalHistory);

  const storeGoal = useAgentStore((s) => s.goal);
  const pendingConfirm = useAgentStore((s) => s.pendingConfirm);
  const showConfig = useAgentStore((s) => s.showConfig);
  const connected = useAgentStore((s) => s.connected);
  const currentPhase = useAgentStore((s) => s.currentPhase);
  const phases = useAgentStore((s) => s.phases);

  const [inputMode, setInputMode] = useState<"idle" | "goal">(goal || tracePath ? "idle" : "goal");
  const [goalInput, setGoalInput] = useState("");
  const [cursorPos, setCursorPos] = useState(0);
  const [historyIdx, setHistoryIdx] = useState(-1);
  const [showTraces, setShowTraces] = useState(false);
  const [isReplay, setIsReplay] = useState(false);

  // Input is "active" when user is typing (goal prompt or confirm comment)
  const inputActive = inputMode === "goal" && !storeGoal;

  useKeybindings(manager, inputActive);

  const startRun = useCallback(
    (g: string, runMode: "learn" | "explore" | "plan" = "learn") => {
      const storeConfigPath = useAgentStore.getState().configPath;
      const effectiveConfig = storeConfigPath
        ? resolve(projectRoot, storeConfigPath)
        : configPath;
      pushGoal(g);
      goalHistory.current = [g, ...goalHistory.current.filter((h) => h !== g)].slice(0, 100);
      useAgentStore.getState().setRunMode(runMode);
      manager.start(g, effectiveConfig, runMode);
      setInputMode("idle");
      setHistoryIdx(-1);
    },
    [manager, configPath, projectRoot]
  );

  // On mount: restore last config if no explicit one was provided
  React.useEffect(() => {
    const store = useAgentStore.getState();

    const cfgToLoad = configPath
      ? configPath
      : (!store.configPath && persistedState.current.lastConfigPath)
        ? persistedState.current.lastConfigPath
        : (!store.configPath && existsSync(resolve(projectRoot, "config.local.yaml")))
          ? "config.local.yaml"
          : null;

    if (cfgToLoad) {
      store.setConfigPath(cfgToLoad);
      setLastConfig(cfgToLoad);
      try {
        const fullPath = resolve(projectRoot, cfgToLoad);
        const raw = parseYaml(readFileSync(fullPath, "utf-8")) || {};
        const patch: Record<string, unknown> = {};
        const be = raw.backend || {};
        const vnc = be.vnc || {};
        patch.backend = {
          kind: be.kind || "vnc",
          host: vnc.host || "127.0.0.1",
          port: vnc.port || 5900,
          hostOs: be.host_os || "unknown",
          passwordEnv: vnc.password_env,
          usernameEnv: vnc.username_env,
          maxWidth: vnc.max_width,
          maxHeight: vnc.max_height,
        };
        const llm = raw.llm || {};
        const roles = llm.roles || {};
        patch.llmRoles = roles;
        if (llm.aws_region) patch.llmAwsRegion = llm.aws_region;
        if (llm.request_timeout_s) patch.llmRequestTimeoutS = llm.request_timeout_s;
        if (llm.creative_temperature !== undefined) patch.llmCreativeTemp = llm.creative_temperature;
        if (llm.analytical_temperature !== undefined) patch.llmAnalyticalTemp = llm.analytical_temperature;
        const exec = raw.executor || {};
        patch.executor = {
          defaultScreenWidth: exec.default_screen_width || 1920,
          defaultScreenHeight: exec.default_screen_height || 1080,
          stepTimeoutS: exec.step_timeout_s || 60,
        };
        const agent = raw.agent || {};
        if (agent.max_retries !== undefined) patch.maxRetries = agent.max_retries;
        if (agent.explore_steps !== undefined) patch.exploreSteps = agent.explore_steps;
        if (agent.no_strategy !== undefined) patch.noStrategy = agent.no_strategy;
        if (agent.verbose !== undefined) patch.verbose = agent.verbose;
        if (agent.record !== undefined) patch.record = agent.record;
        if (agent.record_fps !== undefined) patch.recordFps = agent.record_fps;
        if (agent.yolo !== undefined) patch.yolo = agent.yolo;
        const paths = raw.paths || {};
        if (paths.skills_dir) patch.skillsDir = paths.skills_dir;
        if (paths.traces_dir) patch.tracesDir = paths.traces_dir;
        if (paths.tasks_db) patch.tasksDb = paths.tasks_db;
        const ui = raw.ui || {};
        if (ui.overlay !== undefined) patch.noOverlay = !ui.overlay;
        store.setConfig(patch as any);
      } catch {
        // Config file missing or unparseable — keep defaults
      }
    }
    if (goal && !connected && !storeGoal) {
      startRun(goal);
    }
    if (tracePath && !storeGoal) {
      try {
        const tracesDir = resolve(projectRoot, store.config.tracesDir || "./traces");
        const traceDir = resolveTraceDir(tracesDir, tracePath);
        const { meta, goal: traceGoal } = loadTrace(traceDir);
        store.setGoal(traceGoal || meta.task_name || `[Replay] ${meta.task_id}`);
        setIsReplay(true);
      } catch (err: any) {
        store.setError(`Failed to load trace: ${err.message}`);
      }
    }
  }, [goal, connected, storeGoal, startRun, configPath, projectRoot, tracePath]);

  useInput((input, key) => {
    if (inputActive && !showConfig && !pendingConfirm) {
      if (key.return && goalInput.trim()) {
        const trimmed = goalInput.trim();
        if (trimmed === "/config" || trimmed === "/c") {
          useAgentStore.getState().toggleConfig();
          setGoalInput("");
          setCursorPos(0);
          setHistoryIdx(-1);
        } else if (trimmed === "/traces" || trimmed === "/t") {
          setShowTraces(true);
          setGoalInput("");
          setCursorPos(0);
          setHistoryIdx(-1);
        } else if (trimmed === "/quit" || trimmed === "/q" || trimmed === "/exit") {
          process.exit(0);
        } else if (trimmed === "/help" || trimmed === "/h") {
          setGoalInput("");
          setCursorPos(0);
          setHistoryIdx(-1);
        } else if (trimmed.startsWith("/learn ")) {
          const g = trimmed.slice("/learn ".length).trim();
          if (g) startRun(g, "learn");
        } else if (trimmed.startsWith("/explore ")) {
          const g = trimmed.slice("/explore ".length).trim();
          if (g) startRun(g, "explore");
        } else if (trimmed.startsWith("/plan ")) {
          const g = trimmed.slice("/plan ".length).trim();
          if (g) startRun(g, "plan");
        } else if (trimmed.startsWith("/")) {
          setGoalInput("");
          setCursorPos(0);
          setHistoryIdx(-1);
        } else {
          startRun(trimmed);
        }
      } else if (key.upArrow) {
        const history = goalHistory.current;
        if (history.length > 0) {
          const nextIdx = Math.min(historyIdx + 1, history.length - 1);
          setHistoryIdx(nextIdx);
          setGoalInput(history[nextIdx]);
          setCursorPos(history[nextIdx].length);
        }
      } else if (key.downArrow) {
        if (historyIdx > 0) {
          const nextIdx = historyIdx - 1;
          setHistoryIdx(nextIdx);
          setGoalInput(goalHistory.current[nextIdx]);
          setCursorPos(goalHistory.current[nextIdx].length);
        } else if (historyIdx === 0) {
          setHistoryIdx(-1);
          setGoalInput("");
          setCursorPos(0);
        }
      } else if (key.leftArrow) {
        setCursorPos((p) => Math.max(0, p - 1));
      } else if (key.rightArrow) {
        setCursorPos((p) => Math.min(goalInput.length, p + 1));
      } else if (key.backspace || key.delete) {
        if (cursorPos > 0) {
          setGoalInput((v) => v.slice(0, cursorPos - 1) + v.slice(cursorPos));
          setCursorPos((p) => p - 1);
          setHistoryIdx(-1);
        }
      } else if (input && !key.ctrl && !key.meta && !key.escape) {
        setGoalInput((v) => v.slice(0, cursorPos) + input + v.slice(cursorPos));
        setCursorPos((p) => p + input.length);
        setHistoryIdx(-1);
      }
    }
  });

  const handleApprove = useCallback(
    (id: number) => manager.approve(id),
    [manager]
  );

  const handleDeny = useCallback(
    (id: number, comments: string) => manager.deny(id, comments),
    [manager]
  );

  const handleTraceSelect = useCallback(
    (taskId: string) => {
      const store = useAgentStore.getState();
      store.reset();
      const tracesDir = resolve(projectRoot, store.config.tracesDir || "./traces");
      try {
        const traceDir = resolveTraceDir(tracesDir, taskId);
        const { meta, goal: traceGoal } = loadTrace(traceDir);
        store.setGoal(traceGoal || meta.task_name || `[Replay] ${meta.task_id}`);
        setIsReplay(true);
        setShowTraces(false);
      } catch (err: any) {
        store.setError(`Failed to load trace: ${err.message}`);
        setShowTraces(false);
      }
    },
    [projectRoot]
  );

  if (showTraces) {
    const tracesDir = resolve(projectRoot, useAgentStore.getState().config.tracesDir || "./traces");
    return (
      <Box flexDirection="column" width="100%" height="100%">
        <HeaderBar />
        <Box flexDirection="column" flexGrow={1} overflow="hidden" paddingX={1}>
          <TraceBrowser
            tracesDir={tracesDir}
            onSelect={handleTraceSelect}
            onExit={() => setShowTraces(false)}
          />
        </Box>
        <StatusBar isReplay={isReplay} />
      </Box>
    );
  }

  if (showConfig) {
    return (
      <Box flexDirection="column" width="100%" height="100%">
        <HeaderBar />
        <Box flexDirection="column" flexGrow={1} overflow="hidden">
          <ConfigScreen projectRoot={projectRoot} />
        </Box>
        <StatusBar isReplay={isReplay} />
      </Box>
    );
  }

  return (
    <Box flexDirection="column" width="100%" height="100%">
      <HeaderBar />

      <Box flexDirection="column" flexGrow={1} overflow="hidden" paddingX={1}>
        {/* Show logo + goal prompt when idle */}
        {inputActive && (
          <Box flexDirection="column" marginTop={1}>
            <Text color="cyan" bold>
              {LOGO}
            </Text>
            <Text dimColor>
              {"  "}v0.0.1 — interactive computer control agent
            </Text>
            <Box marginTop={1}>
              <Text>
                <Text color="green" bold>{"❯ "}</Text>
                <Text>{goalInput.slice(0, cursorPos)}</Text>
                <Text color="cyan">█</Text>
                <Text>{goalInput.slice(cursorPos)}</Text>
              </Text>
            </Box>
            <Box marginTop={1}>
              <Text dimColor>
                /learn &lt;goal&gt;  /explore &lt;goal&gt;  /plan &lt;goal&gt;  •  /config  /traces  /help  /quit  •  ↑↓: history
              </Text>
            </Box>
          </Box>
        )}

        {/* Active run — chat-style feed */}
        {storeGoal && (
          <Box flexDirection="column" flexGrow={1}>
            <GoalDisplay goal={storeGoal} isReplay={isReplay} />
            {!pendingConfirm && <ChatFeed />}
            {!pendingConfirm && phases[currentPhase]?.status === "running" &&
              (currentPhase === "planner" || currentPhase === "evaluator") && (
              <Box paddingLeft={2} paddingY={1}>
                <Text color="cyan"><Spinner type="dots" />{" "}</Text>
                <Text color="cyan" bold>
                  {currentPhase === "planner" ? "Planning" : "Evaluating"}
                </Text>
                <Text dimColor>
                  {" — "}{currentPhase === "planner"
                    ? (phases.planner.summary || "generating program...")
                    : (phases.evaluator.summary || "checking success criteria...")}
                </Text>
              </Box>
            )}
            {pendingConfirm && (
              <ConfirmDialog
                request={pendingConfirm}
                onApprove={handleApprove}
                onDeny={handleDeny}
              />
            )}
          </Box>
        )}
      </Box>

      <StatusBar isReplay={isReplay} />
    </Box>
  );
}
