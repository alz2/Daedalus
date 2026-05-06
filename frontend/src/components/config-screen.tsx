import React, { useState, useEffect } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse as parseYaml } from "yaml";
import { useAgentStore } from "../store/agent-store.js";
import { setLastConfig } from "../store/persistence.js";
import type { DaedalusConfig } from "../store/types.js";

interface ConfigScreenProps {
  projectRoot: string;
}

export function ConfigScreen({ projectRoot }: ConfigScreenProps): React.ReactElement {
  const config = useAgentStore((s) => s.config);
  const configPath = useAgentStore((s) => s.configPath);
  const setConfig = useAgentStore((s) => s.setConfig);
  const setConfigPath = useAgentStore((s) => s.setConfigPath);
  const toggleConfig = useAgentStore((s) => s.toggleConfig);

  const { stdout } = useStdout();
  const terminalHeight = stdout?.rows ?? 40;
  const viewportHeight = Math.max(5, terminalHeight - 8);

  const [selected, setSelected] = useState(0);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const [scrollOffset, setScrollOffset] = useState(0);

  const locked = configPath !== null;

  const [configFiles, setConfigFiles] = useState<string[]>([]);
  useEffect(() => {
    try {
      const files = readdirSync(projectRoot)
        .filter((f) => (f.endsWith(".yaml") || f.endsWith(".yml")))
        .sort();
      setConfigFiles(files);
    } catch {
      setConfigFiles([]);
    }
  }, [projectRoot]);

  const fields = getFields(config, configPath, configFiles);

  // Keep selection within scroll viewport
  useEffect(() => {
    if (selected < scrollOffset) {
      setScrollOffset(selected);
    } else if (selected >= scrollOffset + viewportHeight) {
      setScrollOffset(selected - viewportHeight + 1);
    }
  }, [selected, viewportHeight, scrollOffset]);

  useInput((input, key) => {
    if (key.escape) {
      if (editing) {
        setEditing(false);
        setEditValue("");
      } else {
        toggleConfig();
      }
      return;
    }

    if (!editing) {
      // q closes the config screen when not editing a text field
      if (input === "q") {
        toggleConfig();
        return;
      }

      if (key.upArrow) {
        setSelected((s) => Math.max(0, s - 1));
      } else if (key.downArrow) {
        setSelected((s) => Math.min(fields.length - 1, s + 1));
      } else if (key.return) {
        const field = fields[selected];

        if (field.path.startsWith("_sep")) return;

        // Config file toggle: Enter turns it on/off
        if (field.path === "_configPath") {
          if (locked) {
            setConfigPath(null);
            setLastConfig(null);
          } else {
            const firstFile = configFiles[0];
            if (firstFile) {
              setConfigPath(firstFile);
              setLastConfig(firstFile);
              loadConfigFromFile(resolve(projectRoot, firstFile), setConfig);
            }
          }
          return;
        }

        if (locked) return;

        if (field.type === "boolean") {
          applyField(field.path, !field.value, config, setConfig);
        } else if (field.type === "select") {
          const options = field.options!;
          const currentIdx = options.indexOf(String(field.value));
          const nextIdx = (currentIdx + 1) % options.length;
          applyField(field.path, options[nextIdx], config, setConfig);
        } else {
          setEditing(true);
          setEditValue(String(field.value));
        }
      } else if (key.tab || key.leftArrow || key.rightArrow) {
        const field = fields[selected];
        if (field.path.startsWith("_sep")) return;

        // Config file: Tab/arrows cycle between available files
        if (field.path === "_configPath") {
          if (configFiles.length === 0) return;
          const currentIdx = configPath ? configFiles.indexOf(configPath) : -1;
          let nextIdx: number;
          if (key.leftArrow) {
            nextIdx = currentIdx <= 0 ? configFiles.length - 1 : currentIdx - 1;
          } else {
            nextIdx = currentIdx >= configFiles.length - 1 ? 0 : currentIdx + 1;
          }
          const nextFile = configFiles[nextIdx];
          setConfigPath(nextFile);
          setLastConfig(nextFile);
          loadConfigFromFile(resolve(projectRoot, nextFile), setConfig);
          return;
        }

        if (locked) return;

        // Tab cycles select/boolean fields
        if (field.type === "boolean") {
          applyField(field.path, !field.value, config, setConfig);
        } else if (field.type === "select") {
          const options = field.options!;
          const currentIdx = options.indexOf(String(field.value));
          const dir = key.leftArrow ? -1 : 1;
          const nextIdx = (currentIdx + dir + options.length) % options.length;
          applyField(field.path, options[nextIdx], config, setConfig);
        }
      }
    } else {
      if (key.return) {
        const field = fields[selected];
        applyField(field.path, coerce(editValue, field.type), config, setConfig);
        setEditing(false);
        setEditValue("");
      } else if (key.backspace || key.delete) {
        setEditValue((v) => v.slice(0, -1));
      } else if (input && !key.ctrl && !key.meta) {
        setEditValue((v) => v + input);
      }
    }
  });

  const visibleFields = fields.slice(scrollOffset, scrollOffset + viewportHeight);
  const hasMore = scrollOffset + viewportHeight < fields.length;
  const hasAbove = scrollOffset > 0;

  return (
    <Box
      flexDirection="column"
      borderStyle="double"
      borderColor="cyan"
      paddingX={2}
      paddingY={1}
      width="100%"
    >
      <Text bold color="cyan">
        Configuration
      </Text>
      <Text dimColor>
        Esc/q: back  ↑↓: navigate  Enter: toggle/edit  Tab/←→: cycle
        {locked ? "  (locked — press Enter on Config File to unlock)" : ""}
      </Text>
      {hasAbove && (
        <Text dimColor>  ↑ more above ({scrollOffset} hidden)</Text>
      )}
      <Box flexDirection="column" marginTop={hasAbove ? 0 : 1}>
        {visibleFields.map((field, vi) => {
          const i = vi + scrollOffset;
          if (field.path.startsWith("_sep")) {
            return (
              <Box key={field.path}>
                <Text dimColor>  ────────────────────────────────────────</Text>
              </Box>
            );
          }

          const isConfigField = field.path === "_configPath";
          const isLocked = locked && !isConfigField;

          return (
            <Box key={field.path}>
              <Text color={i === selected ? "cyan" : undefined}>
                {i === selected ? "❯ " : "  "}
              </Text>
              <Text
                bold={i === selected}
                color={isLocked ? "gray" : i === selected ? "white" : "gray"}
                dimColor={isLocked}
              >
                {field.label.padEnd(24)}
              </Text>
              {editing && i === selected ? (
                <Text>
                  <Text color="cyan">{editValue}</Text>
                  <Text color="cyan">█</Text>
                </Text>
              ) : (
                <FieldValue field={field} locked={isLocked} isConfigField={isConfigField} />
              )}
            </Box>
          );
        })}
      </Box>
      {hasMore && (
        <Text dimColor>  ↓ more below ({fields.length - scrollOffset - viewportHeight} hidden)</Text>
      )}
    </Box>
  );
}

function FieldValue({
  field,
  locked,
  isConfigField,
}: {
  field: FieldDef;
  locked: boolean;
  isConfigField: boolean;
}): React.ReactElement {
  const color = locked ? "gray" : "yellow";

  if (field.type === "boolean") {
    const boolColor = locked ? "gray" : field.value ? "green" : "red";
    return <Text color={boolColor}>{field.value ? "yes" : "no"}</Text>;
  }
  if (isConfigField) {
    const active = String(field.value) !== "(none)";
    return (
      <Text color={active ? "green" : "yellow"}>
        {active ? `✓ ${field.value}` : "(disabled)"}
        <Text dimColor> ← Enter: on/off  Tab/←→: switch file</Text>
      </Text>
    );
  }
  if (field.type === "select") {
    return (
      <Text color={locked ? "gray" : "yellow"}>
        {String(field.value) || "(none)"}
        {!locked && <Text dimColor> ← Tab/←→</Text>}
      </Text>
    );
  }
  const display = String(field.value);
  return <Text color={color}>{display || <Text dimColor>(empty)</Text>}</Text>;
}

interface FieldDef {
  label: string;
  path: string;
  value: unknown;
  type: "string" | "number" | "boolean" | "select";
  options?: string[];
}

function getFields(
  config: DaedalusConfig,
  configPath: string | null,
  configFiles: string[]
): FieldDef[] {
  return [
    {
      label: "Config File",
      path: "_configPath",
      value: configPath ?? "(none)",
      type: "select",
      options: configFiles,
    },
    { label: "", path: "_sep1", value: "", type: "string" },
    { label: "Backend", path: "backend.kind", value: config.backend.kind, type: "select", options: ["vnc", "mock"] },
    { label: "Host", path: "backend.host", value: config.backend.host, type: "string" },
    { label: "Port", path: "backend.port", value: config.backend.port, type: "number" },
    { label: "Host OS", path: "backend.hostOs", value: config.backend.hostOs, type: "string" },
    { label: "Password Env", path: "backend.passwordEnv", value: config.backend.passwordEnv ?? "", type: "string" },
    { label: "Username Env", path: "backend.usernameEnv", value: config.backend.usernameEnv ?? "", type: "string" },
    { label: "Max Width", path: "backend.maxWidth", value: config.backend.maxWidth ?? "", type: "number" },
    { label: "Max Height", path: "backend.maxHeight", value: config.backend.maxHeight ?? "", type: "number" },
    { label: "", path: "_sep2", value: "", type: "string" },
    { label: "Planner LLM", path: "llmRoles.planner", value: config.llmRoles.planner ?? "", type: "string" },
    { label: "Explorer LLM", path: "llmRoles.explorer", value: config.llmRoles.explorer ?? "", type: "string" },
    { label: "Implementor LLM", path: "llmRoles.implementor", value: config.llmRoles.implementor ?? "", type: "string" },
    { label: "Learner LLM", path: "llmRoles.learner", value: config.llmRoles.learner ?? "", type: "string" },
    { label: "Vision LLM", path: "llmRoles.vision", value: config.llmRoles.vision ?? "", type: "string" },
    { label: "Cheap LLM", path: "llmRoles.cheap", value: config.llmRoles.cheap ?? "", type: "string" },
    { label: "AWS Region", path: "llmAwsRegion", value: config.llmAwsRegion, type: "string" },
    { label: "Request Timeout (s)", path: "llmRequestTimeoutS", value: config.llmRequestTimeoutS, type: "number" },
    { label: "Creative Temp", path: "llmCreativeTemp", value: config.llmCreativeTemp, type: "number" },
    { label: "Analytical Temp", path: "llmAnalyticalTemp", value: config.llmAnalyticalTemp, type: "number" },
    { label: "", path: "_sep3", value: "", type: "string" },
    { label: "Screen Width", path: "executor.defaultScreenWidth", value: config.executor.defaultScreenWidth, type: "number" },
    { label: "Screen Height", path: "executor.defaultScreenHeight", value: config.executor.defaultScreenHeight, type: "number" },
    { label: "Step Timeout (s)", path: "executor.stepTimeoutS", value: config.executor.stepTimeoutS, type: "number" },
    { label: "Max Retries", path: "maxRetries", value: config.maxRetries, type: "number" },
    { label: "Explore Steps", path: "exploreSteps", value: config.exploreSteps, type: "number" },
    { label: "", path: "_sep4", value: "", type: "string" },
    { label: "Record", path: "record", value: config.record, type: "boolean" },
    { label: "Record FPS", path: "recordFps", value: config.recordFps, type: "number" },
    { label: "No Strategy", path: "noStrategy", value: config.noStrategy, type: "boolean" },
    { label: "Verbose", path: "verbose", value: config.verbose, type: "boolean" },
    { label: "", path: "_sep5", value: "", type: "string" },
    { label: "Skills Dir", path: "skillsDir", value: config.skillsDir, type: "string" },
    { label: "Traces Dir", path: "tracesDir", value: config.tracesDir, type: "string" },
    { label: "Tasks DB", path: "tasksDb", value: config.tasksDb, type: "string" },
  ];
}

function loadConfigFromFile(
  filePath: string,
  setConfig: (partial: Partial<DaedalusConfig>) => void
): void {
  try {
    const raw = parseYaml(readFileSync(filePath, "utf-8")) || {};
    const patch: Partial<DaedalusConfig> = {};

    // Backend
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

    // LLM
    const llm = raw.llm || {};
    const roles = llm.roles || {};
    patch.llmRoles = {
      planner: roles.planner,
      explorer: roles.explorer,
      implementor: roles.implementor,
      learner: roles.learner,
      vision: roles.vision,
      cheap: roles.cheap,
    };
    if (llm.aws_region) patch.llmAwsRegion = llm.aws_region;
    if (llm.request_timeout_s) patch.llmRequestTimeoutS = llm.request_timeout_s;
    if (llm.creative_temperature !== undefined) patch.llmCreativeTemp = llm.creative_temperature;
    if (llm.analytical_temperature !== undefined) patch.llmAnalyticalTemp = llm.analytical_temperature;

    // Executor
    const exec = raw.executor || {};
    patch.executor = {
      defaultScreenWidth: exec.default_screen_width || 1920,
      defaultScreenHeight: exec.default_screen_height || 1080,
      stepTimeoutS: exec.step_timeout_s || 60,
    };

    // Agent settings
    const agent = raw.agent || {};
    if (agent.max_retries !== undefined) patch.maxRetries = agent.max_retries;
    if (agent.explore_steps !== undefined) patch.exploreSteps = agent.explore_steps;
    if (agent.no_strategy !== undefined) patch.noStrategy = agent.no_strategy;
    if (agent.verbose !== undefined) patch.verbose = agent.verbose;
    if (agent.record !== undefined) patch.record = agent.record;
    if (agent.record_fps !== undefined) patch.recordFps = agent.record_fps;

    // Paths
    const paths = raw.paths || {};
    if (paths.skills_dir) patch.skillsDir = paths.skills_dir;
    if (paths.traces_dir) patch.tracesDir = paths.traces_dir;
    if (paths.tasks_db) patch.tasksDb = paths.tasks_db;

    // UI
    const ui = raw.ui || {};
    if (ui.overlay !== undefined) patch.noOverlay = !ui.overlay;

    setConfig(patch);
  } catch {
    // If file can't be read/parsed, leave config as-is
  }
}

function coerce(value: string, type: "string" | "number" | "boolean" | "select"): unknown {
  if (type === "number") return Number(value) || 0;
  if (type === "boolean") return value === "true" || value === "yes";
  return value;
}

function applyField(
  path: string,
  value: unknown,
  config: DaedalusConfig,
  setConfig: (partial: Partial<DaedalusConfig>) => void
): void {
  const parts = path.split(".");
  if (parts.length === 1) {
    setConfig({ [parts[0]]: value } as any);
  } else if (parts[0] === "backend") {
    setConfig({
      backend: { ...config.backend, [parts[1]]: value },
    });
  } else if (parts[0] === "executor") {
    setConfig({
      executor: { ...config.executor, [parts[1]]: value },
    });
  } else if (parts[0] === "llmRoles") {
    setConfig({
      llmRoles: { ...config.llmRoles, [parts[1]]: value || undefined },
    });
  }
}
