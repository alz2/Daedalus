import { JsonRpcBridge, JsonRpcNotification, JsonRpcRequest } from "./json-rpc.js";
import { useAgentStore } from "../store/agent-store.js";
import type { DaedalusConfig } from "../store/types.js";
import { applyNotification } from "./apply-event.js";
import { writeFileSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

function buildTempConfig(config: DaedalusConfig, projectRoot: string): string {
  const roles: Record<string, string> = {};
  if (config.llmRoles.planner) roles.planner = config.llmRoles.planner;
  if (config.llmRoles.explorer) roles.explorer = config.llmRoles.explorer;
  if (config.llmRoles.implementor) roles.implementor = config.llmRoles.implementor;
  if (config.llmRoles.learner) roles.learner = config.llmRoles.learner;
  if (config.llmRoles.vision) roles.vision = config.llmRoles.vision;
  if (config.llmRoles.cheap) roles.cheap = config.llmRoles.cheap;

  const yaml: Record<string, unknown> = {
    backend: {
      kind: config.backend.kind,
      host_os: config.backend.hostOs,
      vnc: {
        host: config.backend.host,
        port: config.backend.port,
        ...(config.backend.passwordEnv && { password_env: config.backend.passwordEnv }),
        ...(config.backend.usernameEnv && { username_env: config.backend.usernameEnv }),
        ...(config.backend.maxWidth && { max_width: config.backend.maxWidth }),
        ...(config.backend.maxHeight && { max_height: config.backend.maxHeight }),
      },
    },
    llm: {
      roles,
      aws_region: config.llmAwsRegion,
      request_timeout_s: config.llmRequestTimeoutS,
      creative_temperature: config.llmCreativeTemp,
      analytical_temperature: config.llmAnalyticalTemp,
    },
    paths: {
      skills_dir: config.skillsDir,
      traces_dir: config.tracesDir,
      tasks_db: config.tasksDb,
    },
    executor: {
      default_screen_width: config.executor.defaultScreenWidth,
      default_screen_height: config.executor.defaultScreenHeight,
      step_timeout_s: config.executor.stepTimeoutS,
    },
    agent: {
      max_retries: config.maxRetries,
      explore_steps: config.exploreSteps,
      no_strategy: config.noStrategy,
      verbose: config.verbose,
      record: config.record,
      record_fps: config.recordFps,
      yolo: config.yolo,
    },
    ui: {
      overlay: !config.noOverlay,
      confirm: true,
    },
  };

  const lines: string[] = [];
  function dump(obj: Record<string, unknown>, indent = 0) {
    const pad = "  ".repeat(indent);
    for (const [k, v] of Object.entries(obj)) {
      if (v === undefined || v === null) continue;
      if (typeof v === "object" && !Array.isArray(v)) {
        lines.push(`${pad}${k}:`);
        dump(v as Record<string, unknown>, indent + 1);
      } else {
        lines.push(`${pad}${k}: ${v}`);
      }
    }
  }
  dump(yaml);

  const dir = mkdtempSync(join(tmpdir(), "daedalus-"));
  const filePath = join(dir, "config.yaml");
  writeFileSync(filePath, lines.join("\n") + "\n");
  return filePath;
}

export class ProcessManager {
  private bridge: JsonRpcBridge | null = null;
  private stderrBuffer: string[] = [];

  constructor(private projectRoot: string) {}

  start(goal: string, configPath?: string, mode: "learn" | "explore" | "plan" = "learn"): void {
    const store = useAgentStore.getState();
    const args = ["run", "--goal", goal, "--frontend-mode", "--mode", mode];

    // Always provide --config: use the explicit config file, or generate one
    const effectiveConfig = configPath || buildTempConfig(store.config, this.projectRoot);
    args.push("--config", effectiveConfig);

    args.push("--max-retries", String(store.config.maxRetries));
    args.push("--explore-steps", String(store.config.exploreSteps));

    if (store.config.backend.kind === "vnc") {
      args.push("--backend", "vnc");
      args.push("--host", store.config.backend.host);
      args.push("--port", String(store.config.backend.port));
      if (store.config.backend.passwordEnv) {
        args.push("--password-env", store.config.backend.passwordEnv);
      }
      if (store.config.backend.usernameEnv) {
        args.push("--username-env", store.config.backend.usernameEnv);
      }
    } else {
      args.push("--backend", "mock");
    }

    if (store.config.noStrategy) args.push("--no-strategy");
    if (store.config.noOverlay) args.push("--no-overlay");
    if (store.config.record) args.push("--record");
    if (store.config.verbose) args.push("--verbose");
    if (store.config.yolo) args.push("--yes");

    this.bridge = new JsonRpcBridge("daedalus", args, this.projectRoot);

    this.bridge.on("connected", () => {
      useAgentStore.getState().setConnected(true);
    });

    this.bridge.on("notification", (msg: JsonRpcNotification) => {
      this.handleNotification(msg);
    });

    this.bridge.on("request", (msg: JsonRpcRequest) => {
      this.handleRequest(msg);
    });

    this.bridge.on("stderr", (text: string) => {
      this.stderrBuffer.push(text);
      if (this.stderrBuffer.length > 500) {
        this.stderrBuffer.shift();
      }
    });

    this.bridge.on("exit", (code: number | null) => {
      const store = useAgentStore.getState();
      store.setConnected(false);
      if (code !== 0 && code !== null) {
        const recentStderr = this.stderrBuffer.slice(-10).join("").trim();
        const detail = recentStderr ? `\n${recentStderr}` : "";
        store.setError(`Backend exited with code ${code}${detail}`);
      }
    });

    this.bridge.on("error", (err: Error) => {
      useAgentStore.getState().setError(`Bridge error: ${err.message}`);
    });

    useAgentStore.getState().setGoal(goal);
    this.bridge.start();
  }

  private handleNotification(msg: JsonRpcNotification): void {
    applyNotification(msg.method, msg.params as Record<string, unknown>);
  }

  private handleRequest(msg: JsonRpcRequest): void {
    const store = useAgentStore.getState();
    const typeMap: Record<string, "program" | "criteria" | "skills"> = {
      confirm_program: "program",
      confirm_criteria: "criteria",
      confirm_skills: "skills",
      program: "program",
      criteria: "criteria",
      skills: "skills",
    };
    const confirmType = typeMap[msg.method] ?? "program";
    store.setPendingConfirm({
      id: msg.id,
      type: confirmType,
      payload: msg.params,
    });
  }

  approve(id: number, comments?: string): void {
    this.bridge?.respond(id, { decision: "approve", comments });
    useAgentStore.getState().setPendingConfirm(null);
  }

  deny(id: number, comments: string): void {
    this.bridge?.respond(id, { decision: "deny", comments });
    useAgentStore.getState().setPendingConfirm(null);
  }

  abort(): void {
    this.bridge?.send("abort");
  }

  stop(): void {
    this.bridge?.stop();
    this.bridge = null;
  }

  getStderr(): string[] {
    return this.stderrBuffer;
  }

  get isRunning(): boolean {
    return this.bridge?.isRunning ?? false;
  }
}
