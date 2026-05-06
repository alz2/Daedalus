import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { useAgentStore } from "../store/agent-store.js";
import { applyNotification } from "./apply-event.js";

export interface TraceMeta {
  task_id: string;
  task_name: string;
  program_ref: string | null;
  status: string;
  started: string;
  finished: string;
  events: number;
  screenshots: number;
}

export function loadTrace(tracePath: string): { meta: TraceMeta; goal: string | null } {
  const store = useAgentStore.getState();

  const metaPath = join(tracePath, "meta.json");
  if (!existsSync(metaPath)) {
    throw new Error(`No meta.json found at ${metaPath}`);
  }
  const meta: TraceMeta = JSON.parse(readFileSync(metaPath, "utf-8"));

  // Try per-run bridge_events.jsonl first, fall back to root-level
  let bridgeEventsPath = join(tracePath, "bridge_events.jsonl");
  if (!existsSync(bridgeEventsPath)) {
    const parentBridge = join(tracePath, "..", "bridge_events.jsonl");
    if (existsSync(parentBridge)) {
      bridgeEventsPath = parentBridge;
    } else {
      throw new Error(`No bridge_events.jsonl found at ${bridgeEventsPath}. This trace was recorded without frontend bridge logging.`);
    }
  }

  const content = readFileSync(bridgeEventsPath, "utf-8");
  const lines = content.trim().split("\n").filter(Boolean);

  // Detect goal from plan.yaml or first event data
  let goal: string | null = meta.task_name || null;
  const planPath = join(tracePath, "plan.yaml");
  if (existsSync(planPath)) {
    try {
      const planContent = readFileSync(planPath, "utf-8");
      const goalMatch = planContent.match(/goal:\s*(.+)/);
      if (goalMatch) goal = goalMatch[1].trim();
    } catch {
      // ignore
    }
  }

  // Replay each event through the store
  for (const line of lines) {
    try {
      const record = JSON.parse(line) as { method: string; params: Record<string, unknown>; ts: string };
      applyNotification(record.method, record.params);
    } catch {
      // skip malformed lines
    }
  }

  return { meta, goal };
}

export function listTraces(tracesDir: string): TraceMeta[] {
  if (!existsSync(tracesDir)) return [];

  const entries = readdirSync(tracesDir, { withFileTypes: true });
  const metas: TraceMeta[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith("t_")) continue;
    const metaPath = join(tracesDir, entry.name, "meta.json");
    if (!existsSync(metaPath)) continue;
    try {
      const meta: TraceMeta = JSON.parse(readFileSync(metaPath, "utf-8"));
      metas.push(meta);
    } catch {
      // skip
    }
  }

  // Sort newest first
  metas.sort((a, b) => new Date(b.started).getTime() - new Date(a.started).getTime());
  return metas;
}

export function resolveTraceDir(tracesRoot: string, traceIdOrPath: string): string {
  // If it's an absolute path or relative path that exists, use it directly
  if (existsSync(traceIdOrPath)) return resolve(traceIdOrPath);
  // Otherwise treat as a task_id and look for it in tracesRoot
  const candidate = join(tracesRoot, traceIdOrPath);
  if (existsSync(candidate)) return candidate;
  throw new Error(`Cannot find trace: ${traceIdOrPath}`);
}
