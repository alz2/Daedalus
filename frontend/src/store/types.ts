export type EventLevel = "debug" | "info" | "warn" | "error";

export type PhaseId =
  | "idle"
  | "explorer"
  | "strategy"
  | "planner"
  | "executor"
  | "evaluator"
  | "learner";

export type PhaseStatus = "pending" | "running" | "complete" | "failed" | "skipped";

export interface TraceEvent {
  kind: string;
  ts: string;
  level: EventLevel;
  data: Record<string, unknown>;
}

export interface PhaseState {
  id: PhaseId;
  status: PhaseStatus;
  startedAt?: string;
  completedAt?: string;
  summary?: string;
  progress?: { current: number; total: number };
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: "running" | "success" | "error";
  startedAt: string;
  completedAt?: string;
}

export interface StepResult {
  idx: number;
  skillId: string;
  status: "pending" | "running" | "success" | "error";
  duration?: number;
  error?: string;
}

export interface ProgramStep {
  skillId: string;
  inputs: Record<string, unknown>;
}

export interface Program {
  name: string;
  version: string;
  steps: ProgramStep[];
}

export interface GoalVerdict {
  achieved: boolean;
  summary: string;
  results: Array<{
    passed: boolean;
    kind: string;
    description: string;
    explanation?: string;
  }>;
}

export interface LearnerFeedback {
  summary: string;
  failurePoint?: string;
  suggestions: Array<{
    category: string;
    description: string;
    affectedStepIdx?: number;
  }>;
  newSkillCandidates: Array<{
    proposedId: string;
    description: string;
  }>;
}

export interface ConfirmRequest {
  id: number;
  type: "program" | "criteria" | "skills";
  payload: unknown;
}

export interface BackendConfig {
  kind: "mock" | "vnc";
  host: string;
  port: number;
  passwordEnv?: string;
  usernameEnv?: string;
  maxWidth?: number;
  maxHeight?: number;
  hostOs: string;
}

export interface LLMRoles {
  planner?: string;
  explorer?: string;
  implementor?: string;
  learner?: string;
  vision?: string;
  cheap?: string;
}

export interface ExecutorConfig {
  stepTimeoutS: number;
  defaultScreenWidth: number;
  defaultScreenHeight: number;
}

export interface DaedalusConfig {
  backend: BackendConfig;
  llmRoles: LLMRoles;
  llmAwsRegion: string;
  llmRequestTimeoutS: number;
  llmCreativeTemp: number;
  llmAnalyticalTemp: number;
  executor: ExecutorConfig;
  maxRetries: number;
  exploreSteps: number;
  record: boolean;
  recordFps: number;
  skillsDir: string;
  tracesDir: string;
  tasksDb: string;
  verbose: boolean;
  noOverlay: boolean;
  noStrategy: boolean;
  yolo: boolean;
}

export type ChatMessageKind = "thinking" | "tool_call" | "phase" | "error" | "status" | "learner_feedback";

export interface ChatMessage {
  id: string;
  kind: ChatMessageKind;
  ts: string;
  // For "thinking": the model's text output
  text?: string;
  // For "tool_call": tool details
  toolName?: string;
  toolArgs?: Record<string, unknown>;
  toolResult?: string;
  toolStatus?: "running" | "success" | "error";
  toolImagePath?: string;
  // For "phase": phase transition info
  phase?: PhaseId;
  phaseStatus?: PhaseStatus;
  phaseSummary?: string;
}

export interface AgentState {
  goal: string | null;
  configPath: string | null;
  phases: Record<PhaseId, PhaseState>;
  currentPhase: PhaseId;
  events: TraceEvent[];
  toolCalls: ToolCall[];
  steps: StepResult[];
  program: Program | null;
  verdict: GoalVerdict | null;
  learnerFeedback: LearnerFeedback | null;
  pendingConfirm: ConfirmRequest | null;
  thinkingText: string;
  chatMessages: ChatMessage[];
  config: DaedalusConfig;
  connected: boolean;
  startedAt: string | null;
  taskId: string | null;
  attempt: number;
  error: string | null;
  showConfig: boolean;
}

export const DEFAULT_CONFIG: DaedalusConfig = {
  backend: {
    kind: "vnc",
    host: "127.0.0.1",
    port: 5900,
    hostOs: "unknown",
  },
  llmRoles: {},
  llmAwsRegion: "us-east-1",
  llmRequestTimeoutS: 120,
  llmCreativeTemp: 0.7,
  llmAnalyticalTemp: 0.0,
  executor: {
    stepTimeoutS: 60,
    defaultScreenWidth: 1920,
    defaultScreenHeight: 1080,
  },
  maxRetries: 3,
  exploreSteps: 20,
  record: false,
  recordFps: 30,
  skillsDir: "./skills",
  tracesDir: "./traces",
  tasksDb: "./tasks.db",
  verbose: false,
  noOverlay: true,
  noStrategy: false,
  yolo: false,
};
