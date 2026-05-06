import { create } from "zustand";
import type {
  AgentState,
  ChatMessage,
  ConfirmRequest,
  DaedalusConfig,
  DEFAULT_CONFIG,
  GoalVerdict,
  LearnerFeedback,
  PhaseId,
  PhaseStatus,
  Program,
  StepResult,
  ToolCall,
  TraceEvent,
} from "./types.js";

const INITIAL_PHASES: AgentState["phases"] = {
  idle: { id: "idle", status: "complete" },
  explorer: { id: "explorer", status: "pending" },
  strategy: { id: "strategy", status: "pending" },
  planner: { id: "planner", status: "pending" },
  executor: { id: "executor", status: "pending" },
  evaluator: { id: "evaluator", status: "pending" },
  learner: { id: "learner", status: "pending" },
};

interface AgentActions {
  setGoal: (goal: string) => void;
  setConfigPath: (path: string | null) => void;
  setPhase: (phase: PhaseId, status: PhaseStatus, summary?: string) => void;
  setPhaseProgress: (phase: PhaseId, current: number, total: number) => void;
  pushEvent: (event: TraceEvent) => void;
  pushToolCall: (tc: ToolCall) => void;
  updateToolCall: (id: string, updates: Partial<ToolCall>) => void;
  setProgram: (program: Program) => void;
  setStep: (step: StepResult) => void;
  setVerdict: (verdict: GoalVerdict) => void;
  setLearnerFeedback: (fb: LearnerFeedback) => void;
  setPendingConfirm: (req: ConfirmRequest | null) => void;
  appendThinking: (text: string) => void;
  clearThinking: () => void;
  pushChat: (msg: ChatMessage) => void;
  updateChat: (id: string, updates: Partial<ChatMessage>) => void;
  setConnected: (connected: boolean) => void;
  setStarted: (taskId: string) => void;
  setAttempt: (attempt: number) => void;
  setError: (error: string | null) => void;
  setConfig: (config: Partial<DaedalusConfig>) => void;
  toggleConfig: () => void;
  setContextUsage: (used: number, max: number) => void;
  reset: () => void;
}

export type AgentStore = AgentState & AgentActions;

const defaultConfig: DaedalusConfig = {
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

const INITIAL_STATE: AgentState = {
  goal: null,
  configPath: null,
  phases: { ...INITIAL_PHASES },
  currentPhase: "idle",
  events: [],
  toolCalls: [],
  steps: [],
  program: null,
  verdict: null,
  learnerFeedback: null,
  pendingConfirm: null,
  thinkingText: "",
  chatMessages: [],
  config: defaultConfig,
  connected: false,
  startedAt: null,
  taskId: null,
  attempt: 0,
  error: null,
  showConfig: false,
  contextUsage: null,
};

export const useAgentStore = create<AgentStore>((set) => ({
  ...INITIAL_STATE,

  setGoal: (goal) => set({ goal }),

  setConfigPath: (path) => set({ configPath: path }),

  setPhase: (phase, status, summary) =>
    set((state) => ({
      currentPhase: status === "running" ? phase : state.currentPhase,
      phases: {
        ...state.phases,
        [phase]: {
          ...state.phases[phase],
          status,
          summary,
          startedAt:
            status === "running"
              ? new Date().toISOString()
              : state.phases[phase].startedAt,
          completedAt:
            status === "complete" || status === "failed"
              ? new Date().toISOString()
              : undefined,
        },
      },
    })),

  setPhaseProgress: (phase, current, total) =>
    set((state) => ({
      phases: {
        ...state.phases,
        [phase]: {
          ...state.phases[phase],
          progress: { current, total },
        },
      },
    })),

  pushEvent: (event) =>
    set((state) => ({
      events: [...state.events, event],
    })),

  pushToolCall: (tc) =>
    set((state) => ({
      toolCalls: [...state.toolCalls, tc],
    })),

  updateToolCall: (id, updates) =>
    set((state) => ({
      toolCalls: state.toolCalls.map((tc) =>
        tc.id === id ? { ...tc, ...updates } : tc
      ),
    })),

  setProgram: (program) => set({ program }),

  setStep: (step) =>
    set((state) => {
      const existing = state.steps.findIndex((s) => s.idx === step.idx);
      if (existing >= 0) {
        const updated = [...state.steps];
        updated[existing] = step;
        return { steps: updated };
      }
      return { steps: [...state.steps, step] };
    }),

  setVerdict: (verdict) => set({ verdict }),
  setLearnerFeedback: (fb) => set({ learnerFeedback: fb }),
  setPendingConfirm: (req) => set({ pendingConfirm: req }),

  appendThinking: (text) =>
    set((state) => ({ thinkingText: state.thinkingText + text })),

  clearThinking: () => set({ thinkingText: "" }),

  pushChat: (msg) =>
    set((state) => ({ chatMessages: [...state.chatMessages, msg] })),

  updateChat: (id, updates) =>
    set((state) => ({
      chatMessages: state.chatMessages.map((m) =>
        m.id === id ? { ...m, ...updates } : m
      ),
    })),

  setConnected: (connected) => set({ connected }),

  setStarted: (taskId) =>
    set({ taskId, startedAt: new Date().toISOString() }),

  setAttempt: (attempt) => set({ attempt }),
  setError: (error) => set({ error }),

  setConfig: (partial) =>
    set((state) => ({
      config: { ...state.config, ...partial },
    })),

  toggleConfig: () => set((state) => ({ showConfig: !state.showConfig })),

  setContextUsage: (used, max) => set({ contextUsage: { used, max } }),

  reset: () => set({ ...INITIAL_STATE }),
}));
