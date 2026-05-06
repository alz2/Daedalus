import { useAgentStore } from "../store/agent-store.js";
import type { TraceEvent, PhaseId, PhaseStatus, ChatMessage } from "../store/types.js";

type Store = ReturnType<typeof useAgentStore.getState>;

export function applyEvent(event: TraceEvent, store?: Store): void {
  const s = store ?? useAgentStore.getState();

  switch (event.kind) {
    case "phase_changed":
      s.setPhase(
        event.data.phase as PhaseId,
        event.data.status as PhaseStatus,
        event.data.summary as string | undefined
      );
      s.pushChat({
        id: `phase-${event.data.phase}-${event.data.status}-${Date.now()}-${Math.random().toString(36).slice(2, 5)}`,
        kind: "phase",
        ts: event.ts,
        phase: event.data.phase as PhaseId,
        phaseStatus: event.data.status as PhaseStatus,
        phaseSummary: event.data.summary as string | undefined,
      });
      break;

    case "task_started":
      s.setStarted(event.data.task_id as string);
      break;

    case "skill_started": {
      const tcId = `${event.data.step_idx}-${event.data.skill_id}`;
      s.setStep({
        idx: event.data.step_idx as number,
        skillId: event.data.skill_id as string,
        status: "running",
      });
      s.pushToolCall({
        id: tcId,
        name: event.data.skill_id as string,
        args: (event.data.inputs as Record<string, unknown>) ?? {},
        status: "running",
        startedAt: event.ts,
      });
      s.pushChat({
        id: `tc-${tcId}`,
        kind: "tool_call",
        ts: event.ts,
        toolName: event.data.skill_id as string,
        toolArgs: (event.data.inputs as Record<string, unknown>) ?? {},
        toolStatus: "running",
      });
      break;
    }

    case "skill_finished": {
      const tcId = `${event.data.step_idx}-${event.data.skill_id}`;
      s.setStep({
        idx: event.data.step_idx as number,
        skillId: event.data.skill_id as string,
        status: "success",
        duration: event.data.duration_ms as number,
      });
      s.updateToolCall(tcId, {
        status: "success",
        result: JSON.stringify(event.data.outputs),
        completedAt: event.ts,
      });
      const chatUpdate: Partial<ChatMessage> = {
        toolStatus: "success",
        toolResult: typeof event.data.outputs === "string"
          ? (event.data.outputs as string).slice(0, 200)
          : JSON.stringify(event.data.outputs).slice(0, 200),
      };
      const imagePath =
        (event.data.image_path as string | undefined) ||
        ((event.data.outputs as Record<string, unknown> | undefined)?.image_path as string | undefined);
      if (imagePath) {
        chatUpdate.toolImagePath = imagePath;
      }
      s.updateChat(`tc-${tcId}`, chatUpdate);
      break;
    }

    case "skill_error": {
      const tcId = `${event.data.step_idx}-${event.data.skill_id}`;
      s.setStep({
        idx: event.data.step_idx as number,
        skillId: event.data.skill_id as string,
        status: "error",
        error: event.data.message as string,
      });
      s.updateToolCall(tcId, {
        status: "error",
        result: event.data.message as string,
        completedAt: event.ts,
      });
      s.updateChat(`tc-${tcId}`, {
        toolStatus: "error",
        toolResult: event.data.message as string,
      });
      break;
    }

    case "program_planned":
      s.setProgram(event.data.program as any);
      break;

    case "criteria_generated": {
      const criteria = (event.data.criteria as Array<Record<string, unknown>>) ?? [];
      const goalSummary = event.data.goal_summary as string | undefined;
      let criteriaText = "Success Criteria generated";
      if (goalSummary) criteriaText += `\nGoal: ${goalSummary}`;
      if (criteria.length > 0) {
        criteriaText += `\n${criteria.length} criterion/criteria:`;
        for (const c of criteria.slice(0, 5)) {
          criteriaText += `\n  [${c.kind}] ${c.description}`;
        }
      }
      s.pushChat({
        id: `criteria-${Date.now()}-${Math.random().toString(36).slice(2, 5)}`,
        kind: "thinking",
        ts: event.ts,
        text: criteriaText,
      });
      break;
    }

    case "goal_evaluation":
      s.setVerdict(event.data as any);
      break;

    case "learner_feedback":
      s.setLearnerFeedback(event.data as any);
      s.pushChat({
        id: `learner-fb-${Date.now()}-${Math.random().toString(36).slice(2, 5)}`,
        kind: "learner_feedback",
        ts: event.ts,
        text: JSON.stringify(event.data),
      });
      break;

    case "attempt_started":
      s.setAttempt(event.data.attempt as number);
      break;

    case "explorer_progress":
      s.setPhaseProgress(
        "explorer",
        event.data.current as number,
        event.data.total as number
      );
      break;

    case "status_update": {
      const currentPhase = s.currentPhase;
      if (currentPhase) {
        s.setPhase(currentPhase, "running", event.data.text as string);
      }
      break;
    }

    case "context_usage":
      s.setContextUsage(
        event.data.used_tokens as number,
        event.data.max_tokens as number
      );
      break;

    case "executor_progress":
      s.setPhaseProgress(
        "executor",
        event.data.current as number,
        event.data.total as number
      );
      break;
  }
}

export function applyNotification(method: string, params: Record<string, unknown>, store?: Store): void {
  const s = store ?? useAgentStore.getState();

  switch (method) {
    case "event": {
      const event = params as unknown as TraceEvent;
      s.pushEvent(event);
      applyEvent(event, s);
      break;
    }
    case "thinking": {
      const text = params.text as string;
      s.appendThinking(text);
      const msgs = s.chatMessages;
      const last = msgs[msgs.length - 1];
      if (last && last.kind === "thinking" && !last.toolName) {
        s.updateChat(last.id, { text: (last.text || "") + text });
      } else {
        s.pushChat({
          id: `think-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          kind: "thinking",
          ts: new Date().toISOString(),
          text,
        });
      }
      break;
    }
    case "thinking_clear": {
      s.clearThinking();
      s.pushChat({
        id: `sep-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        kind: "phase",
        ts: new Date().toISOString(),
        phase: s.currentPhase,
        phaseStatus: "running",
      });
      break;
    }
  }
}
