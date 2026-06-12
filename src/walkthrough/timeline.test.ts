import { describe, expect, it, vi } from "vitest";
import type { T } from "../compute/tensor";
import type { TensorView } from "../engine/tensorView";
import {
  type SceneBinding,
  type TimelineSpec,
  type TimelineStep,
  TimelinePlayer,
} from "./timeline";

/** Mock binding: every channel is a vi.fn(); runForward returns `acts`. */
function makeBinding() {
  const acts = new Map<string, T>();
  const binding = {
    views: new Map<string, TensorView>(),
    runForward: vi.fn((_uptoToken: number) => acts),
    applyActivations: vi.fn(),
    setHighlight: vi.fn(),
    setDim: vi.fn(),
    pulse: vi.fn(),
  } satisfies SceneBinding;
  return { binding, acts };
}

function highlight(names: string[], durationMs: number): TimelineStep {
  return { kind: "highlight", names, durationMs };
}

function wait(durationMs: number): TimelineStep {
  return { kind: "wait", durationMs };
}

function loadAndPlay(spec: TimelineSpec) {
  const { binding, acts } = makeBinding();
  const player = new TimelinePlayer(binding);
  player.load(spec);
  player.play();
  return { binding, acts, player };
}

describe("TimelinePlayer sequential execution", () => {
  it("runs steps in order with explicit timestamps", () => {
    const { binding, player } = loadAndPlay({
      steps: [highlight(["a"], 100), { kind: "pulse", from: "x", to: "y", durationMs: 50 }, wait(50)],
    });

    // Entering step 0 happens at play(); the clock opens on the 1st update.
    expect(player.state).toBe("playing");
    expect(player.stepIndex).toBe(0);
    expect(binding.setHighlight).toHaveBeenCalledExactlyOnceWith(["a"], true);
    expect(binding.pulse).not.toHaveBeenCalled();

    player.update(1000); // opens the clock
    expect(player.stepIndex).toBe(0);

    player.update(1099); // 99ms < 100ms: still step 0
    expect(player.stepIndex).toBe(0);
    expect(binding.pulse).not.toHaveBeenCalled();

    player.update(1100); // step 0 done -> enter step 1
    expect(player.stepIndex).toBe(1);
    expect(binding.pulse).toHaveBeenCalledExactlyOnceWith("x", "y");

    player.update(1150); // step 1 done -> enter step 2 (wait)
    expect(player.stepIndex).toBe(2);
    expect(player.state).toBe("playing");

    player.update(1200); // step 2 done -> complete
    expect(player.state).toBe("idle");
  });

  it("a large delta crosses several steps, firing every effect once each", () => {
    const { binding, player } = loadAndPlay({
      steps: [highlight(["a"], 100), { kind: "pulse", from: "x", to: "y", durationMs: 50 }, wait(50)],
    });
    player.update(0);
    player.update(10_000); // jumps past all three steps in one update
    expect(binding.pulse).toHaveBeenCalledTimes(1);
    expect(binding.setHighlight).toHaveBeenCalledWith(["a"], false);
    expect(player.state).toBe("idle");
  });

  it("fires a step's effect once per entry, not once per update", () => {
    const { binding, player } = loadAndPlay({ steps: [highlight(["a"], 100)] });
    player.update(0);
    player.update(30);
    player.update(60);
    const onCalls = binding.setHighlight.mock.calls.filter(([, on]) => on === true);
    expect(onCalls).toEqual([[["a"], true]]);
  });
});

describe("highlight step", () => {
  it("turns highlight on at entry and off at exit", () => {
    const { binding, player } = loadAndPlay({ steps: [highlight(["q", "k"], 100), wait(100)] });
    expect(binding.setHighlight).toHaveBeenCalledExactlyOnceWith(["q", "k"], true);

    player.update(0);
    player.update(100); // exit highlight step
    expect(binding.setHighlight).toHaveBeenCalledTimes(2);
    expect(binding.setHighlight).toHaveBeenLastCalledWith(["q", "k"], false);
  });
});

describe("focus step", () => {
  it("dims others + highlights at entry; restores both at exit", () => {
    const { binding, player } = loadAndPlay({
      steps: [{ kind: "focus", names: ["ssm"], durationMs: 100 }, wait(100)],
    });
    expect(binding.setDim).toHaveBeenCalledExactlyOnceWith(["ssm"]);
    expect(binding.setHighlight).toHaveBeenCalledExactlyOnceWith(["ssm"], true);

    player.update(0);
    player.update(100); // exit focus step
    expect(binding.setHighlight).toHaveBeenLastCalledWith(["ssm"], false);
    expect(binding.setDim).toHaveBeenLastCalledWith(null);
  });
});

describe("stepToken step", () => {
  it("calls runForward(token) then applyActivations with its result at entry", () => {
    const { binding, acts } = loadAndPlay({
      steps: [{ kind: "stepToken", token: 3, durationMs: 100 }],
    });
    expect(binding.runForward).toHaveBeenCalledExactlyOnceWith(3);
    expect(binding.applyActivations).toHaveBeenCalledTimes(1);
    expect(binding.applyActivations.mock.calls[0][0]).toBe(acts); // exact map, not a copy
    expect(binding.runForward.mock.invocationCallOrder[0]).toBeLessThan(
      binding.applyActivations.mock.invocationCallOrder[0],
    );
  });
});

describe("pause / resume", () => {
  it("pause freezes mid-step; resume continues with the REMAINING duration", () => {
    const { player } = loadAndPlay({ steps: [highlight(["a"], 100), wait(100)] });
    player.update(0);
    player.update(60); // 60ms into step 0
    player.pause();
    expect(player.state).toBe("paused");

    player.update(5000); // paused: must not advance
    expect(player.stepIndex).toBe(0);

    player.play(); // resume
    expect(player.state).toBe("playing");
    player.update(9000); // reopens the clock; absolute gap must NOT count
    expect(player.stepIndex).toBe(0);
    player.update(9039); // 60 + 39 = 99ms: still step 0
    expect(player.stepIndex).toBe(0);
    player.update(9040); // 60 + 40 = 100ms: step 0 done
    expect(player.stepIndex).toBe(1);
  });

  it("pause when not playing is a no-op", () => {
    const { binding } = makeBinding();
    const player = new TimelinePlayer(binding);
    player.load({ steps: [wait(100)] });
    player.pause();
    expect(player.state).toBe("idle");
  });
});

describe("stop", () => {
  it("resets to start and clears all highlights + dim", () => {
    const { binding, player } = loadAndPlay({
      steps: [{ kind: "focus", names: ["a"], durationMs: 100 }, highlight(["b"], 100)],
    });
    player.update(0);
    player.update(100); // now inside the highlight(['b']) step
    player.stop();

    expect(player.state).toBe("idle");
    expect(player.stepIndex).toBe(0);
    expect(binding.setDim).toHaveBeenLastCalledWith(null);
    // every name highlighted during the run is switched off
    expect(binding.setHighlight).toHaveBeenLastCalledWith(["a", "b"], false);
  });

  it("update after stop is a no-op", () => {
    const { binding, player } = loadAndPlay({ steps: [highlight(["a"], 100), wait(100)] });
    player.update(0);
    player.stop();
    binding.setHighlight.mockClear();
    player.update(10_000);
    expect(player.stepIndex).toBe(0);
    expect(player.state).toBe("idle");
    expect(binding.setHighlight).not.toHaveBeenCalled();
  });

  it("stop while idle is a no-op (no spurious binding calls)", () => {
    const { binding } = makeBinding();
    const player = new TimelinePlayer(binding);
    player.load({ steps: [wait(100)] });
    player.stop();
    expect(binding.setDim).not.toHaveBeenCalled();
    expect(binding.setHighlight).not.toHaveBeenCalled();
  });
});

describe("loop", () => {
  it("wraps back to step 0 and keeps playing", () => {
    const { binding, player } = loadAndPlay({
      steps: [highlight(["a"], 100), wait(100)],
      loop: true,
    });
    player.update(0);
    player.update(200); // end of last step -> wrap
    expect(player.stepIndex).toBe(0);
    expect(player.state).toBe("playing");
    // entry effect fired again on the wrapped entry
    const onCalls = binding.setHighlight.mock.calls.filter(([, on]) => on === true);
    expect(onCalls).toEqual([
      [["a"], true],
      [["a"], true],
    ]);
  });

  it("does not fire onComplete while looping", () => {
    const { player } = loadAndPlay({ steps: [wait(50)], loop: true });
    const onComplete = vi.fn();
    player.onComplete = onComplete;
    player.update(0);
    player.update(500);
    expect(onComplete).not.toHaveBeenCalled();
    expect(player.state).toBe("playing");
  });
});

describe("callbacks", () => {
  it("onStepStart fires once per entry with (step, index); onComplete fires at the end", () => {
    const { binding } = makeBinding();
    const player = new TimelinePlayer(binding);
    const spec: TimelineSpec = { steps: [highlight(["a"], 100), wait(50)] };
    const started: Array<[string, number]> = [];
    player.onStepStart = (step, index) => started.push([step.kind, index]);
    const onComplete = vi.fn();
    player.onComplete = onComplete;

    player.load(spec);
    player.play();
    player.update(0);
    player.update(40); // still step 0: no new starts
    player.update(100);
    player.update(150);

    expect(started).toEqual([
      ["highlight", 0],
      ["wait", 1],
    ]);
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(player.state).toBe("idle");
    expect(player.stepIndex).toBe(0); // reset for a future play()
  });
});

describe("edge cases", () => {
  it("empty spec: play() completes immediately without touching the binding", () => {
    const { binding } = makeBinding();
    const player = new TimelinePlayer(binding);
    const onComplete = vi.fn();
    player.onComplete = onComplete;
    player.load({ steps: [] });
    player.play();
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(player.state).toBe("idle");
    expect(binding.setHighlight).not.toHaveBeenCalled();
    expect(binding.setDim).not.toHaveBeenCalled();
    expect(binding.runForward).not.toHaveBeenCalled();
  });

  it("update while idle is a no-op", () => {
    const { binding } = makeBinding();
    const player = new TimelinePlayer(binding);
    player.load({ steps: [highlight(["a"], 100)] });
    player.update(0);
    player.update(100);
    expect(player.state).toBe("idle");
    expect(binding.setHighlight).not.toHaveBeenCalled();
  });

  it("play() without a loaded spec is a no-op", () => {
    const { binding } = makeBinding();
    const player = new TimelinePlayer(binding);
    const onComplete = vi.fn();
    player.onComplete = onComplete;
    player.play();
    expect(player.state).toBe("idle");
    expect(onComplete).not.toHaveBeenCalled();
    expect(binding.setHighlight).not.toHaveBeenCalled();
  });

  it("play() while already playing is a no-op (no re-entry of step 0)", () => {
    const { binding, player } = loadAndPlay({ steps: [highlight(["a"], 100)] });
    player.update(0);
    player.play();
    const onCalls = binding.setHighlight.mock.calls.filter(([, on]) => on === true);
    expect(onCalls).toHaveLength(1);
    expect(player.state).toBe("playing");
  });

  it("load() mid-play clears highlights/dim before swapping specs", () => {
    const { binding, player } = loadAndPlay({
      steps: [{ kind: "focus", names: ["a"], durationMs: 100 }],
    });
    player.update(0);
    player.load({ steps: [wait(50)] });
    expect(player.state).toBe("idle");
    expect(player.stepIndex).toBe(0);
    expect(binding.setDim).toHaveBeenLastCalledWith(null);
    expect(binding.setHighlight).toHaveBeenLastCalledWith(["a"], false);
  });

  it("replays from the start after a natural completion", () => {
    const { binding, player } = loadAndPlay({ steps: [highlight(["a"], 100)] });
    player.update(0);
    player.update(100); // complete
    expect(player.state).toBe("idle");

    player.play();
    expect(player.state).toBe("playing");
    const onCalls = binding.setHighlight.mock.calls.filter(([, on]) => on === true);
    expect(onCalls).toHaveLength(2);
  });
});
