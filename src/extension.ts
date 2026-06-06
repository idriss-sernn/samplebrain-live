import {
  initialize,
  type ActivationContext,
} from "@ableton-extensions/sdk";

export function activate(activation: ActivationContext) {
  const ctx = initialize(activation, "1.0.0");

  ctx.commands.registerCommand("samplebrain.process", (arg: unknown) => {
    void import("./runtime.js")
      .then(({ run }) => run(ctx, arg))
      .catch(console.error);
  });

  console.log("SampleBrain: extension loaded");
  ctx.ui.registerContextMenuAction(
    "AudioTrack.ArrangementSelection",
    "Process Selection…",
    "samplebrain.process",
  );
  ctx.ui.registerContextMenuAction("AudioClip", "Process Clip…", "samplebrain.process");
}
