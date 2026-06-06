import {
  initialize,
  type ActivationContext,
} from "@ableton-extensions/sdk";
import { run } from "./runtime.js";

export function activate(activation: ActivationContext) {
  const ctx = initialize(activation, "1.0.0");

  ctx.commands.registerCommand("samplebrain.process", (arg: unknown) => {
    void run(ctx, arg).catch(console.error);
  });

  console.log("SampleBrain: extension loaded");
  ctx.ui.registerContextMenuAction(
    "AudioTrack.ArrangementSelection",
    "SampleBrain: Process Selection...",
    "samplebrain.process",
  );
  ctx.ui.registerContextMenuAction("AudioClip", "SampleBrain: Process Clip...", "samplebrain.process");
}
