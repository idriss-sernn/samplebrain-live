import {
  initialize,
  type ActivationContext,
} from "@ableton-extensions/sdk";

const runtimeModulePath = "./runtime.cjs";

export function activate(activation: ActivationContext) {
  const ctx = initialize(activation, "1.0.0");

  ctx.commands.registerCommand("samplebrain.process", (arg: unknown) => {
    void import(runtimeModulePath)
      .then((module) => (module.run ?? module.default?.run)(ctx, arg))
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
