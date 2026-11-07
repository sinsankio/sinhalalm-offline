import { isLinuxDesktop } from "../../worker/runtime.js";

const LINUX_COMMAND =
  "google-chrome --user-data-dir=$HOME/.sinhalalm-chrome --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan";

export default function GpuHint({ gpu }) {
  const platform = navigator.userAgentData?.platform ?? navigator.platform ?? "";
  const linux = isLinuxDesktop(platform, navigator.userAgent);
  const reason =
    gpu === "software"
      ? "This browser offers WebGPU only in software"
      : "This browser does not offer WebGPU";

  return (
    <div className="gpu-hint">
      <p>
        {reason}, so answers are generated on the processor and are slow. A browser with GPU
        acceleration is several times faster.
      </p>
      {linux && (
        <details>
          <summary>Using the GPU in Chrome on Linux</summary>
          <p>
            Chrome on Linux does not enable WebGPU by default. Quit Chrome completely, then start it
            with a separate profile:
          </p>
          <code>{LINUX_COMMAND}</code>
          <p>
            These flags turn on WebGPU features Chrome has not yet approved for Linux, so use this
            profile only for this application.
          </p>
        </details>
      )}
    </div>
  );
}
