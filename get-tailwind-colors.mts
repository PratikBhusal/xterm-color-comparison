import { execSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

type TailwindColorShades = Record<string, string>;
type TailwindColors = Record<string, string | TailwindColorShades>;

const scriptDir: string = dirname(fileURLToPath(import.meta.url));
const output: string = join(scriptDir, "tailwind-colors.json");

const tmp: string = mkdtempSync(join(tmpdir(), "tw-colors-"));
try {
  execSync("pnpm install tailwindcss", { cwd: tmp, stdio: "inherit" });
  const require: NodeRequire = createRequire(join(tmp, "index.js"));
  const resolved: string = require.resolve("tailwindcss/colors");
  const colorsModule: { default: TailwindColors } = await import(resolved);
  const colors: TailwindColors = colorsModule.default;
  writeFileSync(output, JSON.stringify(colors, null, 2) + "\n");
  console.log(`Wrote ${output}`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
