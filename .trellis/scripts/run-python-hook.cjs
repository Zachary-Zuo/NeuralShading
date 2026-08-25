#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const script = process.argv[2];
const scriptArgs = process.argv.slice(3);
const projectCondaEnv = process.env.TRELLIS_CONDA_ENV || "neural-shading";

if (!script) {
  console.error("Usage: node .trellis/scripts/run-python-hook.cjs <hook.py> [args...]");
  process.exit(2);
}

function candidateList() {
  const candidates = [];
  const envPython = process.env.TRELLIS_PYTHON;
  if (envPython) {
    candidates.push({ command: envPython, args: [], label: "TRELLIS_PYTHON" });
  }

  const condaExe = process.env.CONDA_EXE;
  if (condaExe) {
    const condaRoot = path.dirname(path.dirname(condaExe));
    const envPython = path.join(
      condaRoot,
      "envs",
      projectCondaEnv,
      os.platform() === "win32" ? "python.exe" : "bin/python",
    );
    if (fs.existsSync(envPython)) {
      candidates.push({
        command: envPython,
        args: [],
        label: `conda env ${projectCondaEnv}`,
      });
    }
  }

  const condaArgs = [
    "run",
    "--no-capture-output",
    "-n",
    projectCondaEnv,
    "python",
  ];
  if (condaExe) {
    candidates.push({
      command: condaExe,
      args: condaArgs,
      label: `conda env ${projectCondaEnv}`,
    });
  }
  candidates.push({
    command: "conda",
    args: condaArgs,
    label: `conda env ${projectCondaEnv}`,
  });

  if (os.platform() === "win32") {
    candidates.push(
      { command: "python", args: [], label: "python" },
      { command: "python3", args: [], label: "python3" },
      { command: "py", args: ["-3"], label: "py -3" },
    );
  } else {
    candidates.push(
      { command: "python3", args: [], label: "python3" },
      { command: "python", args: [], label: "python" },
    );
  }

  return candidates;
}

function isPython3(candidate) {
  const result = spawnSync(
    candidate.command,
    [
      ...candidate.args,
      "-c",
      "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)",
    ],
    {
      encoding: "utf8",
      windowsHide: true,
    },
  );
  return result.status === 0;
}

function findPython() {
  for (const candidate of candidateList()) {
    if (isPython3(candidate)) {
      return candidate;
    }
  }
  return null;
}

const python = findPython();
if (!python) {
  console.error(
    `Trellis hook error: no Python 3 executable found. Set TRELLIS_PYTHON, ` +
      `or create the ${projectCondaEnv} Conda environment.`,
  );
  process.exit(127);
}

const result = spawnSync(
  python.command,
  [...python.args, "-X", "utf8", script, ...scriptArgs],
  {
    stdio: "inherit",
    windowsHide: true,
  },
);

if (result.error) {
  console.error(
    `Trellis hook error: failed to run Python via ${python.label}: ${result.error.message}`,
  );
  process.exit(1);
}

process.exit(typeof result.status === "number" ? result.status : 1);
