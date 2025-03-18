#!/usr/bin/env node
const path = require("path");
const fs = require("fs");
const pty = require("node-pty");
const os = require("os");

// Try to find claude executable - assuming it might be in PATH
// or in the same directory as this wrapper
function findClaudeExecutable() {
  // First try the PATH
  const pathEnv = process.env.PATH || "";
  const pathDirs = pathEnv.split(path.delimiter);
  for (const dir of pathDirs) {
    const fullPath = path.join(dir, "claude");
    if (fs.existsSync(fullPath)) {
      return fullPath;
    }
  }
  // Try looking in the same directory as this script
  const localPath = path.join(__dirname, "claude");
  if (fs.existsSync(localPath)) {
    return localPath;
  }
  console.error(
    "Unable to find claude executable. Please make sure it is in your PATH or in the same directory as this wrapper.",
  );
  process.exit(1);
}

const claudePath = findClaudeExecutable();
console.log(`Found Claude at: ${claudePath}`);

// Add --debug to the arguments
const claudeArgs = ["--debug", ...process.argv.slice(2)];
console.log(`Running with arguments: ${claudeArgs.join(" ")}`);

// Create a pseudoterminal
const ptyProcess = pty.spawn(claudePath, claudeArgs, {
  name: "xterm-color",
  cols: process.stdout.columns || 80,
  rows: process.stdout.rows || 30,
  cwd: process.cwd(),
  env: process.env,
});

console.log(`[claudewrapper] Starting Claude with PTY in debug mode`);

// Store chunks to detect inquirer select prompts
let buffer = "";
const INQUIRER_SELECT_PATTERN = /❯.*Yes/;

// Handle PTY output
ptyProcess.onData((data) => {
  process.stdout.write(data); // Forward output
  buffer += data;
  // Detect if this is an inquirer select prompt with a "Yes" option
  if (INQUIRER_SELECT_PATTERN.test(buffer)) {
    console.log('\n[claudewrapper] Auto-selecting "Yes"');
    ptyProcess.write("\r"); // Send return/enter key
    buffer = "";
  }
  // Keep the buffer from growing too large by truncating it
  if (buffer.length > 2000 && !INQUIRER_SELECT_PATTERN.test(buffer)) {
    buffer = buffer.substring(buffer.length - 1000);
  }
});

// Handle user input
process.stdin.on("data", (data) => {
  ptyProcess.write(data.toString());
});

// Make stdin raw to handle special keys properly
if (process.stdin.isTTY) {
  process.stdin.setRawMode(true);
}
process.stdin.resume();

// Handle process exit
ptyProcess.onExit(({ exitCode }) => {
  console.log(`\n[claudewrapper] Claude exited with code ${exitCode}`);
  process.exit(exitCode);
});

// Handle signals
["SIGINT", "SIGTERM", "SIGQUIT"].forEach((signal) => {
  process.on(signal, () => {
    ptyProcess.kill();
    process.exit();
  });
});
