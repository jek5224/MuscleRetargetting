# Rules

## Git Commit Rules
- Always commit after implementing new functions or making significant changes
- Commit before switching branches or running git checkout
- Never discard uncommitted changes without explicit user permission
- If a file has more than 50 lines of changes, commit immediately after implementation
- Create descriptive commit messages that explain what was added/changed

## Before Destructive Git Operations
- Before `git checkout`, `git reset`, or `git stash`: check for uncommitted changes and commit them first
- Warn the user if there are uncommitted changes that could be lost
- Never use `git revert`. Instead, manually undo the changes with edits.
- **NEVER use git to roll back code** — no `git stash`, `git checkout -- <file>`, `git restore`, `git reset`, or any git command that discards or reverts changes. Always fix forward by editing the code directly.

## Code Change Thoroughness
- When modifying code, trace through the ENTIRE code flow to find related changes needed
- Check for variables that are set in one place but used/reset in another
- Look for similar patterns elsewhere in the codebase that may need the same fix
- Don't assume a fix works in isolation - verify data flows correctly through all code paths
- Before declaring a fix complete, search for other usages of the same variable/function

## Journal
- After completing a meaningful task (feature, fix, etc.), log progress to the research journal MCP
- Append to the same day's entry if there are multiple tasks in one day
- Keep entries concise: what was done, key decisions, and any open issues

## A6000 Server
- Always use **slurm** (`sbatch`) to submit jobs on the A6000 server — never run GPU tasks directly
- No `--mem` flag (slurm RealMemory=1, memory requests fail)

## No Guessing
- Do NOT make claims about code behavior, APIs, or research without first reading the actual code or searching for references
- If you don't know, say so — don't fabricate plausible-sounding explanations
- When discussing research ideas or biomechanics, cite sources or explicitly mark speculation as such
- Always read the relevant code before explaining how something works

## No Unsolicited Fallbacks
- Do NOT add fallback logic or edge case handling unless explicitly asked
- If you think a fallback is necessary, ASK the user first before implementing
- Stick to what was requested - don't proactively "protect" against cases the user didn't mention
