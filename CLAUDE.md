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

## Code Change Thoroughness
- When modifying code, trace through the ENTIRE code flow to find related changes needed
- Check for variables that are set in one place but used/reset in another
- Look for similar patterns elsewhere in the codebase that may need the same fix
- Don't assume a fix works in isolation - verify data flows correctly through all code paths
- Before declaring a fix complete, search for other usages of the same variable/function
