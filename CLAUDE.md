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
