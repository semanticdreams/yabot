# yabot - yet another bot

## Running

- Install dependencies: `uv pip install -e .`
- Run Matrix bot: `uv run --project . yabot-matrix`
- Run CLI: `uv run --project . yabot-cli`
- Run daemon: `uv run --project . yabotd`
- Tool install (refresh tool env): `uv tool install --force --editable .`

## Environment variables

- `OPENAI_API_KEY`: required.
- `MATRIX_HOMESERVER`: defaults to `https://matrix.org`.
- `MATRIX_USER`: required for Matrix login.
- `MATRIX_PASSWORD`: required on first login and for cross-signing reset UIA.
- `BOT_NAME`: defaults to `yabot`.
- `ALLOWED_USERS`: comma-separated allowlist; empty means allow all.
- `CROSS_SIGNING_RESET`: set to `1`/`true`/`yes` to force a cross-signing reset.
- `YABOT_DAEMON_URL`: when set, the CLI and Matrix bot connect to a running daemon instead of running the agent locally.
- `YABOT_CLI_DAEMON_AUTOSTART`: when set to `1`/`true`/`yes`, the CLI will start the daemon if it is not reachable.
- `YABOT_DAEMON_HOST`: host interface for the daemon (default `127.0.0.1`).
- `YABOT_DAEMON_PORT`: port for the daemon (default `8765`).
- `YABOT_TRACE_PATH`: optional path for JSONL trace logs; defaults to the per-user log directory.

## Logging & tracing

The agent writes JSONL traces (prompts, tool calls, approvals, responses, metadata) to the per-user log dir.
By default this is under your platform log directory for the app name, with `trace.jsonl` as the filename.
Each trace line includes `schema_version`, `event`, `trace_id`, `room_id`, `conv_id`, and `model` when available.

## Daemon mode (shared state)

Start the daemon once, then point both clients at it:

- `uv run --project . yabotd`
- `YABOT_DAEMON_URL=ws://127.0.0.1:8765 uv run --project . yabot-matrix`
- `YABOT_DAEMON_URL=ws://127.0.0.1:8765 uv run --project . yabot-cli`

## CLI usage

The CLI uses Textual and supports the same commands as the Matrix bot. Use `!help` to see them.

Key bindings:

- `Ctrl+N`: new conversation
- `Ctrl+R`: reset conversation
- `Ctrl+L`: list conversations
- `Ctrl+M`: list models
- `Ctrl+S`: stop current response
- `Ctrl+H`: show help
- `Ctrl+Q`: quit

## Allowlist behavior

If `ALLOWED_USERS` is set, the bot ignores all senders not in the list. This is enforced in the message handler and is silent aside from logs.

## Logging

The bot logs to stdout with INFO level by default. Message handling logs include sender, room ID, and decrypted/verified flags.

## Encrypted rooms and auto-join

- The bot auto-joins when invited to a room.
- It auto-trusts devices in its own trust store after each sync.

## Cross-signing

The bot maintains cross-signing keys in:

- `~/.local/share/yabot/cross_signing.json`
- `~/.local/share/yabot/cross_signing_uia.json` (UIA reset session)

### Normal behavior

On each sync, the bot:

1. Ensures cross-signing keys exist locally.
2. Uploads cross-signing keys if missing on the server.
3. Signs all of the bot's devices with the self-signing key.

This removes "encrypted by a device not verified by its owner" warnings in other clients once they refresh keys.

### Reset flow (Matrix.org)

If server keys are out of sync or need to be replaced:

1. Set `CROSS_SIGNING_RESET=1` in `.env`.
2. Ensure `MATRIX_PASSWORD` is set.
3. Restart the bot once. It will log a reset approval URL.
4. Open the URL in a browser **logged in as the bot account** and approve.
5. Restart the bot again. It should log:
   - `Uploaded cross-signing keys`
   - `Signed N device(s)`
6. Remove `CROSS_SIGNING_RESET` from `.env`.

If the bot keeps logging "Cross-signing reset requires approval", the approval is not tied to the bot account session.

### Refreshing in clients

After a successful reset, some clients need a refresh to clear the warning:

- Log out/in, or
- Use a "Refresh keys" option if the client has one.

## Common issues

### Bot receives messages but does not reply

- Check allowlist: `ALLOWED_USERS` must include the sender.
- Check logs for `room_send` errors (E2EE member sync or permissions).

### `peewee.OperationalError: unable to open database file`

- Ensure the data dir is writable.
- The bot creates `~/.local/share/yabot/nio_store` on startup.

### Cross-signing mismatch

- If you see "Server master key does not match local key file", run the reset flow above.
- If you lost the local key file but server keys exist, do not regenerate without resetting on the server.
