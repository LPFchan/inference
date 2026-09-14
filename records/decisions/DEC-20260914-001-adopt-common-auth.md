# DEC-20260914-001: Adopt Common Auth for Chat and API Access

Opened: 2026-09-14 20-33-07 KST
Recorded by agent: root

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Related ids: none

## Decision

Use `auth.lost.plus` as Grimoire's only public identity and credential authority.

Browser access uses the shared `lp_auth` cookie and the `chat` visibility key. Bearer and `X-API-Key` requests validate against the `chat-v1` machine-token service. The public proxy validates every request and forwards only the immutable account `sub` and shared role to the loopback manager. Account-bound Grimoire state is keyed from that stable `sub`, not from a bearer secret.

Expose common-auth token management in chat settings through a same-origin backend proxy. This view changes the same global/per-service token state shown at `auth.lost.plus`; Grimoire does not copy token state or retain bearer secrets.

## Context

Grimoire previously treated one app-specific bearer secret as browser login, API credential, account identity, and administrator credential. Rotating that secret would also change the history/settings/usage owner. The web UI stored the secret in JavaScript-readable local storage.

Common auth now provides shared browser sessions, immutable string subjects, fleet roles, service visibility, and global or per-service machine tokens. Chat needs both browser and API forms.

## Options Considered

### Keep Grimoire's Existing Authentication

- Avoids a cutover.
- Keeps a second identity, role, and credential implementation.
- Keeps browser credentials in local storage and binds stored data to a rotatable secret.

### Copy Common-Auth Token State Into Grimoire

- Makes the chat settings view locally available.
- Introduces synchronization and two sources of truth for security-sensitive state.

### Validate and Manage Directly Against Common Auth

- Keeps one source of truth.
- Supports both shared browser login and scoped machine tokens.
- Requires the public proxy to fail closed when common auth is unavailable.

## Rationale

Direct validation and management give the requested UX at both `auth.lost.plus` and chat settings without state synchronization. Using `sub` separates durable account ownership from credential rotation. Keeping validation at the public proxy preserves the existing multi-worker/loopback-manager boundary and prevents public clients from asserting internal identity headers.

## Consequences

- Common-auth availability is required for new public Grimoire requests.
- Browser sessions must have `chat` visibility; machine tokens must be active for `chat-v1`.
- The web UI no longer accepts or stores an API key.
- Token secrets appear only in the immediate mint/rotate response.
- Grimoire administrator routes use common auth's `administrator` role.
- Existing bearer and account-owned state require a guarded one-time production migration.
