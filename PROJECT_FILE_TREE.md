# QDR Workbench — Project File Tree

Complete repository structure (466 files / 204 directories). Every file/folder carries a
one-line purpose annotation. This is the structure-first scaffold; all source files are empty
placeholders ready to clone-and-fill.

Legend of planes: **FE** frontend · **CP** control plane · **CE** compute engine ·
**CX** compute plane · **DP** data plane · **EDGE/OPS** edge + operability · **GOV** governance.

```
Project_QDR/
├── README.md                                 # repo map: how the tree mirrors the architecture
├── PROJECT_FILE_TREE.md                       # this document — annotated full tree
├── CODEOWNERS                                 # per-path review ownership (governance)
├── LICENSE                                    # software license
├── SECURITY.md                                # vulnerability disclosure policy
├── CONTRIBUTING.md                            # contribution + branching workflow
├── CHANGELOG.md                               # human-curated release notes
├── .gitignore                                 # ignored paths
├── .gitattributes                             # line-ending / linguist rules
├── .editorconfig                              # editor formatting baseline
├── .dockerignore                              # docker build context excludes
├── .nvmrc                                     # pinned Node version
├── .python-version                            # pinned Python version
├── .env.example                               # documented env-var template
├── package.json                               # pnpm workspace root (JS toolchain)
├── pnpm-workspace.yaml                        # JS workspace member globs
├── turbo.json                                 # Turborepo task graph + caching
├── pyproject.toml                             # uv workspace root (Python members)
├── uv.lock                                    # pinned Python lockfile (reproducibility)
├── Makefile                                   # canonical dev/build/test entrypoints
│
├── .github/                                   # ── CI/CD + repo automation ──
│   ├── pull_request_template.md               # PR checklist
│   ├── dependabot.yml                         # dependency update automation
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md                      # bug issue template
│   │   └── feature_request.md                 # feature issue template
│   └── workflows/
│       ├── ci.yml                             # umbrella CI (lint/typecheck/test matrix)
│       ├── web-ci.yml                         # frontend pipeline
│       ├── api-ci.yml                         # control-plane pipeline
│       ├── compute-engine-ci.yml             # science-package pipeline (incl. H₂ gate)
│       ├── services-ci.yml                    # dispatcher/worker/qpu-broker pipeline
│       ├── contracts-check.yml                # OpenAPI/AsyncAPI drift + codegen validation
│       ├── db-migration-check.yml             # Alembic head + RLS-policy lint
│       ├── terraform-plan.yml                 # IaC plan on PR
│       ├── security-scan.yml                  # SAST/deps/secret scanning
│       ├── cd-staging.yml                     # gated staging deploy
│       └── cd-prod.yml                        # gated production deploy
│
├── apps/
│   ├── web/                                   # ══ FRONTEND (FE) — Next.js RSC IDE ══
│   │   ├── README.md                          # frontend overview + surface map
│   │   ├── package.json                       # web package manifest
│   │   ├── next.config.mjs                    # Next.js config
│   │   ├── tsconfig.json                      # TypeScript config
│   │   ├── tailwind.config.ts                 # Tailwind theme
│   │   ├── postcss.config.mjs                 # PostCSS pipeline
│   │   ├── eslint.config.mjs                  # lint rules
│   │   ├── vitest.config.ts                   # unit-test config
│   │   ├── playwright.config.ts               # e2e-test config
│   │   ├── Dockerfile                         # web container image
│   │   ├── public/
│   │   │   ├── favicon.ico                     # app icon
│   │   │   └── molstar/                        # Mol* static assets
│   │   └── src/
│   │       ├── middleware.ts                   # auth gate + tenant routing (Next middleware)
│   │       ├── app/                            # App Router (RSC)
│   │       │   ├── layout.tsx                   # root layout
│   │       │   ├── page.tsx                     # landing / redirect
│   │       │   ├── globals.css                  # global styles
│   │       │   ├── (auth)/
│   │       │   │   ├── login/page.tsx           # login surface
│   │       │   │   └── accept-invite/page.tsx   # invite acceptance
│   │       │   ├── (workbench)/
│   │       │   │   ├── layout.tsx               # the IDE shell frame
│   │       │   │   ├── library/page.tsx         # surface 1 — library browser
│   │       │   │   ├── configs/[configId]/page.tsx  # surface 2 — FMO config editor
│   │       │   │   ├── structure/page.tsx       # surface 3 — 3D structure viewer
│   │       │   │   ├── runs/page.tsx            # surface 4 — run console / job queue
│   │       │   │   ├── results/[runId]/page.tsx # surface 5 — results dashboard
│   │       │   │   └── workspace/page.tsx       # surface 6 — workspace file tree
│   │       │   ├── actions/
│   │       │   │   ├── README.md                # server-action conventions
│   │       │   │   └── run-actions.ts           # mutations (submit/cancel run) as server actions
│   │       │   └── api/health/route.ts          # BFF health route handler
│   │       ├── components/
│   │       │   ├── ide-shell/
│   │       │   │   ├── IdeShell.tsx             # shell composition
│   │       │   │   └── IdeShell.test.tsx        # co-located unit test
│   │       │   ├── library-browser/
│   │       │   │   ├── LibraryBrowser.tsx       # surface-1 component
│   │       │   │   └── LibraryBrowser.test.tsx
│   │       │   ├── fmo-config-editor/
│   │       │   │   ├── FmoConfigEditor.tsx      # surface-2 component
│   │       │   │   └── FmoConfigEditor.test.tsx
│   │       │   ├── structure-viewer/
│   │       │   │   ├── StructureViewer.tsx      # surface-3 Mol* wrapper
│   │       │   │   └── StructureViewer.test.tsx
│   │       │   ├── run-console/
│   │       │   │   ├── RunConsole.tsx           # surface-4 component
│   │       │   │   └── RunConsole.test.tsx
│   │       │   ├── results-dashboard/
│   │       │   │   ├── ResultsDashboard.tsx     # surface-5 component
│   │       │   │   └── ResultsDashboard.test.tsx
│   │       │   ├── workspace-tree/
│   │       │   │   ├── WorkspaceTree.tsx        # surface-6 component
│   │       │   │   └── WorkspaceTree.test.tsx
│   │       │   └── ui/
│   │       │       ├── README.md                # design-system primitive conventions
│   │       │       ├── Button.tsx               # primitive
│   │       │       ├── Dialog.tsx               # primitive
│   │       │       └── DataTable.tsx            # primitive
│   │       ├── lib/
│   │       │   ├── api-client/
│   │       │   │   ├── README.md                # data-fetching conventions
│   │       │   │   ├── client.ts                # TanStack Query client over /api/v1
│   │       │   │   └── query-keys.ts            # centralized query keys
│   │       │   ├── ws-client/
│   │       │   │   ├── reconnecting-socket.ts   # reconnecting WSS client
│   │       │   │   └── channels.ts              # WS channel subscriptions
│   │       │   ├── rsc/data-access.ts           # server-only data access (RSC/BFF boundary)
│   │       │   ├── molstar/viewer-adapter.ts    # Mol* integration adapter
│   │       │   └── auth/session.ts              # session helpers
│   │       ├── stores/
│   │       │   ├── library-store.ts             # Zustand: library state
│   │       │   ├── config-store.ts              # Zustand: config-editor state
│   │       │   ├── run-store.ts                 # Zustand: run/queue state
│   │       │   └── workspace-store.ts           # Zustand: workspace-tree state
│   │       ├── hooks/
│   │       │   ├── use-live-run.ts              # live run subscription hook
│   │       │   └── use-capabilities.ts          # capability-flag gating hook
│   │       ├── types/
│   │       │   ├── README.md                    # types generated from packages/contracts
│   │       │   └── .gitkeep
│   │       ├── styles/tokens.css                # design tokens
│   │       └── tests/e2e/
│   │           ├── library.spec.ts              # e2e: browse library
│   │           └── run-lifecycle.spec.ts        # e2e: submit→observe→results
│   │
│   └── api/                                   # ══ CONTROL PLANE (CP) — FastAPI modular monolith ══
│       ├── README.md                          # control-plane overview + module map
│       ├── pyproject.toml                     # api package manifest
│       ├── Dockerfile                         # api container image
│       ├── alembic.ini                        # migration config (points to db/migrations)
│       ├── gunicorn.conf.py                   # ASGI server config
│       └── src/qdr_api/
│           ├── __init__.py
│           ├── main.py                        # ASGI app factory
│           ├── settings.py                    # pydantic-settings config
│           ├── api/
│           │   ├── __init__.py
│           │   ├── deps.py                     # shared FastAPI dependencies
│           │   └── v1/
│           │       ├── __init__.py
│           │       └── router.py               # aggregates module routers under /api/v1
│           ├── ws/
│           │   ├── __init__.py
│           │   ├── relay.py                    # /ws pub-sub relay
│           │   ├── manager.py                  # sticky-WS connection manager
│           │   └── channels.py                 # channel registry
│           ├── modules/                        # ── BOUNDED CONTEXTS (identical sibling shape) ──
│           │   ├── __init__.py
│           │   ├── auth/
│           │   │   ├── __init__.py
│           │   │   ├── router.py               # auth endpoints
│           │   │   ├── schemas.py              # request/response DTOs
│           │   │   ├── service.py              # domain logic
│           │   │   ├── repository.py           # persistence access
│           │   │   ├── models.py               # ORM models
│           │   │   └── tests/test_service.py   # co-located tests
│           │   ├── orgs/                        # orgs/teams (same 5-file + tests pattern)
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   └── tests/test_service.py
│           │   ├── projects/                    # projects (same pattern)
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   └── tests/test_service.py
│           │   ├── library/                     # curated drug/protein library (same pattern)
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   └── tests/test_service.py
│           │   ├── configs/                     # configs + estimator context
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   ├── estimator/
│           │   │   │   ├── __init__.py
│           │   │   │   ├── estimator.py         # pure-math estimator entrypoint (no compute)
│           │   │   │   ├── qubit_counter.py     # n=2·n_orb qubit accounting
│           │   │   │   ├── param_counter.py     # ansatz parameter count
│           │   │   │   ├── shot_estimator.py    # shot-budget estimation
│           │   │   │   ├── cost_model.py        # cost projection
│           │   │   │   └── qubit_cap_policy.py  # PRE-SUBMIT ≤40-qubit guard (reads qdr_core.limits)
│           │   │   └── tests/test_estimator.py
│           │   ├── jobs/                        # job intake/state (same pattern)
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   └── tests/test_service.py
│           │   ├── runs/                        # runs / compare
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   ├── compare.py               # multi-run comparison logic
│           │   │   └── tests/test_service.py
│           │   ├── campaigns/                   # batch screening campaigns
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   ├── orchestrator.py          # campaign lifecycle orchestration
│           │   │   ├── fan_out.py               # per-target job fan-out
│           │   │   ├── sweep.py                 # parameter-sweep expansion
│           │   │   └── tests/test_service.py
│           │   ├── billing/                     # billing/quotas (same pattern)
│           │   │   ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │   │   └── tests/test_service.py
│           │   └── backends/                    # backend catalog + capability gating
│           │       ├── __init__.py · router.py · schemas.py · service.py · repository.py · models.py
│           │       └── tests/test_service.py
│           ├── core/                           # ── CROSS-CUTTING HOMES ──
│           │   ├── __init__.py
│           │   ├── rbac/
│           │   │   ├── __init__.py
│           │   │   ├── roles.py                 # owner/admin/member/viewer
│           │   │   ├── capabilities.py          # capability-flag checks (qpu/backend-admin/billing/export)
│           │   │   ├── policy.py                # authorization policy
│           │   │   └── dependencies.py          # RBAC FastAPI dependencies
│           │   ├── rls/
│           │   │   ├── __init__.py
│           │   │   ├── session_context.py       # sets organization_id session var
│           │   │   └── tenant_middleware.py     # per-request tenant binding
│           │   ├── audit/
│           │   │   ├── __init__.py
│           │   │   ├── audit_log.py             # append-only audit writer
│           │   │   └── models.py                # audit ORM model
│           │   ├── db/
│           │   │   ├── __init__.py
│           │   │   ├── engine.py                # SQLAlchemy engine
│           │   │   ├── session.py               # session lifecycle
│           │   │   └── base_model.py            # declarative base
│           │   ├── cache/
│           │   │   ├── __init__.py
│           │   │   ├── redis_client.py          # Redis connection
│           │   │   └── pubsub.py                # pub/sub helpers
│           │   ├── storage/
│           │   │   ├── __init__.py
│           │   │   ├── s3_client.py             # S3 access
│           │   │   └── content_address.py       # content-addressed keys (delegates to qdr_core)
│           │   ├── provenance/
│           │   │   ├── __init__.py
│           │   │   └── artifact_ref.py          # thin adapter over qdr_core.provenance
│           │   ├── observability/
│           │   │   ├── __init__.py
│           │   │   ├── otel.py                  # OpenTelemetry setup
│           │   │   ├── logging.py               # structured logging
│           │   │   └── metrics.py               # metrics registry
│           │   ├── errors/
│           │   │   ├── __init__.py
│           │   │   ├── exceptions.py            # domain exceptions
│           │   │   └── handlers.py              # exception→HTTP handlers
│           │   └── security/
│           │       ├── __init__.py
│           │       ├── passwords.py             # password hashing
│           │       └── tokens.py                # token issue/verify
│           └── tests/
│               ├── conftest.py                  # api test fixtures
│               └── test_health.py               # smoke test
│
├── packages/
│   ├── qdr-core/                              # ══ SHARED PYTHON KERNEL (used by CP/CX/CE) ══
│   │   ├── README.md                          # kernel scope (precise, not a junk drawer)
│   │   ├── pyproject.toml                      # kernel package manifest
│   │   ├── src/qdr_core/
│   │   │   ├── __init__.py
│   │   │   ├── ids.py                          # typed IDs (OrgId/ProjectId/RunId…)
│   │   │   ├── tenant.py                       # TenantContext (organization_id carrier)
│   │   │   ├── errors.py                       # shared error taxonomy
│   │   │   ├── limits.py                       # platform invariants incl. MAX_QUBITS = 40
│   │   │   ├── capabilities.py                 # capability-flag enum (single source)
│   │   │   ├── events/
│   │   │   │   ├── __init__.py
│   │   │   │   └── envelope.py                 # canonical event/message envelope
│   │   │   ├── messaging/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── redis_streams.py            # queue/broker/log-stream client
│   │   │   │   └── channels.py                 # channel/stream name registry
│   │   │   └── provenance/
│   │   │       ├── __init__.py
│   │   │       ├── content_hash.py             # SINGLE canonical content-addressing
│   │   │       └── artifact_ref.py             # canonical artifact reference type
│   │   └── tests/
│   │       ├── test_content_hash.py            # hashing determinism test
│   │       └── test_limits.py                  # cap-invariant test
│   │
│   ├── contracts/                            # ══ SHARED CONTRACTS (contract-first) ══
│   │   ├── README.md                          # contract governance + codegen flow
│   │   ├── package.json                        # contracts package manifest
│   │   ├── codegen.config.ts                   # client/type generation config
│   │   ├── openapi/
│   │   │   ├── openapi.yaml                     # REST /api/v1 spec
│   │   │   └── components/
│   │   │       ├── schemas.yaml                 # shared schema components
│   │   │       └── responses.yaml              # shared response components
│   │   ├── asyncapi/
│   │   │   ├── ws-messages.yaml                 # client-facing WSS /ws messages
│   │   │   └── compute-plane.yaml              # INTERNAL job/lease/heartbeat/log/result/DLQ protocol
│   │   ├── json-schema/
│   │   │   ├── fmo-config.schema.json           # FMO simulation-config schema
│   │   │   ├── run-manifest.schema.json         # run manifest (repro inputs)
│   │   │   └── artifact-provenance.schema.json  # artifact provenance schema
│   │   ├── events/domain-events.yaml            # domain event catalog
│   │   └── generated/
│   │       ├── typescript/.gitkeep             # emitted TS types/client
│   │       └── python/.gitkeep                 # emitted pydantic models
│   │
│   └── compute-engine/                       # ══ COMPUTE ENGINE (CE) — the science ══
│       ├── README.md                          # engine overview: seam + pipeline
│       ├── pyproject.toml                      # engine package manifest
│       └── src/qdr_compute_engine/
│           ├── __init__.py
│           ├── types.py                        # engine-wide types
│           ├── pipeline/                        # ── 9 STAGES + STAGE-0 GATE (ordered, legible) ──
│           │   ├── __init__.py
│           │   ├── pipeline.py                  # orchestrates stages 1→9
│           │   ├── stage_context.py             # typed context passed between stages
│           │   ├── stage_00_fci_gate.py         # stage 0 — H₂→FCI correctness gate
│           │   ├── stage_01_geometry_pose.py    # stage 1 — geometry / pose
│           │   ├── stage_02_fmo_fragmenter.py   # stage 2 — FMO fragmentation
│           │   ├── stage_03_integrals.py        # stage 3 — PySCF integrals (h1,h2,E_nuc)
│           │   ├── stage_04_active_space_selector.py  # stage 4 — active space (n=2·n_orb qubits)
│           │   ├── stage_05_fermionic_hamiltonian.py  # stage 5 — fermionic Hamiltonian
│           │   ├── stage_06_qubit_mapper.py     # stage 6 — JW / parity / BK mapping
│           │   ├── stage_07_ansatz.py           # stage 7 — UCCSD / HEA / ADAPT
│           │   ├── stage_08_solver.py           # stage 8 — VQE / baseline solve
│           │   └── stage_09_validator.py        # stage 9 — ΔE_bind + error budget
│           ├── drivers/                         # ── THE SEAM: SolverBackend implementations ──
│           │   ├── README.md                    # "add a new backend here" checklist
│           │   ├── __init__.py
│           │   ├── solver_backend.py            # THE ONE interface (ABC)
│           │   ├── registry.py                  # backend discovery / factory
│           │   ├── capabilities.py              # per-driver capability flags
│           │   ├── classical/
│           │   │   ├── __init__.py
│           │   │   └── classical_driver.py      # DFT / CCSD(T) / DMRG / FCI
│           │   ├── statevector/
│           │   │   ├── __init__.py
│           │   │   └── statevector_driver.py    # exact statevector simulation
│           │   ├── sampled/
│           │   │   ├── __init__.py
│           │   │   └── sampled_driver.py         # shot-based sampled simulation
│           │   └── qpu/
│           │       ├── __init__.py
│           │       └── qpu_driver.py             # vendor QPU (feature-flagged)
│           ├── geometry/
│           │   ├── __init__.py
│           │   └── pose_builder.py               # pose/geometry preparation
│           ├── fragmentation/
│           │   ├── __init__.py
│           │   └── fmo_fragmenter.py             # FMO fragmentation algorithm
│           ├── integrals/
│           │   ├── __init__.py
│           │   └── pyscf_integrals.py            # PySCF integral extraction
│           ├── active_space/
│           │   ├── __init__.py
│           │   └── active_space_selector.py      # active-space selection
│           ├── hamiltonian/
│           │   ├── __init__.py
│           │   └── fermionic_builder.py          # fermionic Hamiltonian construction
│           ├── mappers/
│           │   ├── __init__.py
│           │   ├── jordan_wigner.py              # JW mapping
│           │   ├── parity.py                     # parity mapping
│           │   └── bravyi_kitaev.py              # BK mapping
│           ├── ansatze/
│           │   ├── __init__.py
│           │   ├── uccsd.py                       # UCCSD ansatz
│           │   ├── hardware_efficient.py          # HEA ansatz
│           │   └── adapt.py                       # ADAPT ansatz
│           ├── solvers/
│           │   ├── __init__.py
│           │   ├── vqe.py                          # VQE solver
│           │   └── classical_baseline.py          # classical baseline solver
│           ├── validation/
│           │   ├── __init__.py
│           │   ├── binding_energy.py              # ΔE_bind computation
│           │   └── error_budget.py               # error-budget accounting
│           ├── reproducibility/
│           │   ├── __init__.py
│           │   ├── seeds.py                        # RNG seed management
│           │   ├── version_pins.py                # pinned-version capture
│           │   └── input_hashing.py              # input hashing (via qdr_core)
│           ├── provenance/
│           │   ├── __init__.py
│           │   └── artifact_writer.py            # writes provenance artifacts
│           └── tests/
│               ├── conftest.py                    # engine test fixtures
│               ├── golden/h2_fci_reference.json   # stage-0 golden reference values
│               ├── test_stage_00_fci_gate.py      # H₂→FCI gate test
│               ├── test_qubit_mapper.py           # mapper correctness test
│               └── test_drivers_contract.py       # SolverBackend contract conformance
│
├── services/                                 # ══ COMPUTE PLANE (CX) ══
│   ├── dispatcher/                            # priority tiers · quotas · fair-share · admission
│   │   ├── README.md                          # dispatcher overview
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   ├── src/qdr_dispatcher/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                         # service entrypoint
│   │   │   ├── queue_consumer.py               # pulls from Redis queue
│   │   │   ├── priority_tiers.py               # priority-tier scheduling
│   │   │   ├── quota_manager.py                # per-org quota enforcement
│   │   │   ├── fair_share.py                   # fair-share arbitration
│   │   │   ├── pool_router.py                  # routes to cpu/gpu/qpu pools
│   │   │   ├── admission_control.py            # DISPATCH-TIME ≤40-qubit + capability check
│   │   │   ├── idempotency.py                  # duplicate-submission guard
│   │   │   ├── retry_policy.py                 # backoff/retry rules
│   │   │   ├── dead_letter.py                  # DLQ routing for poison jobs
│   │   │   └── queue_depth_metrics.py          # queue-depth signals for autoscaler
│   │   └── tests/test_fair_share.py
│   ├── compute-worker/                        # pulls job → runs compute-engine pipeline
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   ├── src/qdr_worker/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                         # worker entrypoint
│   │   │   ├── worker.py                       # lease/execute loop
│   │   │   ├── job_runner.py                   # invokes compute-engine pipeline
│   │   │   ├── heartbeat.py                    # liveness heartbeats
│   │   │   ├── log_streamer.py                 # streams logs → Redis log stream
│   │   │   └── result_publisher.py             # publishes results/artifacts
│   │   └── tests/test_job_runner.py
│   └── qpu-broker/                            # vendor QPU adapters + calibration
│       ├── README.md
│       ├── pyproject.toml
│       ├── Dockerfile
│       ├── src/qdr_qpu_broker/
│       │   ├── __init__.py
│       │   ├── main.py                         # broker entrypoint
│       │   ├── broker.py                       # QPU job brokering
│       │   ├── calibration.py                  # calibration/queue-window handling
│       │   └── vendor_adapters/
│       │       ├── __init__.py
│       │       └── base_adapter.py             # vendor-adapter interface
│       └── tests/test_broker.py
│
├── infra/                                     # ══ EDGE + OPERABILITY (EDGE/OPS) ══
│   ├── README.md                              # infra overview
│   ├── terraform/
│   │   ├── README.md
│   │   ├── modules/
│   │   │   ├── vpc/.gitkeep                     # network
│   │   │   ├── eks/.gitkeep                     # k8s cluster + node pools
│   │   │   ├── rds-postgres/.gitkeep           # Multi-AZ Postgres + replicas
│   │   │   ├── elasticache-redis/.gitkeep      # Redis
│   │   │   ├── s3-artifacts/.gitkeep           # versioned SSE-KMS bucket
│   │   │   ├── cloudfront-waf/.gitkeep         # CDN + WAF
│   │   │   ├── alb/.gitkeep                    # L7 load balancer (TLS/WSS)
│   │   │   ├── kms/.gitkeep                    # encryption keys
│   │   │   ├── iam/.gitkeep                    # roles/policies
│   │   │   └── observability/.gitkeep          # tracing/metrics backends
│   │   ├── environments/
│   │   │   ├── dev/{main.tf, variables.tf, backend.tf}       # dev env composition
│   │   │   ├── staging/{main.tf, variables.tf, backend.tf}   # staging env composition
│   │   │   └── prod/{main.tf, variables.tf, backend.tf}      # prod env composition
│   │   └── global/remote-state.tf              # shared remote-state backend
│   ├── kubernetes/
│   │   ├── README.md
│   │   ├── base/
│   │   │   ├── api/kustomization.yaml           # control-plane deployment
│   │   │   ├── web/kustomization.yaml           # frontend deployment
│   │   │   ├── dispatcher/kustomization.yaml    # dispatcher deployment
│   │   │   ├── compute-worker-cpu/kustomization.yaml   # cpu pool
│   │   │   ├── compute-worker-gpu/kustomization.yaml   # gpu pool
│   │   │   ├── qpu-broker/kustomization.yaml    # qpu broker deployment
│   │   │   └── shared/
│   │   │       ├── namespace.yaml               # namespace
│   │   │       ├── network-policies.yaml        # default-deny networking
│   │   │       ├── service-accounts.yaml        # IRSA service accounts
│   │   │       ├── otel-collector.yaml          # OTel collector
│   │   │       └── external-secrets.yaml        # External Secrets → KMS-backed secrets
│   │   ├── overlays/
│   │   │   ├── dev/kustomization.yaml           # dev overlay
│   │   │   ├── staging/kustomization.yaml       # staging overlay
│   │   │   └── prod/kustomization.yaml          # prod overlay
│   │   └── autoscaling/
│   │       ├── keda-scaledobject-cpu-pool.yaml  # scale cpu pool on queue depth
│   │       └── keda-scaledobject-gpu-pool.yaml  # scale gpu pool on queue depth
│   ├── secrets/                                # SOPS-encrypted secrets (SOC 2)
│   │   ├── README.md                           # secret-management workflow
│   │   ├── .sops.yaml                          # SOPS/KMS rules
│   │   ├── dev.enc.yaml                        # encrypted dev secrets
│   │   ├── staging.enc.yaml                    # encrypted staging secrets
│   │   └── prod.enc.yaml                       # encrypted prod secrets
│   ├── observability/                          # observability-as-code
│   │   ├── README.md
│   │   ├── slo/
│   │   │   ├── api-slos.yaml                    # control-plane SLOs
│   │   │   └── compute-slos.yaml               # compute-plane SLOs
│   │   ├── dashboards/
│   │   │   ├── queue-depth.json                 # dispatcher/queue dashboard
│   │   │   └── pipeline-latency.json            # pipeline-stage latency dashboard
│   │   └── alerts/alert-rules.yaml             # alerting rules
│   └── docker/compose/docker-compose.dev.yml   # local dev stack
│
├── db/                                        # ══ DATA PLANE (DP) ══
│   ├── README.md                              # data-plane overview
│   ├── migrations/
│   │   ├── env.py                              # Alembic environment
│   │   ├── script.py.mako                      # migration template
│   │   └── versions/.gitkeep                   # migration revisions
│   ├── rls/
│   │   ├── README.md                           # RLS policy governance
│   │   ├── default_deny.sql                    # default-deny baseline
│   │   └── policies/
│   │       ├── organizations.sql               # org isolation policy
│   │       ├── projects.sql                    # project isolation policy
│   │       ├── library.sql                     # library isolation policy
│   │       ├── runs.sql                        # runs isolation policy
│   │       └── audit_log.sql                   # audit isolation policy
│   ├── seeds/
│   │   ├── library/.gitkeep                    # curated drug/protein seed data
│   │   └── dev/.gitkeep                        # dev fixtures
│   └── schema/erd.md                           # entity-relationship reference
│
├── docs/                                      # ══ GOVERNANCE (GOV) ══
│   ├── README.md                              # docs index
│   ├── architecture/
│   │   ├── system-context.md                   # C4 system context
│   │   ├── planes.md                           # plane responsibilities
│   │   ├── compute-pipeline.md                 # 9-stage pipeline reference
│   │   ├── data-model.md                       # data model + tenancy
│   │   └── diagrams/.gitkeep                    # diagram sources
│   ├── adr/
│   │   ├── README.md                           # ADR process
│   │   ├── template.md                         # ADR template
│   │   ├── 0000-record-architecture-decisions.md
│   │   ├── 0001-domain-driven-monorepo-topology.md
│   │   ├── 0002-postgres-rls-primary-tenant-isolation.md
│   │   ├── 0003-solver-backend-driver-seam.md
│   │   ├── 0004-contract-first-api.md
│   │   ├── 0005-modular-monolith-control-plane.md
│   │   ├── 0006-forty-qubit-simulator-cap.md
│   │   └── 0007-content-addressed-artifact-provenance.md
│   ├── security/
│   │   ├── threat-model.md                      # threat model
│   │   ├── soc2-controls.md                     # SOC 2 control mapping
│   │   └── gdpr.md                              # GDPR readiness
│   ├── runbooks/
│   │   ├── incident-response.md                 # incident playbook
│   │   └── deploy.md                            # deploy runbook
│   ├── onboarding/getting-started.md            # new-engineer onboarding
│   └── delivery/phase-plan.md                   # P1–P5 / 20-week plan
│
├── tools/                                     # developer tooling (operational, scoped)
│   ├── README.md                              # tooling index
│   ├── codegen/generate-clients.ts             # contract → client generation
│   ├── scripts/
│   │   ├── seed-library.py                      # seed curated library
│   │   └── reset-dev-db.sh                      # reset local database
│   └── lint/
│       ├── ruff.toml                            # shared Python lint config
│       └── .prettierrc                          # shared JS/TS format config
│
└── tests/                                     # cross-system tests
    ├── README.md                              # test-strategy index
    ├── e2e/.gitkeep                            # cross-system journeys
    ├── integration/.gitkeep                    # cross-service integration
    ├── contract/.gitkeep                       # provider/consumer contract tests
    └── load/queue-depth.k6.js                  # load test on queue depth
```

## Where the two critical concerns live

- **Add a new SolverBackend** → `packages/compute-engine/src/qdr_compute_engine/drivers/<name>/<name>_driver.py`,
  implement `drivers/solver_backend.py`, register in `drivers/registry.py`, flag in `drivers/capabilities.py`.
- **The 9 pipeline stages (+ stage 0)** → `packages/compute-engine/src/qdr_compute_engine/pipeline/stage_00…stage_09_*.py`.
- **≤40-qubit cap** → constant in `packages/qdr-core/src/qdr_core/limits.py`, enforced at
  `apps/api/.../configs/estimator/qubit_cap_policy.py` (pre-submit) and
  `services/dispatcher/.../admission_control.py` (dispatch-time).
