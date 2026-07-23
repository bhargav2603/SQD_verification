"""Stage 3b -- projection-based WF-in-DFT active-site embedding.

Runs between integrals (stage 3) and active-space selection (stage 4). This mirrors how
the field structures embedding (Quantinuum InQuanto's embedding driver, PsiEmbed, the
DFT-embedded QSCI/SQD pipeline): embedding is its *own* step that produces an effective
Hamiltonian by modifying the one-electron operator, not something folded into integral
generation.

    03_integrals -> 03b_embedding -> 04_active_space_selector -> ...

Config gate
-----------
When ``config["embedding"]`` is absent the stage is a **pass-through**: it returns the
context unchanged, so non-embedded (R1) runs are byte-identical to a pipeline that never
had this stage. That is the whole point of gating on config -- adding the capability must
not perturb the numbers the correctness gate already certified.

When embedding IS requested the stage will modify the one-electron Hamiltonian:

    h_emb = h_core_A + (g[gamma_tot] - g[gamma_A]) + mu * P_env

with a ROHF/RKS reference (spin-free integrals; the solver targets the (Nalpha, Nbeta)
sector). That construction is not implemented yet -- so a configured embedding request
fails loudly here rather than silently returning un-embedded integrals, which would be a
plausible-looking wrong number.
"""

from qdr_compute_engine.pipeline.stage_context import StageContext

STAGE_NAME = "03b_embedding"


def run(context: StageContext) -> StageContext:
    if not context.config.get("embedding"):
        # Pass-through: the non-embedded path must stay byte-identical.
        return context

    raise NotImplementedError(
        "projection-based WF-in-DFT embedding (config['embedding']) is not yet "
        "implemented; stage 03b currently supports only the non-embedded pass-through. "
        "Configuring embedding fails loudly rather than returning un-embedded integrals."
    )
