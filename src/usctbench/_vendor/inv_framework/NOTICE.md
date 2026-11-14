# Extracted inv_framework components

Source: https://github.com/peterw634/inv_framework

Pinned commit: b258fa90d284f0105259aeffbfd43814f4cd3085

The upstream README declares MIT licensing. That revision has no separate LICENSE
file or explicit copyright-holder statement. This notice records that declaration;
it does not invent a copyright attribution or additional grant. Confirm the
upstream license text before redistributing this extracted subset externally.

SOURCE.json records the original Git blob IDs. Package initializers were reduced
to avoid importing CT or optional diffusion adapters. The neural solvers add
optional loss/observer/before-step hooks and DPS iteration caps. The VP schedule
now uses unique time steps and ends at a clean sample. Remaining model files are
the extracted upstream implementations; formatting changes may be present.

USCT-specific physics, bounded variable mapping, masks, weights, execution policy,
checkpoint validation and reference-prior training live outside this subset.
