id: 6
area: corpus
title: the generality matrix cannot catch an emitter arm that does not exist
opened: 2026-09-01
closed: 2026-09-01

## wanted
A test that would have caught backlog #005 before a user did.

## got
There is none, and there structurally cannot be one in tile_spec as written.
`backend_emit_purity.feature` and the generality matrix both drive the emitters
with canonical snippets, so every arm they exercise is an arm that exists. An
intrinsic with no arm is never fed to them, and the default arm that silently
produces a copy is never reached.

## workaround
Built the detector on the other side, in tile_cli: `emit::ignored_intrinsics`
emits twice, once with the intrinsic renamed to something no emitter can handle,
and reports any operation whose presence changed nothing in the output. That
catches the class without editing emitters this crate does not own.

The test tile_spec is missing is one line of intent: feed
`__tile_definitely_unhandled` to every emitter and assert `Err`. Today most of
them would return `Ok` with a copy. That test belongs there rather than here,
because it is a statement about the emitters, and it would have made #005 a
five-minute fix instead of a two-hour isolation.
