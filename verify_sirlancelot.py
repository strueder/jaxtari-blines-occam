import jax.numpy as jnp


def _set(obj, **kw):
    return obj.replace(**{k: jnp.asarray(v, dtype=jnp.asarray(getattr(obj, k)).dtype) for k, v in kw.items()})


def _castle(base, wing, left, cooldown, fb_active):
    s = _set(base, stage=2, level=2)
    return s.replace(
        dragon=_set(s.dragon, is_active=True, x=70, wing_frame=wing, facing_left=left, shoot_cooldown=cooldown),
        fireballs=_set(s.fireballs, active=fb_active, positions=[[80, 60], [40, 100], [110, 130]], animation_frame=[0, 1, 0]),
        enemies=_set(s.enemies, active=[False] * 4),
    )


def _stale(base):
    s = _set(base, stage=1, level=1)
    return s.replace(
        dragon=_set(s.dragon, is_active=True, x=70),
        fireballs=_set(s.fireballs, active=[True, True, True], positions=[[80, 60], [40, 100], [110, 130]]),
    )


def _beasts(base, level, left, frame, invisible):
    s = _set(base, stage=1, level=level)
    return s.replace(enemies=_set(
        s.enemies,
        active=[True] * 4,
        positions=[[20, 50], [60, 80], [100, 110], [-5, 140]],
        facing_left=left,
        animation_frame=frame,
        is_invisible=invisible,
    ))


def build_poses(env, base):
    off = [False, False, False]
    return [
        ("castle, dragon wing up, facing left [dragon=1 fireballs=0]", _castle(base, 0, True, 0, off)),
        ("castle, dragon wing middle, facing right [dragon=1 fireballs=0]", _castle(base, 1, False, 0, off)),
        ("castle, dragon wing down, facing left [dragon=1 fireballs=0]", _castle(base, 2, True, 0, off)),
        ("castle, dragon shooting, two fireballs [dragon=1 fireballs=2]", _castle(base, 1, False, 240, [True, True, False])),
        ("outdoor, stale dragon and fireball flags [dragon=0 fireballs=0]", _stale(base)),
        ("level 3, beasts both directions, one at x=-5 [enemies=4]", _beasts(base, 3, [True, False, True, False], [0, 1, 0, 1], [False] * 4)),
        ("level 7, flipped 5x1 beasts, two invisible [enemies=2]", _beasts(base, 7, [True, False, False, True], [0, 0, 1, 1], [False, False, True, True])),
    ]