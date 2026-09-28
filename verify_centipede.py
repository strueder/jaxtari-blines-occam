"""Hand-written Centipede scenarios for verify_game.py.

Centipede spawns spider, flea and scorpion behind timers, and the scorpion
additionally behind `wave[1] >= 3`, so a random rollout shows them rarely and
never together. These poses put every enemy on screen at once and add the
edge cases the automatic checks cannot reach.

    uv run python verify_game.py centipede --png
"""
import numpy as np

from verify_kingkong import rep


def _mushroom_row(base, row_y, poisoned):
    """All mushrooms of one pixel row set alive, optionally poisoned."""
    m = np.asarray(base.mushroom_positions).copy()
    hit = m[:, 1] == row_y
    m[hit, 2] = 1 if poisoned else 0
    m[hit, 3] = 3
    return m


def build_poses(env, base):
    """Yield (title, state) for every situation worth looking at."""
    c = env.consts

    spider_y = int(np.asarray(c.SPIDER_Y_POSITIONS)[2])       # 133
    scorp_y = int(np.asarray(c.SCORPION_Y_POSITIONS)[6])      # 61
    x_left = int(np.asarray(c.SCORPION_X_POSITIONS)[0])       # 16
    x_right = int(np.asarray(c.SCORPION_X_POSITIONS)[1])      # 133

    alive = rep(
        base,
        death_counter=0,
        wave=[0, 3],
        player_x=c.PLAYER_START_X,
        player_y=c.PLAYER_START_Y,
        player_spell=[c.PLAYER_START_X + 1, 120, 1],
        spider_position=[70, spider_y, 1],
        flea_position=[88.0, 50.0, 2.0],
        scorpion_position=[70, scorp_y, 1, 1],
    )

    yield "all enemies at once", alive

    # --- spawn edges, both travel directions ------------------------------
    yield ("spider+scorpion at left edge, moving right",
           rep(alive, spider_position=[x_left, spider_y, 1],
               scorpion_position=[x_left, scorp_y, 1, 1]))
    yield ("spider+scorpion at right edge, moving left (sprite flipped)",
           rep(alive, spider_position=[x_right, spider_y, -1],
               scorpion_position=[x_right, scorp_y, -1, 1]))

    # --- scorpion on the topmost and bottommost row it can use ------------
    for label, y in (("top", int(np.asarray(c.SCORPION_Y_POSITIONS)[0])),
                     ("bottom", int(np.asarray(c.SCORPION_Y_POSITIONS)[-1]))):
        yield f"scorpion on {label} row (y={y})", rep(alive, scorpion_position=[70, y, 1, 1])

    # --- spell leaving the screen at the top ------------------------------
    # the spell stays alive for one frame after it has passed y=0
    yield "spell fully above the screen (y=-8)", rep(alive, player_spell=[77, -8, 1])
    yield "spell half above the screen (y=-3)", rep(alive, player_spell=[77, -3, 1])

    # --- death animation: the renderer hides parts of the scene -----------
    yield "death animation, player spark (player not drawn)", rep(alive, death_counter=-1)
    yield "death animation, mushroom phase (centipede not drawn)", rep(alive, death_counter=8)

    # --- flea on its first and last row -----------------------------------
    yield "flea at spawn height", rep(alive, flea_position=[88.0, 5.0, 2.0])
    yield "flea just above the bottom", rep(alive, flea_position=[88.0, 180.0, 2.0])

    # --- poisoned mushrooms, the row a scorpion leaves behind -------------
    row_y = int(np.asarray(base.mushroom_positions)[:, 1][10])
    yield ("poisoned mushroom row",
           rep(alive, mushroom_positions=_mushroom_row(base, row_y, True)))
    yield ("same row, normal mushrooms",
           rep(alive, mushroom_positions=_mushroom_row(base, row_y, False)))