import argparse
import contextlib
import os
import signal
import sys
from datetime import datetime


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Play a JAXAtari game through the OCCAMWrapper.")
    p.add_argument("-g", "--game", required=True)
    p.add_argument("--mode", default="object", choices=["object", "binary", "class", "planes", "all"])
    p.add_argument("-m", "--mods", nargs="*", default=None)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jaxatari-src", default=None)
    p.add_argument("--out-dir", default="recordings")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _ARGS = _parse_args()
    if _ARGS.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if _ARGS.jaxatari_src:
        sys.path.insert(0, os.path.abspath(_ARGS.jaxatari_src))

import av
import numpy as np
import jax
import jax.numpy as jnp
import pygame

import jaxatari
from jaxatari.environment import JAXAtariAction
from jaxatari.wrappers import AtariWrapper

from agents.occam.occam import (
    HEAD_H,
    MASK_MODES,
    OCCAMWrapper,
    _OCCAMViz,
    _extract_object_groups,
    _legend_strip,
    _make_color_palette,
    _strip,
)

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "FIRE")
SIGNALS = tuple(getattr(signal, n) for n in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, n))
BAR_H = 40
PANEL_GAP = 1
ALL_GAP = 2
ALL_SCALE = 2
GAP_GRAY = 70


def _action_parts(action_id):
    for name, value in vars(JAXAtariAction).items():
        if not name.startswith("_") and isinstance(value, int) and value == action_id:
            return frozenset(d for d in DIRECTIONS if d in name)
    return frozenset()


def _pressed_parts():
    k = pygame.key.get_pressed()
    parts = set()
    if k[pygame.K_UP] or k[pygame.K_w]:
        parts.add("UP")
    if k[pygame.K_DOWN] or k[pygame.K_s]:
        parts.add("DOWN")
    if k[pygame.K_LEFT] or k[pygame.K_a]:
        parts.add("LEFT")
    if k[pygame.K_RIGHT] or k[pygame.K_d]:
        parts.add("RIGHT")
    if k[pygame.K_SPACE]:
        parts.add("FIRE")
    return parts


def _choose_action(action_parts, pressed):
    best, best_len = 0, -1
    for idx, parts in enumerate(action_parts):
        if parts <= pressed and len(parts) > best_len:
            best, best_len = idx, len(parts)
    return best


def _agent_panel(obs, channels, height, palette, tint):
    last = np.asarray(obs)[-channels:, :, :, 0]
    if channels == 1:
        img = np.kron(last[0], np.ones((2, 2), np.uint8))
        rgb = np.repeat(img[..., None], 3, axis=-1)
    else:
        h, w = last.shape[1:]
        rows = 2
        cols = -(-channels // rows)
        rgb = np.full((rows * (h + PANEL_GAP), cols * (w + PANEL_GAP), 3), 60, np.uint8)
        for i, plane in enumerate(last):
            r, c = divmod(i, cols)
            col = palette[i + 1].astype(np.float32) / 255.0 if tint else np.ones(3, np.float32)
            y0, x0 = r * (h + PANEL_GAP), c * (w + PANEL_GAP)
            rgb[y0:y0 + h, x0:x0 + w] = (plane[..., None].astype(np.float32) * col).astype(np.uint8)
    pad = max(0, height - rgb.shape[0])
    return np.pad(rgb, ((0, pad), (0, 0), (0, 0)))


def _gray_tile(img, scale):
    tile = np.kron(np.asarray(img, np.uint8), np.ones((scale, scale), np.uint8))
    return np.repeat(tile[..., None], 3, axis=-1)


def _rgb_tile(img, scale):
    return np.repeat(np.repeat(np.asarray(img, np.uint8), scale, axis=0), scale, axis=1)


def _draw_boxes(tile, groups, palette, sx, sy):
    out = tile.copy()
    h, w = out.shape[:2]
    for gi, g in enumerate(groups):
        color = palette[gi + 1]
        fields = [np.asarray(v).reshape(-1) for v in (g.x, g.y, g.width, g.height, g.active)]
        for bx, by, bw, bh, active in zip(*fields):
            if not active or bw <= 0 or bh <= 0:
                continue
            x0 = int(np.clip(np.floor(bx * sx), 0, w - 1))
            y0 = int(np.clip(np.floor(by * sy), 0, h - 1))
            x1 = int(np.clip(np.ceil((bx + bw) * sx) - 1, x0, w - 1))
            y1 = int(np.clip(np.ceil((by + bh) * sy) - 1, y0, h - 1))
            out[y0, x0:x1 + 1] = color
            out[y1, x0:x1 + 1] = color
            out[y0:y1 + 1, x0] = color
            out[y0:y1 + 1, x1] = color
    return out


def _panel_row(panels):
    height = max(p.shape[0] for _, p in panels)
    heads, bodies = [], []
    for i, (name, p) in enumerate(panels):
        if i:
            heads.append(np.zeros((HEAD_H, ALL_GAP, 3), np.uint8))
            bodies.append(np.full((height, ALL_GAP, 3), GAP_GRAY, np.uint8))
        heads.append(_strip(p.shape[1], HEAD_H, name))
        bodies.append(np.pad(p, ((0, height - p.shape[0]), (0, 0), (0, 0))))
    return np.concatenate([np.concatenate(heads, axis=1), np.concatenate(bodies, axis=1)], axis=0)


def _pad_width(img, width):
    return np.pad(img, ((0, 0), (0, width - img.shape[1]), (0, 0)))


def _make_env(base, mode, args):
    env = AtariWrapper(
        base,
        sticky_actions=0.0,
        episodic_life=False,
        first_fire=True,
        noop_max=30,
        full_action_space=False,
    )
    return OCCAMWrapper(
        env,
        mask_mode=mode,
        frame_stack_size=args.frame_stack,
        frame_skip=args.frame_skip,
        max_pooling=True,
        clip_reward=False,
        game_name=args.game,
    )


@jax.jit
def _same_state(a, b):
    leaves = [jnp.all(x == y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))]
    return jnp.all(jnp.stack(leaves))


def _even(frame):
    return np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)))


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


@contextlib.contextmanager
def _deferred_signals():
    received = []
    previous = {sig: signal.signal(sig, lambda signum, frame: received.append(signum)) for sig in SIGNALS}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if received:
            raise KeyboardInterrupt


class _Recorder:
    def __init__(self, fps):
        self.fps = fps
        self.container = None
        self.stream = None
        self.path = None
        self.frames = 0

    @property
    def active(self):
        return self.container is not None

    def start(self, path, frame):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with _deferred_signals():
            self.path = path
            self.frames = 0
            self.container = av.open(
                path + ".part",
                "w",
                format="mp4",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof", "flush_packets": "1"},
            )
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.width = frame.shape[1]
            self.stream.height = frame.shape[0]
            self.stream.pix_fmt = "yuv420p"
            self.stream.codec_context.gop_size = self.fps
        self.add(frame)

    def add(self, frame):
        with _deferred_signals():
            for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                self.container.mux(packet)
            self.frames += 1

    def stop(self):
        with _deferred_signals():
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None
            os.replace(self.path + ".part", self.path)
            print(f"saved {self.frames} frames -> {self.path}")


def main(args):
    for sig in SIGNALS:
        if sig != signal.SIGINT:
            signal.signal(sig, _raise_interrupt)

    base = jaxatari.make(args.game, mods=args.mods or None)
    modes = list(MASK_MODES) if args.mode == "all" else [args.mode]
    envs = {m: _make_env(base, m, args) for m in modes}
    primary = envs[modes[0]]
    viz = _OCCAMViz(primary.base_env, modes[0])
    render = jax.jit(primary.base_env.render)
    palette = np.asarray(_make_color_palette(viz.num_classes))
    action_set = np.asarray(getattr(primary.base_env, "ACTION_SET", np.arange(primary.base_env.action_space().n)))
    action_parts = [_action_parts(int(a)) for a in action_set]

    print(f"jaxatari: {os.path.dirname(jaxatari.__file__)}")
    print(f"groups ({viz.num_classes}): {', '.join(viz.group_names)}")
    print(f"actions: {['+'.join(sorted(p)) or 'NOOP' for p in action_parts]}")

    def legend(width, env_state):
        counts = np.asarray(viz._counts(env_state))
        entries = [
            (name, int(counts[i]), tuple(int(c) for c in palette[i + 1]))
            for i, name in enumerate(viz.group_names)
        ]
        return _legend_strip(width, entries)

    def compose_single(obs, state):
        mode = modes[0]
        env_state = state[mode].atari_state.env_state
        row = np.asarray(viz._frame(env_state), np.uint8)
        agent = _agent_panel(obs[mode], envs[mode].per_frame_channels, row.shape[0], palette, mode == "planes")
        body = np.concatenate([row, agent], axis=1)
        panels = viz.panel_layout() + [("agent input (OCCAMWrapper)", agent.shape[1])]
        header = np.concatenate([_strip(w, HEAD_H, name) for name, w in panels], axis=1)
        header = np.pad(header, ((0, 0), (0, max(0, body.shape[1] - header.shape[1])), (0, 0)))[:, :body.shape[1]]
        return _even(np.concatenate([legend(body.shape[1], env_state), header, body], axis=0))

    def compose_all(obs, state):
        env_state = state[modes[0]].atari_state.env_state
        clean = np.asarray(render(env_state), np.uint8)
        colored = view["colored"]
        small = _rgb_tile(downscaled(env_state), ALL_SCALE)
        groups = _extract_object_groups(observe(env_state))
        sx = small.shape[1] / clean.shape[1]
        sy = small.shape[0] / clean.shape[0]
        panels = [("clean", clean), ("84px + boxes", _draw_boxes(small, groups, palette, sx, sy))]
        for m in ("object", "binary", "class"):
            if colored and m == "class":
                panels.append((m, _rgb_tile(class_colors(env_state), ALL_SCALE)))
            else:
                panels.append((m, _gray_tile(np.asarray(obs[m])[-1, :, :, 0], ALL_SCALE)))
        top = _panel_row(panels)
        planes = np.asarray(obs["planes"])[-envs["planes"].per_frame_channels:, :, :, 0]
        tiles = []
        for i, name in enumerate(viz.group_names):
            tile = _gray_tile(planes[i], ALL_SCALE)
            if colored:
                tile = (tile.astype(np.float32) * (palette[i + 1].astype(np.float32) / 255.0)).astype(np.uint8)
            tiles.append((name, tile))
        bottom = _panel_row(tiles)
        width = max(top.shape[1], bottom.shape[1])
        rows = [
            legend(width, env_state),
            _pad_width(top, width),
            np.full((ALL_GAP, width, 3), GAP_GRAY, np.uint8),
            _strip(width, HEAD_H, "planes (one channel per group)"),
            _pad_width(bottom, width),
        ]
        return _even(np.concatenate(rows, axis=0))

    view = {"colored": True}
    if args.mode == "all":
        res_viz = _OCCAMViz(primary.base_env, "class", obs_res=True)
        class_colors = jax.jit(lambda s: res_viz._mask_rgb(render(s), primary.base_env._get_observation(s)))
        downscaled = jax.jit(lambda s: res_viz._to_obs_res(render(s)))
        observe = jax.jit(primary.base_env._get_observation)

    compose = compose_all if args.mode == "all" else compose_single

    def reset(sub):
        results = {m: envs[m].reset(sub) for m in modes}
        return {m: r[0] for m, r in results.items()}, {m: r[1] for m, r in results.items()}

    key = jax.random.PRNGKey(args.seed)
    key, sub = jax.random.split(key)
    obs, state = reset(sub)
    frame = compose(obs, state)

    pygame.init()
    font = pygame.font.SysFont(None, 22)
    display = pygame.display.Info()
    scale = args.scale
    if display.current_w > 0 and display.current_h > 0:
        fit_w = (display.current_w - 40) // frame.shape[1]
        fit_h = (display.current_h - BAR_H - 100) // frame.shape[0]
        scale = max(1, min(scale, fit_w, fit_h))
    win_w, win_h = frame.shape[1] * scale, frame.shape[0] * scale + BAR_H
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"OCCAM play: {args.game} ({args.mode})")
    button = pygame.Rect(win_w - 130, 4, 122, BAR_H - 8)
    clock = pygame.time.Clock()

    paused = False
    step_once = False
    recorder = _Recorder(args.fps)
    steps = 0
    episode = 0
    ep_return = 0.0
    running = True
    desync_reported = False

    try:
        while running:
            toggle_rec = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_n:
                        step_once = True
                    elif event.key == pygame.K_v:
                        toggle_rec = True
                    elif event.key == pygame.K_c and args.mode == "all":
                        view["colored"] = not view["colored"]
                        frame = compose(obs, state)
                    elif event.key == pygame.K_r:
                        key, sub = jax.random.split(key)
                        obs, state = reset(sub)
                        frame = compose(obs, state)
                        steps, ep_return = 0, 0.0
                        episode += 1
                        if recorder.active:
                            recorder.add(frame)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and button.collidepoint(event.pos):
                    toggle_rec = True

            if toggle_rec:
                if recorder.active:
                    recorder.stop()
                else:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    recorder.start(os.path.join(args.out_dir, f"{args.game}_{args.mode}_{stamp}.mp4"), frame)

            if running and (not paused or step_once):
                step_once = False
                action = _choose_action(action_parts, _pressed_parts())
                results = {m: envs[m].step(state[m], jnp.int32(action)) for m in modes}
                obs = {m: r[0] for m, r in results.items()}
                state = {m: r[1] for m, r in results.items()}
                _, _, reward, terminated, truncated, _ = results[modes[0]]
                if not desync_reported:
                    for m in modes[1:]:
                        if not bool(_same_state(state[m].atari_state.env_state, state[modes[0]].atari_state.env_state)):
                            print(f"[warn] game state of '{m}' diverged from '{modes[0]}' at step {steps + 1}")
                            desync_reported = True
                steps += 1
                ep_return += float(reward)
                if bool(terminated) or bool(truncated):
                    print(f"episode {episode} done: return {ep_return:.0f}, {steps} steps")
                    episode += 1
                    steps, ep_return = 0, 0.0
                frame = compose(obs, state)
                if recorder.active:
                    recorder.add(frame)

            screen.fill((0, 0, 0))
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (frame.shape[1] * scale, frame.shape[0] * scale))
            screen.blit(surf, (0, BAR_H))
            status = f"{args.game} | {args.mode} | episode {episode} | step {steps} | return {ep_return:.0f}"
            if args.mode == "all":
                status += " | colors: occam" if view["colored"] else " | colors: agent gray"
            if paused:
                status += " | PAUSED"
            screen.blit(font.render(status, True, (230, 230, 230)), (8, 4))
            help_text = "arrows/WASD + space   P pause   N step   R reset   V rec   Esc quit"
            if args.mode == "all":
                help_text += "   C colors"
            screen.blit(font.render(help_text, True, (150, 150, 150)), (8, 22))
            pygame.draw.rect(screen, (200, 40, 40) if recorder.active else (70, 70, 70), button, border_radius=6)
            label = font.render(f"STOP ({recorder.frames})" if recorder.active else "REC", True, (255, 255, 255))
            screen.blit(label, label.get_rect(center=button.center))
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        try:
            if recorder.active:
                recorder.stop()
        finally:
            pygame.quit()


if __name__ == "__main__":
    try:
        main(_ARGS)
    except KeyboardInterrupt:
        print("interrupted")