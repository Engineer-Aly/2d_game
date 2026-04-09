"""
Debug module for The Assassin.
- F1 toggles debug mode on/off
- While on: draws live overlay + writes debug_player.csv and debug_vlad.csv
- Player log : frame, time, x, y, tile_x, tile_y, vx, vy, state, on_ground, hp, daggers
- Vlad log   : frame, time, x, y, tile_x, tile_y, vx, vy, state, on_ground, ai_strategy,
               sees_player, ammo, stun, swap_used
"""

import csv
import os
import time
import pygame

PLAYER_LOG = os.path.join(os.path.dirname(__file__), "debug_player.csv")
VLAD_LOG   = os.path.join(os.path.dirname(__file__), "debug_vlad.csv")
TILE       = 48   # must match game.py

_OVERLAY_BG  = (0, 0, 0, 170)
_GREEN       = (80, 255, 120)
_YELLOW      = (255, 220, 60)
_WHITE       = (255, 255, 255)
_RED         = (255, 80, 80)
_ORANGE      = (255, 140, 40)
_CYAN        = (80, 220, 255)

_PLAYER_FIELDS = ["frame", "time", "x", "y", "tile_x", "tile_y",
                  "vx", "vy", "state", "on_ground", "hp", "daggers"]

_VLAD_FIELDS   = ["frame", "time", "x", "y", "tile_x", "tile_y",
                  "vx", "vy", "state", "on_ground",
                  "ai_strategy", "sees_player", "ammo", "stun", "swap_used"]


class DebugLogger:
    def __init__(self):
        self.enabled = False
        self._frame  = 0

        # player log
        self._pfile   = None
        self._pwriter = None
        self._prev_player_xy = None

        # vlad log
        self._vfile   = None
        self._vwriter = None
        self._prev_vlad_xy = None

    # ── toggle ────────────────────────────────────────────────────────────────
    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self._open_logs()
            print(f"[DEBUG] ON  — {PLAYER_LOG}  |  {VLAD_LOG}")
        else:
            self._close_logs()
            print(f"[DEBUG] OFF — logs saved.")

    # ── call once per frame ───────────────────────────────────────────────────
    def update(self, player, vlad=None):
        if not self.enabled:
            return
        self._frame += 1
        self._log_player(player)
        if vlad is not None:
            self._log_vlad(vlad)

    # ── draw overlay ──────────────────────────────────────────────────────────
    def draw(self, surface, player, font, vlad=None):
        if not self.enabled:
            return

        self._draw_panel(surface, font, player, vlad)

        badge = font.render("DBG", True, _RED)
        surface.blit(badge, (surface.get_width() - 50, surface.get_height() - 20))

    # ── internal: log helpers ─────────────────────────────────────────────────
    def _log_player(self, player):
        x, y = int(player.rect.x), int(player.rect.y)
        if (x, y) == self._prev_player_xy:
            return
        self._prev_player_xy = (x, y)
        if self._pwriter:
            self._pwriter.writerow({
                "frame":     self._frame,
                "time":      round(time.time(), 3),
                "x":         x,
                "y":         y,
                "tile_x":    x // TILE,
                "tile_y":    y // TILE,
                "vx":        round(player.vx, 2),
                "vy":        round(player.vy, 2),
                "state":     self._player_state(player),
                "on_ground": int(player.on_ground),
                "hp":        player.hp,
                "daggers":   player.daggers,
            })
            self._pfile.flush()

    def _log_vlad(self, vlad):
        x, y = int(vlad.rect.x), int(vlad.rect.y)
        if (x, y) == self._prev_vlad_xy:
            return
        self._prev_vlad_xy = (x, y)
        if self._vwriter:
            self._vwriter.writerow({
                "frame":       self._frame,
                "time":        round(time.time(), 3),
                "x":           x,
                "y":           y,
                "tile_x":      x // TILE,
                "tile_y":      y // TILE,
                "vx":          round(vlad.vx, 2),
                "vy":          round(vlad.vy, 2),
                "state":       vlad.state,
                "on_ground":   int(vlad.on_ground),
                "ai_strategy": vlad.ai.strategy(),
                "sees_player": int(vlad.sees_player),
                "ammo":        vlad.ammo,
                "stun":        vlad.stun_timer,
                "swap_used":   int(vlad.swap_used),
            })
            self._vfile.flush()

    # ── internal: overlay drawing ─────────────────────────────────────────────
    def _draw_panel(self, surface, font, player, vlad):
        lh  = 18
        pad = 6

        # --- player section ---
        p_state = self._player_state(player)
        p_lines = [
            ("── PLAYER ──", "",                                         _YELLOW),
            ("POS",   f"x={player.rect.x}  y={player.rect.y}",          _WHITE),
            ("TILE",  f"col={player.rect.x//TILE}  row={player.rect.y//TILE}", _YELLOW),
            ("VEL",   f"vx={player.vx:.1f}  vy={player.vy:.1f}",        _GREEN),
            ("STATE", p_state,                                            _GREEN),
            ("GRND",  str(player.on_ground),
                      _GREEN if player.on_ground else _RED),
            ("HP",    str(player.hp),                                    _WHITE),
            ("DAGGERS", str(player.daggers),                             _YELLOW),
        ]

        # --- vlad section ---
        v_lines = []
        if vlad is not None:
            stun_str  = f"{vlad.stun_timer}f" if vlad.stun_timer > 0 else "no"
            swap_str  = "YES" if vlad.swap_used else "no"
            v_lines = [
                ("── VLAD ──",  "",                                      _ORANGE),
                ("POS",  f"x={vlad.rect.x}  y={vlad.rect.y}",           _WHITE),
                ("TILE", f"col={vlad.rect.x//TILE}  row={vlad.rect.y//TILE}", _ORANGE),
                ("VEL",  f"vx={vlad.vx:.1f}  vy={vlad.vy:.1f}",         _GREEN),
                ("STATE", vlad.state,                                     _CYAN),
                ("STRAT", vlad.ai.strategy(),                             _CYAN),
                ("GRND",  str(vlad.on_ground),
                          _GREEN if vlad.on_ground else _RED),
                ("SEES",  str(vlad.sees_player),
                          _RED if vlad.sees_player else _WHITE),
                ("AMMO",  str(vlad.ammo),                                _WHITE),
                ("STUN",  stun_str,
                          _RED if vlad.stun_timer > 0 else _WHITE),
                ("SWAP",  swap_str,
                          _ORANGE if vlad.swap_used else _WHITE),
            ]

        all_lines = p_lines + ["---"] + v_lines + [("FRAME", str(self._frame), _WHITE)]
        w   = 260
        h   = (len(all_lines)) * lh + pad * 2

        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill(_OVERLAY_BG)

        row = 0
        for entry in all_lines:
            if entry == "---":
                pygame.draw.line(panel, (100, 100, 100),
                                 (pad, pad + row * lh + lh // 2),
                                 (w - pad, pad + row * lh + lh // 2))
                row += 1
                continue
            label, value, color = entry
            lbl = font.render(f"{label}:", True, (180, 180, 180))
            val = font.render(value,       True, color)
            y_  = pad + row * lh
            panel.blit(lbl, (pad, y_))
            if value:
                panel.blit(val, (pad + 80, y_))
            row += 1

        x_pos = surface.get_width() - 270   # sit over the LLM chat panel (CHAT_W=280, panel=260)
        surface.blit(panel, (x_pos, 8))

    # ── state helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _player_state(player):
        if player.crouching:
            return "crouching"
        if player.vx != 0:
            return "walking"
        if not player.on_ground:
            return "airborne"
        return "standing"

    # ── file helpers ──────────────────────────────────────────────────────────
    def _open_logs(self):
        self._frame = 0

        self._pfile   = open(PLAYER_LOG, "w", newline="")
        self._pwriter = csv.DictWriter(self._pfile, fieldnames=_PLAYER_FIELDS)
        self._pwriter.writeheader()

        self._vfile   = open(VLAD_LOG, "w", newline="")
        self._vwriter = csv.DictWriter(self._vfile, fieldnames=_VLAD_FIELDS)
        self._vwriter.writeheader()

    def _close_logs(self):
        for f in (self._pfile, self._vfile):
            if f:
                f.close()
        self._pfile = self._pwriter = None
        self._vfile = self._vwriter = None
