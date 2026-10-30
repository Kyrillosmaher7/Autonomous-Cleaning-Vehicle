"""
Tkinter dashboard for watching a :class:`Simulation`.

One dashboard serves every environment version. Anything that depends
on the version (observation window, panels, sizes, colors, speeds)
comes from the simulation and the visualization config.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from cleaning_vehicle.config.schema import VisualizationConfig

from .simulation import Simulation

Position = tuple[int, int]

_TITLE_FONT = ("Arial", 14, "bold")
_PANEL_FONT = ("Arial", 10)
_CELL_FONT = ("Arial", 12, "bold")


class _InfoPanel(ttk.LabelFrame):
    """A titled box of ``name: value`` rows that can be refreshed."""

    def __init__(self, parent: tk.Misc, title: str, rows: list[str]) -> None:
        super().__init__(parent, text=title, padding=8)
        self.pack(fill="x", pady=(0, 8))

        self._values: dict[str, tk.StringVar] = {}

        for row, name in enumerate(rows):
            value = tk.StringVar(value="-")
            self._values[name] = value

            ttk.Label(self, text=f"{name}:", font=_PANEL_FONT).grid(
                row=row, column=0, sticky="w"
            )
            ttk.Label(self, textvariable=value, font=_PANEL_FONT).grid(
                row=row, column=1, sticky="w", padx=(8, 0)
            )

    def update_values(self, **values: str) -> None:
        """Set rows by name; underscores stand in for spaces."""

        for name, text in values.items():
            self._values[name.replace("_", " ")].set(text)


class SimulationDashboard:
    """
    Grid view plus info panels and play/step/reset controls.

    Pass ``master`` to open as a window on top of an existing Tk app
    (the launcher). Without it the dashboard owns its own Tk root and
    :meth:`run` starts the main loop.
    """

    def __init__(
        self,
        simulation: Simulation,
        visualization: VisualizationConfig,
        title: str = "Autonomous Cleaning Vehicle",
        run_details: dict[str, str] | None = None,
        master: tk.Misc | None = None,
    ) -> None:

        self.simulation = simulation
        self.style = visualization
        self.colors = visualization.colors
        self.speed_ms = visualization.speed_default_ms

        self._owns_root = master is None
        self.window: tk.Tk | tk.Toplevel = (
            tk.Tk() if master is None else tk.Toplevel(master)
        )
        self.window.title(title)
        self.window.resizable(False, False)

        self._playing = False
        self._after_id: str | None = None

        self._build_layout(run_details or {})
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    # ==========================================================
    # Layout
    # ==========================================================

    def _build_layout(self, run_details: dict[str, str]) -> None:
        env = self.simulation.environment
        cell = self.style.cell_size

        root = ttk.Frame(self.window, padding=10)
        root.pack()

        self.canvas = tk.Canvas(
            root,
            width=env.cols * cell,
            height=env.rows * cell,
            bg=self.colors["empty"],
            highlightthickness=1,
            highlightbackground=self.colors["grid_line"],
        )
        self.canvas.grid(row=0, column=0, sticky="n")

        side = ttk.Frame(root)
        side.grid(row=0, column=1, sticky="n", padx=(12, 0))

        ttk.Label(side, text="Cleaning Vehicle", font=_TITLE_FONT).pack(
            anchor="w", pady=(0, 8)
        )

        if run_details:
            run = _InfoPanel(side, "Run", list(run_details))
            run.update_values(**run_details)

        self.robot_panel = _InfoPanel(
            side, "Robot", ["Position", "Battery", "Charging"]
        )
        self.battery_bar = ttk.Progressbar(
            self.robot_panel, maximum=env.max_battery, length=180
        )
        self.battery_bar.grid(row=3, column=0, columnspan=2,
                              sticky="ew", pady=(6, 0))

        self.world_panel = _InfoPanel(
            side,
            "Environment",
            ["Dirty cells", "Static obstacles", "Dynamic obstacles"],
        )

        self.observation_panel: _InfoPanel | None = None
        if self.simulation.partial_observation:
            self.observation_panel = _InfoPanel(
                side,
                "Observation",
                ["Vision radius", "Visible dirt", "Visible obstacles"],
            )

        self.progress_panel = _InfoPanel(
            side,
            "Progress",
            ["Step", "Last action", "Last reward", "Total reward", "Cleaned"],
        )

        self._build_controls(root)

    def _build_controls(self, parent: tk.Misc) -> None:
        """Status and buttons in a single row under the grid."""

        bar = ttk.Frame(parent)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.status = tk.StringVar()
        ttk.Label(bar, textvariable=self.status, font=_TITLE_FONT,
                  width=20).pack(side="left")

        self.step_button = ttk.Button(
            bar, text="Step", command=self.step_once
        )
        self.play_button = ttk.Button(
            bar, text="Play", command=self.toggle_play
        )
        self.reset_button = ttk.Button(
            bar, text="Reset", command=self.reset
        )

        for button in (self.step_button, self.play_button, self.reset_button):
            button.pack(side="left", padx=(0, 6))

        ttk.Button(bar, text="Close", command=self.close).pack(side="right")

        self.speed_var = tk.IntVar(value=self.speed_ms)
        ttk.Label(bar, text="Fast").pack(side="right", padx=(4, 12))
        ttk.Scale(
            bar,
            from_=self.style.speed_slowest_ms,
            to=self.style.speed_fastest_ms,
            variable=self.speed_var,
            orient="horizontal",
            length=140,
        ).pack(side="right", padx=(0, 12))
        ttk.Label(bar, text="Slow").pack(side="right")

    # ==========================================================
    # Controls
    # ==========================================================

    def step_once(self) -> None:
        self.simulation.step()
        self.refresh()

    def toggle_play(self) -> None:
        if self._playing:
            self._stop()
        elif not self.simulation.finished:
            self._playing = True
            self._tick()

        self.refresh()

    def reset(self) -> None:
        self._stop()
        self.simulation.reset()
        self.refresh()

    def close(self) -> None:
        self._stop()
        print(self.simulation.summary())
        self.window.destroy()

    def _stop(self) -> None:
        self._playing = False

        if self._after_id is not None:
            self.window.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self) -> None:
        if not self._playing:
            return

        self.simulation.step()
        self.refresh()

        if self.simulation.finished:
            self._stop()
            self.refresh()
            return

        self._after_id = self.window.after(
            int(self.speed_var.get()), self._tick
        )

    # ==========================================================
    # Drawing
    # ==========================================================

    def refresh(self) -> None:
        self._draw_grid()
        self._update_panels()

    def _cell_box(self, position: Position, inset: int = 0) -> tuple[int, ...]:
        size = self.style.cell_size
        row, col = position
        return (
            col * size + inset,
            row * size + inset,
            (col + 1) * size - inset,
            (row + 1) * size - inset,
        )

    def _fill_cell(
        self,
        position: Position,
        color: str,
        label: str = "",
        label_color: str = "black",
    ) -> None:
        box = self._cell_box(position)
        self.canvas.create_rectangle(
            *box, fill=color, outline=self.colors["grid_line"]
        )

        if label:
            self.canvas.create_text(
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2,
                text=label,
                fill=label_color,
                font=_CELL_FONT,
            )

    def _visible_cells(self) -> set[Position]:
        """Cells inside the observation window (partial mode only)."""

        sim = self.simulation

        if not sim.partial_observation:
            return set()

        env = sim.environment
        row, col = sim.state.position
        radius = env.observation_radius

        return {
            (r, c)
            for r in range(row - radius, row + radius + 1)
            for c in range(col - radius, col + radius + 1)
            if 0 <= r < env.rows and 0 <= c < env.cols
        }

    def _draw_grid(self) -> None:
        sim = self.simulation
        env = sim.environment
        state = sim.state
        colors = self.colors

        self.canvas.delete("all")

        visible = self._visible_cells()

        for row in range(env.rows):
            for col in range(env.cols):
                position = (row, col)
                color = colors["visible"] if position in visible \
                    else colors["empty"]
                label = ""
                label_color = "black"

                if position in state.dirty_cells:
                    color, label = colors["dirty"], "D"
                if position in env.obstacles:
                    color, label, label_color = (
                        colors["static_obstacle"], "X", "white"
                    )
                if position in env.dynamic_obstacles:
                    color, label, label_color = (
                        colors["dynamic_obstacle"], "M", "white"
                    )
                if env.is_charging_station(position):
                    color, label = colors["charging"], "C"

                self._fill_cell(position, color, label, label_color)

        self.canvas.create_oval(
            *self._cell_box(state.position, inset=8),
            fill=colors["robot"],
            outline="black",
            width=2,
        )
        self.canvas.create_text(
            *self._cell_center(state.position),
            text="R",
            fill="white",
            font=_CELL_FONT,
        )

    def _cell_center(self, position: Position) -> tuple[float, float]:
        left, top, right, bottom = self._cell_box(position)
        return (left + right) / 2, (top + bottom) / 2

    # ==========================================================
    # Panels
    # ==========================================================

    def _update_panels(self) -> None:
        sim = self.simulation
        env = sim.environment
        state = sim.state

        charging = "Yes" if env.is_charging_station(state.position) else "No"
        low = " (LOW)" if env.is_low_battery(state) else ""

        self.robot_panel.update_values(
            Position=str(state.position),
            Battery=f"{state.battery} / {env.max_battery}{low}",
            Charging=charging,
        )
        self.battery_bar["value"] = state.battery

        self.world_panel.update_values(
            Dirty_cells=str(len(state.dirty_cells)),
            Static_obstacles=str(len(env.obstacles)),
            Dynamic_obstacles=str(len(env.dynamic_obstacles)),
        )

        observation = sim.observation
        if self.observation_panel and observation:
            self.observation_panel.update_values(
                Vision_radius=str(env.observation_radius),
                Visible_dirt=str(len(observation.dirty_cells)),
                Visible_obstacles=str(len(observation.dynamic_obstacles)),
            )

        outcome = sim.last_outcome
        self.progress_panel.update_values(
            Step=f"{sim.step_count} / {sim.max_steps}",
            Last_action=outcome.action.name if outcome else "-",
            Last_reward=f"{outcome.reward:+.1f}" if outcome else "-",
            Total_reward=f"{sim.total_reward:.1f}",
            Cleaned=str(sim.cleaned_count),
        )

        if sim.finished:
            self.status.set(sim.end_reason.upper())
        else:
            self.status.set("RUNNING" if self._playing else "READY")

        finished = sim.finished
        self.step_button.state(["disabled"] if finished else ["!disabled"])
        self.play_button.configure(
            text="Pause" if self._playing else "Play"
        )
        self.play_button.state(["disabled"] if finished else ["!disabled"])

    # ==========================================================
    # Running
    # ==========================================================

    def run(self) -> None:
        """Start the Tk main loop (only for a standalone dashboard)."""

        if self._owns_root:
            self.window.mainloop()
