"""
Control panel: the first screen of the application.

Pick a version and a model, review the configuration that will be
used, then run. Models that need training are trained in a background
thread (with a progress bar) before the dashboard opens.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from cleaning_vehicle.agents.base_agent import BaseAgent
from cleaning_vehicle.config import ConfigCatalog, ConfigError
from cleaning_vehicle.config.schema import ScenarioConfig
from cleaning_vehicle.visualization.dashboard import SimulationDashboard

from .builders import build_simulation, prepare_agent

_POLL_MS = 50


class Launcher:
    """The version / model selection window."""

    def __init__(
        self,
        catalog: ConfigCatalog,
        version: str | None = None,
        model: str | None = None,
    ) -> None:

        self.catalog = catalog
        self._versions = list(catalog.versions().values())
        self._models = list(catalog.models().values())

        if not self._versions or not self._models:
            raise ConfigError(
                f"No available versions or models in {catalog.config_dir}"
            )

        self._results: queue.Queue = queue.Queue()
        self._progress = (0, 1)

        self.root = tk.Tk()
        self.root.title("Autonomous Cleaning Vehicle - Control Panel")
        self.root.resizable(False, False)

        self._build_layout()
        self._select(self.version_box, self._versions, version)
        self._select(self.model_box, self._models, model)
        self._refresh_summary()

    # ==========================================================
    # Layout
    # ==========================================================

    def _build_layout(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Autonomous Cleaning Vehicle",
            font=("Arial", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame, text="Choose what to run, then press Run."
        ).pack(anchor="w", pady=(0, 12))

        self.version_box, self.version_info = self._choice(
            frame, "1. Version", [v.name for v in self._versions]
        )
        self.model_box, self.model_info = self._choice(
            frame, "2. Model", [m.name for m in self._models]
        )

        summary = ttk.LabelFrame(
            frame, text="Configuration (from configs/)", padding=8
        )
        summary.pack(fill="x", pady=(0, 12))
        self.summary = tk.StringVar()
        ttk.Label(
            summary, textvariable=self.summary, justify="left"
        ).pack(anchor="w")

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(frame, textvariable=self.status).pack(anchor="w")

        self.progress = ttk.Progressbar(frame, length=380)
        self.progress.pack(fill="x", pady=(4, 12))

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")

        self.run_button = ttk.Button(
            buttons, text="Run simulation", command=self._on_run
        )
        self.run_button.pack(side="left")
        ttk.Button(buttons, text="Quit", command=self.root.destroy).pack(
            side="right"
        )

    def _choice(
        self,
        parent: tk.Misc,
        title: str,
        names: list[str],
    ) -> tuple[ttk.Combobox, tk.StringVar]:
        """A labelled drop-down with a description line under it."""

        box_frame = ttk.LabelFrame(parent, text=title, padding=8)
        box_frame.pack(fill="x", pady=(0, 8))

        combo = ttk.Combobox(
            box_frame, values=names, state="readonly", width=48
        )
        combo.pack(anchor="w")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_summary())

        info = tk.StringVar()
        ttk.Label(box_frame, textvariable=info, foreground="#555555").pack(
            anchor="w", pady=(4, 0)
        )

        return combo, info

    @staticmethod
    def _select(combo: ttk.Combobox, entries: list, wanted: str | None) -> None:
        ids = [entry.id for entry in entries]
        combo.current(ids.index(wanted) if wanted in ids else 0)

    # ==========================================================
    # Selection
    # ==========================================================

    @property
    def selected_version(self) -> str:
        return self._versions[self.version_box.current()].id

    @property
    def selected_model(self) -> str:
        return self._models[self.model_box.current()].id

    def _refresh_summary(self) -> None:
        self.version_info.set(self._versions[self.version_box.current()].description)
        self.model_info.set(self._models[self.model_box.current()].description)

        try:
            scenario = self.catalog.load(
                self.selected_version, self.selected_model
            )
        except ConfigError as error:
            self.summary.set(f"Invalid configuration:\n{error}")
            self.run_button.state(["disabled"])
            return

        self.summary.set("\n".join(scenario.summary_lines()))
        self.run_button.state(["!disabled"])

    # ==========================================================
    # Running
    # ==========================================================

    def _on_run(self) -> None:
        try:
            scenario = self.catalog.load(
                self.selected_version, self.selected_model
            )
        except ConfigError as error:
            messagebox.showerror("Invalid configuration", str(error))
            return

        self.run_button.state(["disabled"])
        self.progress["value"] = 0

        if scenario.agent.requires_training:
            self.status.set(
                f"Training {scenario.agent.name} "
                f"({scenario.training.episodes} episodes)..."
            )
        else:
            self.status.set("Starting...")

        threading.Thread(
            target=self._prepare_in_background,
            args=(scenario,),
            daemon=True,
        ).start()
        self.root.after(_POLL_MS, self._poll, scenario)

    def _prepare_in_background(self, scenario: ScenarioConfig) -> None:
        """Worker thread: never touches Tk widgets."""

        def report(done: int, total: int) -> None:
            self._progress = (done, total)

        try:
            self._results.put(("ok", prepare_agent(scenario, report)))
        except Exception as error:  # surfaced to the user in _poll
            self._results.put(("error", error))

    def _poll(self, scenario: ScenarioConfig) -> None:
        done, total = self._progress
        self.progress["value"] = 100 * done / total

        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            self.root.after(_POLL_MS, self._poll, scenario)
            return

        self._progress = (0, 1)
        self.progress["value"] = 0
        self.run_button.state(["!disabled"])

        if kind == "error":
            self.status.set("Failed.")
            messagebox.showerror("Could not start the run", str(payload))
            return

        self.status.set("Simulation window opened. Ready for another run.")
        self._open_dashboard(scenario, payload)

    def _open_dashboard(
        self, scenario: ScenarioConfig, agent: BaseAgent
    ) -> SimulationDashboard:
        return SimulationDashboard(
            simulation=build_simulation(scenario, agent),
            visualization=scenario.visualization,
            title=f"{scenario.version_name} - {scenario.agent.name}",
            run_details={
                "Version": scenario.version.upper(),
                "Model": scenario.agent.name,
            },
            master=self.root,
        )

    def run(self) -> None:
        self.root.mainloop()
