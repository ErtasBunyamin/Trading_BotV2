"""Simple Tkinter GUI to visualize trading strategies results."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class TradingApp:
    """Display strategy performance charts and profit table."""

    def __init__(self, results: dict) -> None:
        self.results = results
        self.root = tk.Tk()
        self.root.title("Trading Strategies Simulator")
        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        for name, data in self.results.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=name)
            self._plot_strategy(frame, data)
        btn = ttk.Button(self.root, text="Show Profit Table", command=self._show_table)
        btn.pack(pady=5)

    def _plot_strategy(self, parent: tk.Widget, data: dict) -> None:
        fig = Figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        prices = data["prices"]
        ax.plot(prices, label="Price")
        for idx, price, action in data["signals"]:
            color = "red" if action == "buy" else "green"
            ax.scatter(idx, price, c=color)
        ax.set_title(data["name"])
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_table(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Profit Table")
        tree = ttk.Treeview(win, columns=("profit",), show="headings")
        tree.heading("profit", text="Profit")
        for name, data in self.results.items():
            tree.insert("", tk.END, values=(f"{data['profit']:.2f}",), text=name)
        tree.pack(fill=tk.BOTH, expand=True)

    def run(self) -> None:
        self.root.mainloop()
