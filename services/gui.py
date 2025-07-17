"""Simple Tkinter GUI to display strategy results."""

from __future__ import annotations

from typing import List, Dict, Tuple

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class TradingApp:
    """Display strategy price charts and profits."""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.root = tk.Tk()
        self.root.title("Trading Simulator")
        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        for result in self.results:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=result["name"])
            fig, ax = plt.subplots(figsize=(6, 4))
            prices = result["prices"]
            ax.plot(prices, label="Price")
            buys = [(i, amt) for i, a, amt in result["trades"] if a == "BUY"]
            sells = [(i, amt) for i, a, amt in result["trades"] if a == "SELL"]
            ax.scatter([b[0] for b in buys], [prices[b[0]] for b in buys], color="red", label="Buy")
            ax.scatter([s[0] for s in sells], [prices[s[0]] for s in sells], color="green", label="Sell")
            for idx, amt in buys:
                ax.annotate(f"{amt:.4f}", (idx, prices[idx]), textcoords="offset points", xytext=(0, 5), ha="center", color="red")
            for idx, amt in sells:
                ax.annotate(f"{amt:.4f}", (idx, prices[idx]), textcoords="offset points", xytext=(0, 5), ha="center", color="green")
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        btn = ttk.Button(self.root, text="Show Profit Table", command=self._show_profit)
        btn.pack()

    def _show_profit(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Profit Table")
        cols = ("strategy", "profit", "profit_pct", "bought", "sold")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        tree.heading("strategy", text="Strategy")
        tree.heading("profit", text="Profit (TL)")
        tree.heading("profit_pct", text="Profit (%)")
        tree.heading("bought", text="Bought (BTC)")
        tree.heading("sold", text="Sold (BTC)")
        tree.pack(fill="both", expand=True)
        for result in self.results:
            tree.insert(
                "",
                tk.END,
                values=(
                    result["name"],
                    f"{result['profit']:.2f}",
                    f"{result['profit_pct']:.2f}",
                    f"{result['bought']:.4f}",
                    f"{result['sold']:.4f}",
                ),
            )

    def run(self) -> None:
        self.root.mainloop()
