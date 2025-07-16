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
            buys = [i for i, a in result["trades"] if a == "BUY"]
            sells = [i for i, a in result["trades"] if a == "SELL"]
            ax.scatter(buys, [prices[i] for i in buys], color="red", label="Buy")
            ax.scatter(sells, [prices[i] for i in sells], color="green", label="Sell")
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        btn = ttk.Button(self.root, text="Show Profit Table", command=self._show_profit)
        btn.pack()

    def _show_profit(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Profit Table")
        tree = ttk.Treeview(win, columns=("strategy", "profit"), show="headings")
        tree.heading("strategy", text="Strategy")
        tree.heading("profit", text="Profit (TL)")
        tree.pack(fill="both", expand=True)
        for result in self.results:
            tree.insert("", tk.END, values=(result["name"], f"{result['profit']:.2f}"))

    def run(self) -> None:
        self.root.mainloop()
