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
            buys = [
                (i, amt, price)
                for i, a, amt, price, _ in result["trades"]
                if a == "BUY"
            ]
            sells = [
                (i, amt, price)
                for i, a, amt, price, _ in result["trades"]
                if a == "SELL"
            ]
            ax.scatter([b[0] for b in buys], [b[2] for b in buys], color="red", label="Buy")
            ax.scatter([s[0] for s in sells], [s[2] for s in sells], color="green", label="Sell")
            for idx, amt, trade_price in buys:
                ax.annotate(f"{amt:.4f}", (idx, trade_price), textcoords="offset points", xytext=(0, 5), ha="center", color="red")
            for idx, amt, trade_price in sells:
                ax.annotate(f"{amt:.4f}", (idx, trade_price), textcoords="offset points", xytext=(0, 5), ha="center", color="green")
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        btn = ttk.Button(self.root, text="Show Profit Table", command=self._show_profit)
        btn.pack()
        btn2 = ttk.Button(self.root, text="Show Trades", command=self._show_trades)
        btn2.pack()

    def _show_profit(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Profit Table")
        cols = (
            "strategy",
            "profit",
            "profit_pct",
            "balance",
            "bought",
            "sold",
            "remaining",
            "value",
        )
        tree = ttk.Treeview(win, columns=cols, show="headings")
        tree.heading("strategy", text="Strategy")
        tree.heading("profit", text="Profit (TL)")
        tree.heading("profit_pct", text="Profit (%)")
        tree.heading("balance", text="Balance (TL)")
        tree.heading("bought", text="Bought (BTC)")
        tree.heading("sold", text="Sold (BTC)")
        tree.heading("remaining", text="Remaining (BTC)")
        tree.heading("value", text="Value (TL)")
        tree.pack(fill="both", expand=True)
        for result in self.results:
            tree.insert(
                "",
                tk.END,
                values=(
                    result["name"],
                    f"{result['profit']:.2f}",
                    f"{result['profit_pct']:.2f}",
                    f"{result['final_balance']:.2f}",
                    f"{result['bought']:.4f}",
                    f"{result['sold']:.4f}",
                    f"{result['remaining_btc']:.4f}",
                    f"{result['holding_value']:.2f}",
                ),
            )

    def _show_trades(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Trade Log")
        cols = ("strategy", "candle", "action", "amount", "price", "balance")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for col, text in zip(
            cols,
            ["Strategy", "Candle", "Action", "Amount (BTC)", "Price", "Balance (TL)"],
        ):
            tree.heading(col, text=text)
        tree.pack(fill="both", expand=True)
        for result in self.results:
            for idx, action, amount, price, balance in result["trades"]:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        result["name"],
                        idx,
                        action,
                        f"{amount:.4f}",
                        f"{price:.2f}",
                        f"{balance:.2f}",
                    ),
                )

    def run(self) -> None:
        self.root.mainloop()
