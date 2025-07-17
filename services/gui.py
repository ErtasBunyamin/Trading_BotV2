"""Simple Tkinter GUI to display strategy results."""

from __future__ import annotations

from typing import List, Dict, Tuple

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def fmt(value: float | None, fmt_str: str = ".2f") -> str:
    """Return formatted float or '-' when value is ``None``."""
    return f"{value:{fmt_str}}" if value is not None else "-"


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
            opps = result.get("opportunities", [])
            ax.scatter([b[0] for b in buys], [b[2] for b in buys], color="red", label="Buy")
            ax.scatter([s[0] for s in sells], [s[2] for s in sells], color="green", label="Sell")
            if opps:
                buys_m = [o for o in opps if o[2] == "BUY"]
                sells_m = [o for o in opps if o[2] == "SELL"]
                if buys_m:
                    ax.scatter([o[0] for o in buys_m], [o[1] for o in buys_m], color="purple", marker="x", label="Missed Buy")
                    for i, p, *_ in buys_m:
                        ax.annotate("M", (i, p), textcoords="offset points", xytext=(0, -10), ha="center", color="purple")
                if sells_m:
                    ax.scatter([o[0] for o in sells_m], [o[1] for o in sells_m], color="orange", marker="x", label="Missed Sell")
                    for i, p, *_ in sells_m:
                        ax.annotate("M", (i, p), textcoords="offset points", xytext=(0, -10), ha="center", color="orange")
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
            "profit_threshold",
            "trailing_stop",
            "missed_buy",
            "missed_sell",
            "missed_profit",
            "expected_profit",
            "trades",
            "avg_trade",
            "missed_count",
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
        tree.heading("profit_threshold", text="Profit Th")
        tree.heading("trailing_stop", text="Trailing %")
        tree.heading("missed_buy", text="Missed Buy")
        tree.heading("missed_sell", text="Missed Sell")
        tree.heading("missed_profit", text="Missed Pot")
        tree.heading("expected_profit", text="Expected")
        tree.heading("trades", text="Trades")
        tree.heading("avg_trade", text="Avg Size")
        tree.heading("missed_count", text="Missed Cnt")
        tree.pack(fill="both", expand=True)

        for result in self.results:
            tree.insert(
                "",
                tk.END,
                values=(
                    result["name"],
                    fmt(result.get("profit")),
                    fmt(result.get("profit_pct")),
                    fmt(result.get("final_balance")),
                    fmt(result.get("bought"), ".4f"),
                    fmt(result.get("sold"), ".4f"),
                    fmt(result.get("remaining_btc"), ".4f"),
                    fmt(result.get("holding_value")),
                    fmt(result.get("profit_threshold")),
                    fmt(result.get("trailing_stop_pct")),
                    result.get("missed_buy", 0),
                    result.get("missed_sell", 0),
                    fmt(result.get("missed_profit")),
                    fmt(result.get("expected_profit")),
                    result.get("trade_count", 0),
                    fmt(result.get("avg_trade_size"), ".4f"),
                    result.get("missed_count", 0),
                ),
            )

    def _show_trades(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Trade Log")
        cols = (
            "strategy",
            "candle",
            "action",
            "amount",
            "price",
            "balance",
            "reason",
            "pnl",
            "potential",
        )
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        for col, text in zip(
            cols,
            [
                "Strategy",
                "Candle",
                "Action",
                "Amount (BTC)",
                "Price",
                "Balance (TL)",
                "Reason",
                "PnL",
                "Potential",
            ],
        ):
            tree.heading(col, text=text)

        for result in self.results:
            # Detailed logs contain extra fields
            details = result.get("details", [])
            for log in details:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        result["name"],
                        log.get("idx"),
                        log.get("action"),
                        fmt(log.get("amount"), ".4f"),
                        fmt(log.get("price")),
                        fmt(log.get("balance_after")),
                        log.get("reason", ""),
                        fmt(log.get("pnl")),
                        "",
                    ),
                )

            for idx, price, action, pot in result.get("opportunities", []):
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        result["name"],
                        idx,
                        f"MISSED_{action}",
                        "",
                        fmt(price),
                        "",
                        "missed",
                        "",
                        fmt(pot),
                    ),
                )

    def run(self) -> None:
        self.root.mainloop()
