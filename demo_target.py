import random
import time
import tkinter as tk


IDLE_COLOR = "#2d7dd2"
BITE_COLOR = "#ffd166"
BG_COLOR = "#111827"


class DemoTargetApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Safe Practice Fishing Target")
        self.root.geometry("560x380")
        self.root.configure(bg=BG_COLOR)

        self.status = tk.StringVar(value="Status: Waiting...")
        self.next_bite_at = time.time() + random.uniform(2.0, 5.5)
        self.bite_end_at = 0.0

        title = tk.Label(
            self.root,
            text="Local Demo Target (Training Only)",
            fg="white",
            bg=BG_COLOR,
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(pady=(14, 8))

        self.canvas = tk.Canvas(
            self.root,
            width=420,
            height=230,
            bg="#0b1220",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.canvas.pack(pady=8)

        self.bobber = self.canvas.create_oval(180, 80, 240, 140, fill=IDLE_COLOR, outline="")
        self.canvas.create_text(
            210,
            180,
            text="When circle turns yellow = bite event",
            fill="#cbd5e1",
            font=("Segoe UI", 11),
        )

        status_label = tk.Label(
            self.root,
            textvariable=self.status,
            fg="#cbd5e1",
            bg=BG_COLOR,
            font=("Segoe UI", 12),
        )
        status_label.pack(pady=(8, 4))

        help_label = tk.Label(
            self.root,
            text="Keep this window visible and run safe_fishing_practice.py",
            fg="#94a3b8",
            bg=BG_COLOR,
            font=("Segoe UI", 10),
        )
        help_label.pack()

        self._tick()

    def _tick(self) -> None:
        now = time.time()
        in_bite = now < self.bite_end_at

        if not in_bite and now >= self.next_bite_at:
            bite_duration = random.uniform(0.45, 1.2)
            self.bite_end_at = now + bite_duration
            self.next_bite_at = now + random.uniform(2.0, 5.5)
            in_bite = True

        if in_bite:
            self.canvas.itemconfig(self.bobber, fill=BITE_COLOR)
            self.status.set("Status: BITE!")
        else:
            self.canvas.itemconfig(self.bobber, fill=IDLE_COLOR)
            wait = max(0.0, self.next_bite_at - now)
            self.status.set(f"Status: Waiting... next bite in ~{wait:.1f}s")

        self.root.after(40, self._tick)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    DemoTargetApp().run()
