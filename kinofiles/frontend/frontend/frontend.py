"""KinoFiles: a TV-style surface for the recommendation agent.

Two pages. `/` is the launcher; `/agent` is where the conversation happens —
what the agent just said, the films it proposed, and the one currently in
focus, all on screen at once so a remote (or a voice) only ever has to move
between three things.
"""

import asyncio
import datetime
import json
import random
import uuid

import reflex as rx

from .posters import poster_urls


# --- State ---
class AgentState(rx.State):
    messages: list[dict[str, str]] = []
    current_input: str = ""
    thread_id: str = ""
    is_loading: bool = False
    show_user_query: bool = False
    is_booting: bool = True
    is_recording: bool = False
    current_name: str = random.choice(["Alba", "Carla", "Nuria"])

    # The titles the agent last put on the table, the art found for them, and
    # the one the viewer is looking at. `selected` is only a highlight until
    # `confirm_selection` sends it back as the answer.
    options: list[str] = []
    posters: dict[str, str] = {}
    selected: str = ""
    is_done: bool = False
    search_criteria: dict = {}
    voice_enabled: bool = True

    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled

    def close_player(self):
        self.is_done = False

    def set_current_input(self, val: str):
        self.current_input = val

    def handle_voice(self, result: str):
        """Receive the toggle script's outcome: recording started, or a transcript."""
        outcome = json.loads(result)
        self.is_recording = bool(outcome.get("recording"))

        if error := outcome.get("error"):
            self.messages.append({"role": "agent", "content": f"Voice error: {error}"})
            return

        text = (outcome.get("text") or "").strip()
        if not text:
            return
        self.current_input = text
        return AgentState.send_message

    async def open_tv_agent(self):
        yield rx.redirect("/agent")

    async def select_movie(self, title: str):
        """Pick the film and confirm it immediately."""
        self.selected = title
        async for event in self.confirm_selection():
            yield event

    def hover_movie(self, title: str):
        """Highlight a movie when mouse enters."""
        self.selected = title

    async def handle_key(self, key: str):
        """Handle keyboard arrows and enter to navigate/select."""
        if key == "ArrowRight":
            self.select_next()
        elif key == "ArrowLeft":
            self.select_prev()
        elif key == "Enter":
            if self.selected:
                async for event in self.confirm_selection():
                    yield event

    def select_next(self):
        if not self.options:
            return
        try:
            idx = self.options.index(self.selected)
            self.selected = self.options[(idx + 1) % len(self.options)]
        except ValueError:
            pass

    def select_prev(self):
        if not self.options:
            return
        try:
            idx = self.options.index(self.selected)
            self.selected = self.options[(idx - 1) % len(self.options)]
        except ValueError:
            pass

    @rx.var
    def latest_message(self) -> str:
        if not self.messages:
            return ""
        msg = self.messages[-1]
        if msg["role"] == "user":
            return "You · " + msg["content"].splitlines()[0]
        return msg["content"]

    @rx.var
    def is_user_turn(self) -> bool:
        if not self.messages:
            return False
        return self.messages[-1]["role"] == "user"

    @rx.var
    def show_user_message(self) -> bool:
        if not self.messages:
            return False
        if self.messages[-1]["role"] == "user":
            return self.show_user_query
        return True

    @rx.var
    def has_messages(self) -> bool:
        return len(self.messages) > 0

    @rx.var
    def history(self) -> list[dict[str, str]]:
        """The turns before this one, fading out as they age."""
        past = self.messages[:-1][-3:]
        fades = ["0.2", "0.35", "0.55"][-len(past) :]
        return [
            {
                "content": message["content"].splitlines()[0],
                "opacity": fade,
                "prefix": "You · " if message["role"] == "user" else "",
            }
            for message, fade in zip(past, fades)
        ]

    @rx.var
    def movies(self) -> list[dict[str, str]]:
        """The current options, dressed for the rail."""
        return [
            {
                "title": title,
                "poster": self.posters.get(title, ""),
                "number": str(i),
                "initial": title[:1].upper(),
            }
            for i, title in enumerate(self.options, 1)
        ]

    @rx.var
    def has_movies(self) -> bool:
        return len(self.options) > 0

    @rx.var
    def has_selection(self) -> bool:
        return self.selected != ""

    @rx.var
    def criteria_badges(self) -> list[str]:
        if not self.search_criteria:
            return []
        badges = []
        for k, v in self.search_criteria.items():
            if not v:
                continue
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "title" in item:
                        badges.append(item["title"])
                    else:
                        badges.append(str(item))
            elif isinstance(v, dict) and "title" in v:
                badges.append(v["title"])
            else:
                badges.append(str(v))
        return badges

    @rx.var
    def has_criteria(self) -> bool:
        return len(self.criteria_badges) > 0

    @rx.var
    def selected_poster(self) -> str:
        return self.posters.get(self.selected, "")

    @rx.var
    def selected_initial(self) -> str:
        return self.selected[:1].upper()

    @rx.var
    def movie_count(self) -> str:
        return f"{len(self.options)} movies"

    async def _turn(self, message: str | None):
        """Run one orchestrator turn and show its reply, narration included.

        Called in-process rather than over HTTP: the chat API lives in this
        same server, so a loopback request would only add a serialization
        round trip and break whenever the backend hot-reloads mid-request.
        The call blocks on the LLM and TTS, hence the thread.
        """
        audio_b64 = None
        try:
            data = await asyncio.to_thread(
                chat_with_agent, ChatRequest(thread_id=self.thread_id, message=message)
            )
            if message is not None:
                self.messages.append({"role": "agent", "content": data.get("reply", "")})
                if self.voice_enabled:
                    audio_b64 = data.get("audio")
            else:
                import base64
                from .api import tts
                greeting = f"Hey {self.current_name}, what do you feel like watching?"
                if self.voice_enabled:
                    audio_bytes = await asyncio.to_thread(tts.synthesize, greeting)
                    audio_b64 = base64.b64encode(audio_bytes).decode()

            # Options are the films on offer; a choice means the agent has
            # settled on one, so it takes the spotlight.
            if options := data.get("options"):
                self.options = options
                self.selected = options[0]
            if choice := data.get("choice"):
                self.selected = choice
            self.is_done = data.get("status") == "done"
            if "criteria" in data:
                self.search_criteria = data["criteria"]
        except Exception as e:
            self.messages.append({"role": "agent", "content": f"Error: {str(e)}"})

        self.is_loading = False
        if audio_b64:
            yield rx.call_script(play_audio_script(audio_b64))
        else:
            yield

        # Finding the art means scanning a large catalogue file, so it runs
        # after the reply is already on screen.
        missing = [title for title in self.options if title not in self.posters]
        if missing:
            found = await asyncio.to_thread(poster_urls, missing)
            self.posters = {**self.posters, **found}

    async def reset_agent(self):
        """Reset the conversation state and start fresh."""
        self.messages = []
        self.options = []
        self.selected = ""
        self.is_done = False
        self.is_loading = False
        self.is_booting = True
        self.current_input = ""
        self.thread_id = str(uuid.uuid4())
        self.current_name = random.choice(["Alba", "Carla", "Nuria"])
        self.search_criteria = {}
        yield AgentState.start_agent

    async def start_agent(self):
        self.is_loading = True
        yield
        async for event in self._turn(None):
            yield event
        await asyncio.sleep(0.7)  # Sync fade-in with audio startup
        self.is_booting = False
        yield

    async def _send(self, message: str, shown: str | None = None):
        self.messages.append({"role": "user", "content": shown or message})
        self.current_input = ""
        self.is_loading = True
        self.show_user_query = False
        yield

        async def run_turn():
            events = []
            async for event in self._turn(message):
                events.append(event)
            return events

        turn_task = asyncio.create_task(run_turn())
        done, pending = await asyncio.wait([turn_task], timeout=5.0)

        if not done:
            self.show_user_query = True
            yield
            await turn_task

        for event in turn_task.result():
            yield event

    async def send_message(self):
        if not self.current_input.strip():
            return
        async for event in self._send(self.current_input):
            yield event

    async def confirm_selection(self):
        """Answer the agent's question by picking the film in the spotlight.

        The review node reads a 1-based number, but the viewer clicked a
        poster, so the title is what goes into the transcript.
        """
        if self.is_loading or self.selected not in self.options:
            return
        number = str(self.options.index(self.selected) + 1)
        async for event in self._send(number, shown=self.selected):
            yield event


from .api import ChatRequest, chat_with_agent
from .io_api import api as voice_api


# In dev the page is served by Vite (:3000) while these routes live on the
# Reflex backend (:8000), so a relative fetch would hit Vite and get HTML back.
# `getBackendURL`/`env` are in scope for Reflex's direct eval of call_script;
# the fallback keeps working if that ever stops being true.
BACKEND_URL_JS = """
  const apiURL = (path) => {
    try { return new URL(path, getBackendURL(env.PING).href).href; }
    catch (e) { return `${window.location.protocol}//${window.location.hostname}:8000${path}`; }
  };
"""


def play_audio_script(audio_b64: str) -> str:
    """JS that decodes the base64 wav the orchestrator's reply came with and plays it."""
    return f"""
(async () => {{
  try {{
    const bytes = Uint8Array.from(atob({json.dumps(audio_b64)}), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], {{ type: "audio/wav" }});
    new Audio(URL.createObjectURL(blob)).play();
  }} catch (e) {{}}
}})()
"""

# One click starts recording, the next stops it and transcribes. Returns a
# JSON string so the handler can tell "started" apart from "here's the text".
TOGGLE_RECORDING_JS = """
(async () => {
""" + BACKEND_URL_JS + """
  const recorder = window.__voiceRecorder;

  if (!recorder || recorder.state === "inactive") {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      window.__voiceChunks = [];
      rec.ondataavailable = (e) => window.__voiceChunks.push(e.data);
      rec.start();
      window.__voiceRecorder = rec;
      window.__voiceStream = stream;
      return JSON.stringify({ recording: true });
    } catch (e) {
      return JSON.stringify({ recording: false, error: "Could not open microphone: " + e.message });
    }
  }

  const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
  recorder.stop();
  await stopped;
  window.__voiceStream.getTracks().forEach((t) => t.stop());
  window.__voiceRecorder = null;

  const mime = (recorder.mimeType || "audio/webm").split(";")[0];
  const ext = mime.includes("mp4") ? "mp4" : mime.includes("ogg") ? "ogg" : "webm";
  const blob = new Blob(window.__voiceChunks, { type: mime });
  if (!blob.size) return JSON.stringify({ recording: false, error: "No audio recorded" });

  const form = new FormData();
  form.append("audio", blob, "clip." + ext);
  try {
    const res = await fetch(apiURL("/api/voice"), { method: "POST", body: form });
    if (!res.ok) return JSON.stringify({ recording: false, error: "STT HTTP " + res.status });
    const data = await res.json();
    return JSON.stringify({ recording: false, text: data.text || "" });
  } catch (e) {
    return JSON.stringify({ recording: false, error: e.message });
  }
})()
"""

# --- Design tokens ---
# A light room with white cards floating in it; titan pink is the only accent,
# so anything pink is either the brand or something the viewer can act on.
CANVAS = "#E8E8EB"
SURFACE = "#FFFFFF"
INK = "#15151A"
MUTED = "#8B8B94"
FAINT = "#C6C6CE"
HAIRLINE = "rgba(21, 21, 26, 0.07)"
TINT = "#F2F2F5"
PINK = "#F4434B"
PINK_DEEP = "#DE3A42"
PINK_SOFT = "#FFECED"

SHADOW = "0 24px 60px rgba(20, 20, 30, 0.10), 0 2px 6px rgba(20, 20, 30, 0.04)"
SHADOW_SM = "0 10px 28px rgba(20, 20, 30, 0.07)"
SHADOW_LIFT = "0 30px 60px rgba(20, 20, 30, 0.16)"
RADIUS = "30px"

style = {
    "background_color": "#000",
    "color": INK,
    "font_family": "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif",
    "min_height": "100vh",
    "margin": "0",
    "padding": "0",
}

GLOBAL_CSS = """
<style>
  @keyframes rise { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
  @keyframes glide { from { opacity: 0; transform: translateX(24px); } to { opacity: 1; transform: none; } }
  @keyframes breathe {
    0%, 100% { transform: scale(1) translateX(0); }
    50% { transform: scale(1.12) translateX(12px); }
  }
  @keyframes blink { 0%, 100% { opacity: 0.15; } 50% { opacity: 1; } }
  @keyframes tv-on {
    0%   { clip-path: inset(50% 0 50% 0); filter: brightness(3); opacity: 0; }
    30%  { clip-path: inset(49.5% 0 49.5% 0); filter: brightness(4); opacity: 1; }
    60%  { clip-path: inset(20% 0 20% 0); filter: brightness(1.6); }
    100% { clip-path: inset(0 0 0 0); filter: brightness(1); }
  }
  @keyframes fade-in {
    from { opacity: 0; filter: blur(15px); }
    to { opacity: 1; filter: blur(0); }
  }
  @keyframes pulse-light {
    0% { opacity: 0.3; transform: scale(0.95); }
    50% { opacity: 1; transform: scale(1.05); filter: drop-shadow(0 0 8px rgba(244,67,75,0.6)); }
    100% { opacity: 0.3; transform: scale(0.95); }
  }
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  @keyframes fade-out-delayed {
    0%, 70% { opacity: 1; }
    100% { opacity: 0; display: none; }
  }
  .rail::-webkit-scrollbar { display: none; }
  .rail { scrollbar-width: none; -ms-overflow-style: none; }
</style>
"""


# --- Building blocks ---
def card(*children, **props) -> rx.Component:
    """A white panel floating on the canvas."""
    return rx.box(
        *children,
        bg=SURFACE,
        border=f"1px solid {HAIRLINE}",
        border_radius=RADIUS,
        box_shadow=SHADOW,
        **props,
    )


def caption(text, **props) -> rx.Component:
    """The small grey line that names a section."""
    return rx.text(
        text,
        font_size="0.68rem",
        font_weight="500",
        letter_spacing="0.18em",
        text_transform="uppercase",
        color=MUTED,
        **props,
    )


def pill(content, accent=False, **props) -> rx.Component:
    """A small tag. `accent` may be a Var, so the colours branch at render time."""
    return rx.box(
        rx.text(content, font_size="0.68rem", font_weight="500", line_height="1"),
        bg=rx.cond(accent, PINK_SOFT, TINT),
        color=rx.cond(accent, PINK, MUTED),
        padding="0.42em 0.78em",
        border_radius="999px",
        white_space="nowrap",
        flex_shrink="0",
        **props,
    )


def ambient() -> rx.Component:
    """The soft colour bloom the whole room sits in."""
    return rx.box(
        position="fixed",
        inset="0",
        pointer_events="none",
        z_index="0",
        background_image=(
            "radial-gradient(40% 60% at 5% 50%, rgba(244,67,75,0.16) 0%, rgba(244,67,75,0) 70%),"
            "radial-gradient(30% 50% at 30% 40%, rgba(255,160,180,0.13) 0%, rgba(255,160,180,0) 70%),"
            "radial-gradient(35% 55% at 50% 50%, rgba(180,140,255,0.12) 0%, rgba(180,140,255,0) 70%),"
            "radial-gradient(35% 55% at 75% 45%, rgba(140,160,255,0.11) 0%, rgba(140,160,255,0) 70%),"
            "radial-gradient(30% 50% at 95% 50%, rgba(200,210,255,0.10) 0%, rgba(200,210,255,0) 70%)"
        ),
        filter="blur(40px)",
    )


def status_bar(back: bool = False) -> rx.Component:
    """The TV chrome: who we are on the left, time and weather on the right."""
    brand = rx.hstack(
        rx.image(src="/kinofiles-mark.svg", width="22px", height="22px"),
        rx.text("KinoFiles", font_weight="600", letter_spacing="-0.01em"),
        spacing="2",
        align="center",
    )
    home_button = rx.hstack(
        rx.icon("chevron-left", size=18, color=INK),
        rx.text("Home", font_size="0.9rem", font_weight="500"),
        spacing="1",
        align="center",
        bg=SURFACE,
        border=f"1px solid {HAIRLINE}",
        box_shadow=SHADOW_SM,
        padding="0.5em 1em 0.5em 0.7em",
        border_radius="999px",
        cursor="pointer",
        on_click=rx.redirect("/"),
        _hover={"box_shadow": SHADOW, "transform": "translateY(-1px)"},
        transition="all .2s ease",
    )
    return rx.hstack(
        rx.hstack(*([home_button, brand] if back else [brand]), spacing="4", align="center"),
        rx.hstack(
            rx.icon("sun", size=16, color=MUTED),
            rx.text("18°", color=MUTED, font_size="0.9rem"),
            rx.box(width="1px", height="16px", bg=FAINT),
            rx.vstack(
                rx.text(datetime.datetime.now().strftime("%H:%M"), font_weight="600", font_size="0.95rem", line_height="1"),
                rx.text(datetime.datetime.now().strftime("%A, %B %d"), font_size="0.7rem", color=MUTED, line_height="1"),
                spacing="1",
                align="end",
            ),
            spacing="3",
            align="center",
        ),
        justify="between",
        align="center",
        width="100%",
    )


# --- Agent panel (what the agent is saying) ---
def ghost_line(line: dict[str, str]) -> rx.Component:
    """An older turn, kept on screen but receding."""
    return rx.text(
        line["prefix"] + line["content"],
        font_size="1.05rem",
        color=INK,
        opacity=line["opacity"],
        line_height="1.45",
        max_width="100%",
        overflow="hidden",
        text_overflow="ellipsis",
        white_space="nowrap",
    )


def thinking_dots() -> rx.Component:
    return rx.hstack(
        *[
            rx.box(
                width="7px",
                height="7px",
                border_radius="50%",
                bg=PINK,
                animation=f"blink 1.2s ease-in-out {i * 0.18}s infinite",
            )
            for i in range(3)
        ],
        spacing="2",
        align="center",
        height="16px",
    )


def voice_wave() -> rx.Component:
    """The gradient swoosh that stands in for the agent's voice.

    Nothing clips it: the gradients fade out well inside the box so the blur
    has room to bleed, which is what keeps the edges soft instead of square.
    """
    return rx.box(
        width="min(400px, 100%)",
        height="clamp(64px, 10vh, 110px)",
        pointer_events="none",
        background_image=(
            "radial-gradient(24% 30% at 16% 62%, rgba(244,67,75,0.90) 0%, rgba(244,67,75,0) 100%),"
            "radial-gradient(22% 26% at 38% 42%, rgba(170,120,255,0.70) 0%, rgba(170,120,255,0) 100%),"
            "radial-gradient(24% 28% at 60% 58%, rgba(96,160,255,0.62) 0%, rgba(96,160,255,0) 100%),"
            "radial-gradient(20% 24% at 80% 40%, rgba(255,186,132,0.50) 0%, rgba(255,186,132,0) 100%)"
        ),
        filter="blur(16px) saturate(1.15)",
        style={
            "maskImage": "linear-gradient(90deg, transparent 0%, #000 18%, #000 78%, transparent 100%)",
            "WebkitMaskImage": "linear-gradient(90deg, transparent 0%, #000 18%, #000 78%, transparent 100%)",
        },
        animation=rx.cond(
            AgentState.is_recording,
            "breathe 1.8s ease-in-out infinite",
            "breathe 7s ease-in-out infinite",
        ),
        opacity=rx.cond(AgentState.is_recording | AgentState.is_loading, "1", "0.7"),
        transition="opacity .5s ease",
    )


def mic_button() -> rx.Component:
    return rx.box(
        rx.center(
            rx.cond(
                AgentState.is_recording,
                rx.icon("square", size=18, color="white"),
                rx.icon("mic", size=20, color="white"),
            ),
            width="100%",
            height="100%",
        ),
        width="68px",
        height="68px",
        border_radius="50%",
        border=f"6px solid {SURFACE}",
        bg=PINK,
        cursor="pointer",
        flex_shrink="0",
        position="absolute",
        right="1.4rem",
        bottom="1.4rem",
        z_index="10",
        box_shadow=rx.cond(
            AgentState.is_recording,
            f"0 0 0 10px {PINK_SOFT}, {SHADOW_SM}",
            SHADOW_SM,
        ),
        on_click=rx.call_script(TOGGLE_RECORDING_JS, callback=AgentState.handle_voice),
        _hover={"transform": "scale(1.06)"},
        transition="all .2s ease",
    )


def voice_toggle_button() -> rx.Component:
    return rx.box(
        rx.center(
            rx.cond(
                AgentState.voice_enabled,
                rx.icon("volume-2", size=18, color="white"),
                rx.icon("volume-x", size=18, color="white"),
            ),
            width="100%",
            height="100%",
        ),
        width="48px",
        height="48px",
        border_radius="50%",
        border=f"4px solid {SURFACE}",
        bg=rx.cond(AgentState.voice_enabled, MUTED, FAINT),
        cursor="pointer",
        position="absolute",
        right="2rem",
        bottom="6.6rem",
        z_index="10",
        box_shadow=SHADOW_SM,
        on_click=AgentState.toggle_voice,
        _hover={"transform": "scale(1.06)"},
        transition="all .2s ease",
    )


def agent_panel() -> rx.Component:
    return rx.vstack(
        rx.vstack(
            rx.foreach(AgentState.history, ghost_line),
            spacing="2",
            align="start",
            width="100%",
        ),
        rx.vstack(
            rx.cond(
                AgentState.has_messages,
                rx.cond(
                    AgentState.show_user_message,
                    rx.text(
                        AgentState.latest_message,
                        font_size="clamp(1.8rem, 2.8vw, 3rem)",
                        font_weight="600",
                        letter_spacing="-0.025em",
                        line_height="1.18",
                        color=INK,
                        max_width="100%",
                        animation="rise .5s ease-out",
                        white_space=rx.cond(AgentState.is_user_turn, "nowrap", "normal"),
                        overflow=rx.cond(AgentState.is_user_turn, "hidden", "visible"),
                        text_overflow=rx.cond(AgentState.is_user_turn, "ellipsis", "clip"),
                    ),
                    rx.box(height="0px"),
                ),
                rx.vstack(
                    rx.text("Hey ", AgentState.current_name, ",", font_size="1.6rem", font_weight="500", color=MUTED),
                    rx.text("What do you feel like watching?", font_size="clamp(2rem, 3vw, 3.5rem)", font_weight="600", letter_spacing="-0.03em", line_height="1.1", color=INK, white_space="nowrap"),
                    spacing="2",
                    align="start",
                    animation="rise .5s ease-out",
                )
            ),
            rx.cond(AgentState.is_loading, thinking_dots(), rx.box(height="16px")),
            spacing="2",
            align="start",
        ),
        spacing="4",
        align="start",
        justify="center",
        width="100%",
        flex="1",
        min_width="320px",
    )


# --- Selection (the film in the spotlight) ---
def poster_art(src, initial, **props) -> rx.Component:
    """A poster, or a quiet placeholder when the catalogue has no art."""
    return rx.box(
        rx.cond(
            src != "",
            rx.image(src=src, width="100%", height="100%", object_fit="cover", loading="lazy", alt=""),
            rx.center(
                rx.text(initial, font_size="2rem", font_weight="300", color=FAINT),
                width="100%",
                height="100%",
                bg=TINT,
            ),
        ),
        overflow="hidden",
        bg=TINT,
        **props,
    )


def selection_panel() -> rx.Component:
    """The detail view: the art, the title, and one thing to press."""
    return card(
        rx.vstack(
            rx.hstack(
                rx.cond(AgentState.is_done, caption("Chosen"), rx.box()),
                rx.cond(AgentState.is_done, pill("Now playing", accent=True), rx.box()),
                justify="between",
                align="center",
                width="100%",
            ),
            rx.box(
                poster_art(
                    AgentState.selected_poster,
                    AgentState.selected_initial,
                    width="100%",
                    height="100%",
                    border_radius="22px",
                ),
                rx.center(
                    rx.center(
                        rx.icon("play", size=22, color=PINK, fill=PINK),
                        width="64px",
                        height="64px",
                        border_radius="50%",
                        bg="rgba(255,255,255,0.94)",
                        backdrop_filter="blur(6px)",
                        box_shadow=SHADOW_LIFT,
                        transition="transform .2s ease",
                        _hover={"transform": "scale(1.08)"},
                    ),
                    position="absolute",
                    inset="0",
                    cursor="pointer",
                    on_click=AgentState.confirm_selection,
                ),
                position="relative",
                width="100%",
                flex="1",
                min_height="350px",
            ),
            rx.vstack(
                rx.heading(
                    AgentState.selected,
                    size="7",
                    weight="bold",
                    letter_spacing="-0.03em",
                    line_height="1.12",
                ),
                rx.hstack(pill("Recommendation"), spacing="2", wrap="wrap"),
                spacing="3",
                align="start",
                width="100%",
            ),
            rx.button(
                rx.hstack(
                    rx.icon("play", size=16, fill="white"),
                    rx.text("Watch this", font_weight="600"),
                    spacing="2",
                    align="center",
                ),
                on_click=AgentState.confirm_selection,
                disabled=AgentState.is_loading,
                bg=PINK,
                color="white",
                width="100%",
                height="48px",
                border_radius="999px",
                cursor="pointer",
                _hover={"bg": PINK_DEEP, "transform": "translateY(-1px)"},
                transition="all .2s ease",
                box_shadow=SHADOW_SM,
            ),
            spacing="4",
            align="start",
            width="100%",
            height="100%",
        ),
        padding="1.4rem",
        width="330px",
        flex_shrink="0",
        height="100%",
        overflow="hidden",
        animation="glide .5s ease-out",
    )


# --- Rail (everything on the table) ---
def movie_card(movie: dict[str, str]) -> rx.Component:
    chosen = AgentState.selected == movie["title"]
    return rx.vstack(
        poster_art(
            movie["poster"],
            movie["initial"],
            width="100%",
            height="clamp(160px, 22vh, 230px)",
            border_radius="16px",
            filter=rx.cond(chosen, "none", "grayscale(1) contrast(0.95)"),
            transition="filter .4s ease",
        ),
        rx.hstack(
            pill(movie["number"], accent=chosen),
            rx.text(
                movie["title"],
                font_size="0.9rem",
                font_weight="500",
                color=INK,
                overflow="hidden",
                text_overflow="ellipsis",
                white_space="nowrap",
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        spacing="3",
        align="start",
        width="clamp(110px, 10vw, 150px)",
        flex_shrink="0",
        padding="0.55rem",
        border_radius="24px",
        bg=rx.cond(chosen, TINT, "transparent"),
        box_shadow=rx.cond(chosen, f"inset 0 0 0 1.5px {PINK}", "none"),
        transform=rx.cond(chosen, "translateY(-6px)", "none"),
        cursor="pointer",
        on_click=AgentState.select_movie(movie["title"]),
        on_mouse_enter=AgentState.hover_movie(movie["title"]),
        _hover={"transform": "translateY(-6px)"},
        transition="all .3s cubic-bezier(0.25, 0.8, 0.25, 1)",
    )


def movie_rail() -> rx.Component:
    return card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    caption("Recommendations"),
                    rx.text("Pick one to see it up close", font_size="0.85rem", color=MUTED),
                    spacing="1",
                    align="start",
                ),
                rx.cond(
                    AgentState.has_criteria,
                    rx.hstack(
                        rx.foreach(AgentState.criteria_badges, lambda c: pill(c)),
                        spacing="2",
                        wrap="wrap",
                        justify="end",
                    ),
                    rx.box()
                ),
                justify="between",
                align="center",
                width="100%",
            ),
            rx.box(
                rx.hstack(
                    rx.foreach(AgentState.movies, movie_card),
                    spacing="3",
                    align="start",
                    padding_top="8px",
                ),
                class_name="rail",
                width="100%",
                overflow_x="auto",
                padding_bottom="0.25rem",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        mic_button(),
        voice_toggle_button(),
        padding="1.3rem 1.4rem",
        width="100%",
        flex_shrink="0",
        animation="rise .6s ease-out",
        position="relative",
    )


def empty_rail() -> rx.Component:
    """Stands in for the rail before the agent has proposed anything."""
    return card(
        rx.hstack(
            rx.center(
                rx.icon("clapperboard", size=28, color=FAINT),
                width="64px",
                height="64px",
                border_radius="16px",
                bg=TINT,
                flex_shrink="0",
            ),
            rx.vstack(
                rx.text("No recommendations yet", font_size="1.3rem", font_weight="600"),
                rx.text(
                    "Tell the agent what you're in the mood for and they'll appear here.",
                    font_size="1rem",
                    color=MUTED,
                ),
                spacing="1",
                align="start",
            ),
            spacing="5",
            align="center",
        ),
        mic_button(),
        voice_toggle_button(),
        display="flex",
        align_items="center",
        justify_content="flex-start",
        padding="1.5rem 2rem",
        width="100%",
        min_height="180px",
        position="relative",
    )


def fake_player() -> rx.Component:
    """A fake full-screen video player that appears when a movie is chosen."""
    return rx.box(
        # Background: blurred poster
        rx.box(
            rx.cond(
                AgentState.selected_poster != "",
                rx.image(
                    src=AgentState.selected_poster,
                    position="absolute",
                    inset="0",
                    width="100%",
                    height="100%",
                    object_fit="cover",
                    opacity="0.3",
                    filter="blur(20px)",
                ),
                rx.box()
            ),
            position="absolute",
            inset="0",
            z_index="0",
        ),
        # Center content: Spinner and Title
        rx.center(
            rx.vstack(
                rx.spinner(size="3", color="white"),
                rx.heading(AgentState.selected, size="8", color="white", margin_top="1rem", text_align="center"),
                rx.text("LOADING MOVIE...", font_size="1.2rem", color="rgba(255,255,255,0.7)", text_transform="uppercase", letter_spacing="0.1em"),
                spacing="3",
                align="center",
                z_index="1",
            ),
            width="100%",
            height="100%",
        ),
        # Close button
        rx.icon(
            "x",
            size=30,
            color="white",
            position="absolute",
            top="2rem",
            right="3rem",
            cursor="pointer",
            z_index="2",
            on_click=AgentState.close_player,
            opacity="0.6",
            _hover={"opacity": "1", "transform": "scale(1.1)"},
            bg="rgba(0,0,0,0.4)",
            border_radius="50%",
            padding="4px",
            transition="all 0.2s ease"
        ),
        position="fixed",
        inset="0",
        z_index="100",
        background="linear-gradient(135deg, rgba(15,15,20,1) 0%, rgba(0,0,0,1) 100%)",
        opacity=rx.cond(AgentState.is_done, "1", "0"),
        pointer_events=rx.cond(AgentState.is_done, "auto", "none"),
        transition="opacity 1.2s ease-in-out",
    )


# --- Pages ---
def agent_tv_panel() -> rx.Component:
    return rx.box(
        rx.button(
            id="hidden_key_btn",
            on_click=AgentState.handle_key(rx.Var.create("document.getElementById('hidden_key_val').value")),
            style={"display": "none"},
        ),
        rx.input(id="hidden_key_val", style={"display": "none"}),
        rx.script("""
            document.addEventListener('keydown', function(e) {
                if (['ArrowRight', 'ArrowLeft', 'Enter'].includes(e.key)) {
                    var el = document.getElementById('hidden_key_val');
                    if (el) {
                        el.value = e.key;
                        document.getElementById('hidden_key_btn').click();
                    }
                }
            });
        """),
        rx.box(
            rx.html(GLOBAL_CSS),
            ambient(),
            rx.vstack(
                status_bar(back=True),
                rx.hstack(
                    agent_panel(),
                    spacing="6",
                    align="stretch",
                    width="100%",
                    flex="1",
                    min_height="0",
                    wrap="wrap",
                ),
                rx.hstack(
                    rx.box(
                        rx.cond(AgentState.has_movies, movie_rail(), empty_rail()),
                        flex="1",
                        min_width="0",
                    ),
                    spacing="6",
                    align="end",
                    width="100%",
                    flex_shrink="0",
                ),
                spacing="5",
                width="100%",
                max_width="1320px",
                height="100vh",
                margin="0 auto",
                padding="1.8rem clamp(1.2rem, 4vw, 3rem) 2rem",
                position="relative",
                z_index="1",
                opacity=rx.cond(AgentState.is_booting, "0", "1"),
                transition="opacity 1s ease-in-out",
            ),
            bg=CANVAS,
            height="100vh",
            width="100%",
            overflow="hidden",
            position="relative",
            animation="fade-in .7s ease-out both",
        ),
        fake_player()
    )


def app_tile(app: dict[str, str], index: int) -> rx.Component:
    return rx.center(
        rx.image(
            src=app["icon"],
            width="100%",
            height="100%",
            border_radius="20px",
            object_fit="cover",
        ),
        width="86px",
        height="86px",
        flex_shrink="0",
        padding="0",
        bg=SURFACE,
        border=f"1.5px solid {HAIRLINE}",
        border_radius="22px",
        box_shadow=SHADOW_SM,
        cursor="pointer",
        overflow="hidden",
        _hover={"transform": "translateY(-6px)", "box_shadow": SHADOW},
        transition="all .3s cubic-bezier(0.25, 0.8, 0.25, 1)",
        animation=f"glide .5s ease-out {0.06 * index}s both",
    )


def favourite_apps_row() -> rx.Component:
    apps = [
        {"name": "Netflix", "color": "#E50914", "icon": "https://about.netflix.com/images/meta/netflix-symbol-black.png"},
        {"name": "Prime Video", "color": "#00A8E1", "icon": "https://www.google.com/s2/favicons?domain=primevideo.com&sz=128"},
        {"name": "Disney+", "color": "#113CCF", "icon": "https://www.google.com/s2/favicons?domain=disneyplus.com&sz=128"},
        {"name": "HBO", "color": "#663399", "icon": "https://www.google.com/s2/favicons?domain=max.com&sz=128"},
        {"name": "Apple TV", "color": "#2B2B2E", "icon": "https://www.google.com/s2/favicons?domain=tv.apple.com&sz=128"},
        {"name": "YouTube", "color": "#FF0000", "icon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlx4g7JLK01O3GnMA87AdWdM0zxs9csdWorrUX8S17cQ&s=10"},
        {"name": "Twitch", "color": "#9146FF", "icon": "https://cdn-icons-png.flaticon.com/512/2504/2504946.png"},
        {"name": "Spotify", "color": "#1DB954", "icon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBD_K9JdZ2ghWpJVl-TeDlG8IJqSXZ_Svur3KJdoH_xg&s"},
        {"name": "Browser", "color": "#4285F4", "icon": "https://www.google.com/s2/favicons?domain=google.com&sz=128"},
        {"name": "Crunchyroll", "color": "#F47521", "icon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRkGTfZH3-w93cXUKy9Ngv9f9a_J2BZuZqnyVmFnpsRg&s=10"},
        {"name": "3Cat", "color": "#000000", "icon": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrAllgUZtIE01JQl9jjNSZbZhvuewA65gigXB1M_ncGQ&s=10"},
        {"name": "Music", "color": "#FA243C", "icon": "https://www.google.com/s2/favicons?domain=music.apple.com&sz=128"},
    ]
    return rx.vstack(
        caption("Your apps"),
        rx.divider(border_color=HAIRLINE, width="100%", margin_bottom="0.5rem"),
        rx.box(
            rx.hstack(*[app_tile(app, i) for i, app in enumerate(apps)], spacing="3"),
            class_name="rail",
            width="100%",
            overflow_x="auto",
            padding_y="0.5rem",
        ),
        spacing="3",
        align="start",
        width="100%",
    )


def hero() -> rx.Component:
    return rx.vstack(
        rx.heading(
            "Tell me what you're in the mood for.",
            size="9",
            weight="bold",
            letter_spacing="-0.035em",
            line_height="1.05",
        ),
        rx.text(
            "Your mood in, the perfect movie out.",
            max_width="100%",
            font_size="1.05rem",
            color=MUTED,
            line_height="1.6",
        ),
        rx.button(
            rx.hstack(
                rx.icon("arrow-right", size=17),
                rx.text("Launch agent", font_weight="600"),
                spacing="2",
                align="center",
            ),
            on_click=AgentState.open_tv_agent,
            bg=PINK,
            color="white",
            height="52px",
            padding_x="1.8em",
            border_radius="999px",
            cursor="pointer",
            box_shadow=f"0 14px 30px {PINK}3d",
            _hover={"bg": PINK_DEEP, "transform": "translateY(-2px)"},
            transition="all .2s ease",
            margin_top="0.5rem",
        ),
        spacing="4",
        align="start",
        width="100%",
        animation="rise .7s ease-out",
    )


def index() -> rx.Component:
    return rx.box(
        rx.html(GLOBAL_CSS),
        # The poster wall, washed almost white so the cards stay readable.
        rx.box(
            position="fixed",
            inset="0",
            background_image="url('/movie_posters_bg.jpg')",
            background_size="cover",
            background_position="center",
            filter="grayscale(1)",
            opacity="0.12",
            z_index="0",
        ),
        rx.box(
            position="fixed",
            inset="0",
            background_image=f"linear-gradient(105deg, {CANVAS} 0%, {CANVAS}f2 45%, {CANVAS}c9 100%)",
            z_index="0",
        ),
        ambient(),
        rx.vstack(
            status_bar(),
            rx.box(flex="1"),
            hero(),
            rx.box(height="1.5rem"),
            favourite_apps_row(),
            rx.box(flex="0.4"),
            spacing="5",
            align="start",
            width="100%",
            max_width="1320px",
            min_height="100vh",
            margin="0 auto",
            padding="2.2rem clamp(1.2rem, 4vw, 3rem) 2.5rem",
            position="relative",
            z_index="1",
        ),
        bg=CANVAS,
        min_height="100vh",
        width="100%",
        position="relative",
        animation="tv-on .8s cubic-bezier(0.22, 1, 0.36, 1) both",
    )


app = rx.App(
    style=style,
    api_transformer=voice_api,
    head_components=[
        rx.el.link(rel="apple-touch-icon", sizes="180x180", href="/apple-touch-icon.png"),
        rx.el.link(rel="icon", type="image/png", sizes="32x32", href="/favicon-32x32.png"),
        rx.el.link(rel="icon", type="image/png", sizes="16x16", href="/favicon-16x16.png"),
        rx.el.link(rel="manifest", href="/site.webmanifest"),
    ],
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
)
app.add_page(index, title="KinoFiles OS")
app.add_page(agent_tv_panel, route="/agent", title="KinoFiles Agent TV", on_load=AgentState.reset_agent)
