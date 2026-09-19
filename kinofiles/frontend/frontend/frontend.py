"""KinoFiles: a TV-style surface for the recommendation agent.

Two pages. `/` is the launcher; `/agent` is where the conversation happens —
what the agent just said, the films it proposed, and the one currently in
focus, all on screen at once so a remote (or a voice) only ever has to move
between three things.
"""

import asyncio
import json
import uuid

import reflex as rx

from .posters import poster_urls


# --- State ---
class AgentState(rx.State):
    messages: list[dict[str, str]] = []
    current_input: str = ""
    thread_id: str = ""
    is_loading: bool = False
    is_recording: bool = False

    # The titles the agent last put on the table, the art found for them, and
    # the one the viewer is looking at. `selected` is only a highlight until
    # `confirm_selection` sends it back as the answer.
    options: list[str] = []
    posters: dict[str, str] = {}
    selected: str = ""
    is_done: bool = False

    def set_current_input(self, val: str):
        self.current_input = val

    def handle_voice(self, result: str):
        """Receive the toggle script's outcome: recording started, or a transcript."""
        outcome = json.loads(result)
        self.is_recording = bool(outcome.get("recording"))

        if error := outcome.get("error"):
            self.messages.append({"role": "agent", "content": f"Error de voz: {error}"})
            return

        text = (outcome.get("text") or "").strip()
        if not text:
            return
        self.current_input = text
        return AgentState.send_message

    async def open_tv_agent(self):
        yield rx.redirect("/agent")
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())
            yield AgentState.start_agent

    def select_movie(self, title: str):
        """Move the spotlight to a film without committing to it."""
        self.selected = title

    @rx.var
    def latest_message(self) -> str:
        if not self.messages:
            return "¡Hola! Soy tu agente de KinoFiles. ¿Qué te apetece ver hoy?"
        return self.messages[-1]["content"]

    @rx.var
    def history(self) -> list[dict[str, str]]:
        """The turns before this one, fading out as they age."""
        past = self.messages[:-1][-3:]
        fades = ["0.2", "0.35", "0.55"][-len(past) :]
        return [
            {
                "content": message["content"].splitlines()[0],
                "opacity": fade,
                "prefix": "Tú · " if message["role"] == "user" else "",
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
    def selected_poster(self) -> str:
        return self.posters.get(self.selected, "")

    @rx.var
    def selected_initial(self) -> str:
        return self.selected[:1].upper()

    @rx.var
    def movie_count(self) -> str:
        return f"{len(self.options)} películas"

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
            self.messages.append({"role": "agent", "content": data.get("reply", "")})
            audio_b64 = data.get("audio")

            # Options are the films on offer; a choice means the agent has
            # settled on one, so it takes the spotlight.
            if options := data.get("options"):
                self.options = options
                self.selected = options[0]
            if choice := data.get("choice"):
                self.selected = choice
            self.is_done = data.get("status") == "done"
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

    async def start_agent(self):
        self.is_loading = True
        yield
        async for event in self._turn(None):
            yield event

    async def _send(self, message: str, shown: str | None = None):
        self.messages.append({"role": "user", "content": shown or message})
        self.current_input = ""
        self.is_loading = True
        yield

        async for event in self._turn(message):
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
      return JSON.stringify({ recording: false, error: "No se pudo abrir el micro: " + e.message });
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
  if (!blob.size) return JSON.stringify({ recording: false, error: "No se grabó audio" });

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
    "background_color": CANVAS,
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
            "radial-gradient(45% 55% at 10% 90%, rgba(244,67,75,0.18) 0%, rgba(244,67,75,0) 70%),"
            "radial-gradient(40% 50% at 88% 6%, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0) 70%)"
        ),
    )


def status_bar(back: bool = False) -> rx.Component:
    """The TV chrome: who we are on the left, time and weather on the right."""
    brand = rx.hstack(
        rx.box(width="11px", height="11px", border_radius="50%", bg=PINK),
        rx.text("KinoFiles", font_weight="600", letter_spacing="-0.01em"),
        spacing="2",
        align="center",
    )
    home_button = rx.hstack(
        rx.icon("chevron-left", size=18, color=INK),
        rx.text("Inicio", font_size="0.9rem", font_weight="500"),
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
                rx.text("20:46", font_weight="600", font_size="0.95rem", line_height="1"),
                rx.text("Martes, 12 de octubre", font_size="0.7rem", color=MUTED, line_height="1"),
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
        max_width="30ch",
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
        width="56px",
        height="56px",
        border_radius="50%",
        bg=PINK,
        cursor="pointer",
        flex_shrink="0",
        box_shadow=rx.cond(
            AgentState.is_recording,
            f"0 0 0 10px {PINK_SOFT}, {SHADOW_SM}",
            SHADOW_SM,
        ),
        on_click=rx.call_script(TOGGLE_RECORDING_JS, callback=AgentState.handle_voice),
        _hover={"transform": "scale(1.06)"},
        transition="all .2s ease",
    )


def composer() -> rx.Component:
    """Typing, for when talking to the television is not an option."""
    return rx.hstack(
        mic_button(),
        rx.hstack(
            rx.input(
                placeholder="O escribe lo que te apetece ver…",
                value=AgentState.current_input,
                on_change=AgentState.set_current_input,
                on_key_down=rx.call_script(
                    "if(event.key === 'Enter') { document.getElementById('send_btn').click(); }"
                ),
                variant="soft",
                bg="transparent",
                border="none",
                color=INK,
                width="100%",
                font_size="0.95rem",
                padding_x="0.4em",
                style={
                    "&::placeholder": {"color": FAINT},
                    "&:focus": {"outline": "none", "box_shadow": "none"},
                },
            ),
            rx.box(
                rx.center(rx.icon("arrow-right", size=16, color="white"), width="100%", height="100%"),
                id="send_btn",
                on_click=AgentState.send_message,
                width="34px",
                height="34px",
                border_radius="50%",
                bg=INK,
                cursor="pointer",
                flex_shrink="0",
                _hover={"bg": PINK},
                transition="background .2s ease",
            ),
            spacing="2",
            align="center",
            width="100%",
            bg=SURFACE,
            border=f"1px solid {HAIRLINE}",
            box_shadow=SHADOW_SM,
            border_radius="999px",
            padding="0.45em 0.45em 0.45em 1.1em",
        ),
        spacing="3",
        align="center",
        width="100%",
        max_width="480px",
    )


def agent_panel() -> rx.Component:
    return rx.vstack(
        caption("El agente"),
        rx.vstack(
            rx.foreach(AgentState.history, ghost_line),
            spacing="2",
            align="start",
            width="100%",
        ),
        rx.text(
            AgentState.latest_message,
            font_size="clamp(1.5rem, 2.3vw, 2.5rem)",
            font_weight="600",
            letter_spacing="-0.025em",
            line_height="1.18",
            color=INK,
            max_width="22ch",
            animation="rise .5s ease-out",
        ),
        rx.cond(AgentState.is_loading, thinking_dots(), rx.box(height="16px")),
        voice_wave(),
        composer(),
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
                caption(rx.cond(AgentState.is_done, "Elegida", "En foco")),
                rx.cond(AgentState.is_done, pill("Reproduciendo", accent=True), rx.box()),
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
                min_height="150px",
            ),
            rx.vstack(
                rx.heading(
                    AgentState.selected,
                    size="7",
                    weight="bold",
                    letter_spacing="-0.03em",
                    line_height="1.12",
                ),
                rx.hstack(pill("Recomendación"), spacing="2", wrap="wrap"),
                spacing="3",
                align="start",
                width="100%",
            ),
            rx.button(
                rx.hstack(
                    rx.icon("play", size=16, fill="white"),
                    rx.text("Ver esta", font_weight="600"),
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
            height="clamp(130px, 19vh, 190px)",
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
        _hover={"transform": "translateY(-6px)"},
        transition="all .3s cubic-bezier(0.25, 0.8, 0.25, 1)",
    )


def movie_rail() -> rx.Component:
    return card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    caption("Recomendaciones"),
                    rx.text("Elige una para verla en grande", font_size="0.85rem", color=MUTED),
                    spacing="1",
                    align="start",
                ),
                pill(AgentState.movie_count),
                justify="between",
                align="center",
                width="100%",
            ),
            rx.box(
                rx.hstack(
                    rx.foreach(AgentState.movies, movie_card),
                    spacing="3",
                    align="start",
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
        padding="1.3rem 1.4rem",
        width="100%",
        flex_shrink="0",
        animation="rise .6s ease-out",
    )


def empty_rail() -> rx.Component:
    """Stands in for the rail before the agent has proposed anything."""
    return card(
        rx.hstack(
            rx.center(
                rx.icon("clapperboard", size=20, color=FAINT),
                width="46px",
                height="46px",
                border_radius="16px",
                bg=TINT,
                flex_shrink="0",
            ),
            rx.vstack(
                rx.text("Aún no hay recomendaciones", font_weight="600"),
                rx.text(
                    "Cuéntale al agente qué te apetece y aparecerán aquí.",
                    font_size="0.85rem",
                    color=MUTED,
                ),
                spacing="1",
                align="start",
            ),
            spacing="4",
            align="center",
        ),
        padding="1.5rem",
        width="100%",
    )


# --- Pages ---
def agent_tv_panel() -> rx.Component:
    return rx.box(
        rx.html(GLOBAL_CSS),
        ambient(),
        rx.vstack(
            status_bar(back=True),
            rx.hstack(
                agent_panel(),
                rx.cond(AgentState.has_selection, selection_panel(), rx.box()),
                spacing="6",
                align="stretch",
                width="100%",
                flex="1",
                min_height="0",
                wrap="wrap",
            ),
            rx.cond(AgentState.has_movies, movie_rail(), empty_rail()),
            spacing="5",
            width="100%",
            max_width="1320px",
            height="100vh",
            margin="0 auto",
            padding="1.8rem clamp(1.2rem, 4vw, 3rem) 2rem",
            position="relative",
            z_index="1",
        ),
        bg=CANVAS,
        height="100vh",
        width="100%",
        overflow="hidden",
        position="relative",
    )


def app_tile(app: dict[str, str], index: int) -> rx.Component:
    return rx.vstack(
        rx.box(
            width="34px",
            height="34px",
            border_radius="12px",
            bg=app["color"],
            box_shadow=f"0 8px 18px {app['color']}44",
        ),
        rx.text(app["name"], font_size="0.85rem", font_weight="500"),
        spacing="3",
        align="start",
        justify="center",
        width="132px",
        height="108px",
        flex_shrink="0",
        padding="1rem",
        bg=SURFACE,
        border=f"1px solid {HAIRLINE}",
        border_radius="22px",
        box_shadow=SHADOW_SM,
        cursor="pointer",
        _hover={"transform": "translateY(-6px)", "box_shadow": SHADOW},
        transition="all .3s cubic-bezier(0.25, 0.8, 0.25, 1)",
        animation=f"glide .5s ease-out {0.06 * index}s both",
    )


def favourite_apps_row() -> rx.Component:
    apps = [
        {"name": "TV", "color": "#0055A4"},
        {"name": "Netflix", "color": "#E50914"},
        {"name": "Prime Video", "color": "#00A8E1"},
        {"name": "Disney+", "color": "#113CCF"},
        {"name": "HBO", "color": "#663399"},
        {"name": "Apple TV", "color": "#2B2B2E"},
        {"name": "YouTube", "color": "#FF0000"},
        {"name": "Twitch", "color": "#9146FF"},
    ]
    return rx.vstack(
        caption("Tus apps"),
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
        rx.hstack(
            rx.box(width="7px", height="7px", border_radius="50%", bg=PINK),
            rx.text("KinoFiles AI", font_size="0.8rem", font_weight="500", color=PINK),
            spacing="2",
            align="center",
            bg=PINK_SOFT,
            padding="0.45em 0.9em",
            border_radius="999px",
        ),
        rx.heading(
            "Dime qué te apetece.",
            size="9",
            weight="bold",
            letter_spacing="-0.035em",
            line_height="1.05",
        ),
        rx.text(
            "Tu asistente para explorar el cine: háblale de tu ánimo, de una noche "
            "concreta o de una película que te marcó, y te propondrá qué ver ahora.",
            max_width="42ch",
            font_size="1.05rem",
            color=MUTED,
            line_height="1.6",
        ),
        rx.button(
            rx.hstack(
                rx.icon("mic", size=17),
                rx.text("Lanzar agente", font_weight="600"),
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
    )


app = rx.App(
    style=style,
    api_transformer=voice_api,
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
)
app.add_page(index, title="KinoFiles OS")
app.add_page(agent_tv_panel, route="/agent", title="KinoFiles Agent TV")
