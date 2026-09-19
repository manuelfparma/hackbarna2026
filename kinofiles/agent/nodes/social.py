"""Package social messages for the shared conversational reply layer."""


class SocialHandler:
    def handle(self, request: str) -> dict:
        """Return social context without generating presentation copy here."""
        return {
            "kind": "social",
            "message": request,
            "titles": [],
            "error": None,
        }
