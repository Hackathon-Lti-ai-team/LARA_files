# session.py
# Session-level identity and conversation state

class SessionState:
    def __init__(self):
        self.reset()

    def reset(self):
        # Identity
        self.user_name = None
        self.identity_known = False
        self.allow_name_usage = False

        # Registration flow
        self.asked_registration = False

        # Conversation metadata (extensible)
        self.turn_count = 0

    # -------- Identity management --------
    def set_known_user(self, name: str):
        self.user_name = name
        self.identity_known = True
        self.allow_name_usage = True
        self.asked_registration = True

    def set_anonymous(self):
        self.user_name = None
        self.identity_known = False
        self.allow_name_usage = False
        self.asked_registration = True

    def set_declined_registration(self):
        self.set_anonymous()

    # -------- Conversation helpers --------
    def next_turn(self):
        self.turn_count += 1
        return self.turn_count

    def llm_identity_context(self) -> str:
        """
        Deterministic identity instruction for the LLM.
        Never allows hallucinated names.
        """
        if self.user_name and self.allow_name_usage:
            return f"The user's name is {self.user_name}. Use it naturally and sparingly."
        return "The user's name is unknown. Never invent, guess, or assume a name."
