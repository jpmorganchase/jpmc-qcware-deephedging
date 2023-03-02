from typing import Dict


class TranslationException(Exception):
    def __init__(self, audit: Dict):
        self.audit = audit
        super().__init__(f"Audit: {str(audit)}")
