from typing import Any, Dict, List, Text, Optional

import re
import logging

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.types import DomainDict


# =========================================================
# Logger
# =========================================================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================================================
# Action: Final COVID Score
# =========================================================
class ActionFinalScore(Action):
    """Calcula o score final de probabilidade de COVID-19
    com base nas respostas do formulário.
    """

    def name(self) -> Text:
        return "action_final_score"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        try:
            # =========================
            # Recupera slots (TEXT)
            # =========================
            cold = tracker.get_slot("cold") == "yes"
            fever = tracker.get_slot("fever") == "yes"
            cough = tracker.get_slot("cough") == "yes"
            travel = tracker.get_slot("travel") == "yes"

            isolation_raw = tracker.get_slot("isolation_level")

            try:
                isolation_level = int(isolation_raw)
            except (TypeError, ValueError):
                isolation_level = 5  # pior cenário

            logger.info(
                f"Slots recebidos | cold={cold}, fever={fever}, "
                f"cough={cough}, travel={travel}, isolation={isolation_level}"
            )

            # =========================
            # Score individual
            # =========================
            score = 0.0

            if cold:
                score += 0.2
            if fever:
                score += 0.2
            if cough:
                score += 0.2
            if travel:
                score += 0.2

            # isolamento: quanto menor, maior risco
            isolation_score = (6 - isolation_level) * 0.04
            score += isolation_score

            # limita entre 0 e 1
            score = min(max(score, 0.0), 1.0)

            percentage = int(score * 100)

            # =========================
            # Resposta final
            # =========================
            dispatcher.utter_message(
                text=f"There is an estimated {percentage}% probability that you may have COVID-19."
            )

            logger.info(f"Score final calculado: {percentage}%")

        except Exception as e:
            logger.exception("Erro ao calcular score final")
            dispatcher.utter_message(
                text="Sorry, something went wrong while calculating your diagnosis."
            )

        return []


# =========================================================
# Form Validation
# =========================================================
class ValidateCovidForm(FormValidationAction):
    """Validação dos slots do formulário covid_form"""

    def name(self) -> Text:
        return "validate_covid_form"

    async def required_slots(
        self,
        slots_mapped_in_domain: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Optional[List[Text]]:
        """Define explicitamente a ordem dos slots"""
        return [
            "cold",
            "fever",
            "cough",
            "travel",
            "isolation_level",
        ]

    # -----------------------------------------------------
    # Validação isolamento
    # -----------------------------------------------------
    def validate_isolation_level(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Valida se isolamento está entre 1 e 5"""

        try:
            value = int(re.findall(r"\d+", str(slot_value))[0])
        except Exception:
            value = None

        if value and 1 <= value <= 5:
            return {"isolation_level": value}

        dispatcher.utter_message(response="utter_wrong_isolation_level")
        return {"isolation_level": None}
