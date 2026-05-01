import pandas as pd


class IOput_Manager:
    def __init__(self):
        self.winChance = 0.0
        self.cardsDF = pd.DataFrame(columns=["card", "place"])

    def inputNewCard(self, card: int, place: int):
        """Inputs a single card into the dataframe."""
        if place < 0 or place >= 52:
            print("Card position must be between 0 and 51")
            return False

        if card in self.cardsDF["card"].values:
            print("Card already in list")
            return False

        if place in self.cardsDF["place"].values:
            print("Card already in that spot")
            return False

        new_row = pd.DataFrame({
            "card": [card],
            "place": [place]
        })

        self.cardsDF = pd.concat(
            [self.cardsDF, new_row],
            ignore_index=True
        )
        return True

    def parseCardInput(self, card_input: str) -> int:
        """Parse card input as either numeric ID or rank+suit."""
        token = card_input.strip().upper()
        if not token:
            raise ValueError("Input cannot be empty.")

        if token.isdigit():
            card = int(token)
            if 1 <= card <= 52:
                return card
            raise ValueError("Numeric card must be between 1 and 52.")

        token = token.replace(" ", "")
        if len(token) < 2:
            raise ValueError("Rank and suit required, e.g. 'K S'.")

        suit_map = {
            'S': 0,
            'C': 1,
            'H': 2,
            'D': 3,
        }
        rank_map = {
            'A': 0,
            '2': 1,
            '3': 2,
            '4': 3,
            '5': 4,
            '6': 5,
            '7': 6,
            '8': 7,
            '9': 8,
            '10': 9,
            'J': 10,
            'Q': 11,
            'K': 12,
        }

        suit = token[-1]
        rank = token[:-1]

        if suit not in suit_map:
            raise ValueError("Suit must be one of S, C, H, D.")
        if rank not in rank_map:
            raise ValueError("Rank must be A, 2-10, J, Q, or K.")

        return suit_map[suit] * 13 + rank_map[rank] + 1

    def singleInput(self, i: int):
        """Reads a single card from user input and stores it."""
        while True:
            cardInput = input(
                f"Card #{i} (1-52 or rank+suit, e.g. 2 H, K S): "
            )
            try:
                card = self.parseCardInput(cardInput)
            except ValueError as exc:
                print("Error:", exc)
                continue

            if self.inputNewCard(card, i):
                break

    def inputHandler(self):
        """Handles card entry in the same order as test_input.py."""
        print("Please input the first 7 face up cards left to right")
        print("Format: use the numeric value used in the dataset")

        indexes = [0, 7, 15, 18, 22, 25, 27]
        for index in indexes:
            self.singleInput(index)

        print("Please input the remaining cards (28-51)")
        print("Format: use the numeric value used in the dataset")

        for index in range(28, 52):
            self.singleInput(index)

    def printWinChance(self, confidence, prediction):
        """Prints a formatted win chance from model output."""
        value = 0.0
        if hasattr(confidence, "__len__") and len(confidence) > 0:
            try:
                if hasattr(confidence[0], "__len__"):
                    value = max(confidence[0])
                else:
                    value = float(confidence[0])
            except Exception:
                value = float(confidence)
        else:
            value = float(confidence)

        if hasattr(prediction, "__len__") and not isinstance(prediction, str):
            prediction = prediction[0]
        if prediction:
            prediction = "winning"
        else:            prediction = "losing"
        print(f"{round(value * 100, 2)}% chance of {prediction}")

    def printCardsList(self):
        """Prints the stored cards sorted by position."""
        if self.cardsDF.empty:
            print("No cards have been entered.")
            return

        print(self.cardsDF.sort_values("place").reset_index(drop=True))

    def getDFCardsList(self):
        """Returns a single-row DataFrame with x0..x51 columns."""
        row = {f"x{i}": 0 for i in range(52)}
        for _, card_row in self.cardsDF.iterrows():
            try:
                place = int(card_row["place"])
                card = int(card_row["card"])
            except (ValueError, TypeError):
                continue

            if 0 <= place < 52:
                row[f"x{place}"] = card

        return pd.DataFrame([row])
