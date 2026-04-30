import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler as StdS
from sklearn.neighbors import KNeighborsClassifier as KNN


def main() -> None:
    data = pd.read_csv("./winning_deck_results.csv", index_col=0)
    data['won'] = data['won'].str.strip().map(
        {'True': 1, 'False': 0}).astype(int)
    y = data['won']
    known = pd.concat(
        [data.iloc[:, [2, 9, 17, 20, 24, 27, 29]],
         data.iloc[:, 30:]], axis=1
    )
    known.columns = list(range(known.shape[1]))
    # Feature engineering
    X = remap_features(data, known)
    # Pipeline = scaling + model (prevents leakage too)
    model = make_pipeline(StdS(), KNN(n_neighbors=71))

    model.fit(X, y)  # type: ignore
    acc = model.score(X, y)  # type: ignore
    print(f"Accuracy: {acc*100:.2f}%")

    # Put user input here
    pred = model.predict(user_input)[0]  # type: ignore
    prob = model.predict_proba(user_input)[0][pred]  # type: ignore
    print(f"{prob * 100:.2f}% chance of {'winning' if pred else 'losing'}")


def remap_features(data: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    known = known.iloc[:, 2:].fillna(0)

    card_ids = data.to_numpy(dtype=int)
    mask = card_ids > 0

    card_ids = np.where(mask, card_ids - 1, 0)
    suits = card_ids // 13
    ranks = card_ids % 13
    colors = (suits >= 2).astype(int)

    # Handle unknowns
    suits[~mask] = -1
    colors[~mask] = -1
    ranks[~mask] = 0

    # Build feature dataframe (vectorized)
    features = pd.concat(
        [
            pd.DataFrame({
                f"{col}_rank": ranks[:, i],
                f"{col}_suit": suits[:, i],
                f"{col}_color": colors[:, i],
            })
            for i, col in enumerate(known.columns)
        ],
        axis=1
    )

    # Vectorized move count (faster than nested loops)
    features["num_valid_moves"] = [
        count_valid_moves(r, c) for r, c in zip(ranks, colors)
    ]

    return features


def count_valid_moves(ranks: np.ndarray, colors: np.ndarray):
    return sum(
        1
        for i in range(len(ranks))  # type: ignore
        for j in range(len(ranks))  # type: ignore
        if i != j
        and ranks[i] > 0
        and ranks[j] > 0
        and colors[i] != colors[j]
        and ranks[i] == ranks[j] - 1
    )


if __name__ == '__main__':
    main()
